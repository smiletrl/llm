# 预训练参数与分布式步数

训练入口类似

```bash
python -m scripts.base_train \
  --depth=4 \
  --max-seq-len=512 \
  --device-batch-size=4 \
  --eval-tokens=512 \
  --core-metric-every=-1 \
  --total-batch-size=4096 \
  --num-iterations=1 \
  --eval-every=-1 
```

文件 `scripts/base_train.py` 提供了大量可选的训练参数，比如上例中的 `--depth`, `--max-seq-len`。了解这些参数，方便理解模型训练流程。

*注：在上述单卡测试配置下，单次 `total-batch-size` 是 `device-batch-size * max-seq-len` (Micro-batch) 的 Token 数量2倍 ，此时 grad_accum_steps = 2，需要做梯度累积两次才能更新权重。*

## 分布式训练步骤

**梯度累积步数（Gradient Accumulation Steps）， `grad_accum_steps`**，指为了更新一次模型权重（单次 Optimizer Step），需要让硬件凑够多少次前向/反向传播。

```python
# 计算梯度累积步数
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # 单个设备/GPU 一次能处理的token数量
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # 所有的设备/GPU 一次能处理的token数量
assert total_batch_size % world_tokens_per_fwdbwd == 0, f"total_batch_size ({total_batch_size}) must be a multiple of {world_tokens_per_fwdbwd}."
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
```

**各参数概念校准**

* **`max_seq_len`**：即大模型的最大上下文长度（Context Window / 上下文窗口）。也就是日常大家讨论“某模型支持 128K / 1M 上下文”时所指的核心指标。在代码与显卡层面，它决定了注意力矩阵（Attention Matrix）在时间维度上的最大跨度。在预训练阶段，为了极致压榨算力，通常不是只塞一句话，而是由多篇文章或对话通过 `<|eos|>` 等特殊字符紧密拼接打包，尽量塞满整个窗口。
* **`device_batch_size`（又称 Micro-batch Size）**：单张 GPU 在单次 Forward/Backward 操作中实际处理的序列数量。如果显存不够（OOM），通常优先调小此项。
* **`tokens_per_fwdbwd`**：单张卡单次前向/反向处理的 Token 数量（ $device\_batch\_size \times max\_seq\_len$ ）。
* **`world_tokens_per_fwdbwd`**：当前集群所有参与训练的 GPU（World Size）同时做一次前向/反向所能消耗的总 Token 数量。做完一次前向跟反向传播计算，仅仅是累积梯度，没有做优化器的参数更新。
* **`total_batch_size`**：**模型每做一次参数更新（One Optimizer Step）所需要消耗的目标 Token 总量**（业界常称 Global Batch Size）。例如训练 LLaMA 或 GPT-3 时，目标 Global Batch Size 可能是 200 万 或 400 万 tokens。**不是**指全部训练语料的总量。整套语料（Corpus）的总量在代码中对应的是 `total_tokens`。
* **`num_iterations`**：整个训练任务实际执行的模型参数更新总步数（Optimizer Steps）。每一步代表集群执行完一个完整的 Global Batch 训练并执行一次优化器更新。如果显式配置（例如 `--num-iterations=1`），脚本将直接运行指定步数（非常适合本地或小规模联调）；若未显式配置，脚本通常会依据 Scaling Laws（如 `--target-param-data-ratio` 或 `--target-flops`）自动计算出最优的迭代步数。

### 核心计算关系

大模型训练受限于 GPU 显存，无法直接把几百万 token 的 Global Batch 一次性塞进显卡，因此拆解为：

$$\text{Global Batch Size} = \text{Micro Batch Size} \times \text{Sequence Length} \times \text{World Size (GPU数量)} \times \text{Gradient Accumulation Steps}$$

在代码中的样例体现为：

```python
# 每次权重更新目标消费 524,288 tokens (total_batch_size)
# 假设 8 卡 A100，每卡 micro_batch=4，seq_len=2048
# tokens_per_fwdbwd = 4 * 2048 = 8,192
# world_tokens_per_fwdbwd = 8,192 * 8 = 65,536 (8卡一次 fwd/bwd 吞吐 6.5w tokens)
# grad_accum_steps = 524,288 // 65,536 = 8
```

这意味着集群所有卡需要连续执行 **8 次** Micro-step 的 Forward + Backward（把梯度累加起来），然后调用一次 `optimizer.step()` 更新模型参数。

### 训练循环流程

`scripts/base_train.py` 里的核心循环结构

```python
# 1. 连续循环 grad_accum_steps 次，只做前向和反向计算，累积梯度
for micro_step in range(grad_accum_steps):
    loss = model(x, y)
    loss = loss / grad_accum_steps  # 梯度累加前做均值归一化
    loss.backward()                 # 这里只往 .grad 里叠加梯度，完全没有更新模型参数！
    x, y = next(train_loader)

# 2. 只有当上面所有微步（Micro-steps）跑完、凑齐了目标 Token 数量后，才做真正的一步更新：
optimizer.step()     # 这一行才会真正根据累加好的总梯度去修改模型权重 W
model.zero_grad()    # 把累加的梯度清零，迎接下一个 Global Batch
```

为什么需要**梯度累积步数（Gradient Accumulation Steps）`grad_accum_steps`**：

1. **显存装不下（硬件限制）**：
研究发现，大模型训练要想稳定收敛，每次参数更新往往需要几百万 Token 的“大视野”（Global Batch Size，比如 50 万或 400 万 tokens）。
但显卡显存有限，单次 Forward/Backward（`world_tokens_per_fwdbwd`）如果塞几百万 Token 直接就 OOM 爆显存了。所以把这个大 Batch 拆成多次小批次，**显存每次只跑一小块，但数学上把它们的梯度加起来，等效于一次性看了一个巨大的 Batch**。

2. **通信开销太高（性能限制）**：
在分布式多卡训练（DDP）中，每次更新参数前，多张卡之间必须通过网络把各自算出的梯度同步（All-Reduce）。如果每做一个微小的 Forward/Backward 就做一次跨网络同步并更新参数，GPU 大部分时间都会干等着网卡传数据。
通过累积多次反向传播再更新，可以大幅减少同步频率，显著提升训练吞吐率。
