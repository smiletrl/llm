# 预训练参数与分布式步数

训练入口类似

```bash
python -m scripts.base_train \
  --depth=4 \
  --max-seq-len=512 \
  --device-batch-size=4 \
  --eval-tokens=512 \
  --core-metric-every=-1 \
  --total-batch-size=2048 \
  --num-iterations=1 \
  --eval-every=-1 
```

文件 `scripts/base_train.py` 提供了大量可选的训练参数，比如上例中的 `--depth`, `--max-seq-len`。了解这些参数，方便理解模型训练流程。

*注：在上述单卡测试配置下，单次 `device-batch-size * max-seq-len` (Micro-batch) 的 Token 刚好等于 total-batch-size，此时 grad_accum_steps = 1，不需要做梯度累积直接更新权重。*

## 分布式训练步数

**梯度累积步数（Gradient Accumulation Steps）， `grad_accum_steps`**，指为了更新一次模型权重（单次 Optimizer Step），需要让硬件凑够多少次前向/反向传播。

```python
# 计算梯度累积步数
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # 单个设备/GPU 一次能处理的token数量
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # 所有的设备/GPU 一次能处理的token数量
assert total_batch_size % world_tokens_per_fwdbwd == 0, f"total_batch_size ({total_batch_size}) must be a multiple of {world_tokens_per_fwdbwd}."
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
```

**各参数概念校准**

* **`max_seq_len`**：输入模型的一个序列（Context Window）的最大 token 数。通常并不是一句话，而是由多篇文章/对话通过 EOS 拼接打包含满该长度的 token 序列。
* **`device_batch_size`（又称 Micro-batch Size）**：单张 GPU 在单次 Forward/Backward 操作中实际处理的序列数量。如果显存 OOM，通常优先调小此项。
* **`tokens_per_fwdbwd`**：单张卡单次前向/反向处理的 Token 数量（ $device\_batch\_size \times max\_seq\_len$ ）。
* **`world_tokens_per_fwdbwd`**：当前集群所有参与训练的 GPU（World Size）同时做一次前向/反向所能消耗的总 Token 数量。
* **`total_batch_size`**：**模型每做一次参数更新（One Optimizer Step）所需要消耗的目标 Token 总量**（业界常称 Global Batch Size）。例如训练 LLaMA 或 GPT-3 时，目标 Global Batch Size 可能是 200 万 或 400 万 tokens。**不是**指全部训练语料的总量。整套语料（Corpus）的总量在代码中对应的是 `total_tokens`。
* **`num_iterations`**：整个训练任务实际执行的模型参数更新总步数（Optimizer Steps）。每一步代表集群执行完一个完整的 Global Batch 训练并执行一次优化器更新。如果显式配置（例如 `--num-iterations=1`），脚本将直接运行指定步数（非常适合本地或小规模联调）；若未显式配置，脚本通常会依据 Scaling Laws（如 `--target-param-data-ratio` 或 `--target-flops`）自动计算出最优的迭代步数。

**梯度累积步数**
* **`grad_accum_steps`**：**单次优化器更新之前，每张 GPU 需要连续做多少次前向与反向传播来累加梯度**。**不是**指跑完所有语料需要的总步数。

---

**核心计算关系**

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
