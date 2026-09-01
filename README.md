# nanochat-guide

不提供代码。对照 [karpathy/nanochat](https://github.com/karpathy/nanochat)，精读一份最小可行大模型：从分词、预训练、对齐，到推理和对话。

## 这不是什么

- 不是 nanochat 的 fork，也不维护一份代码副本
- 不是从 NLP 讲起的系统课
- 不是 vLLM / SGLang 生产系统教程

代码只在上游。这里只讲：这段代码在全链路的哪一步、为什么这样写。

## 怎么读

1. Clone 上游，不必先跑完 `runs/speedrun.sh`：

   ```bash
   git clone https://github.com/karpathy/nanochat.git
   ```

2. 按下面顺序读。已完成的章节链到正文，未写的只标对应文件。
3. 读设计即可，不要求 8×H100。

建议有 Transformer 和 PyTorch 基础。

## 目录

状态：✅ 已有正文 · 🚧 草稿 · ⏳ 未写

| # | 章节 | 状态 | 上游主要文件 |
|---|---|---|---|
| 0 | 仓库地图：一次训练到能聊天经过哪些阶段 | ⏳ | `runs/speedrun.sh`，`scripts/` |
| 1 | Tokenizer | ⏳ | `scripts/tok_train.py`，`scripts/tok_eval.py`，`nanochat/tokenizer.py` |
| 2 | 模型结构 | ⏳ | `nanochat/gpt.py` |
| 3 | [旋转位置编码 RoPE](RotaryEmbedding.md) | ✅ | `nanochat/gpt.py` |
| 4 | [FlashAttention](FlashAttention.md) | 🚧 | `nanochat/flash_attention.py` |
| 5 | [KV Cache 与推理引擎](KVCache.md) | ✅ | `nanochat/engine.py`，`nanochat/gpt.py` |
| 6 | 数据与预训练循环 | ⏳ | `nanochat/dataset.py`，`nanochat/dataloader.py`，`scripts/base_train.py` |
| 7 | 优化器、精度与 checkpoint | ⏳ | `nanochat/optim.py`，`nanochat/common.py`，`nanochat/checkpoint_manager.py` |
| 8 | [FP8](FP8.md) | ✅ | `nanochat/fp8.py` |
| 9 | SFT / 对话对齐 | ⏳ | `scripts/chat_sft.py`，`tasks/` |
| 10 | 强化学习 | ⏳ | `scripts/chat_rl.py` |
| 11 | 评测 | ⏳ | `scripts/base_eval.py`，`scripts/chat_eval.py`，`nanochat/core_eval.py`，`tasks/` |
| 12 | 对话入口 | ⏳ | `scripts/chat_cli.py` |

已有四篇按主题写成，尚未按上表重排。目录是目标结构，不是已完成篇幅。

## 进度

进行中。已完成 RoPE、KV Cache、FP8；FlashAttention 为草稿。

每章固定四块：问题是什么、最小直觉、nanochat 里对应哪几行、工业系统通常还会多什么。最后一块目前可能很短。

## 上游

- 代码：[karpathy/nanochat](https://github.com/karpathy/nanochat)
- 原作者说明：[Discussions](https://github.com/karpathy/nanochat/discussions)

本仓库与 Andrej Karpathy / nanochat 没有官方关系。