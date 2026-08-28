# RotaryEmbedding

旋转位置编码取代Google 2017《Attention Is All You Need》论文中的绝对位置编码，采用相对位置的编码方式，是当前大模型训练的位置编码的常见训练方式。

Nanochat中的步骤，主要在文件`nanochat/gpt.py`中

- 预先计算出不同旋转角度的正弦余弦值
- 旋转各自的Q，K向量，计算旋转后Q*K的点积。

## References

- [苏剑林. (Mar. 23, 2021). 《Transformer升级之路：2、博采众长的旋转式位置编码 》](https://spaces.ac.cn/archives/8265)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
