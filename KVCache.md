# KV cache

在做大模型注意力计算时，因为 $K$ 都需要做[位置编码（Rotary Position Embedding）](./RotaryEmbedding.md)计算， 我们可以将公式中的 $K/V$ 数据缓存起来，提高计算效率。

- 场景一：预测序列 sequence 的下一个token

大模型计算注意力的公式，

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \cdot V
$$

在模型推理阶段，前向传播时，当前sequence已经生成出来的最后一个token的 $Q$ 需要跟seq前面的每一个token的 $K$ 以及 $V$ 进行计算。如果每生成一个新的token就把这个seq已经生成的token的 $K V$ 都算一遍，将是极大的计算浪费。因为这里的token会重复计算多次。

举例：当前已经推理出来的seq是 [12, 34, 178, 19], 这里用一个整数表示vocabulary里的一个token id。在预测下一个token时，将19的 $Q$ 跟 $12, 34, 178$ 的 $K/V$ 做运算。假定新的token的id是 17， 我们得到seq 为 [12, 34, 178, 19, 17]。 继续预测下一个token时，我们需要将17的 $Q$ 跟 $12, 34, 178, 19$ 的 $K/V$ 做运算。显然，token $12, 34, 178$ 的 $K/V$ 需要重复计算两次。

由此， 我们可以提前将seq里的每个token的 $K/V$ 保存起来，每生成一个新的token，就将token的 $K/V$ 加到前面token的缓存中。预测下一个token的时候，就可以直接拿来用了。

- 场景二：根据一段提示词，同时生成多个回答。

此时，就可以提前将提示词（prompt token）的 $K/V$ 缓存起来，然后复制多份（生成回答的数量）。这样每一个回答都可以直接使用相同的 $K/V$ 缓存。这种策略一般称为预填充(Prefill)。

## 实现方式

`karpathy/nanochat` 在文件`nanochat/engine.py`中，定义了一个 `KVCache` 类:

```python

class KVCache:
    """
    这个类的写法，是为了适配 Flash Attention 3的flash_attn_with_kvcache 接口
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        # 提前分配好缓存的张量: (n_layers, B, T, H, D)
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        # 对于每个batch，seq已经生成好的长度(FA3 需要int32)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # 这个是karpathy的一个测试策略，可以先不用管
        self.prev_embedding = None
    
    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) views for a specific layer."""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]
    
    def prefill(self, other):
        """
         从另一个缓存中复制数据到当前缓存中。主要用于先做batch=1的prefill，然后并行生成多个回答结果。
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)
```

这个类用来存储缓存信息。同时提供了读取缓存跟prefill缓存的方法。这个类没有提供写入缓存的方法。

写入缓存在 FlashAttention 接口内部实现。见 `nanochat/flash_attention.py`,

```python
    if USE_FA3:
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )
```

这里传入的参数 k_cache, v_cache 直接对应了上面 KVCache 类内部的缓存的地址。FlashAttention 内部直接更新对应的cache。

调用路径在 `nanochat/gpt.py` 中的类 `class CausalSelfAttention(nn.Module):` 的前向传播中

```python
k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
y = flash_attn.flash_attn_with_kvcache(
    q, k_cache, v_cache,
    k=k, v=v,
    cache_seqlens=kv_cache.cache_seqlens,
    causal=True,
    window_size=window_size,
)
```

推理路径的完整调用在 `nanochat/engine.py` 中

```python
# 1) 运行 batch = 1的提示词的预填充（prefill）
m = self.model.config
kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
kv_cache_prefill = KVCache(
    batch_size=1,
    seq_len=len(tokens),
    device=device,
    dtype=dtype,
    **kv_model_kwargs,
)
ids = torch.tensor([tokens], dtype=torch.long, device=device)
logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)

# 2) 对每个回答样本复制KV缓存
kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
kv_cache_decode = KVCache(
    batch_size=num_samples,
    seq_len=kv_length_hint,
    device=device,
    dtype=dtype,
    **kv_model_kwargs,
)
kv_cache_decode.prefill(kv_cache_prefill)
del kv_cache_prefill # no need to keep this memory around
```
