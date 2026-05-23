import math
import numpy as np
import torch
import torch.nn.functional as F

# Attempt MindSpore imports; fallback to stubs if unavailable
try:
    import mindspore
    from mindspore import nn as ms_nn
    from mindspore import ops as ms_ops
    HAS_MINDSPORE = True
except Exception:
    HAS_MINDSPORE = False
    mindspore = None
    ms_nn = None
    ms_ops = None


# -----------------------------------------------------------------------------
# PTA-side custom implementations
# -----------------------------------------------------------------------------

class _RMSNorm_PT(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed * self.weight


class _SwiGLU_PT(torch.nn.Module):
    def __init__(self, hidden: int, ffn: int):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, ffn, bias=False)
        self.up_proj = torch.nn.Linear(hidden, ffn, bias=False)
        self.down_proj = torch.nn.Linear(ffn, hidden, bias=False)
        self.act = torch.nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class _RoPE_PT(torch.nn.Module):
    def __init__(self, dim: int, max_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len

    def forward(self, x, seq_len: int = None):
        if seq_len is None:
            # Input convention: (seq_len, batch, hidden) -> seq_len is dim 0
            seq_len = x.shape[0]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


class _ALiBi_PT(torch.nn.Module):
    def __init__(self, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, x):
        # ALiBi: add a bias matrix based on distance
        seq_len = x.shape[-2] if x.dim() >= 3 else x.shape[0]
        bias = torch.arange(seq_len, device=x.device).unsqueeze(0) - torch.arange(seq_len, device=x.device).unsqueeze(1)
        # scale per head
        slopes = torch.tensor([2 ** (-8 * (i + 1) / self.num_heads) for i in range(self.num_heads)], device=x.device)
        bias = bias.unsqueeze(0).unsqueeze(0) * slopes.view(1, self.num_heads, 1, 1)
        return x + bias


class _ScaledDotProductAttention_PT(torch.nn.Module):
    def __init__(self, dropout_p: float = 0.0):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, q, k, v, attn_mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, v)


class _TopKGating_PT(torch.nn.Module):
    def __init__(self, hidden: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.router = torch.nn.Linear(hidden, num_experts, bias=False)
        self.top_k = top_k
        self.num_experts = num_experts

    def forward(self, x):
        router_logits = self.router(x)
        weights, selected_experts = torch.topk(torch.softmax(router_logits, dim=-1), self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights, selected_experts


class _MLA_Q_Projection_PT(torch.nn.Module):
    def __init__(self, hidden: int, q_lora_rank: int, num_heads: int, head_dim: int):
        super().__init__()
        self.q_down = torch.nn.Linear(hidden, q_lora_rank, bias=False)
        self.q_up = torch.nn.Linear(q_lora_rank, num_heads * head_dim, bias=False)

    def forward(self, x):
        return self.q_up(self.q_down(x))


class _MLA_KV_Projection_PT(torch.nn.Module):
    def __init__(self, hidden: int, kv_lora_rank: int, num_heads: int, head_dim: int):
        super().__init__()
        self.kv_down = torch.nn.Linear(hidden, kv_lora_rank, bias=False)
        self.k_up = torch.nn.Linear(kv_lora_rank, num_heads * head_dim, bias=False)
        self.v_up = torch.nn.Linear(kv_lora_rank, num_heads * head_dim, bias=False)

    def forward(self, x):
        compressed = self.kv_down(x)
        return self.k_up(compressed), self.v_up(compressed)


# -----------------------------------------------------------------------------
# MSA-side custom implementations (when MindSpore is available)
# -----------------------------------------------------------------------------

if HAS_MINDSPORE:
    class _RMSNorm_MS(ms_nn.Cell):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = mindspore.Parameter(ms_ops.ones((dim,), mindspore.float32))

        def construct(self, x):
            normed = x * ms_ops.rsqrt(ms_ops.pow(x, 2).mean(-1, keepdim=True) + self.eps)
            return normed * self.weight

    class _SwiGLU_MS(ms_nn.Cell):
        def __init__(self, hidden: int, ffn: int):
            super().__init__()
            self.gate_proj = ms_nn.Dense(hidden, ffn, has_bias=False)
            self.up_proj = ms_nn.Dense(hidden, ffn, has_bias=False)
            self.down_proj = ms_nn.Dense(ffn, hidden, has_bias=False)
            self.act = ms_nn.SiLU()

        def construct(self, x):
            return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))

    class _ScaledDotProductAttention_MS(ms_nn.Cell):
        def __init__(self, dropout_p: float = 0.0):
            super().__init__()
            self.dropout = ms_nn.Dropout(keep_prob=1.0 - dropout_p)

        def construct(self, q, k, v, attn_mask=None):
            d_k = q.shape[-1]
            scores = ms_ops.matmul(q, k.swapaxes(-2, -1)) / math.sqrt(d_k)
            if attn_mask is not None:
                scores = ms_ops.masked_fill(scores, attn_mask == 0, float('-inf'))
            attn = ms_ops.softmax(scores, axis=-1)
            attn = self.dropout(attn)
            return ms_ops.matmul(attn, v)

    class _TopKGating_MS(ms_nn.Cell):
        def __init__(self, hidden: int, num_experts: int, top_k: int = 2):
            super().__init__()
            self.router = ms_nn.Dense(hidden, num_experts, has_bias=False)
            self.top_k = top_k
            self.num_experts = num_experts

        def construct(self, x):
            router_logits = self.router(x)
            weights, selected_experts = ms_ops.top_k(ms_ops.softmax(router_logits, axis=-1), self.top_k)
            weights = weights / weights.sum(axis=-1, keepdims=True)
            return weights, selected_experts

    class _MLA_Q_Projection_MS(ms_nn.Cell):
        def __init__(self, hidden: int, q_lora_rank: int, num_heads: int, head_dim: int):
            super().__init__()
            self.q_down = ms_nn.Dense(hidden, q_lora_rank, has_bias=False)
            self.q_up = ms_nn.Dense(q_lora_rank, num_heads * head_dim, has_bias=False)

        def construct(self, x):
            return self.q_up(self.q_down(x))

    class _MLA_KV_Projection_MS(ms_nn.Cell):
        def __init__(self, hidden: int, kv_lora_rank: int, num_heads: int, head_dim: int):
            super().__init__()
            self.kv_down = ms_nn.Dense(hidden, kv_lora_rank, has_bias=False)
            self.k_up = ms_nn.Dense(kv_lora_rank, num_heads * head_dim, has_bias=False)
            self.v_up = ms_nn.Dense(kv_lora_rank, num_heads * head_dim, has_bias=False)

        def construct(self, x):
            compressed = self.kv_down(x)
            return self.k_up(compressed), self.v_up(compressed)

    class _RoPE_MS(ms_nn.Cell):
        def __init__(self, dim: int, max_len: int = 2048, base: float = 10000.0):
            super().__init__()
            inv_freq = 1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) / dim))
            self.inv_freq = mindspore.Tensor(inv_freq)
            self.max_len = max_len

        def construct(self, x, seq_len: int = None):
            if seq_len is None:
                # Input convention: (seq_len, batch, hidden) -> seq_len is dim 0
                seq_len = x.shape[0]
            t = ms_ops.arange(seq_len).astype(mindspore.float32).reshape(-1, 1)
            freqs = t * self.inv_freq.reshape(1, -1)
            emb = ms_ops.concat((freqs, freqs), axis=-1)
            cos = ms_ops.cos(emb)
            sin = ms_ops.sin(emb)
            return cos, sin

    class _ALiBi_MS(ms_nn.Cell):
        def __init__(self, num_heads: int = 8):
            super().__init__()
            self.num_heads = num_heads

        def construct(self, x):
            seq_len = x.shape[-2] if x.ndim >= 3 else x.shape[0]
            arange = ms_ops.arange(seq_len).astype(mindspore.float32)
            bias = ms_ops.expand_dims(arange, 0) - ms_ops.expand_dims(arange, 1)
            slopes = mindspore.Tensor([2 ** (-8 * (i + 1) / self.num_heads) for i in range(self.num_heads)], mindspore.float32)
            bias = ms_ops.expand_dims(ms_ops.expand_dims(bias, 0), 0) * ms_ops.expand_dims(ms_ops.expand_dims(slopes, 1), 2)
            return x + bias

    class _LayerNorm_MS(ms_nn.Cell):
        def __init__(self, shape, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.gamma = mindspore.Parameter(ms_ops.ones(shape, mindspore.float32))
            self.beta = mindspore.Parameter(ms_ops.zeros(shape, mindspore.float32))

        def construct(self, x):
            # numpy fallback to avoid CANN LayerNormV3
            x_np = x.asnumpy()
            mean = x_np.mean(axis=-1, keepdims=True)
            var = np.var(x_np, axis=-1, keepdims=True)
            normed = (x_np - mean) / np.sqrt(var + self.eps)
            gamma_np = self.gamma.asnumpy()
            beta_np = self.beta.asnumpy()
            out = normed * gamma_np + beta_np
            return mindspore.Tensor(out.astype(np.float32))

    def _cross_entropy_ms(input_t, target):
        # numpy fallback to avoid CANN OnesLike
        input_np = input_t.asnumpy()
        target_np = target.asnumpy().astype(np.int64)
        # numeric stability: subtract max
        shifted = input_np - input_np.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=-1, keepdims=True)
        n = input_np.shape[0]
        log_probs = -np.log(probs[np.arange(n), target_np] + 1e-8)
        return mindspore.Tensor(np.float32(log_probs.mean()))

    def _mean_ms(x, dim=-1, keepdim=True):
        # numpy fallback to avoid CANN ReduceMean
        return mindspore.Tensor(x.asnumpy().mean(axis=dim, keepdims=keepdim).astype(np.float32))

else:
    _RMSNorm_MS = None
    _SwiGLU_MS = None
    _ScaledDotProductAttention_MS = None
    _TopKGating_MS = None
    _MLA_Q_Projection_MS = None
    _MLA_KV_Projection_MS = None
    _RoPE_MS = None
    _ALiBi_MS = None
    _LayerNorm_MS = None
    _cross_entropy_ms = None
    _mean_ms = None


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

OPERATOR_REGISTRY = {
    "embedding": {
        "pta": lambda num_emb=40000, emb_dim=1024: torch.nn.Embedding(num_emb, emb_dim),
        "msa": lambda num_emb=40000, emb_dim=1024: ms_nn.Embedding(num_emb, emb_dim) if HAS_MINDSPORE else None,
        "input_shape": (64,),  # (seq_len,)
        "input_type": "int",
        "input_range": (0, 40000),
    },
    "layernorm": {
        "pta": lambda shape=(1024,), eps=1e-5: torch.nn.LayerNorm(shape, eps=eps),
        "msa": lambda shape=(1024,), eps=1e-5: _LayerNorm_MS(shape, eps) if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 1024),
    },
    "rmsnorm": {
        "pta": lambda dim=1024, eps=1e-6: _RMSNorm_PT(dim, eps),
        "msa": lambda dim=1024, eps=1e-6: _RMSNorm_MS(dim, eps) if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 1024),
    },
    "linear": {
        "pta": lambda in_f=1024, out_f=4096, bias=True: torch.nn.Linear(in_f, out_f, bias=bias),
        "msa": lambda in_c=1024, out_c=4096, bias=True: ms_nn.Dense(in_c, out_c, has_bias=bias) if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 1024),
    },
    "matmul": {
        "pta": lambda: lambda a, b: torch.matmul(a, b),
        "msa": lambda: lambda a, b: ms_ops.matmul(a, b) if HAS_MINDSPORE else None,
        "input_shape": ((32, 16, 64), (32, 64, 64)),  # two tensors
        "multi_input": True,
    },
    "scaled_dot_product_attention": {
        "pta": lambda dropout=0.0: _ScaledDotProductAttention_PT(dropout),
        "msa": lambda dropout=0.0: _ScaledDotProductAttention_MS(dropout) if HAS_MINDSPORE else None,
        "input_shape": ((32, 2, 16, 64), (32, 2, 16, 64), (32, 2, 16, 64)),
        "multi_input": True,
    },
    "flash_attention": {
        "pta": lambda: None,  # Placeholder; requires flash_attn package
        "msa": lambda: None,
        "input_shape": (32, 2, 1024),
        "skip": True,  # Skip if flash_attn not installed
    },
    "softmax": {
        "pta": lambda: torch.nn.Softmax(dim=-1),
        "msa": lambda: ms_nn.Softmax(axis=-1) if HAS_MINDSPORE else None,
        "input_shape": (32, 16, 2, 2),
    },
    "gelu": {
        "pta": lambda: torch.nn.GELU(approximate='tanh'),
        "msa": lambda: ms_nn.GELU(approximate=True) if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 4096),
    },
    "silu": {
        "pta": lambda: torch.nn.SiLU(),
        "msa": lambda: ms_nn.SiLU() if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 4096),
    },
    "swiglu": {
        "pta": lambda hidden=1024, ffn=4096: _SwiGLU_PT(hidden, ffn),
        "msa": lambda hidden=1024, ffn=4096: _SwiGLU_MS(hidden, ffn) if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 1024),
    },
    "rope": {
        "pta": lambda dim=128, max_len=2048, base=10000.0: _RoPE_PT(dim, max_len, base),
        "msa": lambda dim=128, max_len=2048, base=10000.0: _RoPE_MS(dim, max_len, base) if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 1024),
    },
    "alibi": {
        "pta": lambda num_heads=8: _ALiBi_PT(num_heads),
        "msa": lambda num_heads=8: _ALiBi_MS(num_heads) if HAS_MINDSPORE else None,
        "input_shape": (2, 8, 32, 32),  # (batch, heads, seq, seq) attention scores
    },
    "dropout": {
        "pta": lambda p=0.1: torch.nn.Dropout(p=p),
        "msa": lambda p=0.1: ms_nn.Dropout(keep_prob=1.0 - p) if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 1024),
    },
    "add": {
        "pta": lambda: lambda a, b: torch.add(a, b),
        "msa": lambda: lambda a, b: ms_ops.add(a, b) if HAS_MINDSPORE else None,
        "input_shape": ((32, 2, 1024), (32, 2, 1024)),
        "multi_input": True,
    },
    "mul": {
        "pta": lambda: lambda a, b: torch.mul(a, b),
        "msa": lambda: lambda a, b: ms_ops.mul(a, b) if HAS_MINDSPORE else None,
        "input_shape": ((32, 2, 1024), (32, 2, 1024)),
        "multi_input": True,
    },
    "cross_entropy": {
        "pta": lambda: lambda input_t, target: F.cross_entropy(input_t, target.long(), reduction='mean'),
        "msa": lambda: lambda input_t, target: _cross_entropy_ms(input_t, target) if HAS_MINDSPORE else None,
        "input_shape": ((64, 40000), (64,)),
        "multi_input": True,
        "input_types": ["float", "int"],
    },
    "topk_gating": {
        "pta": lambda hidden=1024, num_experts=4, top_k=2: _TopKGating_PT(hidden, num_experts, top_k),
        "msa": lambda hidden=1024, num_experts=4, top_k=2: _TopKGating_MS(hidden, num_experts, top_k) if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 1024),
    },
    "mla_q_projection": {
        "pta": lambda hidden=1024, q_lora_rank=64, num_heads=16, head_dim=64: _MLA_Q_Projection_PT(hidden, q_lora_rank, num_heads, head_dim),
        "msa": lambda hidden=1024, q_lora_rank=64, num_heads=16, head_dim=64: _MLA_Q_Projection_MS(hidden, q_lora_rank, num_heads, head_dim) if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 1024),
    },
    "mla_kv_projection": {
        "pta": lambda hidden=1024, kv_lora_rank=64, num_heads=16, head_dim=64: _MLA_KV_Projection_PT(hidden, kv_lora_rank, num_heads, head_dim),
        "msa": lambda hidden=1024, kv_lora_rank=64, num_heads=16, head_dim=64: _MLA_KV_Projection_MS(hidden, kv_lora_rank, num_heads, head_dim) if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 1024),
    },
    "transpose": {
        "pta": lambda: lambda x: torch.transpose(x, -2, -1),
        "msa": lambda: lambda x: ms_ops.swapaxes(x, -2, -1) if HAS_MINDSPORE else None,
        "input_shape": (32, 16, 2, 2),
    },
    "masked_fill": {
        "pta": lambda: lambda x, mask: x.masked_fill(mask == 0, -1e4),
        "msa": lambda: lambda x, mask: ms_ops.masked_fill(x, mask == 0, -1e4) if HAS_MINDSPORE else None,
        "input_shape": ((32, 16, 2, 2), (32, 16, 2, 2)),
        "multi_input": True,
        "input_types": ["float", "int"],
        "input_ranges": [(0, 50000), (0, 2)],  # mask: 0 or 1
    },
    "concat": {
        "pta": lambda dim=-1: lambda a, b: torch.cat((a, b), dim=dim),
        "msa": lambda dim=-1: lambda a, b: ms_ops.concat((a, b), axis=dim) if HAS_MINDSPORE else None,
        "input_shape": ((32, 2, 512), (32, 2, 512)),
        "multi_input": True,
    },
    "where": {
        "pta": lambda: lambda cond, x, y: torch.where(cond, x, y),
        "msa": lambda: lambda cond, x, y: ms_ops.where(cond, x, y) if HAS_MINDSPORE else None,
        "input_shape": ((32, 2, 1024), (32, 2, 1024), (32, 2, 1024)),
        "multi_input": True,
        "input_types": ["bool", "float", "float"],
    },
    "exp": {
        "pta": lambda: torch.exp,
        "msa": lambda: ms_ops.exp if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 1024),
    },
    "log": {
        "pta": lambda: lambda x: torch.log(torch.clamp(x, min=1e-4)),
        "msa": lambda: lambda x: ms_ops.log(ms_ops.clip_by_value(x, 1e-4, 1e9)) if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 1024),
    },
    "sqrt": {
        "pta": lambda: torch.sqrt,
        "msa": lambda: ms_ops.sqrt if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 1024),
    },
    "pow": {
        "pta": lambda: lambda x, y: torch.pow(x, y),
        "msa": lambda: lambda x, y: ms_ops.pow(x, y) if HAS_MINDSPORE else None,
        "input_shape": ((32, 2, 1024), (32, 2, 1024)),
        "multi_input": True,
    },
    "clamp": {
        "pta": lambda min_v=-1.0, max_v=1.0: lambda x: torch.clamp(x, min_v, max_v),
        "msa": lambda min_v=-1.0, max_v=1.0: lambda x: ms_ops.clip_by_value(x, min_v, max_v) if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 1024),
    },
    "split": {
        "pta": lambda split_size=512, dim=-1: lambda x: torch.split(x, split_size, dim),
        "msa": lambda split_size=512, dim=-1: lambda x: ms_ops.split(x, split_size, axis=dim) if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 1024),
    },
    "reshape": {
        "pta": lambda shape=(64, 1024): lambda x: torch.reshape(x, shape),
        "msa": lambda shape=(64, 1024): lambda x: ms_ops.reshape(x, shape) if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 1024),
    },
    "mean": {
        "pta": lambda dim=-1, keepdim=True: lambda x: torch.mean(x, dim, keepdim),
        "msa": lambda dim=-1, keepdim=True: lambda x: _mean_ms(x, dim, keepdim) if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 1024),
    },
    "stack": {
        "pta": lambda dim=0: lambda a, b: torch.stack((a, b), dim=dim),
        "msa": lambda dim=0: lambda a, b: ms_ops.stack((a, b), axis=dim) if HAS_MINDSPORE else None,
        "input_shape": ((32, 2, 1024), (32, 2, 1024)),
        "multi_input": True,
    },
    "expand": {
        "pta": lambda target_shape=(32, 2, 1024): lambda x: x.expand(target_shape),
        "msa": lambda target_shape=(32, 2, 1024): lambda x: ms_ops.broadcast_to(x, target_shape) if HAS_MINDSPORE else None,
        "input_shape": (1, 2, 1024),
    },
    "tril": {
        "pta": lambda diagonal=0: lambda x: torch.tril(x, diagonal),
        "msa": lambda diagonal=0: lambda x: mindspore.Tensor(np.tril(x.asnumpy(), k=diagonal)) if HAS_MINDSPORE else None,
        "input_shape": (32, 32),
    },
    "sum": {
        "pta": lambda dim=-1, keepdim=True: lambda x: torch.sum(x, dim, keepdim),
        "msa": lambda dim=-1, keepdim=True: lambda x: ms_ops.sum(x, dim, keepdim) if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 1024),
    },
    "max": {
        "pta": lambda dim=-1, keepdim=True: lambda x: torch.max(x, dim, keepdim).values,
        "msa": lambda dim=-1, keepdim=True: lambda x: ms_ops.max(x, dim, keepdim) if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 1024),
    },
    "argmax": {
        "pta": lambda dim=-1: lambda x: torch.argmax(x, dim),
        "msa": lambda dim=-1: lambda x: mindspore.Tensor(x.asnumpy().argmax(axis=dim)) if HAS_MINDSPORE else None,
        "input_shape": (32, 40000),
    },
    "abs": {
        "pta": lambda: torch.abs,
        "msa": lambda: ms_ops.abs if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 1024),
    },
    "rsqrt": {
        "pta": lambda: lambda x: torch.rsqrt(torch.clamp(x, min=1e-4)),
        "msa": lambda: lambda x: ms_ops.rsqrt(ms_ops.clip_by_value(x, 1e-4, 1e9)) if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 1024),
    },
    "eq": {
        "pta": lambda: lambda a, b: torch.eq(a, b),
        "msa": lambda: lambda a, b: ms_ops.equal(a, b) if HAS_MINDSPORE else None,
        "input_shape": ((32, 2, 1024), (32, 2, 1024)),
        "multi_input": True,
    },
    "gather": {
        "pta": lambda dim=-1: lambda x, index: torch.gather(x, dim, index),
        "msa": lambda dim=-1: lambda x, index: ms_ops.gather_elements(x, dim, index) if HAS_MINDSPORE else None,
        "input_shape": ((32, 40000), (32, 10)),
        "multi_input": True,
        "input_types": ["float", "int"],
        "input_ranges": [(0, 50000), (0, 40000)],
    },
    "pad": {
        "pta": lambda pad=(0, 2): lambda x: F.pad(x, pad),
        "msa": lambda pad=(0, 2): lambda x: ms_ops.pad(x, pad) if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 1024),
    },
    "sin": {
        "pta": lambda: torch.sin,
        "msa": lambda: ms_ops.sin if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 1024),
    },
    "cos": {
        "pta": lambda: torch.cos,
        "msa": lambda: ms_ops.cos if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 1024),
    },
    "permute": {
        "pta": lambda: lambda x: x.permute(0, 2, 1),
        "msa": lambda: lambda x: ms_ops.transpose(x, (0, 2, 1)) if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 1024),
    },
    "cumsum": {
        "pta": lambda dim=-1: lambda x: torch.cumsum(x, dim),
        "msa": lambda dim=-1: lambda x: mindspore.Tensor(x.asnumpy().cumsum(axis=dim)) if HAS_MINDSPORE else None,
        "input_shape": (32, 2, 1024),
    },
}


def get_operator_factory(op_name: str, backend: str):
    entry = OPERATOR_REGISTRY.get(op_name)
    if entry is None:
        raise ValueError(f"Unknown operator: {op_name}")
    factory = entry.get(backend)
    if factory is None:
        raise ValueError(f"No factory for {op_name} on {backend}")
    return factory, entry


def list_operators():
    return list(OPERATOR_REGISTRY.keys())
