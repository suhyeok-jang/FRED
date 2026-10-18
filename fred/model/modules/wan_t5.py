"""Wan umT5-XXL text encoder (``models_t5_umt5-xxl-enc-bf16.pth``) and its tokenizer."""
import html
import math
import os
import re
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

WAN_T5_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
WAN_T5_FILE = "models_t5_umt5-xxl-enc-bf16.pth"
WAN_T5_TOKENIZER_ID = "Wan-AI/Wan2.1-T2V-1.3B"
WAN_T5_TOKENIZER_SUBDIR = "google/umt5-xxl"


def fp16_clamp(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.float16 and torch.isinf(x).any():
        clamp = torch.finfo(x.dtype).max - 1000
        x = torch.clamp(x, min=-clamp, max=clamp)
    return x


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class T5LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            x = x.type_as(self.weight)
        return self.weight * x


class T5Attention(nn.Module):
    def __init__(self, dim, dim_attn, num_heads, dropout=0.1):
        assert dim_attn % num_heads == 0
        super().__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, mask=None, pos_bias=None):
        """x [B, L1, C], context [B, L2, C] or None, mask [B, L2] or [B, L1, L2] or None."""
        context = x if context is None else context
        b, n, c = x.size(0), self.num_heads, self.head_dim
        q = self.q(x).view(b, -1, n, c)
        k = self.k(context).view(b, -1, n, c)
        v = self.v(context).view(b, -1, n, c)

        attn_bias = x.new_zeros(b, n, q.size(1), k.size(1))
        if pos_bias is not None:
            attn_bias += pos_bias
        if mask is not None:
            assert mask.ndim in [2, 3]
            mask = mask.view(b, 1, 1, -1) if mask.ndim == 2 else mask.unsqueeze(1)
            attn_bias.masked_fill_(mask == 0, torch.finfo(x.dtype).min)

        attn = torch.einsum("binc,bjnc->bnij", q, k) + attn_bias   # T5 does not scale
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum("bnij,bjnc->binc", attn, v)
        x = x.reshape(b, -1, n * c)
        x = self.o(x)
        x = self.dropout(x)
        return x


class T5FeedForward(nn.Module):
    def __init__(self, dim, dim_ffn, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn
        self.gate = nn.Sequential(nn.Linear(dim, dim_ffn, bias=False), GELU())
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x) * self.gate(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class T5RelativeEmbedding(nn.Module):
    def __init__(self, num_buckets, num_heads, bidirectional, max_dist=128):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, lq, lk):
        device = self.embedding.weight.device
        rel_pos = torch.arange(lk, device=device).unsqueeze(0) - torch.arange(lq, device=device).unsqueeze(1)
        rel_pos = self._relative_position_bucket(rel_pos)
        rel_pos_embeds = self.embedding(rel_pos)
        return rel_pos_embeds.permute(2, 0, 1).unsqueeze(0).contiguous()   # [1, N, Lq, Lk]

    def _relative_position_bucket(self, rel_pos):
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).long() * num_buckets
            rel_pos = torch.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = 0
            rel_pos = -torch.min(rel_pos, torch.zeros_like(rel_pos))
        max_exact = num_buckets // 2
        rel_pos_large = max_exact + (torch.log(rel_pos.float() / max_exact) / math.log(self.max_dist / max_exact) *
                                     (num_buckets - max_exact)).long()
        rel_pos_large = torch.min(rel_pos_large, torch.full_like(rel_pos_large, num_buckets - 1))
        rel_buckets += torch.where(rel_pos < max_exact, rel_pos, rel_pos_large)
        return rel_buckets


class T5SelfAttention(nn.Module):
    def __init__(self, dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos=True, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)

    def forward(self, x, mask=None, pos_bias=None):
        e = pos_bias if self.shared_pos else self.pos_embedding(x.size(1), x.size(1))
        x = fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.ffn(self.norm2(x)))
        return x


class WanT5Encoder(nn.Module):
    """umT5-XXL encoder with the .pth's key layout: ``text_encoder(ids, mask) -> [B, L, 4096]``."""

    def __init__(self, vocab=256384, dim=4096, dim_attn=4096, dim_ffn=10240, num_heads=64, num_layers=24,
                 num_buckets=32, shared_pos=False, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.shared_pos = shared_pos
        self.token_embedding = nn.Embedding(vocab, dim)
        self.pos_embedding = T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True) if shared_pos else None
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos, dropout) for _ in range(num_layers)
        ])
        self.norm = T5LayerNorm(dim)

    def forward(self, ids, mask=None):
        x = self.token_embedding(ids)
        x = self.dropout(x)
        e = self.pos_embedding(x.size(1), x.size(1)) if self.shared_pos else None
        for block in self.blocks:
            x = block(x, mask, pos_bias=e)
        x = self.norm(x)
        x = self.dropout(x)
        return x


def _whitespace_clean(text: str) -> str:
    """Wan's ``clean='whitespace'``: ftfy fix (when installed) + double html unescape + collapse whitespace."""
    try:
        import ftfy  # noqa: PLC0415
        text = ftfy.fix_text(text)
    except ModuleNotFoundError:
        pass
    text = html.unescape(html.unescape(text)).strip()
    return re.sub(r"\s+", " ", text).strip()


class WanT5Tokenizer:
    """``tokenizer(prompts, return_mask=True) -> (ids [B, L], mask [B, L])``, padded to max_length (512)."""

    def __init__(self, tok, seq_len: int):
        self.tok = tok
        self.seq_len = int(seq_len)

    def __call__(self, prompts, return_mask: bool = False):
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        enc = self.tok([_whitespace_clean(p) for p in prompts], padding="max_length", max_length=self.seq_len,
                       truncation=True, return_tensors="pt")
        if return_mask:
            return enc["input_ids"], enc["attention_mask"]
        return enc["input_ids"]


def load_wan_t5(model_id: str = WAN_T5_MODEL_ID, tokenizer_id: str = WAN_T5_TOKENIZER_ID, *,
                torch_dtype: torch.dtype = torch.bfloat16, device: Union[str, torch.device] = "cpu",
                max_sequence_length: int = 512) -> Tuple[WanT5Encoder, WanT5Tokenizer]:
    """umT5 encoder + tokenizer in eval mode on ``device``."""
    from huggingface_hub import hf_hub_download, snapshot_download  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    pth = hf_hub_download(model_id, WAN_T5_FILE)
    tok_root = snapshot_download(tokenizer_id, allow_patterns=[f"{WAN_T5_TOKENIZER_SUBDIR}/*"])
    with torch.device("meta"):
        enc = WanT5Encoder()
    state = torch.load(pth, map_location="cpu", weights_only=True)
    enc.load_state_dict(state, strict=True, assign=True)
    enc = enc.to(device=device, dtype=torch_dtype).eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(os.path.join(tok_root, WAN_T5_TOKENIZER_SUBDIR))
    return enc, WanT5Tokenizer(tok, max_sequence_length)
