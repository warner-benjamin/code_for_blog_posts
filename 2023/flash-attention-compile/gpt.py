"""
The codebase for gpt.py is inspired by:
nanoGPT https://github.com/karpathy/nanoGPT - Copyright (c) 2022 Andrej Karpathy - MIT License
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from flash_attn import flash_attn_qkvpacked_func, flash_attn_func


class FlashAttn(str, Enum):
    none = 'none'
    pytorch = 'pytorch'
    flash = 'flash'
    flash_qkv = 'flash_qkv'

class GPTType(str, Enum):
    small = 'small'
    medium = 'medium'
    large = 'large'


class FeedForward(nn.Module):
    def __init__(self, hidden_size:int, expand_size:int, act:nn.Module=nn.GELU,
                 drop:float=0.1, bias:bool=True):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, expand_size, bias=bias)
        self.act = act()
        self.fc2 = nn.Linear(expand_size, hidden_size, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x:Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CausalAttention(nn.Module):
    def __init__(self, hidden_size:int, num_heads:int, context_size:int,
                 attn_drop:float=0.1, out_drop:float=0.1, bias:bool=True,
                 flashattn:FlashAttn=FlashAttn.pytorch):
        super().__init__()
        self._packed = flashattn == FlashAttn.flash_qkv
        self._flash = flashattn == FlashAttn.flash
        self._pytorch = flashattn == FlashAttn.pytorch
        assert hidden_size % num_heads == 0
        self.nh = num_heads

        self.Wqkv = nn.Linear(hidden_size, hidden_size * 3, bias=bias)
        if self._packed or self._flash or self._pytorch:
            self.attn_drop = attn_drop
        else:
            self.attn_drop = nn.Dropout(attn_drop)

        self.Wo = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_drop = nn.Dropout(out_drop)


        if self._pytorch:
            # PyTorch documentation states using the `torch.backends.cuda.sdp_kernel()` context manager is
            # prefered over this method, but the manager causes a graph break with `compile` and this doesn't
            # https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
            if not torch.backends.cuda.flash_sdp_enabled():
                torch.backends.cuda.enable_flash_sdp(True)
        elif not (self._flash and self._packed):
            self.register_buffer('causal_mask',
                torch.triu(torch.ones([context_size, context_size], dtype=torch.bool), diagonal=1)
                    .view(1, 1, context_size, context_size), persistent=False
            )

    def forward(self, x: Tensor):
        B, S, C = x.shape
        if self._packed:
            qkv = self.Wqkv(x).reshape(B, S, 3, self.nh, C//self.nh)
            x = flash_attn_qkvpacked_func(qkv, dropout_p=self.attn_drop if self.training else 0, causal=True)
        elif self._flash:
            q, k, v = self.Wqkv(x).reshape(B, S, 3, self.nh, C//self.nh).unbind(dim=2)
            x = flash_attn_func(q, k, v, dropout_p=self.attn_drop if self.training else 0, causal=True)
        else:
            q, k, v = self.Wqkv(x).reshape(B, S, 3, self.nh, C//self.nh).transpose(3, 1).unbind(dim=2)

            if self._pytorch:
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop, is_causal=True)
            else:
                attn = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
                attn = attn.masked_fill(self.causal_mask[:, :, :S, :S], float('-inf'))
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                x = attn @ v

        x = x.transpose(1, 2).reshape(B, S, C)
        return self.out_drop(self.Wo(x))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size:int, num_heads:int, context_size:int, expand_size:int,
                 act:nn.Module=nn.GELU, attn_drop:float=0.1, out_drop:float=0.1,
                 ffn_drop:float=0.1, bias:bool=True, flashattn:FlashAttn=FlashAttn.pytorch):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = CausalAttention(
            hidden_size=hidden_size, num_heads=num_heads, context_size=context_size,
            attn_drop=attn_drop, out_drop=out_drop, bias=bias, flashattn=flashattn
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(
            hidden_size=hidden_size, expand_size=expand_size, act=act,
            drop=ffn_drop, bias=bias,
        )

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, num_layers:int, vocab_size:int, hidden_size:int, num_heads:int,
                 context_size:int, expand_size:int, act:nn.Module=nn.GELU,
                 embed_drop:float=0.1, attn_drop:float=0.1, out_drop:float=0.1,
                 ffn_drop:float=0.1, head_norm:bool=True, tie_weights:bool=True,
                 head_bias:bool=True, bias:bool=True,
                 flashattn:FlashAttn=FlashAttn.pytorch):
        super().__init__()
        self.vocab_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(context_size, hidden_size)
        self.embed_drop = nn.Dropout(embed_drop)

        self.tfm_blocks = nn.ModuleList([TransformerBlock(
                hidden_size=hidden_size, num_heads=num_heads, context_size=context_size,
                expand_size=expand_size, act=act, bias=bias, attn_drop=attn_drop,
                out_drop=out_drop, ffn_drop=ffn_drop, flashattn=flashattn)
            for _ in range(num_layers)])

        self.head_norm = nn.LayerNorm(hidden_size) if head_norm else nn.Identity()
        self.head = nn.Linear(hidden_size, vocab_size, bias=head_bias)
        if tie_weights:
            self.head.weight = self.vocab_embed.weight

        pos = torch.arange(0, context_size, dtype=torch.long)
        self.register_buffer('pos', pos, persistent=False)

        self.apply(self._init_weights)

    def forward(self, x: Tensor):
        tokens = self.vocab_embed(x)
        pos = self.pos_embed(self.pos[:x.shape[1]])
        x = self.embed_drop(tokens + pos)

        for block in self.tfm_blocks:
            x = block(x)

        x = self.head_norm(x)
        return self.head(x)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module._get_name() == 'fc2':
                # GPT-2 style FFN init
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02/math.sqrt(2 * self.num_layers))
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class GPTForCausalLM(GPT):
    def __init__(self, loss_fn:nn.Module=nn.CrossEntropyLoss(), **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn

    def forward(self, input_ids: Tensor, labels: Tensor):
        logits = super().forward(input_ids)
        loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return {'logits': logits, 'loss': loss}


@dataclass
class GPTSmall:
    vocab_size:int=16384
    num_layers:int=12
    hidden_size:int=768
    num_heads:int=12
    context_size:int=256
    expand_size:int=3072
    act:nn.Module=nn.GELU
    embed_drop:float=0
    attn_drop:float=0
    out_drop:float=0
    ffn_drop:float=0
    head_norm:bool=True
    tie_weights:bool=True
    head_bias:bool=True
    bias:bool=False
    flashattn:FlashAttn=FlashAttn.pytorch

    def as_kwargs(self):
        return self.__dict__

@dataclass
class GPTMedium:
    vocab_size:int=16384
    num_layers:int=24
    hidden_size:int=1024
    num_heads:int=16
    context_size:int=256
    expand_size:int=4096
    act:nn.Module=nn.GELU
    embed_drop:float=0
    attn_drop:float=0
    out_drop:float=0
    ffn_drop:float=0
    head_norm:bool=True
    tie_weights:bool=True
    head_bias:bool=False
    bias:bool=False
    flashattn:FlashAttn=FlashAttn.pytorch

    def as_kwargs(self):
        return self.__dict__


@dataclass
class GPTLarge:
    vocab_size:int=16384
    num_layers:int=36
    hidden_size:int=1280
    num_heads:int=20
    context_size:int=256
    expand_size:int=5120
    act:nn.Module=nn.GELU
    embed_drop:float=0
    attn_drop:float=0
    out_drop:float=0
    ffn_drop:float=0
    head_norm:bool=True
    tie_weights:bool=True
    head_bias:bool=False
    bias:bool=False
    flashattn:FlashAttn=FlashAttn.pytorch

    def as_kwargs(self):
        return self.__dict__
