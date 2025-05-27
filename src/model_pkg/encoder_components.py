# model_pkg/encoder_components.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input_tensor, self.weight.shape, self.weight, self.bias, 1e-5)

class MultiHeadSelfAttention(nn.Module):
    """ Multi-Head Self-Attention module for an encoder (bidirectional) """
    def __init__(self, n_embd: int, n_head: int, dropout: float, bias: bool):
        super().__init__()
        assert n_embd % n_head == 0, "Embedding dimension must be divisible by number of heads"
        
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head # Dimension of each head
        self.dropout_val = dropout # Renamed to avoid conflict with nn.Dropout layer

        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Flash Attention support (PyTorch >= 2.0) for efficiency
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: Using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape and transpose for multi-head attention
        # (B, T, n_embd) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Self-attention
        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels
            # For encoders, is_causal=False.
            # `attention_mask` should be a boolean mask where True indicates elements to *NOT* attend to.
            # If input `attention_mask` is (B, T) with 1=real, 0=pad, invert it.
            flash_attn_mask = None
            if attention_mask is not None: # (B, T_key)
                # Expand to (B, 1, 1, T_key) and then invert for scaled_dot_product_attention
                # where True means "mask this key token"
                flash_attn_mask = ~(attention_mask.unsqueeze(1).unsqueeze(2).bool()) # (B, 1, 1, T_key)
                # scaled_dot_product_attention will broadcast this mask across query positions and heads.

            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=flash_attn_mask, dropout_p=self.dropout_val if self.training else 0, is_causal=False
            )
        else:
            # Manual implementation of attention
            # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if attention_mask is not None:
                # `attention_mask` (B, T_key) -> (B, 1, 1, T_key) for broadcasting
                # `masked_fill` expects mask where True means fill.
                # Here, if attention_mask is 0 (pad), we want to fill with -inf.
                att = att.masked_fill(attention_mask.unsqueeze(1).unsqueeze(1) == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """ Feed-Forward Network module """
    def __init__(self, n_embd: int, dropout: float, bias: bool):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu    = nn.GELU() # GELU activation
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class EncoderBlock(nn.Module):
    """ A single Transformer Encoder block """
    def __init__(self, n_embd: int, n_head: int, dropout: float, bias: bool):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = MultiHeadSelfAttention(n_embd=n_embd, n_head=n_head, dropout=dropout, bias=bias)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd=n_embd, dropout=dropout, bias=bias)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x
