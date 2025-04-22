import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from net.transformer_utils import LayerNorm # Assuming LayerNorm is defined here

# --- Modern FFN Components ---

class SwiGLU(nn.Module):
    """ SwiGLU FFN Layer """
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, bias=False):
        super().__init__()
        out_dim = out_dim or in_dim
        # Standard SwiGLU expansion is 2/3 * 4 = 8/3
        hidden_dim = hidden_dim or int(in_dim * 8 / 3)

        # Ensure hidden_dim is divisible for chunking if needed,
        # but original SwiGLU paper uses approximation
        # hidden_dim = (hidden_dim + 1) // 2 * 2 # Example: Make divisible by 2

        self.w12 = nn.Conv2d(in_dim, hidden_dim * 2, kernel_size=1, bias=bias) # Combined weights for gate & up projection
        self.w3 = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=bias)     # Down projection

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=1) # Split into two parts: one for activation, one for gating
        hidden = F.silu(x1) * x2     # Swish Gated Linear Unit
        return self.w3(hidden)

class ModernFFN(nn.Module):
    """ Feed Forward Network using SwiGLU """
    def __init__(self, dim, ffn_expansion_factor=8/3, bias=False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.ffn = SwiGLU(in_dim=dim, hidden_dim=hidden_features, out_dim=dim, bias=bias)

    def forward(self, x):
        return self.ffn(x)

# --- Cross Attention Block (CAB) - Largely unchanged from original ---

class CAB(nn.Module):
    """ Cross Attention Block """
    def __init__(self, dim, num_heads, bias=False):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y): # x is query, y is key/value source
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

# --- Lightweight Cross Attention Block (LCABlock) using Pre-Norm ---

class LCABlock(nn.Module):
    """ Standardized Lightweight Cross Attention Block with Pre-Normalization """
    def __init__(self, dim, num_heads, ffn_expansion_factor=8/3, bias=False):
        super().__init__()
        self.norm1_x = LayerNorm(dim)
        self.norm1_y = LayerNorm(dim)
        self.attn = CAB(dim, num_heads, bias=bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = ModernFFN(dim, ffn_expansion_factor, bias=bias)

    def forward(self, x, y):
        # Pre-Normalization -> Cross Attention -> Residual
        x_res = x
        x_norm = self.norm1_x(x)
        y_norm = self.norm1_y(y)
        x = x_res + self.attn(x_norm, y_norm)

        # Pre-Normalization -> FFN -> Residual
        x_res = x
        x_norm = self.norm2(x)
        x = x_res + self.ffn(x_norm)
        return x 