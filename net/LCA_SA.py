import torch
import torch.nn as nn
from einops import rearrange
from net.transformer_utils import *
import torch.nn.functional as F # Added for F.normalize

# +++ Define SpatialAttention Module (copied from CIDNet_MSSA.py) +++
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        # Apply attention: x * self.sigmoid(y)
        # Return the attention map instead, let CAB apply it if needed, or apply here.
        # Let's apply it here for direct integration.
        return x * self.sigmoid(y)
# +++ End Definition +++


# Cross Attention Block with integrated Spatial Attention
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        # Initialize Spatial Attention for q, k, v
        self.sa_q = SpatialAttention()
        self.sa_k = SpatialAttention()
        self.sa_v = SpatialAttention()

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        # Apply Spatial Attention
        q = self.sa_q(q)
        k = self.sa_k(k)
        v = self.sa_v(v) # Apply to value as well

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1) # Use F.normalize
        k = F.normalize(k, dim=-1) # Use F.normalize

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn,dim=-1) # Use F.softmax

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


# Intensity Enhancement Layer (copied, unchanged)
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x


# Lightweight Cross Attention using modified CAB
class HV_LCA(nn.Module):
    def __init__(self, dim,num_heads, bias=False):
        super(HV_LCA, self).__init__()
        self.gdfn = IEL(dim) # IEL and CDL have same structure
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias) # Uses the modified CAB

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x),self.norm(y))
        x = self.gdfn(self.norm(x)) # Original IEL application remains
        return x

class I_LCA(nn.Module):
    def __init__(self, dim,num_heads, bias=False):
        super(I_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        self.ffn = CAB(dim, num_heads, bias=bias) # Uses the modified CAB

    def forward(self, x, y):
        x = x + self.ffn(self.norm(x),self.norm(y))
        x = x + self.gdfn(self.norm(x)) # Original IEL application remains
        return x 