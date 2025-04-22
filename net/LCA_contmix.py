import torch
import torch.nn as nn
import torch.nn.functional as F # 添加 F 导入
from einops import rearrange
from net.transformer_utils import *
# Assuming ContMix implementation exists here:
# from net.overlock.contmix import ContMix 
from net.OverLoCK.models.contmix_module import ContMix

# Cross Attention Block using ContMix for depthwise parts
class CAB(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=3, smk_size=3, bias=False):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.dim = dim # 保存 dim
        ctx_dim_q = dim // 4
        ctx_dim_kv = (dim * 2) // 4
        if ctx_dim_q == 0 or ctx_dim_kv == 0:
             raise ValueError(f"Dimension {dim} is too small for ctx_dim calculation.")

        # 添加上下文投影层
        self.ctx_proj_q = nn.Conv2d(dim, ctx_dim_q, 1, bias=bias)
        self.ctx_proj_kv = nn.Conv2d(dim, ctx_dim_kv, 1, bias=bias) # 输入是 y，维度是 dim

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # 修改 ContMix 初始化：移除无效参数，添加 ctx_dim, kernel_size, smk_size
        self.q_dwconv = ContMix(dim=dim, ctx_dim=ctx_dim_q, kernel_size=kernel_size, smk_size=smk_size, num_heads=num_heads, bias=bias)

        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        # 修改 ContMix 初始化：移除无效参数，添加 ctx_dim, kernel_size, smk_size
        # 注意：这里的 dim 是 dim*2
        self.kv_dwconv = ContMix(dim=dim*2, ctx_dim=ctx_dim_kv, kernel_size=kernel_size, smk_size=smk_size, num_heads=num_heads, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        # 生成 q 和 ctx_q
        q_pre = self.q(x)
        ctx_q = self.ctx_proj_q(x)
        q = self.q_dwconv(q_pre, ctx_q) # 传递 ctx_q

        # 生成 kv 和 ctx_kv
        kv_pre = self.kv(y)
        ctx_kv = self.ctx_proj_kv(y) # 从 y 生成 ctx_kv
        kv = self.kv_dwconv(kv_pre, ctx_kv) # 传递 ctx_kv
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn, dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    

# Intensity Enhancement Layer / Color Denoise Layer (IEL/CDL) using ContMix
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, kernel_size=3, smk_size=3, num_heads=2, bias=False):
        super(IEL, self).__init__()

        # 计算 ContMix 内部需要的 head 数量
        contmix_internal_heads = num_heads * 2

        # 计算初步的 hidden_features
        initial_hidden_features = int(dim * ffn_expansion_factor)

        # 向上取整 hidden_features 到 contmix_internal_heads 的倍数
        if initial_hidden_features == 0:
            hidden_features = contmix_internal_heads # 避免为 0
        else:
            hidden_features = ((initial_hidden_features + contmix_internal_heads - 1) // contmix_internal_heads) * contmix_internal_heads

        # 确保调整后的 hidden_features 至少为 contmix_internal_heads
        hidden_features = max(hidden_features, contmix_internal_heads)

        # 计算不同阶段的 ctx_dim (基于调整后的 hidden_features)
        hidden_features_x2 = hidden_features * 2
        ctx_dim_dw = hidden_features_x2 // 4
        ctx_dim_dw12 = hidden_features // 4

        # 重新检查调整后的维度是否导致 ctx_dim 为 0 （理论上调整后不会）
        if ctx_dim_dw == 0 or ctx_dim_dw12 == 0:
             raise ValueError(f"Adjusted hidden dimension {hidden_features} results in zero ctx_dim.")

        # 使用调整后的 hidden_features_x2 初始化输入投影层
        self.project_in = nn.Conv2d(dim, hidden_features_x2, kernel_size=1, bias=bias)

        # 添加上下文投影层 (从各自的输入特征投影)
        # 维度基于调整后的 hidden_features
        self.ctx_proj_dw = nn.Conv2d(hidden_features_x2, ctx_dim_dw, 1, bias=bias)
        self.ctx_proj_dw1 = nn.Conv2d(hidden_features, ctx_dim_dw12, 1, bias=bias)
        self.ctx_proj_dw2 = nn.Conv2d(hidden_features, ctx_dim_dw12, 1, bias=bias)

        # 修改 ContMix 初始化 (使用调整后的 hidden_features)
        self.dwconv = ContMix(dim=hidden_features_x2, ctx_dim=ctx_dim_dw, kernel_size=kernel_size, smk_size=smk_size, num_heads=num_heads, bias=bias)
        # 修改 ContMix 初始化
        self.dwconv1 = ContMix(dim=hidden_features, ctx_dim=ctx_dim_dw12, kernel_size=kernel_size, smk_size=smk_size, num_heads=num_heads, bias=bias)
        # 修改 ContMix 初始化
        self.dwconv2 = ContMix(dim=hidden_features, ctx_dim=ctx_dim_dw12, kernel_size=kernel_size, smk_size=smk_size, num_heads=num_heads, bias=bias)

        # 修改输出投影层的输入维度
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh()
        
    def forward(self, x):
        x_in = self.project_in(x)

        # 生成并传递 ctx_dw
        ctx_dw = self.ctx_proj_dw(x_in)
        x_dw = self.dwconv(x_in, ctx_dw) # Apply shared DW conv with context

        x1, x2 = x_dw.chunk(2, dim=1) # Split for gating
        
        # 生成并传递 ctx1 和 ctx2
        ctx1 = self.ctx_proj_dw1(x1)
        ctx2 = self.ctx_proj_dw2(x2)

        # Apply branch-specific DW conv, activation, and residual with context
        x1_processed = self.Tanh(self.dwconv1(x1, ctx1)) + x1 
        x2_processed = self.Tanh(self.dwconv2(x2, ctx2)) + x2
        
        x = x1_processed * x2_processed # Gating
        x = self.project_out(x)
        return x
  
  
# Lightweight Cross Attention modules using the modified CAB and IEL
class HV_LCA(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=3, smk_size=3, bias=False):
        super(HV_LCA, self).__init__()
        # Note: Original paper implementation details for CDL/IEL might differ slightly in structure
        # Pass kernel_size, smk_size, num_heads to IEL
        self.gdfn = IEL(dim, kernel_size=kernel_size, smk_size=smk_size, num_heads=num_heads, bias=bias) # CDL functionality (Color Denoise) using IEL structure with ContMix
        self.norm = LayerNorm(dim)
        # Pass kernel_size, smk_size to CAB
        self.ffn = CAB(dim, num_heads, kernel_size=kernel_size, smk_size=smk_size, bias=bias) # Cross Attention Block with ContMix
        
    def forward(self, x, y):
        # Apply cross-attention
        attn_out = self.ffn(self.norm(x), self.norm(y))
        x = x + attn_out
        # Apply GDFN/CDL part
        ffn_out = self.gdfn(self.norm(x))
        x = x + ffn_out # In original I_LCA it was added, let's assume same for HV_LCA's CDL part
        return x
    
class I_LCA(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=3, smk_size=3, bias=False):
        super(I_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        # Pass kernel_size, smk_size, num_heads to IEL
        # IEL (Intensity Enhancement) using IEL structure with ContMix
        self.gdfn = IEL(dim, bias=bias) 
        # Cross Attention Block with ContMix
        self.ffn = CAB(dim, num_heads, bias=bias)
        
    def forward(self, x, y):
        # Apply cross-attention
        attn_out = self.ffn(self.norm(x), self.norm(y))
        x = x + attn_out
        # Apply GDFN/IEL part
        ffn_out = self.gdfn(self.norm(x))
        x = x + ffn_out 
        return x 