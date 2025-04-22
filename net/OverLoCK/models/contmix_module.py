import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from natten.functional import na2d_av # 从 natten 库导入核心函数
from timm.models.layers import LayerNorm2d # 假设使用 timm 的 LayerNorm

# 注意： DynamicConvBlock 中的某些部分（如 lepe, se_layer, gate, proj, mlp, 残差连接）
# 未包含在此 ContMix 模块中，因为它专注于核心的动态卷积机制。
# 如果需要这些功能，需要在调用 ContMix 模块的外部实现。

class ContMix(nn.Module):
    """
    提取自 OverLoCK 的 DynamicConvBlock 的核心 Context-Mixing 动态卷积部分。
    根据输入特征 x 和上下文特征 ctx 动态生成卷积核，并应用它。
    包含两种 kernel_size: 一个主要的 kernel_size 和一个 smk_size (small kernel size)。
    """
    def __init__(self,
                 dim: int,
                 ctx_dim: int, # 期望的上下文特征维度 (通常是 dim // 4)
                 kernel_size: int = 7,
                 smk_size: int = 5,
                 num_heads: int = 2, # 注意：这里是外部传入的 head 数量，内部会 * 2
                 bias: bool = False): # 添加 bias 参数以匹配常规卷积层接口
        super().__init__()

        self.dim = dim
        self.ctx_dim = ctx_dim
        self.kernel_size = kernel_size
        self.smk_size = smk_size
        # 内部实际使用的 head 数量是传入的两倍，与 DynamicConvBlock 一致
        # 一半用于 smk_size，一半用于 kernel_size
        self.num_heads = num_heads * 2
        head_dim = dim // self.num_heads
        if head_dim == 0:
             raise ValueError(f"dim ({dim}) must be >= num_heads*2 ({self.num_heads})")
        self.scale = head_dim ** -0.5

        # --- 从 DynamicConvBlock 提取的层 ---
        # 用于生成 Query (来自主特征 x)
        self.weight_query = nn.Sequential(
            # DynamicConvBlock 使用了 dim//2, 我们这里保持和输入 dim 一致
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias), # 输出维度改为 dim
            LayerNorm2d(dim) if LayerNorm2d else nn.Identity(), # 使用 timm 的 LayerNorm 或占位符
        )

        # 用于生成 Key (来自上下文特征 ctx)
        self.weight_key = nn.Sequential(
            nn.AdaptiveAvgPool2d(7), # 与 DynamicConvBlock 保持一致
            # DynamicConvBlock 使用了 dim//2, 我们这里保持和输入 dim 一致
            nn.Conv2d(ctx_dim, dim, kernel_size=1, bias=bias), # 输入 ctx_dim, 输出 dim
            LayerNorm2d(dim) if LayerNorm2d else nn.Identity(), # 使用 timm 的 LayerNorm 或占位符
        )

        # 将 Query-Key 的结果投影到动态核权重
        # 输入通道是固定的 7x7=49 (因为 weight_key 中有 AdaptiveAvgPool2d(7))
        # 输出通道 L_new = K^2 + SMK^2
        # 改回 Conv1d 以匹配 rearrange 后的输入形状 [B*G, L_in, N]
        self.weight_proj = nn.Conv1d(49, kernel_size**2 + smk_size**2, kernel_size=1, bias=bias)

        # 动态卷积后的最终投影 (可选，但 DynamicConvBlock 中有)
        self.dyconv_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim), # 使用 BatchNorm 而不是 LayerNorm，与 DynamicConvBlock 一致
        )
        # --- 结束提取的层 ---

        # --- 从 DynamicConvBlock 提取的 RPB (相对位置偏置) 相关 ---
        self.get_rpb()
        # --- 结束 RPB 相关 ---

    def get_rpb(self):
        self.rpb_size1 = 2 * self.smk_size - 1
        self.rpb1 = nn.Parameter(torch.empty(self.num_heads // 2, self.rpb_size1, self.rpb_size1)) # 每个核尺寸用一半 heads
        self.rpb_size2 = 2 * self.kernel_size - 1
        self.rpb2 = nn.Parameter(torch.empty(self.num_heads // 2, self.rpb_size2, self.rpb_size2)) # 每个核尺寸用一半 heads
        nn.init.zeros_(self.rpb1)
        nn.init.zeros_(self.rpb2)

    @torch.no_grad()
    def generate_idx(self, kernel_size):
        rpb_size = 2 * kernel_size - 1
        idx_h = torch.arange(0, kernel_size, device=self.rpb1.device)
        idx_w = torch.arange(0, kernel_size, device=self.rpb1.device)
        idx_k = ((idx_h.unsqueeze(-1) * rpb_size) + idx_w).view(-1)
        return (idx_h, idx_w, idx_k)

    def apply_rpb(self, attn, rpb, height, width, kernel_size, idx_h, idx_w, idx_k):
        num_heads_for_rpb = rpb.shape[0] # 获取当前 RPB 使用的 head 数 (self.num_heads // 2)
        num_repeat_h = torch.ones(kernel_size, dtype=torch.long, device=attn.device)
        num_repeat_w = torch.ones(kernel_size, dtype=torch.long, device=attn.device)
        num_repeat_h[kernel_size//2] = height - (kernel_size-1)
        num_repeat_w[kernel_size//2] = width - (kernel_size-1)

        bias_hw = (idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (2*kernel_size-1)) + idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + idx_k
        bias_idx = bias_idx.reshape(-1, int(kernel_size**2))
        bias_idx = torch.flip(bias_idx, [0])

        rpb = torch.flatten(rpb, 1, 2)[:, bias_idx]
        # Reshape RPB to match attention shape: [B, G, H, W, K*K] where G is num_heads_for_rpb
        # attn shape is [B, G, H, W, K*K]
        rpb = rpb.reshape(1, num_heads_for_rpb, height, width, kernel_size**2)

        return attn + rpb

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        B_ctx, C_ctx, H_ctx, W_ctx = ctx.shape

        if C != self.dim:
            raise ValueError(f"Input feature dim ({C}) doesn't match module dim ({self.dim})")
        if C_ctx != self.ctx_dim:
             raise ValueError(f"Context feature dim ({C_ctx}) doesn't match module ctx_dim ({self.ctx_dim})")

        # 1. 生成 Query 和 Key
        query = self.weight_query(x) * self.scale
        key = self.weight_key(ctx) # ctx 输入到 key

        # 2. 计算注意力权重 (动态核的基础)
        # key 的 H, W 被 pool 到 7x7
        query = rearrange(query, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        # key reshape 需要匹配 pool 后的 7x7=49
        key = rearrange(key, 'b (g c) h w -> b g c (h w)', g=self.num_heads, h=7, w=7)
        # weight: [B, G, N, L], N=(H*W), L=49
        weight = einsum(query, key, 'b g c n, b g c l -> b g n l')

        # 3. 投影权重以匹配所需的核尺寸
        # Reshape for Conv1d: [B, G, N, L] -> [B*G*N, L] ? No. [B*G, L, N] ? Maybe.
        # Let's try [B*G, L, N]
        B, G, N, L = weight.shape
        weight = rearrange(weight, 'b g n l -> (b g) l n') # Shape: [B*G, L, N] = [64, 49, 16384]

        # Apply Conv1d(49, L_new, 1) where L_new = K^2+SMK^2 = 10
        weight = self.weight_proj(weight) # Shape: [B*G, L_new, N] = [64, 10, 16384]
        L_new = weight.shape[1] # Get L_new dimension size

        # Reshape back to target format: [B, G, H, W, L_new]
        weight = rearrange(weight, '(b g) l (h w) -> b g h w l', b=B, g=G, h=H, w=W, l=L_new)
        # weight shape: [B, G, H, W, L_new] = [16, 4, 128, 128, 10]

        # 4. 分割权重并应用 RPB
        # 当前 weight shape: [B, G, H, W, L_new] = [16, 4, 128, 128, 10]
        # 将 heads (G) 分成两半，分别用于 smk_size 和 kernel_size
        # weight 需要包含两种核的权重，所以 L_new 应该是 K^2 + SMK^2 = 10
        # 分割 L_new 维度
        attn_weights = rearrange(weight, 'b g h w l -> b g l h w') # Move L dim before H,W for split
        attn1_weights, attn2_weights = torch.split(attn_weights, [self.smk_size**2, self.kernel_size**2], dim=2) # Split along L dim
        # attn1_weights: [B, G, SMK^2, H, W]
        # attn2_weights: [B, G, K^2, H, W]

        # 还要将 G 分成两半给 RPB
        attn1_weights_g1, attn1_weights_g2 = torch.chunk(attn1_weights, 2, dim=1) # Shape: [B, G/2, SMK^2, H, W]
        attn2_weights_g1, attn2_weights_g2 = torch.chunk(attn2_weights, 2, dim=1) # Shape: [B, G/2, K^2, H, W]

        # Rearrange for apply_rpb which expects [B, G, H, W, K*K]
        attn1 = rearrange(attn1_weights_g1, 'b g l h w -> b g h w l') # Shape: [B, G/2, H, W, SMK^2]
        attn2 = rearrange(attn2_weights_g2, 'b g l h w -> b g h w l') # Shape: [B, G/2, H, W, K^2]

        rpb1_idx = self.generate_idx(self.smk_size)
        rpb2_idx = self.generate_idx(self.kernel_size)
        attn1 = self.apply_rpb(attn1, self.rpb1, H, W, self.smk_size, *rpb1_idx)
        attn2 = self.apply_rpb(attn2, self.rpb2, H, W, self.kernel_size, *rpb2_idx)

        # 5. 应用 Softmax
        attn1 = torch.softmax(attn1, dim=-1)
        attn2 = torch.softmax(attn2, dim=-1)

        # 6. 准备 Value (将输入 x 拆分给两个 head 组)
        # value: [2, B, G/2, H, W, C_head]
        value = rearrange(x, 'b (m g c) h w -> m b g h w c', m=2, g=self.num_heads // 2)

        # 7. 执行动态卷积 (NATTEN)
        # x1: [B, G/2, H, W, C_head]
        x1 = na2d_av(attn1, value[0], kernel_size=self.smk_size)
        # x2: [B, G/2, H, W, C_head]
        x2 = na2d_av(attn2, value[1], kernel_size=self.kernel_size)

        # 8. 合并结果并投影
        # x: [B, G, H, W, C_head]
        x = torch.cat([x1, x2], dim=1)
        # x: [B, C, H, W] where C = G * C_head
        x = rearrange(x, 'b g h w c -> b (g c) h w', h=H, w=W)
        x = self.dyconv_proj(x)

        return x 