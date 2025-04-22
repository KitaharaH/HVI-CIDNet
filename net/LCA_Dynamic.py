import torch
import torch.nn as nn
import torch.nn.functional as F
from net.OverLoCK.models.overlock import DynamicConvBlock
# 可能需要从 OverLoCK 导入其他辅助模块/函数，如果 DynamicConvBlock 依赖它们
# 例如 LayerNorm2d 或其他激活/归一化层，具体取决于 DynamicConvBlock 的实现细节
# 导入 timm 的 LayerNorm2d 作为尝试
from timm.models.layers import LayerNorm2d

class LCA_Dynamic_Base(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=7, ctx_dim_ratio=2, dynamic_block_params={}):
        super().__init__()
        # Ensure dim is divisible by num_heads for potential internal attention mechanisms
        if dim % num_heads != 0:
            # Find the nearest higher multiple of num_heads if needed, but maybe error is better
            # For now, let's raise an error, or adjust num_heads/dim externally.
            raise ValueError(f"Dimension {dim} must be divisible by num_heads {num_heads}")
        # if dim % ctx_dim_ratio != 0: # 这个检查不再严格需要，因为 ctx_dim 不直接用于 DynamicConvBlock 的关键部分
        #     raise ValueError(f"Dimension {dim} must be divisible by ctx_dim_ratio {ctx_dim_ratio}")
            
        # self.ctx_dim = dim // ctx_dim_ratio # 不再需要这个成员变量
        self.num_heads = num_heads
        self.dim = dim
        
        # 计算 DynamicConvBlock 内部期望的 context 维度
        internal_ctx_dim = self.dim // 4
        if internal_ctx_dim == 0:
            raise ValueError(f"Dimension {dim} is too small, results in internal_ctx_dim=0")

        # Projection for spatial context (h_x)
        self.conv_hx = nn.Conv2d(self.dim, internal_ctx_dim, kernel_size=1, bias=False)
        
        # Projection for channel context (h_r)
        self.conv_ctx = nn.Conv2d(self.dim, internal_ctx_dim, kernel_size=1, bias=False) # 输出 internal_ctx_dim
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Default parameters for DynamicConvBlock, can be overridden by dynamic_block_params
        default_params = {
            'smk_size': 5,
            'mlp_ratio': 4,
            'ls_init_value': 1e-5, # Using a common default from OverLoCK
            'res_scale': True, # Using a common default from OverLoCK
            'drop_path': 0.,
            'norm_layer': LayerNorm2d, # Using timm's LayerNorm2d
            'use_gemm': True, 
            'deploy': False,
            'use_checkpoint': False,
        }
        default_params.update(dynamic_block_params) # Allow overriding defaults

        # Instantiate DynamicConvBlock
        self.dynamic_block = DynamicConvBlock(
            dim=self.dim,
            ctx_dim=self.dim,
            kernel_size=kernel_size,
            num_heads=self.num_heads,
            **default_params
        )

    def forward(self, x, y):
        # x: feature from the main path (e.g., I-path)
        # y: feature from the cross path (e.g., HV-path)

        # Ensure dimensions match if needed (should match in CIDNet structure)
        assert x.shape == y.shape, f"Input shapes must match for LCA_Dynamic: x={x.shape}, y={y.shape}"

        # Prepare context inputs for DynamicConvBlock
        # h_x = y  # 旧方式
        h_x = self.conv_hx(y) # 新方式：将 y 映射到 internal_ctx_dim

        h_r_ = self.conv_ctx(y) # 新方式：将 y 映射到 internal_ctx_dim
        h_r = self.pool(h_r_) # Shape: (B, internal_ctx_dim, 1, 1)
        # DynamicConvBlock forward signature seems to be (x, h_x, h_r)
        # Let's assume it handles h_r in (B, C, 1, 1) format internally

        # Pass through DynamicConvBlock
        # 现在 h_x 和 h_r 的通道数 (internal_ctx_dim) 应该与 DynamicConvBlock 内部 LayerScale 的期望匹配
        out = self.dynamic_block(x, h_x, h_r)

        return out

class I_LCA_Dynamic(LCA_Dynamic_Base):
    # Wrapper for consistency, inherits everything
    pass

class HV_LCA_Dynamic(LCA_Dynamic_Base):
    # Wrapper for consistency, inherits everything
    pass 