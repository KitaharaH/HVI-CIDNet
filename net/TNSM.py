import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from net.transformer_utils import LayerNorm

class DynamicNoiseMap(nn.Module):
    """
    动态噪声图模块，负责学习和生成噪声特征图
    """
    def __init__(self, dim, reduction=4, bias=False):
        super(DynamicNoiseMap, self).__init__()
        
        # 使用特征压缩-扩展结构来学习噪声特征
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 使用1x1卷积进行通道压缩和扩展
        reduced_dim = max(8, dim // reduction)
        self.fc1 = nn.Conv2d(dim, reduced_dim, kernel_size=1, bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_dim, dim, kernel_size=1, bias=bias)
        
        # 噪声特征学习分支
        self.noise_branch = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        )
        
        # --- Add 1x1 Conv to output single channel map --- 
        self.final_conv = nn.Conv2d(dim, 1, kernel_size=1, bias=bias)
        # --- End Add --- 
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 全局信息聚合
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        
        # 特征压缩与扩展
        avg_out = self.fc2(self.relu(self.fc1(avg_out)))
        max_out = self.fc2(self.relu(self.fc1(max_out)))
        
        # 全局特征
        global_feat = self.sigmoid(avg_out + max_out)
        
        # 局部噪声特征学习
        local_noise = self.noise_branch(x)
        
        # --- Apply final_conv before sigmoid --- 
        noise_feat = global_feat * local_noise
        noise_map = self.sigmoid(self.final_conv(noise_feat)) # Output shape: (b, 1, h, w)
        # --- End Apply --- 
        
        return noise_map

class NoiseAwareAttentionCABStyle(nn.Module):
    """
    噪声感知注意力机制 (仿照 LCA.py 中的 CAB 实现)
    计算通道/特征注意力 (c_ph x c_ph)，并通过噪声图调制 Value (V)。
    """
    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 特征提取 (与 CAB 相同)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        # 噪声调制器 (用于调制 Value)
        self.noise_scaler = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=1, bias=bias),
            nn.Sigmoid()
        )

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y, noise_map=None):
        b, c, h, w = x.shape

        q_feat = self.q_dwconv(self.q(x))
        kv_feat = self.kv_dwconv(self.kv(y))
        k_feat, v_feat = kv_feat.chunk(2, dim=1)

        # --- 使用 CAB 的 Rearrange 模式 --- 
        # Shape: (b, head, c_ph, h*w)
        q = rearrange(q_feat, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k_feat, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v_feat, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 归一化 (沿 seq_len 维度，与 CAB 不同，但可能更适合这种模式下的点积)
        # 或者保留 CAB 的方式 (沿 c_ph 维度)? -> 暂时保留 CAB 方式
        # q = torch.nn.functional.normalize(q, dim=-1)
        # k = torch.nn.functional.normalize(k, dim=-1)

        # --- CAB-Style Attention Calculation --- 
        # attn = (q @ k.transpose(-2, -1)) shape: (b, head, c_ph, c_ph)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn, dim=-1) # Softmax 沿最后一个维度 (c_ph)

        # --- Noise Modulation on Value (V) --- 
        if noise_map is not None:
            # noise_map shape: (b, 1, h, w)
            noise_keep_factor = self.noise_scaler(noise_map)
            # noise_keep_factor shape: (b, dim, h, w)

            # Rearrange to match V's shape
            # Shape: (b, head, c_ph, h*w)
            noise_keep_factor = rearrange(noise_keep_factor, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

            # Modulate V based on spatial noise
            v = v * noise_keep_factor
        
        # --- Output Calculation --- 
        # out = (attn @ v) shape: (b, head, c_ph, h*w)
        out = (attn @ v)

        # Rearrange back to original spatial format
        # Shape: (b, c, h, w)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class AdaptiveFilter(nn.Module):
    """
    自适应滤波模块，根据噪声图自适应调整特征
    """
    def __init__(self, dim, bias=False):
        super(AdaptiveFilter, self).__init__()
        
        # 噪声区域处理路径
        self.noise_process = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        )
        
        # 细节保留路径
        self.detail_preserve = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        )
        
        # 融合层
        self.fusion = nn.Conv2d(dim*2, dim, kernel_size=1, bias=bias)
        self.norm = LayerNorm(dim)
        
    def forward(self, x, noise_map):
        # 噪声分支 - 对噪声区域进行平滑处理
        noise_branch = self.noise_process(x)
        
        # 细节分支 - 保留图像细节
        detail_branch = self.detail_preserve(x)
        
        # 根据噪声图融合两个分支
        # 噪声较高的区域，更多地使用噪声分支的结果
        # 噪声较低的区域，更多地保留原始细节
        weighted_noise = noise_map * noise_branch
        weighted_detail = (1.0 - noise_map) * detail_branch
        
        # 特征融合
        fused = torch.cat([weighted_noise, weighted_detail], dim=1)
        out = self.fusion(fused)
        out = self.norm(out)
        
        return out

class TrainableNoiseSuppression(nn.Module):
    """
    可训练的噪声抑制模块(TNSM)，集成了动态噪声图、
    噪声感知注意力和自适应滤波
    """
    def __init__(self, dim, num_heads, bias=False):
        super(TrainableNoiseSuppression, self).__init__()
        
        # 动态噪声图生成器
        self.noise_map_generator = DynamicNoiseMap(dim, bias=bias)
        
        # --- 使用新的 CAB 风格的噪声感知注意力 --- 
        self.noise_attention = NoiseAwareAttentionCABStyle(dim, num_heads, bias=bias)
        
        # 自适应滤波器
        self.adaptive_filter = AdaptiveFilter(dim, bias=bias)
        
        # 标准化层
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        
    def forward(self, x, y=None):
        # 如果没有提供辅助输入，则使用自身
        if y is None:
            y = x
            
        # 生成动态噪声图
        noise_map = self.noise_map_generator(x)
        
        # 应用噪声感知注意力 (CAB Style)
        x_norm = self.norm1(x)
        y_norm = self.norm1(y) # y 也需要归一化
        attn_out = self.noise_attention(x_norm, y_norm, noise_map)
        x = x + attn_out
        
        # 应用自适应滤波
        x_norm = self.norm2(x)
        filtered = self.adaptive_filter(x_norm, noise_map)
        x = x + filtered
        
        return x, noise_map

# HV通道的TNSM模块
class HV_TNSM(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(HV_TNSM, self).__init__()
        self.tnsm = TrainableNoiseSuppression(dim, num_heads, bias)
        
    def forward(self, x, y):
        return self.tnsm(x, y)

# I通道的TNSM模块
class I_TNSM(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(I_TNSM, self).__init__()
        self.tnsm = TrainableNoiseSuppression(dim, num_heads, bias)
        
    def forward(self, x, y):
        return self.tnsm(x, y) 