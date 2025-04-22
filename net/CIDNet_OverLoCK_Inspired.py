import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
# Assuming NormDownsample/NormUpsample are defined or imported correctly
# If they are in transformer_utils, ensure that file is accessible
# from net.transformer_utils import *
from net.LCA import *
from huggingface_hub import PyTorchModelHubMixin
import torch.nn.functional as F

# +++ 定义基础模块 (从 CIDNet_MSSA.py 复制并修改) +++

# Modified NormDownsample with Large Kernel option
class NormDownsampleLK(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=False, kernel_size=7, stride=2):
        super(NormDownsampleLK, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.norm = nn.InstanceNorm2d(out_channels) if use_norm else nn.Identity()
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

# Modified NormUpsample with Large Kernel option (using Transposed Conv)
# Note: PixelShuffle could also be used here if preferred.
class NormUpsampleLK(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=False, kernel_size=7, stride=2):
        super(NormUpsampleLK, self).__init__()
        padding = kernel_size // 2
        # Output padding might be needed depending on kernel_size and stride to match output shape
        output_padding = stride - 1 if stride > 1 else 0
        self.trans_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False)
        self.norm = nn.InstanceNorm2d(out_channels) if use_norm else nn.Identity()
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, skip_input):
        x = self.trans_conv(x)
        x = self.norm(x)
        x = self.relu(x)
        # Concatenate with skip connection
        x = torch.cat([x, skip_input], dim=1)
        # Add a fusion conv like in PixelShuffle version
        # The input channels to fusion conv: out_channels + skip_input.size(1)
        # The output channels should be out_channels
        # fuse_conv = nn.Conv2d(out_channels + skip_input.size(1), out_channels, kernel_size=3, padding=1, bias=False)
        # x = fuse_conv(x)
        # For simplicity now, return concatenated. Fusion needs extra layers.
        return x

# --- Original Blocks (if needed, define NormDownsample/NormUpsample without LK) ---
class NormDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=False):
        super(NormDownsample, self).__init__()
        # Original uses kernel_size=4, stride=2, padding=1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(out_channels) if use_norm else nn.Identity()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class NormUpsample(nn.Module):
     def __init__(self, in_channels, out_channels, use_norm=False):
         super(NormUpsample, self).__init__()
         # Assuming original used Transposed Conv for symmetry
         self.trans_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
         self.norm = nn.InstanceNorm2d(out_channels) if use_norm else nn.Identity()
         self.relu = nn.LeakyReLU(0.2, inplace=True)
     def forward(self, x, skip_input):
        x = self.relu(self.norm(self.trans_conv(x)))
        x = torch.cat([x, skip_input], dim=1)
        # Add fusion conv similar to NormUpsampleLK if needed
        return x
# --- End Original Blocks ---

class CIDNet_OverLoCK_Inspired(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False,
                 large_kernel_size=7 # Parameter to control large kernel size
        ):
        super(CIDNet_OverLoCK_Inspired, self).__init__()

        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        lk = large_kernel_size # Short alias

        # HV_ways Encoder
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0,bias=False)
            )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm = norm) # Keep first level standard
        self.HVE_block2 = NormDownsampleLK(ch2, ch3, use_norm = norm, kernel_size=lk) # Use Large Kernel
        self.HVE_block3 = NormDownsampleLK(ch3, ch4, use_norm = norm, kernel_size=lk) # Use Large Kernel

        # HV_ways Decoder
        # Need fusion layers after concatenation
        self.HVD_block3 = NormUpsampleLK(ch4, ch3, use_norm = norm, kernel_size=lk) # Use Large Kernel
        self.fusion_hv3 = nn.Conv2d(ch3 + ch3, ch3, kernel_size=3, padding=1, bias=False) # Fuse skip

        self.HVD_block2 = NormUpsampleLK(ch3, ch2, use_norm = norm, kernel_size=lk) # Use Large Kernel
        self.fusion_hv2 = nn.Conv2d(ch2 + ch2, ch2, kernel_size=3, padding=1, bias=False) # Fuse skip

        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm = norm) # Keep last level standard
        self.fusion_hv1 = nn.Conv2d(ch1 + ch1, ch1, kernel_size=3, padding=1, bias=False) # Fuse skip

        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0,bias=False)
        )

        # I_ways Encoder
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0,bias=False),
            )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm = norm) # Keep first level standard
        self.IE_block2 = NormDownsampleLK(ch2, ch3, use_norm = norm, kernel_size=lk) # Use Large Kernel
        self.IE_block3 = NormDownsampleLK(ch3, ch4, use_norm = norm, kernel_size=lk) # Use Large Kernel

        # I_ways Decoder
        self.ID_block3 = NormUpsampleLK(ch4, ch3, use_norm=norm, kernel_size=lk) # Use Large Kernel
        self.fusion_i3 = nn.Conv2d(ch3 + ch3, ch3, kernel_size=3, padding=1, bias=False) # Fuse skip

        self.ID_block2 = NormUpsampleLK(ch3, ch2, use_norm=norm, kernel_size=lk) # Use Large Kernel
        self.fusion_i2 = nn.Conv2d(ch2 + ch2, ch2, kernel_size=3, padding=1, bias=False) # Fuse skip

        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm) # Keep last level standard
        self.fusion_i1 = nn.Conv2d(ch1 + ch1, ch1, kernel_size=3, padding=1, bias=False) # Fuse skip

        self.ID_block0 =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0,bias=False),
            )

        # LCA Modules remain the same
        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)

        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)

        self.trans = RGB_HVI()
        # --- SpatialAttention modules are removed --- 

    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes)
        
        # --- Encoder --- 
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        i_jump0 = i_enc0 # ch1
        hv_jump0 = hv_0 # ch1

        i_enc2_pre = self.I_LCA1(i_enc1, hv_1) # ch2
        hv_2_pre = self.HV_LCA1(hv_1, i_enc1) # ch2
        i_jump1 = i_enc2_pre # ch2
        hv_jump1 = hv_2_pre # ch2
        i_enc2 = self.IE_block2(i_enc2_pre) # ch3 (Using NormDownsampleLK)
        hv_2 = self.HVE_block2(hv_2_pre) # ch3 (Using NormDownsampleLK)

        i_enc3_pre = self.I_LCA2(i_enc2, hv_2) # ch3
        hv_3_pre = self.HV_LCA2(hv_2, i_enc2) # ch3
        i_jump2 = i_enc3_pre # ch3
        hv_jump2 = hv_3_pre # ch3
        i_enc3 = self.IE_block3(i_enc3_pre) # ch4 (Using NormDownsampleLK)
        hv_3 = self.HVE_block3(hv_3_pre) # ch4 (Using NormDownsampleLK)

        i_enc4 = self.I_LCA3(i_enc3, hv_3) # ch4
        hv_4 = self.HV_LCA3(hv_3, i_enc3) # ch4

        # --- Bottleneck LCA --- 
        i_dec4 = self.I_LCA4(i_enc4,hv_4) # ch4
        hv_4_postlca = self.HV_LCA4(hv_4, i_enc4) # ch4 

        # --- Decoder with Large Kernels and Fusion --- 
        hv_3_upsampled_concat = self.HVD_block3(hv_4_postlca, hv_jump2) # Output: 2*ch3 (Using NormUpsampleLK)
        hv_3 = self.fusion_hv3(hv_3_upsampled_concat) # Output: ch3
        # hv_3 = self.sa_hv3(hv_3) # SA removed

        i_dec3_upsampled_concat = self.ID_block3(i_dec4, i_jump2) # Output: 2*ch3 (Using NormUpsampleLK)
        i_dec3 = self.fusion_i3(i_dec3_upsampled_concat) # Output: ch3
        # i_dec3 = self.sa_i3(i_dec3) # SA removed

        i_dec2_prelca = self.I_LCA5(i_dec3, hv_3) # ch2
        hv_2_prelca = self.HV_LCA5(hv_3, i_dec3) # ch2

        hv_2_upsampled_concat = self.HVD_block2(hv_2_prelca, hv_jump1) # Output: 2*ch2 (Using NormUpsampleLK)
        hv_2 = self.fusion_hv2(hv_2_upsampled_concat) # Output: ch2
        # hv_2 = self.sa_hv2(hv_2) # SA removed

        i_dec2_upsampled_concat = self.ID_block2(i_dec2_prelca, i_jump1) # Output: 2*ch2 (Using NormUpsampleLK)
        i_dec2 = self.fusion_i2(i_dec2_upsampled_concat) # Output: ch2
        # i_dec2 = self.sa_i2(i_dec2) # SA removed

        i_dec1_prelca = self.I_LCA6(i_dec2, hv_2) # ch1
        hv_1_prelca = self.HV_LCA6(hv_2, i_dec2) # ch1

        i_dec1_upsampled_concat = self.ID_block1(i_dec1_prelca, i_jump0) # Output: 2*ch1 (Using Standard NormUpsample)
        i_dec1 = self.fusion_i1(i_dec1_upsampled_concat) # Output: ch1
        # i_dec1 = self.sa_i1(i_dec1) # SA removed
        i_dec0 = self.ID_block0(i_dec1) # Output: 1 channel

        hv_1_upsampled_concat = self.HVD_block1(hv_1_prelca, hv_jump0) # Output: 2*ch1 (Using Standard NormUpsample)
        hv_1 = self.fusion_hv1(hv_1_upsampled_concat) # Output: ch1
        # hv_1 = self.sa_hv1(hv_1) # SA removed
        hv_0 = self.HVD_block0(hv_1) # Output: 2 channels

        # --- Final Output --- 
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi # Residual connection
        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb

    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi 