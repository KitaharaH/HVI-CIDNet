import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import * # Assuming NormDownsample is here
from net.LCA import *
from huggingface_hub import PyTorchModelHubMixin
import torch.nn.functional as F

# +++ 定义 SpatialAttention 模块 +++
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
        return x * self.sigmoid(y)
# +++ 结束定义 +++

# +++ 定义 PixelShuffle 上采样模块 +++
class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=False, upscale_factor=2):
        super(PixelShuffleUpsample, self).__init__()
        self.upscale_factor = upscale_factor
        # PixelShuffle requires channels = out_channels * (upscale_factor ** 2)
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.norm = nn.InstanceNorm2d(out_channels) if use_norm else nn.Identity()
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, skip_input):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.norm(x)
        x = self.relu(x)
        # Concatenate with skip connection after upsampling
        x = torch.cat([x, skip_input], dim=1)
        # Optional: Add another conv layer to fuse concatenated features
        # You might need to adjust channel dimensions here if you add this
        # fuse_conv = nn.Conv2d(out_channels + skip_input.size(1), out_channels, kernel_size=3, padding=1, bias=False)
        # x = fuse_conv(x)
        return x

# Assuming NormDownsample remains the same from transformer_utils
# If NormDownsample is not in transformer_utils, it needs to be defined here or imported correctly.
# For simplicity, let's assume it exists. Example structure:
class NormDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=False):
        super(NormDownsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(out_channels) if use_norm else nn.Identity()
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class CIDNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False
        ):
        super(CIDNet, self).__init__()


        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads

        # HV_ways
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0,bias=False)
            )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm = norm)

        # --- Modified Upsampling Blocks ---
        # Note: The input to the upsample block needs careful channel handling.
        # The skip connection is concatenated *after* upsampling in PixelShuffleUpsample.
        # So, the input channel to the conv inside PixelShuffleUpsample is just the channel from the previous layer.
        # The output channel of PixelShuffleUpsample (after fusing skip connection) needs consideration.
        # Let's adjust the fusion logic within PixelShuffleUpsample or add fusion layers after it.
        # For now, PixelShuffleUpsample concatenates and returns features. We need subsequent layers to handle this.

        # Example: HVD_block3 takes ch4 from previous layer, skip is ch3.
        # PixelShuffleUpsample conv takes ch4, outputs ch3*(2^2). PixelShuffle makes it ch3.
        # Concatenates with ch3 (skip), output is 2*ch3.
        # We need a conv layer after this to reduce channels back to ch3 if the next layer expects ch3.

        # Let's add fusion conv layers after the upsampling+concat
        self.HVD_block3 = PixelShuffleUpsample(ch4, ch3, use_norm = norm)
        self.fusion_hv3 = nn.Conv2d(ch3 + ch3, ch3, kernel_size=3, padding=1, bias=False) # Fuse skip

        self.HVD_block2 = PixelShuffleUpsample(ch3, ch2, use_norm = norm)
        self.fusion_hv2 = nn.Conv2d(ch2 + ch2, ch2, kernel_size=3, padding=1, bias=False) # Fuse skip

        self.HVD_block1 = PixelShuffleUpsample(ch2, ch1, use_norm = norm)
        self.fusion_hv1 = nn.Conv2d(ch1 + ch1, ch1, kernel_size=3, padding=1, bias=False) # Fuse skip
        # --- End Modified Upsampling Blocks ---

        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0,bias=False)
        )


        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0,bias=False),
            )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm = norm)

        # --- Modified Upsampling Blocks ---
        self.ID_block3 = PixelShuffleUpsample(ch4, ch3, use_norm=norm)
        self.fusion_i3 = nn.Conv2d(ch3 + ch3, ch3, kernel_size=3, padding=1, bias=False) # Fuse skip

        self.ID_block2 = PixelShuffleUpsample(ch3, ch2, use_norm=norm)
        self.fusion_i2 = nn.Conv2d(ch2 + ch2, ch2, kernel_size=3, padding=1, bias=False) # Fuse skip

        self.ID_block1 = PixelShuffleUpsample(ch2, ch1, use_norm=norm)
        self.fusion_i1 = nn.Conv2d(ch1 + ch1, ch1, kernel_size=3, padding=1, bias=False) # Fuse skip
        # --- End Modified Upsampling Blocks ---

        self.ID_block0 =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0,bias=False),
            )

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

        # SpatialAttention instances remain the same
        self.sa_hv3 = SpatialAttention()
        self.sa_i3 = SpatialAttention()
        self.sa_hv2 = SpatialAttention()
        self.sa_i2 = SpatialAttention()
        self.sa_hv1 = SpatialAttention()
        self.sa_i1 = SpatialAttention()


    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes)
        # low
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        i_jump0 = i_enc0 # ch1
        hv_jump0 = hv_0 # ch1

        i_enc2_pre = self.I_LCA1(i_enc1, hv_1) # ch2
        hv_2_pre = self.HV_LCA1(hv_1, i_enc1) # ch2
        i_jump1 = i_enc2_pre # ch2 - Renamed from v_jump1
        hv_jump1 = hv_2_pre # ch2
        i_enc2 = self.IE_block2(i_enc2_pre) # ch3
        hv_2 = self.HVE_block2(hv_2_pre) # ch3

        i_enc3_pre = self.I_LCA2(i_enc2, hv_2) # ch3
        hv_3_pre = self.HV_LCA2(hv_2, i_enc2) # ch3
        i_jump2 = i_enc3_pre # ch3 - Renamed from v_jump2
        hv_jump2 = hv_3_pre # ch3
        i_enc3 = self.IE_block3(i_enc2) # ch4 - Error in original? Should be i_enc3 = self.IE_block3(i_enc3_pre)
        i_enc3 = self.IE_block3(i_enc3_pre) # Corrected ch4
        hv_3 = self.HVE_block3(hv_2) # ch4 - Error in original? Should be hv_3 = self.HVE_block3(hv_3_pre)
        hv_3 = self.HVE_block3(hv_3_pre) # Corrected ch4


        i_enc4 = self.I_LCA3(i_enc3, hv_3) # ch4
        hv_4 = self.HV_LCA3(hv_3, i_enc3) # ch4

        i_dec4 = self.I_LCA4(i_enc4,hv_4) # ch4
        hv_4_postlca = self.HV_LCA4(hv_4, i_enc4) # ch4 - Renamed hv_4 to avoid overwrite


        # Decoder with PixelShuffle and Fusion
        hv_3_upsampled = self.HVD_block3(hv_4_postlca, hv_jump2) # Output: 2*ch3 (after concat)
        hv_3 = self.fusion_hv3(hv_3_upsampled) # Output: ch3
        hv_3 = self.sa_hv3(hv_3) # Apply SA after fusion

        i_dec3_upsampled = self.ID_block3(i_dec4, i_jump2) # Output: 2*ch3 (after concat)
        i_dec3 = self.fusion_i3(i_dec3_upsampled) # Output: ch3
        i_dec3 = self.sa_i3(i_dec3) # Apply SA after fusion


        i_dec2_prelca = self.I_LCA5(i_dec3, hv_3) # ch2
        hv_2_prelca = self.HV_LCA5(hv_3, i_dec3) # ch2


        hv_2_upsampled = self.HVD_block2(hv_2_prelca, hv_jump1) # Output: 2*ch2 (after concat)
        hv_2 = self.fusion_hv2(hv_2_upsampled) # Output: ch2
        hv_2 = self.sa_hv2(hv_2) # Apply SA after fusion

        i_dec2_upsampled = self.ID_block2(i_dec2_prelca, i_jump1) # Output: 2*ch2 (after concat)
        i_dec2 = self.fusion_i2(i_dec2_upsampled) # Output: ch2
        i_dec2 = self.sa_i2(i_dec2) # Apply SA after fusion


        i_dec1_prelca = self.I_LCA6(i_dec2, hv_2) # ch1
        hv_1_prelca = self.HV_LCA6(hv_2, i_dec2) # ch1


        i_dec1_upsampled = self.ID_block1(i_dec1_prelca, i_jump0) # Output: 2*ch1 (after concat)
        i_dec1 = self.fusion_i1(i_dec1_upsampled) # Output: ch1
        i_dec1 = self.sa_i1(i_dec1) # Apply SA after fusion
        i_dec0 = self.ID_block0(i_dec1) # Output: 1 channel

        hv_1_upsampled = self.HVD_block1(hv_1_prelca, hv_jump0) # Output: 2*ch1 (after concat)
        hv_1 = self.fusion_hv1(hv_1_upsampled) # Output: ch1
        hv_1 = self.sa_hv1(hv_1) # Apply SA after fusion
        hv_0 = self.HVD_block0(hv_1) # Output: 2 channels


        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb

    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi 