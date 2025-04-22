import torch
import torch.nn as nn
import torch.nn.functional as F # Added for upsampling
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.LCA import *
from net.FusionModules import DecoderAttentionFusion # Added
from huggingface_hub import PyTorchModelHubMixin

class CIDNet_EMCAD_Fusion(nn.Module, PyTorchModelHubMixin): # Renamed class
    def __init__(self,
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False
        ):
        super(CIDNet_EMCAD_Fusion, self).__init__()


        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads

        # HV_ways Encoder
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0,bias=False)
            )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm = norm)

        # I_ways Encoder
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0,bias=False),
            )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm = norm)

        # Upsampling Layers
        self.upsample_43 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_32 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_21 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # HV_ways Decoder Fusion Blocks
        self.HVD_fuse3 = DecoderAttentionFusion(ch4, ch3, ch3, use_norm=norm)
        self.HVD_fuse2 = DecoderAttentionFusion(ch3, ch2, ch2, use_norm=norm)
        self.HVD_fuse1 = DecoderAttentionFusion(ch2, ch1, ch1, use_norm=norm)
        self.HVD_block0 = nn.Sequential( # Final output conv remains
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0,bias=False)
        )

        # I_ways Decoder Fusion Blocks
        self.ID_fuse3 = DecoderAttentionFusion(ch4, ch3, ch3, use_norm=norm)
        self.ID_fuse2 = DecoderAttentionFusion(ch3, ch2, ch2, use_norm=norm)
        self.ID_fuse1 = DecoderAttentionFusion(ch2, ch1, ch1, use_norm=norm)
        self.ID_block0 =  nn.Sequential( # Final output conv remains
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0,bias=False),
            )

        # LCA Blocks (remain the same)
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

    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes)

        # ---- Encoder ----
        # Level 0
        i_enc0 = self.IE_block0(i)
        hv_enc0 = self.HVE_block0(hvi)
        i_jump0 = i_enc0
        hv_jump0 = hv_enc0

        # Level 1
        i_enc1 = self.IE_block1(i_enc0)
        hv_enc1 = self.HVE_block1(hv_enc0)
        # LCA 1
        i_feat1 = self.I_LCA1(i_enc1, hv_enc1)
        hv_feat1 = self.HV_LCA1(hv_enc1, i_enc1)
        i_jump1 = i_feat1 # Use LCA output as skip connection
        hv_jump1 = hv_feat1 # Use LCA output as skip connection

        # Level 2
        i_enc2 = self.IE_block2(i_feat1)
        hv_enc2 = self.HVE_block2(hv_feat1)
        # LCA 2
        i_feat2 = self.I_LCA2(i_enc2, hv_enc2)
        hv_feat2 = self.HV_LCA2(hv_enc2, i_enc2)
        i_jump2 = i_feat2 # Use LCA output as skip connection
        hv_jump2 = hv_feat2 # Use LCA output as skip connection

        # Level 3
        i_enc3 = self.IE_block3(i_feat2)
        hv_enc3 = self.HVE_block3(hv_feat2)
        # LCA 3
        i_feat3 = self.I_LCA3(i_enc3, hv_enc3)
        hv_feat3 = self.HV_LCA3(hv_enc3, i_enc3)
        # No skip connection needed from the bottleneck

        # ---- Bottleneck ----
        # LCA 4
        i_dec4 = self.I_LCA4(i_feat3, hv_feat3) # Renaming for clarity (start of decoder)
        hv_dec4 = self.HV_LCA4(hv_feat3, i_feat3) # Renaming for clarity (start of decoder)


        # ---- Decoder ----
        # Level 3 Decode
        hv_dec4_up = self.upsample_43(hv_dec4)
        hv_dec3 = self.HVD_fuse3(hv_dec4_up, hv_jump2) # Fuse with skip from enc lvl 2
        i_dec4_up = self.upsample_43(i_dec4)
        i_dec3 = self.ID_fuse3(i_dec4_up, i_jump2)   # Fuse with skip from enc lvl 2

        # LCA 5
        i_feat_dec2_pre = self.I_LCA5(i_dec3, hv_dec3)
        hv_feat_dec2_pre = self.HV_LCA5(hv_dec3, i_dec3)

        # Level 2 Decode
        hv_feat_dec2_pre_up = self.upsample_32(hv_feat_dec2_pre)
        hv_dec2 = self.HVD_fuse2(hv_feat_dec2_pre_up, hv_jump1) # Fuse with skip from enc lvl 1
        i_feat_dec2_pre_up = self.upsample_32(i_feat_dec2_pre)
        i_dec2 = self.ID_fuse2(i_feat_dec2_pre_up, i_jump1)   # Fuse with skip from enc lvl 1

        # LCA 6
        i_feat_dec1_pre = self.I_LCA6(i_dec2, hv_dec2)
        hv_feat_dec1_pre = self.HV_LCA6(hv_dec2, i_dec2)

        # Level 1 Decode
        hv_feat_dec1_pre_up = self.upsample_21(hv_feat_dec1_pre)
        hv_dec1 = self.HVD_fuse1(hv_feat_dec1_pre_up, hv_jump0) # Fuse with skip from enc lvl 0
        i_feat_dec1_pre_up = self.upsample_21(i_feat_dec1_pre)
        i_dec1 = self.ID_fuse1(i_feat_dec1_pre_up, i_jump0)   # Fuse with skip from enc lvl 0

        # Final Output Layers
        i_dec0 = self.ID_block0(i_dec1)
        hv_dec0 = self.HVD_block0(hv_dec1)

        # Combine and transform back to RGB
        output_hvi = torch.cat([hv_dec0, i_dec0], dim=1) + hvi # Residual connection
        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb

    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi 