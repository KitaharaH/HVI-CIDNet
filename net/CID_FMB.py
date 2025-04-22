import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.LCA import *
from net.FMB import FMB # Import FMB
from huggingface_hub import PyTorchModelHubMixin

class CIDNet(nn.Module, PyTorchModelHubMixin): # Rename class
    def __init__(self,
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False,
                 ffn_scale=2.0 # Add ffn_scale for FMB
        ):
        super(CIDNet, self).__init__() # Update super call


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

        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm = norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm = norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm = norm)
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

        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0,bias=False),
            )

        # Instantiate FMB blocks
        self.hv_fmb1 = FMB(ch2, ffn_scale=ffn_scale)
        self.i_fmb1 = FMB(ch2, ffn_scale=ffn_scale)
        self.hv_fmb2 = FMB(ch3, ffn_scale=ffn_scale)
        self.i_fmb2 = FMB(ch3, ffn_scale=ffn_scale)
        self.hv_fmb3 = FMB(ch4, ffn_scale=ffn_scale)
        self.i_fmb3 = FMB(ch4, ffn_scale=ffn_scale)
        self.hv_fmb_b = FMB(ch4, ffn_scale=ffn_scale) # Bottleneck FMB
        self.i_fmb_b = FMB(ch4, ffn_scale=ffn_scale)  # Bottleneck FMB
        self.hv_fmb5 = FMB(ch3, ffn_scale=ffn_scale)
        self.i_fmb5 = FMB(ch3, ffn_scale=ffn_scale)
        self.hv_fmb6 = FMB(ch2, ffn_scale=ffn_scale)
        self.i_fmb6 = FMB(ch2, ffn_scale=ffn_scale)

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
        # Encoder Path
        # Stage 0
        i_enc0 = self.IE_block0(i)
        hv_0 = self.HVE_block0(hvi)
        i_jump0 = i_enc0
        hv_jump0 = hv_0

        # Stage 1
        i_enc1 = self.IE_block1(i_enc0)
        hv_1 = self.HVE_block1(hv_0)
        # Apply FMB before LCA1
        i_enc1_enhanced = self.i_fmb1(i_enc1)
        hv_1_enhanced = self.hv_fmb1(hv_1)
        i_enc2 = self.I_LCA1(i_enc1_enhanced, hv_1_enhanced)
        hv_2 = self.HV_LCA1(hv_1_enhanced, i_enc1_enhanced)
        v_jump1 = i_enc2 # Skip connection from I path (after LCA1)
        hv_jump1 = hv_2 # Skip connection from HV path (after LCA1)

        # Stage 2
        i_enc2 = self.IE_block2(i_enc2) # Downsample I path
        hv_2 = self.HVE_block2(hv_2)   # Downsample HV path
        # Apply FMB before LCA2
        i_enc2_enhanced = self.i_fmb2(i_enc2)
        hv_2_enhanced = self.hv_fmb2(hv_2)
        i_enc3 = self.I_LCA2(i_enc2_enhanced, hv_2_enhanced)
        hv_3 = self.HV_LCA2(hv_2_enhanced, i_enc2_enhanced)
        v_jump2 = i_enc3 # Skip connection from I path (after LCA2)
        hv_jump2 = hv_3 # Skip connection from HV path (after LCA2)

        # Stage 3 (Bottleneck input)
        i_enc3 = self.IE_block3(i_enc2) # Downsample I path to bottleneck
        hv_3 = self.HVE_block3(hv_2)   # Downsample HV path to bottleneck
        # Apply FMB before LCA3
        i_enc3_enhanced = self.i_fmb3(i_enc3)
        hv_3_enhanced = self.hv_fmb3(hv_3)
        i_enc4 = self.I_LCA3(i_enc3_enhanced, hv_3_enhanced) # Bottleneck features I
        hv_4 = self.HV_LCA3(hv_3_enhanced, i_enc3_enhanced) # Bottleneck features HV

        # Bottleneck LCA
        # Apply FMB before LCA4
        i_enc4_enhanced = self.i_fmb_b(i_enc4)
        hv_4_enhanced = self.hv_fmb_b(hv_4)
        i_dec4 = self.I_LCA4(i_enc4_enhanced, hv_4_enhanced) # Output from bottleneck LCA (I path)
        hv_dec4 = self.HV_LCA4(hv_4_enhanced, i_enc4_enhanced) # Output from bottleneck LCA (HV path)

        # Decoder Path
        # Stage 3 Decode
        hv_3_up = self.HVD_block3(hv_dec4, hv_jump2) # Upsample HV + Skip
        i_dec3_up = self.ID_block3(i_dec4, v_jump2)   # Upsample I + Skip
        # Apply FMB before LCA5
        i_dec3_up_enhanced = self.i_fmb5(i_dec3_up)
        hv_3_up_enhanced = self.hv_fmb5(hv_3_up)
        i_dec2_lca5 = self.I_LCA5(i_dec3_up_enhanced, hv_3_up_enhanced) # Output of LCA5 (I path)
        hv_2_lca5 = self.HV_LCA5(hv_3_up_enhanced, i_dec3_up_enhanced) # Output of LCA5 (HV path)

        # Stage 2 Decode
        hv_2_up = self.HVD_block2(hv_2_lca5, hv_jump1) # Upsample HV + Skip
        i_dec2_up = self.ID_block2(i_dec2_lca5, v_jump1)   # Upsample I + Skip
        # Apply FMB before LCA6
        i_dec2_up_enhanced = self.i_fmb6(i_dec2_up)
        hv_2_up_enhanced = self.hv_fmb6(hv_2_up)
        i_dec1 = self.I_LCA6(i_dec2_up_enhanced, hv_2_up_enhanced) # Output of LCA6 (I path)
        hv_1 = self.HV_LCA6(hv_2_up_enhanced, i_dec2_up_enhanced) # Output of LCA6 (HV path)

        # Stage 1 Decode
        i_dec1 = self.ID_block1(i_dec1, i_jump0) # Upsample I + Skip
        hv_1 = self.HVD_block1(hv_1, hv_jump0)   # Upsample HV + Skip

        # Final Output layers
        i_dec0 = self.ID_block0(i_dec1) # Final I conv
        hv_0 = self.HVD_block0(hv_1)   # Final HV conv

        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb

    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi 