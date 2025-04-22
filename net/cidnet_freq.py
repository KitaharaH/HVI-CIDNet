import torch
import torch.nn as nn
# Ensure necessary imports from the original CIDNet context are present
# These might include specific utility modules if they were not copied
from net.HVI_transform import RGB_HVI
from net.transformer_utils import NormDownsample, NormUpsample # Assuming these are still needed directly
from net.lca_freq import LCA_Freq # Import the new LCA module
from huggingface_hub import PyTorchModelHubMixin

class CIDNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False,
                 lca_bias=False, # Added option for bias in LCA
                 freq_p=0.25     # Added option for frequency mask threshold
        ):
        super(CIDNet, self).__init__()

        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads # head1 seems unused in original LCA calls

        # HV_ways (Keep original encoder/decoder blocks)
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(2, ch1, 3, stride=1, padding=0,bias=False) # Input channels for HV should be 2
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


        # I_ways (Keep original encoder/decoder blocks)
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

        # --- Replace LCA modules with LCA_Freq ---
        # Note: Using common `bias=lca_bias` and `freq_p=freq_p` for all LCA_Freq instances.
        # Set `residual_gdfn` based on original HV_LCA (False) or I_LCA (True).

        # Original HV_LCA instances replaced with LCA_Freq(..., residual_gdfn=False)
        self.HV_LCA1 = LCA_Freq(dim=ch2, num_heads=head2, residual_gdfn=False, bias=lca_bias, freq_p=freq_p)
        self.HV_LCA2 = LCA_Freq(dim=ch3, num_heads=head3, residual_gdfn=False, bias=lca_bias, freq_p=freq_p)
        self.HV_LCA3 = LCA_Freq(dim=ch4, num_heads=head4, residual_gdfn=False, bias=lca_bias, freq_p=freq_p)
        self.HV_LCA4 = LCA_Freq(dim=ch4, num_heads=head4, residual_gdfn=False, bias=lca_bias, freq_p=freq_p)
        self.HV_LCA5 = LCA_Freq(dim=ch3, num_heads=head3, residual_gdfn=False, bias=lca_bias, freq_p=freq_p)
        self.HV_LCA6 = LCA_Freq(dim=ch2, num_heads=head2, residual_gdfn=False, bias=lca_bias, freq_p=freq_p)

        # Original I_LCA instances replaced with LCA_Freq(..., residual_gdfn=True)
        self.I_LCA1 = LCA_Freq(dim=ch2, num_heads=head2, residual_gdfn=True, bias=lca_bias, freq_p=freq_p)
        self.I_LCA2 = LCA_Freq(dim=ch3, num_heads=head3, residual_gdfn=True, bias=lca_bias, freq_p=freq_p)
        self.I_LCA3 = LCA_Freq(dim=ch4, num_heads=head4, residual_gdfn=True, bias=lca_bias, freq_p=freq_p)
        self.I_LCA4 = LCA_Freq(dim=ch4, num_heads=head4, residual_gdfn=True, bias=lca_bias, freq_p=freq_p)
        self.I_LCA5 = LCA_Freq(dim=ch3, num_heads=head3, residual_gdfn=True, bias=lca_bias, freq_p=freq_p)
        self.I_LCA6 = LCA_Freq(dim=ch2, num_heads=head2, residual_gdfn=True, bias=lca_bias, freq_p=freq_p)

        self.trans = RGB_HVI()

    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        # Assuming HVI channels are [H, V, I] -> H=0, V=1, I=2
        hv_rgb = hvi[:,0:2,:,:].to(dtypes) # Extract HV channels (channels 0, 1)
        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes) # Extract I channel (channel 2)


        # --- Encoder ---
        # Level 0
        i_enc0 = self.IE_block0(i)
        hv_enc0 = self.HVE_block0(hv_rgb) # Input should be HV (2 channels)

        # Level 1
        i_enc1 = self.IE_block1(i_enc0)
        hv_enc1 = self.HVE_block1(hv_enc0)
        i_jump0 = i_enc0 # Storing pre-downsample features for skip connection
        hv_jump0 = hv_enc0

        # LCA 1
        i_enc2_pre = self.I_LCA1(i_enc1, hv_enc1) # I stream updated using HV stream
        hv_enc2_pre = self.HV_LCA1(hv_enc1, i_enc1) # HV stream updated using I stream
        i_jump1 = i_enc2_pre # Storing features after LCA for skip connection
        hv_jump1 = hv_enc2_pre

        # Level 2
        i_enc2 = self.IE_block2(i_enc2_pre)
        hv_enc2 = self.HVE_block2(hv_enc2_pre)

        # LCA 2
        i_enc3_pre = self.I_LCA2(i_enc2, hv_enc2)
        hv_enc3_pre = self.HV_LCA2(hv_enc2, i_enc2)
        i_jump2 = i_enc3_pre
        hv_jump2 = hv_enc3_pre

        # Level 3
        i_enc3 = self.IE_block3(i_enc3_pre) # Corrected: use i_enc3_pre
        hv_enc3 = self.HVE_block3(hv_enc3_pre) # Corrected: use hv_enc3_pre

        # LCA 3 (Bottleneck LCA - applied before down/upsample)
        i_enc4 = self.I_LCA3(i_enc3, hv_enc3)
        hv_enc4 = self.HV_LCA3(hv_enc3, i_enc3)

        # --- Decoder ---
        # LCA 4 (Bottleneck LCA - applied before upsample)
        i_dec4_pre = self.I_LCA4(i_enc4, hv_enc4)
        hv_dec4_pre = self.HV_LCA4(hv_enc4, i_enc4)

        # Level 3 Decode
        hv_dec3_pre = self.HVD_block3(hv_dec4_pre, hv_jump2) # Upsample HV, fuse with skip
        i_dec3_pre = self.ID_block3(i_dec4_pre, i_jump2)   # Upsample I, fuse with skip

        # LCA 5
        i_dec2_pre = self.I_LCA5(i_dec3_pre, hv_dec3_pre)
        hv_dec2_pre = self.HV_LCA5(hv_dec3_pre, i_dec3_pre)

        # Level 2 Decode
        hv_dec2 = self.HVD_block2(hv_dec2_pre, hv_jump1) # Upsample HV, fuse with skip
        # Corrected: ID_block2 input should be i_dec2_pre, not i_dec3_pre
        i_dec2 = self.ID_block2(i_dec2_pre, i_jump1)   # Upsample I, fuse with skip

        # LCA 6
        i_dec1_pre = self.I_LCA6(i_dec2, hv_dec2)
        hv_dec1_pre = self.HV_LCA6(hv_dec2, i_dec2)

        # Level 1 Decode
        i_dec1 = self.ID_block1(i_dec1_pre, i_jump0) # Upsample I, fuse with skip
        hv_dec1 = self.HVD_block1(hv_dec1_pre, hv_jump0) # Upsample HV, fuse with skip

        # Output layers
        i_dec0 = self.ID_block0(i_dec1)
        hv_dec0 = self.HVD_block0(hv_dec1)

        # Combine HVI and transform back to RGB
        # Original: output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        # Assuming hv_dec0 is dim 2 and i_dec0 is dim 1
        output_hvi = torch.cat([hv_dec0, i_dec0], dim=1)
        output_hvi = output_hvi + hvi # Add residual HVI

        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb

    # Keep the HVIT method if needed
    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi 