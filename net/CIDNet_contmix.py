import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
# Import the modified LCA modules that use ContMix
from net.LCA_contmix import HV_LCA, I_LCA 
from huggingface_hub import PyTorchModelHubMixin

class CIDNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False
        ):
        super(CIDNet, self).__init__()
        
        
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        
        # HV_ways (Encoder/Decoder blocks remain standard Conv)
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
        
        
        # I_ways (Encoder/Decoder blocks remain standard Conv)
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
        
        # Instantiate the LCA modules imported from LCA_contmix.py
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
        
        # --- Encoder --- 
        # Level 0
        i_enc0 = self.IE_block0(i)
        hv_0 = self.HVE_block0(hvi)
        i_jump0 = i_enc0 # Skip connection
        hv_jump0 = hv_0 # Skip connection
        
        # Level 1
        i_enc1 = self.IE_block1(i_enc0)
        hv_1 = self.HVE_block1(hv_0)
        # Cross Attention + Feature Enhancement/Denoise using ContMix LCA
        i_enc2_pre = self.I_LCA1(i_enc1, hv_1) 
        hv_2_pre = self.HV_LCA1(hv_1, i_enc1)
        i_jump1 = i_enc2_pre # Skip connection
        hv_jump1 = hv_2_pre # Skip connection
        
        # Level 2
        i_enc2 = self.IE_block2(i_enc2_pre)
        hv_2 = self.HVE_block2(hv_2_pre)
        # Cross Attention + Feature Enhancement/Denoise using ContMix LCA
        i_enc3_pre = self.I_LCA2(i_enc2, hv_2)
        hv_3_pre = self.HV_LCA2(hv_2, i_enc2)
        i_jump2 = i_enc3_pre # Skip connection
        hv_jump2 = hv_3_pre # Skip connection
        
        # Level 3 (Bottleneck)
        i_enc3 = self.IE_block3(i_enc3_pre)
        hv_3 = self.HVE_block3(hv_3_pre)
        # Cross Attention + Feature Enhancement/Denoise using ContMix LCA
        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)
        
        # --- Bottleneck LCA --- 
        i_dec4_pre = self.I_LCA4(i_enc4, hv_4)
        hv_4_dec = self.HV_LCA4(hv_4, i_enc4)
        
        # --- Decoder --- 
        # Level 3
        hv_3_dec = self.HVD_block3(hv_4_dec, hv_jump2) # Upsample HV
        i_dec3 = self.ID_block3(i_dec4_pre, i_jump2) # Upsample I
        # Cross Attention + Feature Enhancement/Denoise using ContMix LCA
        i_dec2_pre = self.I_LCA5(i_dec3, hv_3_dec)
        hv_2_dec = self.HV_LCA5(hv_3_dec, i_dec3)
        
        # Level 2
        hv_2_final = self.HVD_block2(hv_2_dec, hv_jump1) # Upsample HV
        i_dec2 = self.ID_block2(i_dec2_pre, i_jump1) # Upsample I
        # Cross Attention + Feature Enhancement/Denoise using ContMix LCA
        i_dec1_pre = self.I_LCA6(i_dec2, hv_2_final)
        hv_1_dec = self.HV_LCA6(hv_2_final, i_dec2)
        
        # Level 1
        i_dec1 = self.ID_block1(i_dec1_pre, i_jump0) # Upsample I
        hv_1_final = self.HVD_block1(hv_1_dec, hv_jump0) # Upsample HV
        
        # Level 0 (Output layers)
        i_dec0 = self.ID_block0(i_dec1)
        hv_0_final = self.HVD_block0(hv_1_final)
        
        # Combine outputs
        output_hvi = torch.cat([hv_0_final, i_dec0], dim=1) + hvi # Add residual connection
        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb
    
    # Keep helper methods if needed
    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi 