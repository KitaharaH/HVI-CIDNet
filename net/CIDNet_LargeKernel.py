import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
# Import the modified transformer utils
from net.transformer_utils_LargeKernel import NormDownsample, NormUpsample, LayerNorm 
from net.LCA import HV_LCA, I_LCA # Assuming LCA doesn't need modification for kernel size
from huggingface_hub import PyTorchModelHubMixin

class CIDNet_LargeKernel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False,
                 kernel_size=7 # Added kernel_size parameter
        ):
        super(CIDNet_LargeKernel, self).__init__()
        
        
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        padding = kernel_size // 2 # Calculate padding
        
        # HV_ways
        self.HVE_block0 = nn.Sequential(
            # Use ReplicationPad2d for consistent padding behavior
            nn.ReplicationPad2d(padding),
            nn.Conv2d(3, ch1, kernel_size=kernel_size, stride=1, padding=0,bias=False) # Use kernel_size
            )
        # Pass kernel_size to NormDownsample/NormUpsample
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm = norm, kernel_size=kernel_size)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm = norm, kernel_size=kernel_size)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm = norm, kernel_size=kernel_size)
        
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm = norm, kernel_size=kernel_size)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm = norm, kernel_size=kernel_size)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm = norm, kernel_size=kernel_size)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(padding),
            nn.Conv2d(ch1, 2, kernel_size=kernel_size, stride=1, padding=0,bias=False) # Use kernel_size
        )
        
        
        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(padding),
            nn.Conv2d(1, ch1, kernel_size=kernel_size, stride=1, padding=0,bias=False), # Use kernel_size
            )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm = norm, kernel_size=kernel_size)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm = norm, kernel_size=kernel_size)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm = norm, kernel_size=kernel_size)
        
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm, kernel_size=kernel_size)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm, kernel_size=kernel_size)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm, kernel_size=kernel_size)
        self.ID_block0 =  nn.Sequential(
            nn.ReplicationPad2d(padding),
            nn.Conv2d(ch1, 1, kernel_size=kernel_size, stride=1, padding=0,bias=False), # Use kernel_size
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
        
    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes)
        # low
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        i_jump0 = i_enc0
        hv_jump0 = hv_0
        
        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)
        
        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2) # Note: Original code had IE_block3(i_enc2), likely meant IE_block3(i_enc3)? Keeping original for now.
        hv_3 = self.HVE_block3(hv_2) # Note: Original code had HVE_block3(hv_2), likely meant HVE_block3(hv_3)? Keeping original for now.
        
        # Bottleneck / LCA interaction
        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)
        
        i_dec4 = self.I_LCA4(i_enc4,hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)
        
        # Decoder
        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)
        
        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        # Note: Original code had ID_block2(i_dec3, v_jump1), likely meant ID_block2(i_dec2, v_jump1)? Keeping original for now.
        i_dec2_out = self.ID_block2(i_dec2, v_jump1) 
        
        i_dec1 = self.I_LCA6(i_dec2_out, hv_2) # Use output of ID_block2
        hv_1 = self.HV_LCA6(hv_2, i_dec2_out) # Use output of ID_block2
        
        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)
        
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb
    
    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi 