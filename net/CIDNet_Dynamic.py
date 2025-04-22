# 预期在 net/CIDNet.py 中的代码
import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
# Import standard transformer utils
from net.transformer_utils import NormDownsample, NormUpsample#, LayerNorm # LayerNorm not directly used here
# Import the NEW Dynamic LCA modules
from net.LCA_Dynamic import HV_LCA_Dynamic, I_LCA_Dynamic
from huggingface_hub import PyTorchModelHubMixin

class CIDNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 channels=[36, 40, 80, 160],
                 heads=[2, 4, 8], # Heads for LCA stages (ch2, ch3, ch4 levels)
                 norm=False,
                 dynamic_lca_kernel_size=7,
                 dynamic_lca_ctx_ratio=2
        ):
        super(CIDNet, self).__init__()


        [ch1, ch2, ch3, ch4] = channels
        # Ensure heads list matches channel stages where LCA is applied (ch2, ch3, ch4)
        if len(heads) != 3:
             raise ValueError(f"Length of heads list ({len(heads)}) should match the number of LCA stages (3 for ch2, ch3, ch4 levels)")
        head2, head3, head4 = heads[0], heads[1], heads[2]
        # Use the same heads for corresponding channel sizes in decoder: head4, head3, head2


        # HV_ways (Keep standard convolutions here)
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


        # I_ways (Keep standard convolutions here)
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

        # Instantiate DYNAMIC LCA modules
        # Encoder LCAs
        self.I_LCA1 = I_LCA_Dynamic(ch2, head2, kernel_size=dynamic_lca_kernel_size, ctx_dim_ratio=dynamic_lca_ctx_ratio)
        self.HV_LCA1 = HV_LCA_Dynamic(ch2, head2, kernel_size=dynamic_lca_kernel_size, ctx_dim_ratio=dynamic_lca_ctx_ratio)
        self.I_LCA2 = I_LCA_Dynamic(ch3, head3, kernel_size=dynamic_lca_kernel_size, ctx_dim_ratio=dynamic_lca_ctx_ratio)
        self.HV_LCA2 = HV_LCA_Dynamic(ch3, head3, kernel_size=dynamic_lca_kernel_size, ctx_dim_ratio=dynamic_lca_ctx_ratio)
        self.I_LCA3 = I_LCA_Dynamic(ch4, head4, kernel_size=dynamic_lca_kernel_size, ctx_dim_ratio=dynamic_lca_ctx_ratio)
        self.HV_LCA3 = HV_LCA_Dynamic(ch4, head4, kernel_size=dynamic_lca_kernel_size, ctx_dim_ratio=dynamic_lca_ctx_ratio)
        
        # Bottleneck LCAs
        self.I_LCA4 = I_LCA_Dynamic(ch4, head4, kernel_size=dynamic_lca_kernel_size, ctx_dim_ratio=dynamic_lca_ctx_ratio) 
        self.HV_LCA4 = HV_LCA_Dynamic(ch4, head4, kernel_size=dynamic_lca_kernel_size, ctx_dim_ratio=dynamic_lca_ctx_ratio) 
        
        # Decoder LCAs
        self.I_LCA5 = I_LCA_Dynamic(ch3, head3, kernel_size=dynamic_lca_kernel_size, ctx_dim_ratio=dynamic_lca_ctx_ratio)
        self.HV_LCA5 = HV_LCA_Dynamic(ch3, head3, kernel_size=dynamic_lca_kernel_size, ctx_dim_ratio=dynamic_lca_ctx_ratio)
        self.I_LCA6 = I_LCA_Dynamic(ch2, head2, kernel_size=dynamic_lca_kernel_size, ctx_dim_ratio=dynamic_lca_ctx_ratio)
        self.HV_LCA6 = HV_LCA_Dynamic(ch2, head2, kernel_size=dynamic_lca_kernel_size, ctx_dim_ratio=dynamic_lca_ctx_ratio)

        self.trans = RGB_HVI()

    def forward(self, x):
        # Forward pass logic remains identical to original CIDNet,
        # just calls the dynamic LCA modules instead of static ones.
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

        # Stage 1 LCA + Encoder
        i_enc2_lca = self.I_LCA1(i_enc1.contiguous(), hv_1)
        hv_2_lca = self.HV_LCA1(hv_1.contiguous(), i_enc1)
        v_jump1 = i_enc2_lca[0] # Use LCA output[0] for skip connection
        hv_jump1 = hv_2_lca[0] # Use LCA output[0] for skip connection
        i_enc2 = self.IE_block2(i_enc2_lca[0]) # Pass LCA output[0] to next encoder block
        hv_2 = self.HVE_block2(hv_2_lca[0]) # Pass LCA output[0] to next encoder block

        # Stage 2 LCA + Encoder
        i_enc3_lca = self.I_LCA2(i_enc2.contiguous(), hv_2)
        hv_3_lca = self.HV_LCA2(hv_2.contiguous(), i_enc2)
        v_jump2 = i_enc3_lca[0] # Use LCA output[0]
        hv_jump2 = hv_3_lca[0] # Use LCA output[0]
        i_enc3 = self.IE_block3(i_enc3_lca[0]) # Pass LCA output[0] to next encoder block
        hv_3 = self.HVE_block3(hv_3_lca[0]) # Pass LCA output[0] to next encoder block

        # Stage 3 LCA (Bottleneck Entry)
        i_enc4_lca = self.I_LCA3(i_enc3.contiguous(), hv_3)
        hv_4_lca = self.HV_LCA3(hv_3.contiguous(), i_enc3)
        i_enc4 = i_enc4_lca[0] # Take the first element
        hv_4 = hv_4_lca[0] # Take the first element

        # Stage 4 LCA (Bottleneck Interaction)
        i_dec4_lca_tuple = self.I_LCA4(i_enc4.contiguous(), hv_4)
        hv_4_lca_tuple = self.HV_LCA4(hv_4.contiguous(), i_enc4)
        i_dec4_lca = i_dec4_lca_tuple[0] # Result for I-path decoder
        hv_4_lca = hv_4_lca_tuple[0] # Result for HV-path decoder

        # Stage 5 LCA + Decoder
        # Note: The input to HVD/ID blocks are the features *after* the bottleneck LCA
        hv_3_dec_in = self.HVD_block3(hv_4_lca, hv_jump2) # Input from HV_LCA4
        i_dec3_dec_in = self.ID_block3(i_dec4_lca, v_jump2) # Input from I_LCA4
        i_dec2_lca_tuple = self.I_LCA5(i_dec3_dec_in.contiguous(), hv_3_dec_in)
        hv_2_lca_tuple = self.HV_LCA5(hv_3_dec_in.contiguous(), i_dec3_dec_in)
        i_dec2_lca = i_dec2_lca_tuple[0] # Use LCA output[0]
        hv_2_lca = hv_2_lca_tuple[0] # Use LCA output[0]

        # Stage 6 LCA + Decoder
        hv_2_dec_in = self.HVD_block2(hv_2_lca, hv_jump1) # Use LCA output from prev stage
        i_dec2_dec_in = self.ID_block2(i_dec2_lca, v_jump1) # Use LCA output from prev stage
        i_dec1_lca_tuple = self.I_LCA6(i_dec2_dec_in.contiguous(), hv_2_dec_in)
        hv_1_lca_tuple = self.HV_LCA6(hv_2_dec_in.contiguous(), i_dec2_dec_in)
        i_dec1_lca = i_dec1_lca_tuple[0] # Use LCA output[0]
        hv_1_lca = hv_1_lca_tuple[0] # Use LCA output[0]

        # Final Decoder Blocks
        i_dec1 = self.ID_block1(i_dec1_lca, i_jump0) # Use LCA output[0]
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1_lca, hv_jump0) # Use LCA output
        hv_0 = self.HVD_block0(hv_1)

        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        # Need to read HVI_transform to confirm the exact method name and args
        # Assuming PHVIT takes the HVI tensor
        output_rgb = self.trans.PHVIT(output_hvi) 

        return output_rgb

    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi 