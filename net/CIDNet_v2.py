import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
# Assuming NormDownsample, NormUpsample, LayerNorm are in transformer_utils
from net.transformer_utils import NormDownsample, NormUpsample
from net.LCA_v2 import LCABlock # Import the new block
from huggingface_hub import PyTorchModelHubMixin
import torch.nn.functional as F # Needed for concat skip connection alternative

class CIDNet(nn.Module, PyTorchModelHubMixin):
    """ CIDNet Version 2 using LCABlock with SwiGLU FFN """
    def __init__(self,
                 channels=[36, 36, 72, 144], # Note: ch1 should match embedding dim if used directly
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=8/3,
                 bias=False,
                 norm=False # Keep norm option for initial Down/Up sample layers?
                 ):
        super().__init__()

        # Allow channels to be asymmetric, e.g., [embed_dim, dec1_dim, dec2_dim, dec3_dim]
        # For simplicity, keeping the original symmetric channel structure for now
        if len(channels) != 4:
            raise ValueError("channels list must contain 4 values")
        ch1, ch2, ch3, ch4 = channels

        if len(heads) != 4:
            raise ValueError("heads list must contain 4 values")
        head1, head2, head3, head4 = heads # head1 not used in LCA, kept for consistency

        # H/V Path Encoders/Decoders (Initial embedding and final projection)
        self.HVE_block0 = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(2, ch1, 3, stride=1, padding=0, bias=bias))
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.HVD_block0 = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=bias))

        # Intensity Path Encoders/Decoders (Initial embedding and final projection)
        self.IE_block0 = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(1, ch1, 3, stride=1, padding=0, bias=bias))
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=bias))

        # --- LCA Blocks --- (Cross-attention between I and H/V paths)
        self.enc_lca1 = LCABlock(dim=ch2, num_heads=head2, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
        self.enc_lca2 = LCABlock(dim=ch3, num_heads=head3, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
        self.enc_lca3 = LCABlock(dim=ch4, num_heads=head4, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

        self.bottleneck_lca = LCABlock(dim=ch4, num_heads=head4, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

        self.dec_lca1 = LCABlock(dim=ch3, num_heads=head3, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
        self.dec_lca2 = LCABlock(dim=ch2, num_heads=head2, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
        # No LCA block at the highest resolution (ch1) in this design

        self.trans = RGB_HVI()

    def forward(self, x):
        dtypes = x.dtype
        hvi_orig = self.trans.HVIT(x) # Keep original for residual connection
        i = hvi_orig[:, 2, :, :].unsqueeze(1).to(dtypes)
        hv = hvi_orig[:, 0:2, :, :].to(dtypes) # Separate HV: [B, 2, H, W]

        # --- Encoder --- Store outputs for skip connections
        # Level 0 - Initial Embedding
        i_enc0 = self.IE_block0(i)
        hv_enc0 = self.HVE_block0(hv)

        # Level 1
        i_enc1_in = self.IE_block1(i_enc0)
        hv_enc1_in = self.HVE_block1(hv_enc0)
        # Cross Attention (Update I based on HV, Update HV based on I)
        i_enc1_out = self.enc_lca1(i_enc1_in, hv_enc1_in)
        hv_enc1_out = self.enc_lca1(hv_enc1_in, i_enc1_in)

        # Level 2
        i_enc2_in = self.IE_block2(i_enc1_out)
        hv_enc2_in = self.HVE_block2(hv_enc1_out)
        # Cross Attention
        i_enc2_out = self.enc_lca2(i_enc2_in, hv_enc2_in)
        hv_enc2_out = self.enc_lca2(hv_enc2_in, i_enc2_in)

        # Level 3 (Bottleneck Input)
        i_enc3_in = self.IE_block3(i_enc2_out)
        hv_enc3_in = self.HVE_block3(hv_enc2_out)
        # Cross Attention
        i_enc3_out = self.enc_lca3(i_enc3_in, hv_enc3_in)
        hv_enc3_out = self.enc_lca3(hv_enc3_in, i_enc3_in)

        # --- Bottleneck --- Cross attention between I and HV paths
        i_bottle = self.bottleneck_lca(i_enc3_out, hv_enc3_out)
        hv_bottle = self.bottleneck_lca(hv_enc3_out, i_enc3_out)

        # --- Decoder --- Use skip connections from corresponding encoder level *outputs*
        # Level 3 Decode
        # Upsample + Skip Connection (Concatenation Option)
        # Note: Original used addition in NormUpsample. If concat is preferred, NormUpsample needs modification
        # or implement concat here.
        # Assuming NormUpsample handles skip connection internally via addition for simplicity now.
        hv_dec3_in = self.HVD_block3(hv_bottle, hv_enc2_out)
        i_dec3_in = self.ID_block3(i_bottle, i_enc2_out)
        # Cross Attention
        i_dec3_out = self.dec_lca1(i_dec3_in, hv_dec3_in)
        hv_dec3_out = self.dec_lca1(hv_dec3_in, i_dec3_in)

        # Level 2 Decode
        hv_dec2_in = self.HVD_block2(hv_dec3_out, hv_enc1_out)
        i_dec2_in = self.ID_block2(i_dec3_out, i_enc1_out)
        # Cross Attention
        i_dec2_out = self.dec_lca2(i_dec2_in, hv_dec2_in)
        hv_dec2_out = self.dec_lca2(hv_dec2_in, i_dec2_in)

        # Level 1 Decode (Upsample + Skip)
        hv_dec1 = self.HVD_block1(hv_dec2_out, hv_enc0)
        i_dec1 = self.ID_block1(i_dec2_out, i_enc0)
        # No LCA block at this level

        # Level 0 Output Projection
        hv_out = self.HVD_block0(hv_dec1) # [B, 2, H, W]
        i_out = self.ID_block0(i_dec1)   # [B, 1, H, W]

        # Combine features and add residual from original HVI
        output_hvi = torch.cat([hv_out, i_out], dim=1) + hvi_orig # [B, 3, H, W]

        # Transform back to RGB
        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb

    # Keep the HVIT helper method if needed
    def HVIT(self, x):
        hvi = self.trans.HVIT(x)
        return hvi 