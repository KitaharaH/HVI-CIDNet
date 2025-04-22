import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.LCA import *
from huggingface_hub import PyTorchModelHubMixin
# Import ASSA and FRFN, and helpers
from net.ASSA import WindowAttention_sparse
from net.FRFN import FRFN
from einops import rearrange
import math
import torch.nn.functional as F

# Helper function to apply FRFN block
def apply_frfn_block(x, frfn_module):
    B, C, H, W = x.shape
    if H == 0 or W == 0:
        print(f"Warning: Zero dimension encountered (H={H}, W={W}). Skipping FRFN block.")
        return x
    x_reshaped = rearrange(x, 'b c h w -> b (h w) c')
    x_reshaped = frfn_module(x_reshaped)
    x_out = rearrange(x_reshaped, 'b (h w) c -> b c h w', h=H, w=W)
    return x_out

# Helper function to apply ASSA and FRFN block
def apply_assa_frfn_block(x, assa_module, frfn_module):
    B, C, H, W = x.shape
    if H == 0 or W == 0:
        print(f"Warning: Zero dimension encountered (H={H}, W={W}). Skipping ASSA-FRFN block.")
        return x
    x_reshaped = rearrange(x, 'b c h w -> b (h w) c')
    x_reshaped = assa_module(x_reshaped)
    x_reshaped = frfn_module(x_reshaped)
    x_out = rearrange(x_reshaped, 'b (h w) c -> b c h w', h=H, w=W)
    return x_out

# --- Add Window Partitioning/Reversing --- 
def window_partition(x, win_size):
    """ Partition the input feature map into non-overlapping windows. """
    B, C, H, W = x.shape
    win_h, win_w = win_size
    # Ensure H and W are divisible by window size
    pad_h = (win_h - H % win_h) % win_h
    pad_w = (win_w - W % win_w) % win_w
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h)) # Pad last dim first (W), then second last (H)
        B, C, H, W = x.shape # Update H, W after padding
    
    x = x.view(B, C, H // win_h, win_h, W // win_w, win_w)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, win_h * win_w, C)
    # Shape: (B * num_windows_h * num_windows_w, win_h*win_w, C)
    return windows, (H, W) # Return original (padded) H, W for reversing

def window_reverse(windows, win_size, H, W):
    """ Reverse the window partitioning operation. """
    win_h, win_w = win_size
    num_windows_h = H // win_h
    num_windows_w = W // win_w
    B_nw = windows.shape[0] # B * num_windows_h * num_windows_w
    C = windows.shape[2]
    # Calculate B: B_nw = B * num_windows_h * num_windows_w
    B = B_nw // (num_windows_h * num_windows_w)
    
    x = windows.view(B, num_windows_h, num_windows_w, win_h, win_w, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H, W)
    return x

# --- Modified Helper functions using windowing --- 
def apply_frfn_block_windowed(x, frfn_module, win_size):
    B, C, H, W = x.shape
    if H == 0 or W == 0 or H < win_size[0] or W < win_size[1]: # Skip if too small
        print(f"Warning: Feature map size ({H},{W}) too small for window size {win_size}. Skipping FRFN block.")
        return x
        
    windows, (Hp, Wp) = window_partition(x, win_size)
    # Input to frfn: (B*num_win, win_N, C)
    windows_out = frfn_module(windows) 
    x_out = window_reverse(windows_out, win_size, Hp, Wp)
    
    # Crop back if padding was applied
    if Hp > H or Wp > W:
        x_out = x_out[:, :, :H, :W]
        
    return x_out

def apply_assa_frfn_block_windowed(x, assa_module, frfn_module, win_size):
    B, C, H, W = x.shape
    if H == 0 or W == 0 or H < win_size[0] or W < win_size[1]: # Skip if too small
        print(f"Warning: Feature map size ({H},{W}) too small for window size {win_size}. Skipping ASSA-FRFN block.")
        return x
        
    windows, (Hp, Wp) = window_partition(x, win_size)
    # Input to assa/frfn: (B*num_win, win_N, C)
    windows_out = assa_module(windows)
    windows_out = frfn_module(windows_out)
    x_out = window_reverse(windows_out, win_size, Hp, Wp)
    
    # Crop back if padding was applied
    if Hp > H or Wp > W:
        x_out = x_out[:, :, :H, :W]
        
    return x_out

class CIDNet(nn.Module, PyTorchModelHubMixin):
    def __init__(self,
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False,
                 win_size=(8, 8), # Add win_size for ASSA
                 frfn_hidden_dim_multiplier=2 # Multiplier for FRFN hidden dim
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

        # LCA Modules
        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4) # Bottleneck LCA
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)

        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4) # Bottleneck LCA
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)

        # --- AST Blocks --- 
        # Encoder FRFN Blocks (No ASSA in Encoder based on diagram)
        self.frfn_i1 = FRFN(dim=ch2, hidden_dim=ch2 * frfn_hidden_dim_multiplier)
        self.frfn_hv1 = FRFN(dim=ch2, hidden_dim=ch2 * frfn_hidden_dim_multiplier)
        self.frfn_i2 = FRFN(dim=ch3, hidden_dim=ch3 * frfn_hidden_dim_multiplier)
        self.frfn_hv2 = FRFN(dim=ch3, hidden_dim=ch3 * frfn_hidden_dim_multiplier)
        self.frfn_i3 = FRFN(dim=ch4, hidden_dim=ch4 * frfn_hidden_dim_multiplier)
        self.frfn_hv3 = FRFN(dim=ch4, hidden_dim=ch4 * frfn_hidden_dim_multiplier)
        
        # Bottleneck ASSA + FRFN Blocks
        self.assa_i_bn = WindowAttention_sparse(dim=ch4, win_size=win_size, num_heads=head4)
        self.frfn_i_bn = FRFN(dim=ch4, hidden_dim=ch4 * frfn_hidden_dim_multiplier)
        self.assa_hv_bn = WindowAttention_sparse(dim=ch4, win_size=win_size, num_heads=head4)
        self.frfn_hv_bn = FRFN(dim=ch4, hidden_dim=ch4 * frfn_hidden_dim_multiplier)

        # Decoder ASSA + FRFN Blocks
        # Stage 3 (Input dim=ch4, after upsample)
        self.assa_i_d3 = WindowAttention_sparse(dim=ch3, win_size=win_size, num_heads=head3) # Dim is ch3 after upsample
        self.frfn_i_d3 = FRFN(dim=ch3, hidden_dim=ch3 * frfn_hidden_dim_multiplier)
        self.assa_hv_d3 = WindowAttention_sparse(dim=ch3, win_size=win_size, num_heads=head3) # Dim is ch3 after upsample
        self.frfn_hv_d3 = FRFN(dim=ch3, hidden_dim=ch3 * frfn_hidden_dim_multiplier)

        # Stage 2 (Input dim=ch3, after upsample)
        self.assa_i_d2 = WindowAttention_sparse(dim=ch2, win_size=win_size, num_heads=head2) # Dim is ch2 after upsample
        self.frfn_i_d2 = FRFN(dim=ch2, hidden_dim=ch2 * frfn_hidden_dim_multiplier)
        self.assa_hv_d2 = WindowAttention_sparse(dim=ch2, win_size=win_size, num_heads=head2) # Dim is ch2 after upsample
        self.frfn_hv_d2 = FRFN(dim=ch2, hidden_dim=ch2 * frfn_hidden_dim_multiplier)

        # Stage 1 (Input dim=ch2, after upsample)
        self.assa_i_d1 = WindowAttention_sparse(dim=ch1, win_size=win_size, num_heads=head1) # Dim is ch1 after upsample
        self.frfn_i_d1 = FRFN(dim=ch1, hidden_dim=ch1 * frfn_hidden_dim_multiplier)        
        self.assa_hv_d1 = WindowAttention_sparse(dim=ch1, win_size=win_size, num_heads=head1) # Dim is ch1 after upsample
        self.frfn_hv_d1 = FRFN(dim=ch1, hidden_dim=ch1 * frfn_hidden_dim_multiplier)

        self.trans = RGB_HVI()
        self.win_size = win_size

    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes)

        # ---- Encoder ----
        # Stage 0
        i_enc0 = self.IE_block0(i)
        hv_0 = self.HVE_block0(hvi)
        i_jump0 = i_enc0
        hv_jump0 = hv_0

        # Stage 1
        i_enc1 = self.IE_block1(i_enc0) # ch2
        hv_1 = self.HVE_block1(hv_0)   # ch2
        # Apply FRFN block windowed (No ASSA here)
        i_enc1 = apply_frfn_block_windowed(i_enc1, self.frfn_i1, self.win_size)
        hv_1 = apply_frfn_block_windowed(hv_1, self.frfn_hv1, self.win_size)
        # LCA
        i_enc1_lca = self.I_LCA1(i_enc1, hv_1)
        hv_1_lca = self.HV_LCA1(hv_1, i_enc1)
        i_jump1 = i_enc1_lca
        hv_jump1 = hv_1_lca

        # Stage 2
        i_enc2 = self.IE_block2(i_enc1_lca) # ch3
        hv_2 = self.HVE_block2(hv_1_lca)   # ch3
        # Apply FRFN block windowed
        i_enc2 = apply_frfn_block_windowed(i_enc2, self.frfn_i2, self.win_size)
        hv_2 = apply_frfn_block_windowed(hv_2, self.frfn_hv2, self.win_size)
        # LCA
        i_enc2_lca = self.I_LCA2(i_enc2, hv_2)
        hv_2_lca = self.HV_LCA2(hv_2, i_enc2)
        i_jump2 = i_enc2_lca
        hv_jump2 = hv_2_lca

        # Stage 3
        i_enc3 = self.IE_block3(i_enc2_lca) # ch4
        hv_3 = self.HVE_block3(hv_2_lca)   # ch4
        # Apply FRFN block windowed
        i_enc3 = apply_frfn_block_windowed(i_enc3, self.frfn_i3, self.win_size)
        hv_3 = apply_frfn_block_windowed(hv_3, self.frfn_hv3, self.win_size)
        # LCA
        i_enc3_lca = self.I_LCA3(i_enc3, hv_3)
        hv_3_lca = self.HV_LCA3(hv_3, i_enc3)
        # No jump connection needed here

        # ---- Bottleneck ----
        # LCA interaction first
        i_bn_lca = self.I_LCA4(i_enc3_lca, hv_3_lca)
        hv_bn_lca = self.HV_LCA4(hv_3_lca, i_enc3_lca)
        # Apply ASSA + FRFN block windowed after LCA
        i_bn_out = apply_assa_frfn_block_windowed(i_bn_lca, self.assa_i_bn, self.frfn_i_bn, self.win_size)
        hv_bn_out = apply_assa_frfn_block_windowed(hv_bn_lca, self.assa_hv_bn, self.frfn_hv_bn, self.win_size)

        # ---- Decoder ----
        # Stage 3
        # Upsample first
        i_dec3_up = self.ID_block3(i_bn_out, i_jump2) # ch3
        hv_3_up = self.HVD_block3(hv_bn_out, hv_jump2) # ch3
        # Apply ASSA + FRFN block windowed after Upsample
        i_dec3_ast = apply_assa_frfn_block_windowed(i_dec3_up, self.assa_i_d3, self.frfn_i_d3, self.win_size)
        hv_3_ast = apply_assa_frfn_block_windowed(hv_3_up, self.assa_hv_d3, self.frfn_hv_d3, self.win_size)
        # LCA
        i_dec3_lca = self.I_LCA5(i_dec3_ast, hv_3_ast)
        hv_3_lca = self.HV_LCA5(hv_3_ast, i_dec3_ast)

        # Stage 2
        # Upsample first
        i_dec2_up = self.ID_block2(i_dec3_lca, i_jump1) # ch2
        hv_2_up = self.HVD_block2(hv_3_lca, hv_jump1)   # ch2
        # Apply ASSA + FRFN block windowed after Upsample
        i_dec2_ast = apply_assa_frfn_block_windowed(i_dec2_up, self.assa_i_d2, self.frfn_i_d2, self.win_size)
        hv_2_ast = apply_assa_frfn_block_windowed(hv_2_up, self.assa_hv_d2, self.frfn_hv_d2, self.win_size)
        # LCA
        i_dec2_lca = self.I_LCA6(i_dec2_ast, hv_2_ast)
        hv_2_lca = self.HV_LCA6(hv_2_ast, i_dec2_ast)

        # Stage 1
        # Upsample first
        i_dec1_up = self.ID_block1(i_dec2_lca, i_jump0) # ch1
        hv_1_up = self.HVD_block1(hv_2_lca, hv_jump0)   # ch1
        # Apply ASSA + FRFN block windowed after Upsample
        i_dec1_ast = apply_assa_frfn_block_windowed(i_dec1_up, self.assa_i_d1, self.frfn_i_d1, self.win_size)
        hv_1_ast = apply_assa_frfn_block_windowed(hv_1_up, self.assa_hv_d1, self.frfn_hv_d1, self.win_size)

        # Stage 0 (Final Convolution)
        i_dec0 = self.ID_block0(i_dec1_ast)
        hv_0 = self.HVD_block0(hv_1_ast)

        # Combine and transform back to RGB (CIDNet's original final stage)
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb

    def HVIT(self,x):
        hvi = self.trans.HVIT(x)
        return hvi
    
    
