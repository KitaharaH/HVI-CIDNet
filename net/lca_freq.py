import torch
import torch.nn as nn
import torch.fft
from einops import rearrange
import torch.nn.functional as F # Added for F.relu

# Copied LayerNorm from net.transformer_utils
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            # Assuming LayerNorm_BiasFree exists elsewhere or define it here if needed
            # self.body = LayerNorm_BiasFree(dim)
            # Using standard LayerNorm as fallback
             self.body = nn.LayerNorm(dim)
        else:
            self.body = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        # LayerNorm expects shape (N, ..., C)
        x_permuted = x.permute(0, 2, 3, 1) # N, H, W, C
        normed_x = self.body(x_permuted)
        return normed_x.permute(0, 3, 1, 2) # N, C, H, W


# Copied CAB from net/LCA.py
class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn,dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

# Copied IEL from net/LCA.py
class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.Tanh = nn.Tanh() # Consider replacing with ReLU or GeLU if needed

    def forward(self, x):
        #x_in = x # Not used in original IEL forward
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # Original IEL used Tanh, keeping it for now, but ReLU might be more standard
        x1 = self.Tanh(self.dwconv1(x1)) # Removed residual connection here as per original IEL
        x2 = self.Tanh(self.dwconv2(x2)) # Removed residual connection here as per original IEL
        x = x1 * x2
        x = self.project_out(x)
        return x

# --- Frequency Enhancement Components ---

def create_high_freq_mask(h, w, p=0.25, device='cpu'):
    """Creates a mask to select high frequencies (corners of the spectrum)."""
    center_h, center_w = h // 2, w // 2
    radius_h, radius_w = int(center_h * (1 - p)), int(center_w * (1 - p))

    mask = torch.ones((h, w), device=device)
    mask[center_h - radius_h : center_h + radius_h, center_w - radius_w : center_w + radius_w] = 0
    # This creates a high-pass mask (blocking the center low frequencies) after fftshift

    # Ensure mask shape is broadcastable: (1, 1, H, W)
    return mask.unsqueeze(0).unsqueeze(0)


class FrequencyEnhancementPath(nn.Module):
    """Applies FFT, high-pass filtering, IFFT, and spatial refinement."""
    def __init__(self, dim, refine_channels_mult=1, p=0.25, bias=False):
        super().__init__()
        self.p = p
        refine_channels = int(dim * refine_channels_mult)
        # Spatial refinement block
        self.refine = nn.Sequential(
            nn.Conv2d(dim, refine_channels, kernel_size=3, padding=1, bias=bias),
            nn.ReLU(inplace=True), # Using ReLU for refinement
            nn.Conv2d(refine_channels, dim, kernel_size=3, padding=1, bias=bias)
        )
        self.register_buffer('mask', None, persistent=False) # Use register_buffer for non-parameter tensors


    def forward(self, x):
        b, c, h, w = x.shape
        dtype = x.dtype
        device = x.device

        # 1. FFT
        x_fft = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))

        # 2. Create or get mask (cache it using register_buffer)
        if self.mask is None or self.mask.shape[-2:] != (h, w):
             self.mask = create_high_freq_mask(h, w, self.p, device=device)


        # 3. Apply mask (high-pass)
        x_fft_filtered_shifted = x_fft_shifted * self.mask # Keep only high frequencies

        # 4. Inverse shift and IFFT
        x_fft_filtered = torch.fft.ifftshift(x_fft_filtered_shifted, dim=(-2,-1))
        x_freq_spatial = torch.fft.ifft2(x_fft_filtered, dim=(-2, -1), norm='ortho')
        x_freq_spatial = torch.real(x_freq_spatial) # Take real part

        # 5. Spatial Refinement
        out = self.refine(x_freq_spatial.to(dtype)) # Ensure dtype matches input
        return out

# --- Modified LCA with Frequency Path ---

class LCA_Freq(nn.Module):
    def __init__(self, dim, num_heads, residual_gdfn=True, ffn_expansion_factor=2.66, freq_p=0.25, bias=False):
        """
        LCA module augmented with a parallel frequency enhancement path.

        Args:
            dim (int): Feature dimension.
            num_heads (int): Number of attention heads in CAB.
            residual_gdfn (bool): If True, adds the fused (IEL + Freq) output to the state after attention (like I_LCA).
                                   If False, uses the fused output to replace the state after attention (like HV_LCA).
            ffn_expansion_factor (float): Expansion factor for IEL hidden features.
            freq_p (float): Proportion of frequency spectrum center to block (0.0 to 1.0). High pass filter threshold.
            bias (bool): Whether to use bias in conv layers.
        """
        super(LCA_Freq, self).__init__()
        self.residual_gdfn = residual_gdfn
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)          # Cross Attention (Spatial Path 1)
        self.gdfn = IEL(dim, ffn_expansion_factor, bias)  # Original IEL/CDL (Spatial Path 2a)
        self.freq_path = FrequencyEnhancementPath(dim, p=freq_p, bias=bias) # Frequency Path (Spatial Path 2b)

        # Fusion layer for IEL and Freq paths
        self.fusion_conv = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        # Input x comes from one stream (e.g., intensity), y from the other (e.g., hue/value)
        identity = x # Store original x

        # 1. Cross Attention (between x and y)
        x_norm = self.norm(x)
        y_norm = self.norm(y)
        attn_out = self.ffn(x_norm, y_norm)
        x = x + attn_out # Update x using information from y (state after attention)

        # 2. Prepare input for parallel IEL/Freq paths
        x_for_branches = self.norm(x) # Normalize the updated x

        # 3. Run IEL (Spatial Path 2a) and Frequency Path (Spatial Path 2b) in parallel
        cdl_output = self.gdfn(x_for_branches)
        freq_output = self.freq_path(x_for_branches)

        # 4. Fuse the outputs of IEL and Freq paths
        fused_branch_output = self.fusion_conv(torch.cat([cdl_output, freq_output], dim=1))

        # 5. Combine fused output with the main path based on residual_gdfn flag
        # Follows the structure: x = x + attn; x = op(norm(x)) OR x = x + op(norm(x))
        if self.residual_gdfn:
            # Additive behavior (like I_LCA)
            out = x + fused_branch_output # Add fused features to the state *after* attention
        else:
            # Replacement behavior (like HV_LCA)
            # Replace the state *after* attention with the fused output
            out = fused_branch_output

        return out 