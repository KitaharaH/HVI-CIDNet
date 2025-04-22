import torch
import torch.nn as nn
import torch.nn.functional as F
from net.transformer_utils import LayerNorm # Assuming LayerNorm is in transformer_utils

class ChannelAttention(nn.Module):
    """
    Simple Channel Attention block
    """
    def __init__(self, num_features, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(num_features, num_features // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DecoderAttentionFusion(nn.Module):
    """
    Fuses upsampled features from the previous decoder stage with 
    skip connection features from the encoder using Channel Attention.
    """
    def __init__(self, upsampled_channels, skip_channels, output_channels, use_norm=False):
        super(DecoderAttentionFusion, self).__init__()
        
        # Convolution to align skip connection channels if necessary
        # If skip_channels might not equal output_channels
        if skip_channels != output_channels:
            self.conv_skip = nn.Conv2d(skip_channels, output_channels, kernel_size=1, bias=False)
        else:
            self.conv_skip = nn.Identity()
            
        # Convolution to align upsampled channels if necessary
        # If upsampled_channels might not equal output_channels
        if upsampled_channels != output_channels:
             self.conv_upsampled = nn.Conv2d(upsampled_channels, output_channels, kernel_size=1, bias=False)
        else:
             self.conv_upsampled = nn.Identity()

        # Combined channels after potential alignment and concatenation
        combined_channels = 2 * output_channels 
        
        # Convolution after concatenation
        self.conv_concat = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(combined_channels, output_channels, kernel_size=3, padding=0, bias=False),
            LayerNorm(output_channels) if use_norm else nn.Identity(),
            nn.GELU() # Using GELU as seen in other blocks potentially
        )
        
        # Channel Attention applied to the concatenated features before final conv
        self.channel_attention = ChannelAttention(combined_channels)
        
        # Final convolution to produce the output features
        self.conv_final = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(combined_channels, output_channels, kernel_size=3, padding=0, bias=False),
            LayerNorm(output_channels) if use_norm else nn.Identity(),
            nn.GELU()
        )

    def forward(self, x_upsampled, x_skip):
        # Align channels if needed
        x_upsampled = self.conv_upsampled(x_upsampled)
        x_skip = self.conv_skip(x_skip)
        
        # Concatenate features
        x_concat = torch.cat([x_upsampled, x_skip], dim=1)
        
        # Apply channel attention
        x_att = self.channel_attention(x_concat)
        
        # Final convolution
        x_out = self.conv_final(x_att)
        
        return x_out 