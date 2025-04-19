import torch
import torch.nn as nn
from net.HVI_transform import RGB_HVI
from net.transformer_utils import *
from net.LCA import *
from net.TNSM import HV_TNSM, I_TNSM
from huggingface_hub import PyTorchModelHubMixin
import torch.nn.functional as F

class CIDNet_TNSM(nn.Module, PyTorchModelHubMixin):
    """
    集成TNSM模块的CIDNet网络
    这个版本在原始CIDNet的基础上加入了可训练的噪声抑制模块
    """
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False,
                 use_tnsm=True
        ):
        super(CIDNet_TNSM, self).__init__()
        
        self.use_tnsm = use_tnsm
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
        
        # 原始的LCA模块
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
        
        # 新增的TNSM模块
        if self.use_tnsm:
            self.HV_TNSM1 = HV_TNSM(ch2, head2)
            self.HV_TNSM2 = HV_TNSM(ch3, head3)
            self.HV_TNSM3 = HV_TNSM(ch4, head4)
            self.HV_TNSM4 = HV_TNSM(ch4, head4)
            self.HV_TNSM5 = HV_TNSM(ch3, head3)
            self.HV_TNSM6 = HV_TNSM(ch2, head2)
            
            self.I_TNSM1 = I_TNSM(ch2, head2)
            self.I_TNSM2 = I_TNSM(ch3, head3)
            self.I_TNSM3 = I_TNSM(ch4, head4)
            self.I_TNSM4 = I_TNSM(ch4, head4)
            self.I_TNSM5 = I_TNSM(ch3, head3)
            self.I_TNSM6 = I_TNSM(ch2, head2)
        
        self.trans = RGB_HVI()
        
        # 噪声图融合网络
        if self.use_tnsm:
            self.noise_fusion = nn.Sequential(
                nn.Conv2d(12, 3, kernel_size=3, padding=1, bias=False),
                nn.Sigmoid()
            )
        
    def forward(self, x):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        i = hvi[:,2,:,:].unsqueeze(1).to(dtypes)
        
        # 存储每个层级生成的噪声图
        noise_maps = []
        
        # 低层特征提取
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        i_jump0 = i_enc0
        hv_jump0 = hv_0
        
        # 第1层交互
        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)
        
        # 添加TNSM模块处理
        if self.use_tnsm:
            i_enc2_tnsm, i_noise1 = self.I_TNSM1(i_enc2, hv_2)
            hv_2_tnsm, hv_noise1 = self.HV_TNSM1(hv_2, i_enc2)
            
            # 使用TNSM的输出
            i_enc2 = i_enc2_tnsm
            hv_2 = hv_2_tnsm
            
            # 存储噪声图
            noise_maps.append(i_noise1)
            noise_maps.append(hv_noise1)
        
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)
        
        # 第2层交互
        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)
        
        # 添加TNSM模块处理
        if self.use_tnsm:
            i_enc3_tnsm, i_noise2 = self.I_TNSM2(i_enc3, hv_3)
            hv_3_tnsm, hv_noise2 = self.HV_TNSM2(hv_3, i_enc3)
            
            # 使用TNSM的输出
            i_enc3 = i_enc3_tnsm
            hv_3 = hv_3_tnsm
            
            # 存储噪声图
            noise_maps.append(i_noise2)
            noise_maps.append(hv_noise2)
        
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)
        
        # 第3层交互
        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)
        
        # 添加TNSM模块处理
        if self.use_tnsm:
            i_enc4_tnsm, i_noise3 = self.I_TNSM3(i_enc4, hv_4)
            hv_4_tnsm, hv_noise3 = self.HV_TNSM3(hv_4, i_enc4)
            
            # 使用TNSM的输出
            i_enc4 = i_enc4_tnsm
            hv_4 = hv_4_tnsm
            
            # 存储噪声图
            noise_maps.append(i_noise3)
            noise_maps.append(hv_noise3)
        
        # 瓶颈层交互
        i_dec4 = self.I_LCA4(i_enc4, hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)
        
        # 添加TNSM模块处理
        if self.use_tnsm:
            i_dec4_tnsm, i_noise4 = self.I_TNSM4(i_dec4, hv_4)
            hv_4_tnsm, hv_noise4 = self.HV_TNSM4(hv_4, i_dec4)
            
            # 使用TNSM的输出
            i_dec4 = i_dec4_tnsm
            hv_4 = hv_4_tnsm
            
            # 存储噪声图
            noise_maps.append(i_noise4)
            noise_maps.append(hv_noise4)
        
        # 上采样和跳跃连接
        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        
        # 第4层交互
        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)
        
        # 添加TNSM模块处理
        if self.use_tnsm:
            i_dec2_tnsm, i_noise5 = self.I_TNSM5(i_dec2, hv_2)
            hv_2_tnsm, hv_noise5 = self.HV_TNSM5(hv_2, i_dec2)
            
            # 使用TNSM的输出
            i_dec2 = i_dec2_tnsm
            hv_2 = hv_2_tnsm
            
            # 存储噪声图
            noise_maps.append(i_noise5)
            noise_maps.append(hv_noise5)
        
        # 上采样和跳跃连接
        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)
        
        # 第5层交互
        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)
        
        # 添加TNSM模块处理
        if self.use_tnsm:
            i_dec1_tnsm, i_noise6 = self.I_TNSM6(i_dec1, hv_1)
            hv_1_tnsm, hv_noise6 = self.HV_TNSM6(hv_1, i_dec1)
            
            # 使用TNSM的输出
            i_dec1 = i_dec1_tnsm
            hv_1 = hv_1_tnsm
            
            # 存储噪声图
            noise_maps.append(i_noise6)
            noise_maps.append(hv_noise6)
        
        # 最终上采样和输出
        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)
        
        # 合并HVI通道并转换回RGB
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)
        
        # 如果使用TNSM，返回噪声图用于可视化和分析
        if self.use_tnsm and self.training:
            # --- Resize noise maps to match output_rgb resolution --- 
            target_h, target_w = output_rgb.shape[-2:] # Use output_rgb size as target
            if len(noise_maps) > 0:
                resized_noise_maps = []
                # print("Resizing noise maps to:", target_h, target_w) # Optional debug print
                for i, nm in enumerate(noise_maps):
                    # print(f"  Original noise_maps[{i}]: {nm.shape}") # Optional debug print
                    if nm.shape[-2:] != (target_h, target_w):
                        # Use bilinear interpolation for resizing
                        nm_resized = F.interpolate(nm, size=(target_h, target_w), mode='bilinear', align_corners=False)
                        # print(f"  Resized noise_maps[{i}]: {nm_resized.shape}") # Optional debug print
                        resized_noise_maps.append(nm_resized)
                    else:
                        resized_noise_maps.append(nm)
                
                # Concatenate the resized maps
                stacked_noise = torch.cat(resized_noise_maps, dim=1)
                fused_noise = self.noise_fusion(stacked_noise)  # B x 3 x target_H x target_W
                return output_rgb, fused_noise # Return fused noise map
            else:
                 # This case should ideally not happen if use_tnsm is True and TNSM modules are active
                 # Returning None might be problematic later. Raising error is safer during debug.
                 raise ValueError("noise_maps list is empty during training with use_tnsm=True")
            # --- End Resize --- 
            
            # --- Remove old resize logic based on noise_maps[0] --- 
            # if len(noise_maps) > 0:
            #     target_h, target_w = noise_maps[0].shape[-2:] # Get target size from first map
            #     resized_noise_maps = []
            #     for nm in noise_maps:
            #         if nm.shape[-2:] != (target_h, target_w):
            #             nm_resized = F.interpolate(nm, size=(target_h, target_w), mode='bilinear', align_corners=False)
            #             resized_noise_maps.append(nm_resized)
            #         else:
            #             resized_noise_maps.append(nm)
            #     
            #     stacked_noise = torch.cat(resized_noise_maps, dim=1)  
            # else:
            #     stacked_noise = None 
            # fused_noise = self.noise_fusion(stacked_noise)  # B x 3 x H x W
            # return output_rgb, fused_noise
            # --- End Remove --- 

        else: # Not training or not using TNSM
            # Return None for noise map when not needed or not available
            return output_rgb, None
    
    def HVIT(self, x):
        hvi = self.trans.HVIT(x)
        return hvi 