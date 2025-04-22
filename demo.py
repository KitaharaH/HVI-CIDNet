import numpy as np
import torch
import argparse
from PIL import Image
# from net.CIDNet import CIDNet
# from net.CIDNet_MSSA import CIDNet
from net.CIDNet_PixelShuffle import CIDNet
import torchvision.transforms as transforms
import torch.nn.functional as F
import os

# 参数设置
parser = argparse.ArgumentParser(description='HVI-CIDNet推理')
parser.add_argument('--input', type=str, default='/root/autodl-tmp/LMOT_DARK_YOLO/images/train/LMOT-02_000001.png', help='输入图片路径')
parser.add_argument('--output_dir', type=str, default='output/demo', help='输出目录')
parser.add_argument('--weight', type=str, default='/root/HVI-CIDNet/weights/PixelShuffle_P1en2/epoch_10.pth', help='权重文件路径')
parser.add_argument('--gamma', type=float, default=1.0, help='gamma曲线参数，越低越亮')
parser.add_argument('--alpha_s', type=float, default=1.0, help='饱和度参数，越高越饱和')
parser.add_argument('--alpha_i', type=float, default=1.0, help='亮度参数，越高越亮')
parser.add_argument('--cpu', action='store_true', help='使用CPU推理')
args = parser.parse_args()

# 确保输出目录存在
os.makedirs(args.output_dir, exist_ok=True)

# 加载模型
print(f"正在加载模型: {args.weight}")
if args.cpu:
    eval_net = CIDNet().cpu()
else:
    eval_net = CIDNet().cuda()

eval_net.trans.gated = True
eval_net.trans.gated2 = True
eval_net.load_state_dict(torch.load(args.weight, map_location=lambda storage, loc: storage))
eval_net.eval()

# 设置饱和度和亮度参数
eval_net.trans.alpha_s = args.alpha_s
eval_net.trans.alpha = args.alpha_i

# 读取并处理图像
print(f"处理图像: {args.input}")
input_img = Image.open(args.input).convert('RGB')
pil2tensor = transforms.Compose([transforms.ToTensor()])
input_tensor = pil2tensor(input_img)

# 添加填充以适应网络
factor = 8
h, w = input_tensor.shape[1], input_tensor.shape[2]
H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
padh = H - h if h % factor != 0 else 0
padw = W - w if w % factor != 0 else 0
input_tensor = F.pad(input_tensor.unsqueeze(0), (0, padw, 0, padh), 'reflect')

# 模型推理
with torch.no_grad():
    if args.cpu:
        output = eval_net(input_tensor**args.gamma)
    else:
        output = eval_net(input_tensor.cuda()**args.gamma)

# 处理输出
if args.cpu:
    output = torch.clamp(output, 0, 1)
else:
    output = torch.clamp(output.cuda(), 0, 1)
    
output = output[:, :, :h, :w]
enhanced_img = transforms.ToPILImage()(output.squeeze(0))

# 保存结果
input_name = os.path.basename(args.input)
output_path = os.path.join(args.output_dir, f"enhanced_{input_name}")
enhanced_img.save(output_path)
print(f"增强后的图像已保存至: {output_path}")