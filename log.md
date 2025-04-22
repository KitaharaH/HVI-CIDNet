


<think>
根据您提供的代码库上下文，我需要分析并总结这个项目的目录结构。让我从提供的文件和它们的路径来重建整个项目的结构。

从文件路径和导入语句中，我可以看到以下目录和文件：

1. 根目录
   - Readme.md - 项目的主要文档
   - .gitignore - Git忽略文件
   - app.py - 图形界面应用
   - eval.py - 评估脚本
   - eval_SID_blur.py - 专门评估SID和模糊数据集的脚本
   - eval_hf.py (提到但没有完整显示)
   - measure.py - 测量性能指标的脚本
   - measure_SID_blur.py - 特定于SID和模糊数据集的测量脚本
   - measure_niqe_bris.py (提到但没有完整显示)
   - net_test.py (提到但没有完整显示)
   - train.py - 训练脚本
   - requirements.txt (提到但没有显示)

2. data目录
   - data/options.py - 包含配置选项
   - data/data.py (导入但没有显示)
   - data/scheduler.py (导入但没有显示)

3. loss目录
   - loss/losses.py - 包含各种损失函数
   - loss/vgg_arch.py - VGG网络架构相关

4. net目录
   - net/CIDNet.py - 主要网络模型

5. 其他目录（从代码中推断）
   - weights/ - 存储模型权重
   - output/ - 存储输出结果
   - output_hf/ - 存储Hugging Face输出
   - results/ - 存储训练结果和指标
   - datasets/ - 存储数据集
   - pic/ - 存储图片，用于README等

6. 特定数据集目录结构（从README中可以看到详细的数据集组织）
   - datasets/DICM
   - datasets/LIME
   - datasets/LOLdataset
   - datasets/LOLv2
   - datasets/LOL_blur
   - datasets/MEF
   - datasets/NPE
   - datasets/SICE
   - datasets/Sony_total_dark
   - datasets/VV

这个项目是一个低光照图像增强的深度学习项目，使用了一种称为HVI-CIDNet的模型。它支持多种数据集的训练和评估，并提供了图形界面和命令行界面来使用模型。

完整的目录结构应该是这样的（基于提供的信息推断）：
</think>

# HVI-CIDNet项目目录结构

## 主要目录结构

```
HVI-CIDNet/
├── pic/                      # 图片资源目录（用于README展示）
├── datasets/                 # 数据集目录
├── net/                      # 网络模型定义
│   └── CIDNet.py             # CIDNet模型实现
├── data/                     # 数据处理相关
│   ├── data.py               # 数据加载和处理
│   ├── options.py            # 配置选项
│   └── scheduler.py          # 学习率调度器
├── loss/                     # 损失函数
│   ├── losses.py             # 各种损失函数实现
│   └── vgg_arch.py           # VGG特征提取架构
├── weights/                  # 模型权重存储目录
├── output/                   # 评估输出图像目录
├── output_hf/                # Hugging Face模型输出
├── results/                  # 训练结果和指标
│   └── training/             # 训练过程中生成的图像和指标
├── flagged/                  # Gradio界面标记的图像
└── 主要脚本文件
```

## 主要脚本文件

```
├── app.py                    # Gradio演示界面
├── eval.py                   # 评估模型性能的主脚本
├── eval_SID_blur.py          # 针对SID和模糊数据集的评估脚本
├── eval_hf.py                # 使用Hugging Face模型评估的脚本
├── measure.py                # 测量指标的脚本
├── measure_SID_blur.py       # 测量SID和模糊数据集指标
├── measure_niqe_bris.py      # 测量NIQE和BRISQUE指标
├── net_test.py               # 测试网络参数、FLOPs和运行时间
├── train.py                  # 模型训练脚本
├── requirements.txt          # 项目依赖
└── Readme.md                 # 项目说明文档
```

## 数据集目录结构

```
datasets/
├── DICM/                     # DICM数据集
├── LIME/                     # LIME数据集
├── LOLdataset/               # LOLv1数据集
│   ├── our485/               # 训练集
│   │   ├── low/
│   │   └── high/
│   └── eval15/               # 评估集
│       ├── low/
│       └── high/
├── LOLv2/                    # LOLv2数据集
│   ├── Real_captured/        # 真实捕获数据
│   │   ├── Train/
│   │   └── Test/
│   └── Synthetic/            # 合成数据
│       ├── Train/
│       └── Test/
├── LOL_blur/                 # LOL模糊数据集
├── MEF/                      # MEF数据集
├── NPE/                      # NPE数据集
├── SICE/                     # SICE数据集
├── Sony_total_dark/          # Sony总暗数据集
└── VV/                       # VV数据集
```

这是HVI-CIDNet项目的目录结构，该项目是一个低光照图像增强的深度学习项目，它提出了一种新的颜色空间HVI并实现了高效的网络CIDNet来处理低光照图像。
