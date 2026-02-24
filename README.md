# 双流 Wan 模型：多视角全景图补全

基于 [Wan2.1-I2V-14B](https://github.com/Wan-Video/Wan2.1) 的双流视频扩散框架，用于多视角 360° 全景图像补全。系统同时处理 **RGB** 和 **Depth** 两个流，通过跨流交互与 Plücker 射线相机位姿条件注入，实现从稀疏观测视角生成几何一致的完整全景图。

## 概述

给定参考位姿（pose 0）的完整全景图（RGB + Depth），模型通过填充投影产生的缺失区域，生成新视角下的完整全景图。核心设计包括：

- **双流架构**：两个并行的 Wan Transformer 分支（RGB 流 & Depth 流），通过分布式训练分别运行在两张 GPU 上
- **跨流交互**：在可配置的 Transformer 层间隔处，RGB 和 Depth 流之间进行特征交换，支持梯度反传
- **Plücker 位姿条件**：3D CNN（`PluckerNet`）将 6D Plücker 射线图编码为位姿特征，通过 Cross-Attention 注入，实现几何感知生成
- **LoRA 微调**：使用低秩适配（LoRA）对预训练 Wan 模型进行高效微调

## 架构

```
GPU 0 (RGB 流)                        GPU 1 (Depth 流)
┌──────────────────┐                  ┌──────────────────┐
│  Wan Transformer  │ ── 跨流交互 ──  │  Wan Transformer  │
│  + LoRA 适配器    │  Cross-Attn     │  + LoRA 适配器    │
│                   │ ◄──────────────►│                   │
│  + Pose CrossAttn │                 │  + Pose CrossAttn │
│    (PluckerNet)   │                 │    (PluckerNet)   │
└──────────────────┘                  └──────────────────┘
```

交互层在可配置的间隔（默认每 4 层）交替执行 **Pose Cross-Attention** 和 **跨流 Cross-Attention**。

## 项目结构

```
.
├── train_dual_stream.py       # 分布式训练脚本（双 GPU）
├── inference_dual_stream.py   # 分布式推理脚本（双 GPU）
├── dual_wan_model_v2.py       # 双流模型定义（PluckerNet + 交互层）
├── real_dataset_dual.py       # HM3D 数据集加载器（含 Plücker 射线图生成）
├── comm_utils.py              # 支持梯度反传的分布式通信原语
└── README.md
```

## 环境配置

### Python 依赖

```bash
pip install torch torchvision
pip install transformers diffusers peft
pip install opencv-python numpy pillow
```

### 外部依赖

- **[VideoX-Fun](https://github.com/alibaba/VideoX-Fun)**：提供 `WanTransformer3DModel`、`AutoencoderKLWan`、`WanT5EncoderModel`、`CLIPModel`
- **[Wan2.1-I2V-14B-720P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P)**：预训练基座模型权重

### 硬件要求

- **2× NVIDIA GPU**（每张 GPU 承载一个流），推荐 2× A100 80GB 或同等规格
- 需要 NCCL 后端支持分布式通信

## 数据准备

数据集采用 **HM3D 多视角全景图** 格式：

```
data/
  {area}/
    {room}/
      input/
        panorama_0000.png                      # pose 0 完整 RGB 全景图
        panorama_0000_depth.npy                # pose 0 完整深度图
        pano0000_to_pano0001_rgb.png           # pose 0→1 投影的残缺 RGB
        pano0000_to_pano0001_depth_range.npy   # pose 0→1 投影的残缺深度
        pano0000_to_pano0001_mask.png          # 投影 mask（可选）
        pose_0000.json ~ pose_XXXX.json        # 相机位姿（position + 四元数）
        description.txt                        # 文本描述
      output/
        panorama_0000.png ~ panorama_XXXX.png              # GT 完整 RGB
        panorama_0000_depth.npy ~ panorama_XXXX_depth.npy  # GT 完整深度
```

**帧数约束**：Wan 要求 `num_frames % 4 == 1`（即 5, 9, 13, ...）。少于 5 帧的房间会被丢弃；超出时保留首尾帧，中间随机采样至最近的合法帧数。

## 训练

使用 2 张 GPU 启动分布式训练：

```bash
torchrun --nproc_per_node=2 --master_port=29500 train_dual_stream.py \
    --base_model_dir checkpoints/Wan-AI/Wan2.1-I2V-14B-720P \
    --output_dir ./output_dual_stream \
    --train_batch_size 1 \
    --max_train_steps 1000 \
    --learning_rate 1e-4 \
    --lora_rank 32 \
    --lora_alpha 32 \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --lr_scheduler constant_with_warmup \
    --lr_warmup_steps 100 \
    --checkpointing_steps 50 \
    --interaction_interval 4 \
    --resolution 720 \
    --num_frames 17
```

### 主要训练参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--base_model_dir` | （必填） | 预训练 Wan2.1-I2V-14B-720P 路径 |
| `--lora_rank` | 32 | LoRA 秩 |
| `--interaction_interval` | 4 | 每隔 N 层进行一次跨流交互 |
| `--mixed_precision` | bf16 | 混合精度模式（`no`、`fp16`、`bf16`） |
| `--gradient_checkpointing` | False | 启用梯度检查点以节省显存 |
| `--weighting_scheme` | logit_normal | Flow Matching 损失加权方案 |

## 推理

```bash
torchrun --nproc_per_node=2 --master_port=29504 inference_dual_stream.py \
    --base_model_dir checkpoints/Wan-AI/Wan2.1-I2V-14B-720P \
    --checkpoint_dir ./output_dual_stream \
    --checkpoint_step 50 \
    --data_root ./data \
    --output_dir ./inference_results \
    --num_inference_steps 50 \
    --guidance_scale 1.0
```

## 技术细节

### 通信原语（`comm_utils.py`）

自定义 `torch.autograd.Function` 实现，支持**梯度在分布式通信中反传**：

- `ExchangeFeatures`：GPU 0 与 GPU 1 之间的点对点特征交换，支持梯度反传
- `AllGatherWithGrad`：支持梯度的 All-Gather 操作，用于收集所有 rank 的特征

### Plücker 射线图

相机位姿被编码为 6D Plücker 坐标（3D 方向 + 3D 矩），基于等距柱状投影（ERP）全景图几何（水平 360°、垂直 180°）计算。经 3D CNN（`PluckerNet`）处理后，通过 Cross-Attention 在交替的交互点注入模型。

### 损失函数

训练结合多种损失：

- **Flow Matching MSE 损失**：支持 logit-normal、uniform 等加权方案
- **潜空间结构一致性损失**：基于 patch 级别相关性
- **潜空间全局一致性损失**：确保生成的全局连贯性
- **SSIM 损失**：在解码后的视频帧上计算

## 致谢

本项目基于以下工作构建：

- [Wan2.1](https://github.com/Wan-Video/Wan2.1) — 视频生成基座模型
- [VideoX-Fun](https://github.com/alibaba/VideoX-Fun) — 视频生成工具包
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers) — 扩散模型库
- [PEFT](https://github.com/huggingface/peft) — 参数高效微调
