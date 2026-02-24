"""
双流 Wan 模型推理脚本

用法:
  torchrun --nproc_per_node=2 --master_port=29504 inference_dual_stream.py \
    --base_model_dir checkpoints/Wan-AI/Wan2.1-I2V-14B-720P \
    --checkpoint_dir ./output_dual_stream \
    --checkpoint_step 50 \
    --data_root ./data \
    --output_dir ./inference_results \
    --num_inference_steps 50 \
    --guidance_scale 1.0
"""
import argparse
import gc
import json
import math
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datetime import timedelta
from PIL import Image
from transformers import AutoTokenizer
from diffusers import FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, inject_adapter_in_model

# 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dual_wan_model_v2 import DualStreamWanModel, create_preprocess_fn, sinusoidal_embedding_1d
from comm_utils import sync_tensor
from real_dataset_dual import DualStreamRealDataset

try:
    from videox_fun.models import (
        WanTransformer3DModel,
        AutoencoderKLWan,
        WanT5EncoderModel,
        CLIPModel
    )
except ImportError:
    print("Error: videox_fun not found")
    sys.exit(1)


# ============================================================
# 工具函数
# ============================================================

def setup_distributed():
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=600)
    )
    rank = dist.get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    return rank, local_rank, device


def latent_to_images(vae, latents, device, weight_dtype):
    """
    VAE decode: latents [B, C, F, H, W] → images [B, F, 3, H_out, W_out]
    返回 numpy uint8 [B, F, H, W, 3]
    """
    vae.to(device, dtype=weight_dtype)
    with torch.no_grad():
        # VAE decode 期望 [B, C, F, H, W]
        decoded = vae.decode(latents.to(device, dtype=weight_dtype)).sample
    vae.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()

    # decoded: [B, C, F, H, W] → [B, F, H, W, C]
    decoded = decoded.permute(0, 2, 3, 4, 1)  # [B, F, H, W, C]
    decoded = ((decoded + 1.0) / 2.0).clamp(0, 1)
    images = (decoded * 255).to(torch.uint8).cpu().numpy()
    return images


def save_frames(images, save_dir, prefix="frame", room_name=""):
    """
    保存帧图像
    images: [F, H, W, 3] uint8 numpy
    """
    os.makedirs(save_dir, exist_ok=True)
    for f_idx in range(images.shape[0]):
        img = Image.fromarray(images[f_idx])
        fname = f"{prefix}_{f_idx:04d}.png"
        img.save(os.path.join(save_dir, fname))


# ============================================================
# 模型加载
# ============================================================

def load_model_for_inference(args, rank, device, weight_dtype):
    """
    加载完整的推理模型
    """
    # 1. 加载基础 Transformer
    print(f"[Rank {rank}] Loading base transformer...")
    trans_sub_dir = os.path.join(args.base_model_dir, "transformer")
    if os.path.exists(os.path.join(trans_sub_dir, "config.json")):
        load_dir = trans_sub_dir
    else:
        load_dir = args.base_model_dir

    transformer = WanTransformer3DModel.from_pretrained(
        load_dir, torch_dtype=weight_dtype
    )

    # 2. 注入 LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q", "k", "v","ffn.0","ffn.2"],
        bias="none"
    )
    transformer = inject_adapter_in_model(lora_config, transformer)

    # 3. 加载 LoRA 权重
    if rank == 0:
        lora_path = os.path.join(
            args.checkpoint_dir, f"lora_rgb_step{args.checkpoint_step}.pt"
        )
    else:
        lora_path = os.path.join(
            args.checkpoint_dir, f"lora_depth_step{args.checkpoint_step}.pt"
        )

    print(f"[Rank {rank}] Loading LoRA from {lora_path}")
    
    lora_state = torch.load(lora_path, map_location="cpu")
    
    # Debug: 打印 key 对比
    model_keys = [k for k in transformer.state_dict().keys() if 'lora' in k.lower()]
    ckpt_keys = list(lora_state.keys())
    print(f"[Rank {rank}] Model LoRA keys sample: {model_keys[:10]}")
    print(f"[Rank {rank}] Ckpt LoRA keys sample: {ckpt_keys[:10]}")
    
    # 自动适配 key: 尝试常见的前缀映射
    def adapt_lora_keys(state_dict, model_state_dict):
        model_lora_keys = {k for k in model_state_dict if 'lora' in k.lower()}
        ckpt_keys = set(state_dict.keys())
        
        # --- 步骤 1: 处理 PEFT 常见的 .default. 命名冲突 ---
        # 如果权重里没有 .default. 而模型里有，则强制补上
        processed_ckpt_state = {}
        has_default_in_model = any(".default." in k for k in model_lora_keys)
        
        for k, v in state_dict.items():
            new_k = k
            if has_default_in_model and ".default." not in k:
                # 把 lora_A.weight 变成 lora_A.default.weight
                new_k = k.replace("lora_A", "lora_A.default").replace("lora_B", "lora_B.default")
            processed_ckpt_state[new_k] = v
        
        # --- 步骤 2: 现有的前缀转换逻辑 (使用处理后的 processed_ckpt_state) ---
        state_dict = processed_ckpt_state
        ckpt_keys = set(state_dict.keys())

        if ckpt_keys & model_lora_keys:
            return state_dict
            
        # ... (保留你原来的 prefixes_to_try 逻辑) ...
        prefixes_to_try = [
            ("base_model.model.", ""),
            ("", "base_model.model."),
        ]
        
        for old_prefix, new_prefix in prefixes_to_try:
            # ... 原样保留你这里的代码 ...
            adapted = {}
            for k, v in state_dict.items():
                if k.startswith(old_prefix):
                    new_key = new_prefix + k[len(old_prefix):]
                else:
                    new_key = k
                adapted[new_key] = v
            
            if set(adapted.keys()) & model_lora_keys:
                print(f"  Adapted keys with: '{old_prefix}' → '{new_prefix}'")
                return adapted
        
        # 如果还是不匹配，用模糊匹配：找 ckpt key 的后缀在 model key 中的匹配
        adapted = {}
        model_suffix_map = {}
        for mk in model_lora_keys:
            # 取最后几段作为后缀
            parts = mk.split('.')
            for i in range(len(parts)):
                suffix = '.'.join(parts[i:])
                model_suffix_map[suffix] = mk
        
        for ck, v in state_dict.items():
            parts = ck.split('.')
            matched = False
            for i in range(len(parts)):
                suffix = '.'.join(parts[i:])
                if suffix in model_suffix_map:
                    adapted[model_suffix_map[suffix]] = v
                    matched = True
                    break
            if not matched:
                adapted[ck] = v  # 保留原 key
        
        matched_count = len(set(adapted.keys()) & model_lora_keys)
        print(f"  Fuzzy matched {matched_count}/{len(ckpt_keys)} keys")
        return adapted
    
    adapted_state = adapt_lora_keys(lora_state, transformer.state_dict())
    missing, unexpected = transformer.load_state_dict(adapted_state, strict=False)

    lora_loaded = sum(1 for k in adapted_state if k in transformer.state_dict())
    print(f"[Rank {rank}] LoRA loaded: {lora_loaded} params matched, "
          f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    # 验证：missing 里不应该有 lora 相关的 key
    missing_lora = [k for k in missing if 'lora' in k.lower()]
    if missing_lora:
        print(f"[Rank {rank}] WARNING: {len(missing_lora)} LoRA keys still missing!")
        print(f"  Examples: {missing_lora[:3]}")
    else:
        print(f"[Rank {rank}] All LoRA keys loaded successfully.")
    print(f"[Rank {rank}] LoRA loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # 4. 创建 DualStreamWanModel
    if rank == 0:
        dual_model = DualStreamWanModel(
            rgb_model=transformer,
            depth_model=None,
            interaction_interval=args.interaction_interval
        )
    else:
        dual_model = DualStreamWanModel(
            rgb_model=None,
            depth_model=transformer,
            interaction_interval=args.interaction_interval
        )

    # 5. 加载交互层权重（两个 rank 都需要）
    cross_attn_path = os.path.join(
        args.checkpoint_dir, f"cross_attn_step{args.checkpoint_step}.pt"
    )
    pose_cross_attn_path = os.path.join(
        args.checkpoint_dir, f"pose_cross_attn_step{args.checkpoint_step}.pt"
    )
    plucker_net_path = os.path.join(
        args.checkpoint_dir, f"plucker_net_step{args.checkpoint_step}.pt"
    )

    print(f"[Rank {rank}] Loading cross_attn, pose_cross_attn, plucker_net...")
    dual_model.cross_attn_layers.load_state_dict(
        torch.load(cross_attn_path, map_location="cpu")
    )
    dual_model.pose_cross_attn_layers.load_state_dict(
        torch.load(pose_cross_attn_path, map_location="cpu")
    )
    dual_model.plucker_net.load_state_dict(
        torch.load(plucker_net_path, map_location="cpu")
    )

    # 6. 设置设备
    dual_model.setup_for_rank(rank, device)
    dual_model.to(dtype=weight_dtype)
    dual_model.eval()

    # Fix precision
    try:
        from videox_fun.models.wan_transformer3d import WanLayerNorm
        for name, module in dual_model.named_modules():
            if isinstance(module, WanLayerNorm):
                module.float()
    except ImportError:
        pass

    return dual_model


# ============================================================
# 采样循环
# ============================================================

@torch.no_grad()
def sample(
    dual_model,
    noise_scheduler,
    latents_incomplete_rgb, latents_incomplete_depth,
    mask_rgb, mask_depth,
    clip_fea_rgb, clip_fea_depth,
    context,
    plucker_rgb, plucker_depth,
    preprocess_fn,
    num_inference_steps=50,
    guidance_scale=1.0,
    rank=0,
    device="cuda",
    weight_dtype=torch.bfloat16,
):
    """
    Euler 采样去噪

    Flow matching: x_t = (1 - sigma) * x_0 + sigma * noise
    v-prediction: v = noise - x_0
    更新: x_{t-1} = x_t - dt * v_pred  (由 scheduler 处理)
    """
    bsz = latents_incomplete_rgb.shape[0]
    latent_shape = latents_incomplete_rgb.shape  # [B, 16, F_l, H_l, W_l]

    # 初始化纯噪声
    latents = torch.randn(latent_shape, device=device, dtype=weight_dtype)
    latents = sync_tensor(latents, rank)

    # 设置 timesteps
    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = noise_scheduler.timesteps

    # 条件 y = [mask, incomplete_latent * mask]
    y_rgb = torch.cat([mask_rgb, latents_incomplete_rgb * mask_rgb], dim=1)
    y_depth = torch.cat([mask_depth, latents_incomplete_depth * mask_depth], dim=1)

    f, h, w = latent_shape[2], latent_shape[3] // 2, latent_shape[4] // 2
    seq_len = f * h * w

    print(f"[Rank {rank}] Starting sampling with {num_inference_steps} steps...")

    for i, t in enumerate(timesteps):
        t_batch = t.unsqueeze(0).expand(bsz).to(device)
        t_batch = sync_tensor(t_batch, rank)

        # 模型前向
        pred = dual_model(
            latents, y_rgb, clip_fea_rgb,
            latents, y_depth, clip_fea_depth,
            t_batch, context, seq_len,
            preprocess_fn,
            plucker_rgb, plucker_depth
        )

        # scheduler step
        # FlowMatchEulerDiscreteScheduler 的 step 接口
        scheduler_output = noise_scheduler.step(pred, t, latents)
        latents = scheduler_output.prev_sample

        if (i + 1) % 10 == 0 or i == 0:
            print(f"[Rank {rank}] Step {i+1}/{num_inference_steps} done")

    return latents


# ============================================================
# 主函数
# ============================================================

def main():
    args = parse_args()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    rank, local_rank, device = setup_distributed()
    assert dist.get_world_size() == 2, "Inference requires exactly 2 GPUs"

    weight_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16

    # ---- 加载模型 ----
    # 串行加载避免内存爆炸
    if rank == 0:
        dual_model = load_model_for_inference(args, rank, device, weight_dtype)
    dist.barrier()
    if rank == 1:
        dual_model = load_model_for_inference(args, rank, device, weight_dtype)
    dist.barrier()
    gc.collect()
    torch.cuda.empty_cache()

    # T5: 只在 rank 0 加载
    if rank == 0:
        print("[Rank 0] Loading T5...")
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(args.base_model_dir, "google/umt5-xxl")
        )
        
        # 定义 T5 的架构参数 (必须和训练时一致)
        t5_config = {
            "vocab": 256384, 
            "dim": 4096, 
            "dim_attn": 4096, 
            "dim_ffn": 10240, 
            "num_heads": 64, 
            "num_layers": 24, 
            "num_buckets": 32, 
            "shared_pos": False
        }
        
        t5_path = os.path.join(args.base_model_dir, "models_t5_umt5-xxl-enc-bf16.pth")
        
        # 先用 config 初始化模型，再加载权重
        text_encoder = WanT5EncoderModel(**t5_config).to(dtype=weight_dtype)
        if os.path.exists(t5_path):
            # 使用 torch.load 加载权重
            state_dict = torch.load(t5_path, map_location="cpu")
            text_encoder.load_state_dict(state_dict, strict=False)
        
        text_encoder.to(device).eval()
    else:
        tokenizer = None
        text_encoder = None

    dist.barrier()

    # VAE + CLIP: 串行加载
    if rank == 0:
        print("[Rank 0] Loading VAE + CLIP...")
        vae_path = os.path.join(args.base_model_dir, "Wan2.1_VAE.pth")
        vae = AutoencoderKLWan.from_pretrained(vae_path).eval()

        clip_path = os.path.join(
            args.base_model_dir,
            "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
        )
        clip_encoder = CLIPModel.from_pretrained(clip_path).to(dtype=weight_dtype).eval()
    dist.barrier()
    if rank == 1:
        print("[Rank 1] Loading VAE + CLIP...")
        vae_path = os.path.join(args.base_model_dir, "Wan2.1_VAE.pth")
        vae = AutoencoderKLWan.from_pretrained(vae_path).eval()

        clip_path = os.path.join(
            args.base_model_dir,
            "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
        )
        clip_encoder = CLIPModel.from_pretrained(clip_path).to(dtype=weight_dtype).eval()
    dist.barrier()
    gc.collect()

    # ---- Noise scheduler ----
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=args.train_sampling_steps,
        shift=5.0
    )

    preprocess_fn = create_preprocess_fn(device, weight_dtype)

    # ---- 数据集 ----
    dataset = DualStreamRealDataset(
        data_root=args.data_root,
        height=128,
        width=256,
    )

    print(f"[Rank {rank}] Dataset: {len(dataset)} samples")

    # ---- 推理循环 ----
    os.makedirs(args.output_dir, exist_ok=True)

    num_samples = min(args.num_samples, len(dataset))

    for sample_idx in range(num_samples):
        print(f"\n[Rank {rank}] ===== Sample {sample_idx+1}/{num_samples} =====")

        sample = dataset[sample_idx]

        # 加 batch 维度
        for key in sample:
            if isinstance(sample[key], torch.Tensor):
                sample[key] = sample[key].unsqueeze(0).to(device)

        # ---- 广播数据确保一致 ----
        tensor_keys = [
            'gt_rgb', 'gt_depth',
            'incomplete_rgb', 'incomplete_depth',
            'mask_rgb', 'mask_depth',
            'plucker_rgb', 'plucker_depth',
        ]
        for key in tensor_keys:
            if key in sample:
                # 无论发送还是接收，先移动到 device 并强制连续化
                # 注意：必须赋值回 sample[key]，否则 broadcast 的是旧的内存地址
                sample[key] = sample[key].to(device).contiguous()
                dist.broadcast(sample[key], src=0)

        # ---- 编码 ----
        # T5 文本编码 (确保护持 GPU 状态)
        if rank == 0:
            print(sample)
            text = sample['text']
            # 确保 tokenizer 的输出也在 GPU
            prompt_ids = tokenizer([text], padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                context = text_encoder(prompt_ids.input_ids)[0].contiguous()
        else:
            # Rank 1 创建占位符
            context = torch.zeros(1, 512, 4096, device=device, dtype=weight_dtype).contiguous()
        
        # 同步 Context
        dist.broadcast(context, src=0)

        # ---- VAE & CLIP 保持在 GPU 直到推理结束 (推理通常显存够) ----
        vae.to(device, dtype=weight_dtype)
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling() # 开启分块解码，防止 OOM

        with torch.no_grad():
            # 直接使用已经在 GPU 上的数据
            latents_incomplete_rgb = vae.encode(sample['incomplete_rgb'].to(weight_dtype)).latent_dist.sample()
            latents_incomplete_depth = vae.encode(sample['incomplete_depth'].to(weight_dtype)).latent_dist.sample()

        # CLIP 推理
        clip_encoder.to(device, dtype=weight_dtype)
        with torch.no_grad():
            rgb_first = sample['gt_rgb'][:, :, 0:1, :, :]
            depth_first = sample['gt_depth'][:, :, 0:1, :, :]
            clip_fea_rgb = clip_encoder([rgb_first[0]])
            clip_fea_depth = clip_encoder([depth_first[0]])
        clip_encoder.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        # Mask: 下采样到 latent 空间
        mask_rgb = F.interpolate(
            sample['mask_rgb'].to(device, dtype=weight_dtype),
            size=latents_incomplete_rgb.shape[2:]
        )
        mask_depth = F.interpolate(
            sample['mask_depth'].to(device, dtype=weight_dtype),
            size=latents_incomplete_depth.shape[2:]
        )

        # Plücker
        plucker_rgb = sample['plucker_rgb'].to(device, dtype=weight_dtype)
        plucker_depth = sample['plucker_depth'].to(device, dtype=weight_dtype)

        # ---- 采样 ----
        predicted_latents = sample_fn(
            dual_model=dual_model,
            noise_scheduler=noise_scheduler,
            latents_incomplete_rgb=latents_incomplete_rgb,
            latents_incomplete_depth=latents_incomplete_depth,
            mask_rgb=mask_rgb,
            mask_depth=mask_depth,
            clip_fea_rgb=clip_fea_rgb,
            clip_fea_depth=clip_fea_depth,
            context=context,
            plucker_rgb=plucker_rgb,
            plucker_depth=plucker_depth,
            preprocess_fn=preprocess_fn,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            rank=rank,
            device=device,
            weight_dtype=weight_dtype,
        )

        # ---- VAE decode ----
        vae.to(device, dtype=weight_dtype)
        with torch.no_grad():
            decoded = vae.decode(predicted_latents.to(weight_dtype)).sample
        vae.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        # [B, C, F, H, W] → [B, F, H, W, C] → uint8
        decoded = decoded.permute(0, 2, 3, 4, 1)
        decoded = ((decoded + 1.0) / 2.0).clamp(0, 1)
        images = (decoded * 255).to(torch.uint8).cpu().numpy()

        # ---- 保存 ----
        sample_dir = os.path.join(args.output_dir, f"sample_{sample_idx:04d}")

        if rank == 0:
            # 保存 RGB 结果
            rgb_dir = os.path.join(sample_dir, "predicted_rgb")
            save_frames(images[0], rgb_dir, prefix="rgb")

            # 也保存 GT 和 输入 做对比
            gt_dir = os.path.join(sample_dir, "gt_rgb")
            gt_images = sample['gt_rgb'].permute(0, 2, 3, 4, 1)
            gt_images = ((gt_images + 1.0) / 2.0).clamp(0, 1)
            gt_images = (gt_images * 255).to(torch.uint8).cpu().numpy()
            save_frames(gt_images[0], gt_dir, prefix="gt_rgb")

            inc_dir = os.path.join(sample_dir, "input_incomplete_rgb")
            inc_images = sample['incomplete_rgb'].permute(0, 2, 3, 4, 1)
            inc_images = ((inc_images + 1.0) / 2.0).clamp(0, 1)
            inc_images = (inc_images * 255).to(torch.uint8).cpu().numpy()
            save_frames(inc_images[0], inc_dir, prefix="incomplete_rgb")

            print(f"[Rank 0] RGB results saved to {rgb_dir}")

        else:
            # 保存 Depth 结果
            depth_dir = os.path.join(sample_dir, "predicted_depth")
            save_frames(images[0], depth_dir, prefix="depth")

            gt_dir = os.path.join(sample_dir, "gt_depth")
            gt_images = sample['gt_depth'].permute(0, 2, 3, 4, 1)
            gt_images = ((gt_images + 1.0) / 2.0).clamp(0, 1)
            gt_images = (gt_images * 255).to(torch.uint8).cpu().numpy()
            save_frames(gt_images[0], gt_dir, prefix="gt_depth")

            print(f"[Rank 1] Depth results saved to {depth_dir}")

        dist.barrier()

    # ---- 清理 ----
    if rank == 0:
        # T5 清理
        if text_encoder is not None:
            del text_encoder
            gc.collect()
            torch.cuda.empty_cache()

    dist.destroy_process_group()
    print(f"[Rank {rank}] Inference complete!")


# 给 sample 函数一个别名，避免和 dataset sample 冲突
sample_fn = sample


def parse_args():
    parser = argparse.ArgumentParser(description="Dual Stream Wan Inference")

    # 路径
    parser.add_argument("--base_model_dir", type=str, required=True,
                        help="Wan 基础模型目录")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="训练 checkpoint 目录 (包含 lora_rgb/depth, cross_attn 等)")
    parser.add_argument("--checkpoint_step", type=int, required=True,
                        help="加载哪一步的 checkpoint")
    parser.add_argument("--data_root", type=str, required=True,
                        help="测试数据根目录")
    parser.add_argument("--output_dir", type=str, default="./inference_results")

    # 采样参数
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=5,
                        help="推理多少个样本")
    parser.add_argument("--train_sampling_steps", type=int, default=1000)

    # 模型配置 (需要和训练时一致)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--interaction_interval", type=int, default=4)

    # 精度
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--allow_tf32", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main()
