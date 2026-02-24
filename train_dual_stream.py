import argparse
import gc
import os
import sys
import logging
from contextlib import nullcontext
from datetime import timedelta
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_loss_weighting_for_sd3
from diffusers.optimization import get_scheduler
from peft import LoraConfig, inject_adapter_in_model, get_peft_model_state_dict
from real_dataset_dual import DualStreamRealDataset, collate_fn_dual

# 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 本地
from dual_wan_model_v2 import DualStreamWanModel, create_preprocess_fn
from dummy_dataset_dual_v2 import DummyDualStreamDataset, collate_fn_dual
from comm_utils import sync_tensor, sync_gradients_for_shared_params

# VideoX-Fun
try:
    from videox_fun.models import (
        WanTransformer3DModel, 
        AutoencoderKLWan, 
        WanT5EncoderModel, 
        CLIPModel
    )
    from videox_fun.utils.discrete_sampler import DiscreteSampling
except ImportError:
    print("Warning: videox_fun not found, using placeholder imports")
    WanTransformer3DModel = None
    AutoencoderKLWan = None
    WanT5EncoderModel = None
    CLIPModel = None
    DiscreteSampling = None


def setup_logging(rank):
    logging.basicConfig(
        format=f'[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)


def setup_distributed():
    """初始化分布式环境"""
    dist.init_process_group(
    backend="nccl",
    init_method="env://",
    timeout=timedelta(seconds=600)
)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    return rank, world_size, local_rank, device


def cleanup_distributed():
    """清理分布式环境"""
    dist.destroy_process_group()


def clear_memory():
    """清理显存"""
    gc.collect()
    torch.cuda.empty_cache()


def custom_mse_loss(pred, target, weighting=None, threshold=50.0):
    """MSE Loss，阈值裁剪"""
    pred, target = pred.float(), target.float()
    diff = pred - target
    mse_loss = F.mse_loss(pred, target, reduction='none')
    
    mask = (diff.abs() <= threshold).float()
    masked_loss = mse_loss * mask
    
    if weighting is not None:
        while weighting.dim() < masked_loss.dim():
            weighting = weighting.unsqueeze(-1)
        masked_loss = masked_loss * weighting
        
    return masked_loss.mean()

def latent_structure_consistency_loss(pred_latent, gt_latent, patch_size=2):
    """
    Latent space 跨帧 patch-wise 结构一致性 loss
    
    鼓励模型预测的跨帧结构关系与 GT 一致。
    
    Args:
        pred_latent: [B, C, F, H, W] 模型估计的 clean latent
        gt_latent:   [B, C, F, H, W] GT latent
        patch_size:  空间 patch 大小
    """
    B, C, F, H, W = pred_latent.shape
    
    # 切 patch: [B, C, F, nH, nW, p, p]
    pred_patches = pred_latent.unfold(3, patch_size, patch_size) \
                              .unfold(4, patch_size, patch_size)
    gt_patches = gt_latent.unfold(3, patch_size, patch_size) \
                          .unfold(4, patch_size, patch_size)
    
    nH, nW = pred_patches.shape[3], pred_patches.shape[4]
    
    # reshape: [B, F, nH*nW, C*p*p]
    pred_patches = pred_patches.permute(0, 2, 3, 4, 1, 5, 6) \
                               .reshape(B, F, nH * nW, -1)
    gt_patches = gt_patches.permute(0, 2, 3, 4, 1, 5, 6) \
                           .reshape(B, F, nH * nW, -1)
    
    # L2 normalize
    pred_norm = torch.nn.functional.normalize(pred_patches, dim=-1)
    gt_norm = torch.nn.functional.normalize(gt_patches, dim=-1)
    
    loss = 0.0
    count = 0
    
    for f in range(F - 1):
        # 相邻帧 patch 相似度矩阵 [B, num_patches, num_patches]
        pred_sim = torch.bmm(pred_norm[:, f], pred_norm[:, f+1].transpose(1, 2))
        gt_sim = torch.bmm(gt_norm[:, f], gt_norm[:, f+1].transpose(1, 2))
        
        loss += torch.nn.functional.mse_loss(pred_sim, gt_sim.detach())
        count += 1
    
    return loss / max(count, 1)


def latent_global_consistency_loss(pred_latent):
    """
    全局统计量一致性 loss
    
    鼓励不同帧（视角）的 latent 全局 mean/std 一致，保证风格统一。
    
    Args:
        pred_latent: [B, C, F, H, W]
    """
    per_frame_mean = pred_latent.mean(dim=[3, 4])  # [B, C, F]
    per_frame_std = pred_latent.std(dim=[3, 4])     # [B, C, F]
    
    mean_var = per_frame_mean.var(dim=2).mean()
    std_var = per_frame_std.var(dim=2).mean()
    
    return mean_var + std_var


def sigma_dependent_weight(sigmas, low=0.2, high=0.8):
    """
    Sigma-dependent weighting: sigma 小时权重大，sigma 大时权重小
    
    使用平滑的余弦衰减：
    - sigma < low  → weight ≈ 1.0
    - sigma > high → weight ≈ 0.0
    - 中间平滑过渡
    
    Args:
        sigmas: [B, 1, 1, 1, 1] 当前噪声水平
        low:    开始衰减的 sigma 阈值
        high:   衰减到零的 sigma 阈值
        
    Returns:
        weight: scalar, batch 平均权重
    """
    import math
    sigma_flat = sigmas.flatten()  # [B]
    
    # 线性映射到 [0, 1] 区间，再用余弦平滑
    t = ((sigma_flat - low) / (high - low)).clamp(0, 1)
    weight = 0.5 * (1 + torch.cos(t * math.pi))  # 1 → 0 平滑过渡
    
    return weight.mean()

def fix_wan_precision_issues(model):
    """精度问题"""
    try:
        from videox_fun.models.wan_transformer3d import WanLayerNorm
        for name, module in model.named_modules():
            if isinstance(module, WanLayerNorm):
                module.float()
            elif isinstance(module, nn.LayerNorm):
                module.to(torch.bfloat16)
    except ImportError:
        pass

def vae_decode_video(vae, latents):
    """
    latents: [B, C, F, H, W]  (latent space)
    return:  images:  [B, 3, F, H*scale, W*scale]  (pixel space, typically in [-1,1] or [0,1])
    """
    # diffusers/wan 一般都有 scaling_factor（可能在 vae.config 或 vae 上）
    sf = getattr(vae, "scaling_factor", None)
    if sf is None and hasattr(vae, "config"):
        sf = getattr(vae.config, "scaling_factor", 1.0)
    if sf is None:
        sf = 1.0

    x = latents / sf

    out = vae.decode(x)
    # 兼容不同返回结构
    if hasattr(out, "sample"):
        img = out.sample
    else:
        img = out

    return img

def _ssim_2d(x, y, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    x,y: [N, C, H, W] in [0,1] (推荐先 clamp 到 0~1)
    return: [N] ssim per-sample
    """
    pad = window_size // 2
    mu_x = F.avg_pool2d(x, window_size, 1, pad)
    mu_y = F.avg_pool2d(y, window_size, 1, pad)

    sigma_x  = F.avg_pool2d(x * x, window_size, 1, pad) - mu_x * mu_x
    sigma_y  = F.avg_pool2d(y * y, window_size, 1, pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, window_size, 1, pad) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2) + 1e-8)

    # 对空间和通道平均
    return ssim_map.mean(dim=[1,2,3])

def ssim_loss_video(pred, target, window_size=11):
    """
    pred/target: [B, C, F, H, W]
    返回: scalar loss
    """
    B, C, Fm, H, W = pred.shape
    x = pred.permute(0,2,1,3,4).reshape(B*Fm, C, H, W)
    y = target.permute(0,2,1,3,4).reshape(B*Fm, C, H, W)

    # 建议把 decode 输出映射到 [0,1]
    x = x.clamp(0, 1)
    y = y.clamp(0, 1)

    ssim_val = _ssim_2d(x, y, window_size=window_size)  # [B*F]
    return (1.0 - ssim_val).mean()
def image_patch_features(img, patch_size=8):
    """
    img: [B, C, F, H, W] in pixel space
    return: feats [B, F, N, D], where D = C*patch*patch
    """
    B, C, Fm, H, W = img.shape
    assert H % patch_size == 0 and W % patch_size == 0

    patches = img.unfold(3, patch_size, patch_size).unfold(4, patch_size, patch_size)
    # [B, C, F, nH, nW, p, p]
    nH, nW = patches.shape[3], patches.shape[4]
    patches = patches.permute(0, 2, 3, 4, 1, 5, 6).reshape(B, Fm, nH*nW, -1)
    return patches  # [B,F,N,D]

def ray_guided_photo_loss(decoded_img, plucker_map, patch_size_img=8, patch_size_plucker=8,
                          temperature=0.1, top_k=8):
    """
    decoded_img: [B, C, F, H, W] (pixel)
    plucker_map: [B, 6, F, H0, W0] (原分辨率或任意；内部会 pool 到 img 分辨率)
    """
    B, C, Fm, H, W = decoded_img.shape

    # 1) plucker 下采样到 img 分辨率（保证 patch 对齐）
    plucker_resized = F.adaptive_avg_pool3d(plucker_map, (Fm, H, W))  # [B,6,F,H,W]

    # 2) plucker 切 patch 并均值到 [B,F,N,6]
    ps = patch_size_plucker
    ppatch = plucker_resized.unfold(3, ps, ps).unfold(4, ps, ps)  # [B,6,F,nH,nW,ps,ps]
    nH, nW = ppatch.shape[3], ppatch.shape[4]
    N = nH * nW
    pp = ps * ps
    ppatch = ppatch.permute(0,2,3,4,1,5,6).reshape(B, Fm, N, 6, pp).mean(dim=-1)  # [B,F,N,6]

    # 3) img 切 patch 特征 [B,F,N,D]
    feats = image_patch_features(decoded_img, patch_size=patch_size_img)
    feats = F.normalize(feats, dim=-1)

    loss = 0.0
    count = 0
    for f in range(Fm - 1):
        geo_weight = plucker_ray_distance_weight(
            ppatch[:, f], ppatch[:, f+1],
            temperature=temperature,
            top_k=top_k
        ).detach()  # [B,N,N]

        # cosine sim: [B,N,N]
        sim = torch.bmm(feats[:, f], feats[:, f+1].transpose(1,2))

        # 目标：几何相近的对，sim 越大越好 -> (1 - sim)
        diff = (1.0 - sim).clamp(min=0.0)
        loss_f = (diff * geo_weight).sum() / (geo_weight.sum() + 1e-8)

        loss += loss_f
        count += 1

    return loss / max(count, 1)



def plucker_ray_distance_weight(plucker_f1, plucker_f2, temperature=0.1, top_k=8):
    """
    计算两帧之间 patch 对的 3D 光线最短距离，用于判断是否看同一区域
    
    dist = |d1·m2 + d2·m1| / ||d1 × d2||
    
    只有 dist ≈ 0 时，两条光线才真正接近相交。
    额外用 top-k 过滤，每个 patch 只保留几何最近的 k 个匹配。
    
    Args:
        plucker_f1: [B, N, 6]  帧 f 的 patch-level Plücker 坐标
        plucker_f2: [B, N, 6]  帧 f+1 的 patch-level Plücker 坐标
        temperature: 高斯核温度参数
        top_k: 每个 patch 保留的最近邻数量
        
    Returns:
        weight: [B, N, N]  稀疏权重矩阵，高值 = 看同一区域
    """
    d1 = plucker_f1[..., :3]   # [B, N, 3]
    m1 = plucker_f1[..., 3:]   # [B, N, 3]
    d2 = plucker_f2[..., :3]   # [B, M, 3]
    m2 = plucker_f2[..., 3:]   # [B, M, 3]
    
    # ---- reciprocal product: |d1·m2 + d2·m1| ----
    term1 = torch.einsum('bni,bmi->bnm', d1, m2)  # [B, N, M]
    term2 = torch.einsum('bni,bmi->bnm', m1, d2)  # [B, N, M]
    reciprocal = (term1 + term2).abs()              # [B, N, M]
    
    # ---- ||d1 × d2||: 方向叉积的模长 ----
    # d1: [B, N, 3] vs d2: [B, M, 3]
    # 需要计算所有 (i, j) 对的叉积模长
    # d1_expanded: [B, N, 1, 3], d2_expanded: [B, 1, M, 3]
    d1_exp = d1.unsqueeze(2)  # [B, N, 1, 3]
    d2_exp = d2.unsqueeze(1)  # [B, 1, M, 3]
    cross_prod = torch.cross(
        d1_exp.expand(-1, -1, d2.shape[1], -1),
        d2_exp.expand(-1, d1.shape[1], -1, -1),
        dim=-1
    )  # [B, N, M, 3]
    cross_norm = cross_prod.norm(dim=-1)  # [B, N, M]
    
    # ---- 光线最短距离 ----
    # 当 cross_norm ≈ 0 时，光线几乎平行，距离无意义 → 给大值排除掉
    dist = reciprocal / (cross_norm + 1e-6)  # [B, N, M]
    
    # 平行光线标记（叉积模长太小 = 几乎平行）
    parallel_mask = (cross_norm < 1e-4)
    dist = dist.masked_fill(parallel_mask, 1e6)  # 平行的给大距离
    
    # ---- 高斯核权重 ----
    weight = torch.exp(-(dist ** 2) / temperature)  # [B, N, M]
    
    # ---- Top-k 过滤：每个 patch 只保留 k 个最近邻 ----
    if top_k > 0 and top_k < weight.shape[-1]:
        # 沿 M 维度取 top-k
        topk_vals, topk_idx = weight.topk(top_k, dim=-1)  # [B, N, k]
        
        # 构建稀疏 mask
        sparse_weight = torch.zeros_like(weight)
        sparse_weight.scatter_(-1, topk_idx, topk_vals)
        weight = sparse_weight
    
    return weight

def pose_guided_depth_consistency_loss(
    pred_depth_latent, gt_depth_latent, 
    plucker_map, patch_size=2
):
    """
    基于 Plücker 互矩的 Depth latent 跨帧一致性 loss
    
    只约束光线相交的 patch 对（= 看同一 3D 区域的像素），
    鼓励这些位置的 depth latent 结构一致。
    
    Args:
        pred_depth_latent: [B, C, F, H, W]
        gt_depth_latent:   [B, C, F, H, W]
        plucker_map:       [B, 6, F, H, W]
        patch_size:        空间 patch 大小
    """
    B, C, F_2 , H_l, W_l = pred_depth_latent.shape
    
    # ---- Step 1: Plücker map 下采样到 latent 分辨率并切 patch ----
    plucker_resized = F.adaptive_avg_pool3d(
        plucker_map, (F_2, H_l, W_l)
    )  # [B, 6, F, H_l, W_l]
    
    plucker_patches = plucker_resized.unfold(3, patch_size, patch_size) \
                                     .unfold(4, patch_size, patch_size)
    nH, nW = plucker_patches.shape[3], plucker_patches.shape[4]
    # [B, F, nH*nW, 6*p*p]
    plucker_patches = plucker_patches.permute(0, 2, 3, 4, 1, 5, 6) \
                                     .reshape(B, F_2, nH * nW, -1)
    
    # 对每个 patch 取 Plücker 坐标的均值（代表这个 patch 的平均光线）
    # reshape 成 [B, F, num_patches, 6, p*p] 再 mean
    num_patches = nH * nW
    pp = patch_size * patch_size
    plucker_patches = plucker_patches.reshape(B, F_2, num_patches, 6, pp).mean(dim=-1)
    # 现在是 [B, F, num_patches, 6]
    
    # ---- Step 2: Depth latent 切 patch ----
    pred_patches = pred_depth_latent.unfold(3, patch_size, patch_size) \
                                    .unfold(4, patch_size, patch_size)
    gt_patches = gt_depth_latent.unfold(3, patch_size, patch_size) \
                                .unfold(4, patch_size, patch_size)
    
    pred_patches = pred_patches.permute(0, 2, 3, 4, 1, 5, 6) \
                               .reshape(B, F_2, num_patches, -1)
    gt_patches = gt_patches.permute(0, 2, 3, 4, 1, 5, 6) \
                           .reshape(B, F_2, num_patches, -1)
    
    pred_norm = F.normalize(pred_patches, dim=-1)
    gt_norm = F.normalize(gt_patches, dim=-1)
    
    # ---- Step 3: 逐帧对计算 loss ----
    loss = 0.0
    count = 0
    
    for f in range(F_2 - 1):
        # Plücker 互矩权重 [B, N, N]
        geo_weight = plucker_ray_distance_weight(
            plucker_patches[:, f],
            plucker_patches[:, f + 1],
            temperature=0.1,
            top_k=8
        ).detach()
        
        # Depth latent 相似度
        pred_sim = torch.bmm(pred_norm[:, f], pred_norm[:, f+1].transpose(1, 2))
        gt_sim = torch.bmm(gt_norm[:, f], gt_norm[:, f+1].transpose(1, 2))
        
        # 几何加权: 只在光线相交的 patch 对上约束
        diff = (pred_sim - gt_sim.detach()) ** 2
        weighted_diff = diff * geo_weight
        
        # 归一化（除以权重总和，避免权重大的帧对主导）
        loss += weighted_diff.sum() / (geo_weight.sum() + 1e-8)
        count += 1
    
    return loss / max(count, 1)

def broadcast_batch(batch, rank, device):
    """
    广播 batch 确保两个 rank 处理相同数据，以 rank 0 为准
    """
    tensor_keys = [
            'gt_rgb', 'gt_depth',
            'incomplete_rgb', 'incomplete_depth',
            'mask_rgb', 'mask_depth',
            'plucker_rgb', 'plucker_depth',
        ]
    for key in tensor_keys:
        if key in batch:
            tensor = batch[key].to(device)
            dist.broadcast(tensor, src=0)
            batch[key] = tensor
    
    return batch


def load_text_encoder(args, device, weight_dtype):
    """加载 T5 文本编码器"""
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
    
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.base_model_dir, "google/umt5-xxl")
    )
    
    t5_path = args.t5_path or os.path.join(
        args.base_model_dir, "models_t5_umt5-xxl-enc-bf16.pth"
    )
    
    try:
        text_encoder = WanT5EncoderModel.from_pretrained(t5_path, torch_dtype=weight_dtype)
    except:
        text_encoder = WanT5EncoderModel(**t5_config).to(dtype=weight_dtype)
        if os.path.exists(t5_path):
            text_encoder.load_state_dict(
                torch.load(t5_path), 
                strict=False
            )
    text_encoder.to(device)
    text_encoder.eval().requires_grad_(False)
    return tokenizer, text_encoder


def load_vae(args, weight_dtype):
    """加载 VAE"""
    vae_path = args.vae_path or os.path.join(args.base_model_dir, "Wan2.1_VAE.pth")
    vae = AutoencoderKLWan.from_pretrained(vae_path)
    vae.eval().requires_grad_(False)
    return vae


def load_clip_encoder(args, weight_dtype):
    """加载 CLIP"""
    clip_path = args.clip_path or os.path.join(
        args.base_model_dir, 
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    )
    clip_encoder = CLIPModel.from_pretrained(clip_path)
    clip_encoder.to(dtype=weight_dtype).eval().requires_grad_(False)
    return clip_encoder
#不可用
def vae_decode_video_chunked(vae, latents, chunk_size=8):
    """
    分批 VAE decode，避免一次性 OOM
    latents: [B, C, F, H, W]
    return:  [B, 3, F, H', W']
    """
    sf = getattr(vae, "scaling_factor", None)
    if sf is None and hasattr(vae, "config"):
        sf = getattr(vae.config, "scaling_factor", 1.0)
    if sf is None:
        sf = 1.0
    
    x = latents / sf
    F = x.shape[2]
    decoded_chunks = []
    
    for i in range(0, F, chunk_size):
        chunk = x[:, :, i:i+chunk_size]
        with torch.no_grad():
            out = vae.decode(chunk)
            if hasattr(out, "sample"):
                decoded_chunks.append(out.sample)
            else:
                decoded_chunks.append(out)
        torch.cuda.empty_cache()  # 每个 chunk 解码完立刻释放
    
    return torch.cat(decoded_chunks, dim=2)  # 沿帧维度拼回来

def vae_decode_with_grad(vae, latents):
    from torch.utils.checkpoint import checkpoint
    
    sf = getattr(vae.config, "scaling_factor", 1.0) if hasattr(vae, "config") else 1.0
    x = latents / sf
    
    def _decode(z):
        out = vae.decode(z)
        return out.sample if hasattr(out, "sample") else out
    
    return checkpoint(_decode, x, use_reentrant=False)
    
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

def load_transformer(args, weight_dtype):
    """加载 Transformer"""
   # trans_sub_dir = os.path.join(args.base_model_dir, "transformer")
   # if os.path.exists(os.path.join(trans_sub_dir, "config.json")):
   #     load_dir = trans_sub_dir
   # else:
   #     load_dir = args.base_model_dir
    load_dir = "checkpoints/Wan-AI/Wan2.1-I2V-14B-720P"
    transformer = WanTransformer3DModel.from_pretrained(load_dir, torch_dtype=weight_dtype)

    # 2. 注入 LoRA
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q", "k", "v","ffn.0","ffn.2"],
        bias="none"
    )
    transformer = inject_adapter_in_model(lora_config, transformer)

    
    lora_state = torch.load("checkpoints/Wan-AI/wan_lora/checkpoints/pano_video_gen_720p.bin", map_location="cpu")
    
    # Debug: 打印 key 对比
    model_keys = [k for k in transformer.state_dict().keys() if 'lora' in k.lower()]
    ckpt_keys = list(lora_state.keys())
    print(f"[Rank  Model LoRA keys sample: {model_keys[:10]}")
    print(f"[Rank Ckpt LoRA keys sample: {ckpt_keys[:10]}")
    
    # 自动适配 key: 尝试常见的前缀映射
    
    adapted_state = adapt_lora_keys(lora_state, transformer.state_dict())
    missing, unexpected = transformer.load_state_dict(adapted_state, strict=False)

    lora_loaded = sum(1 for k in adapted_state if k in transformer.state_dict())

    
    # 验证：missing 里不应该有 lora 相关的 key
    missing_lora = [k for k in missing if 'lora' in k.lower()]
    if missing_lora:
        print(f"WARNING: {len(missing_lora)} LoRA keys still missing!")
        print(f"Examples: {missing_lora[:3]}")
    else:
        print(f" All LoRA keys loaded successfully.")
    print(f" LoRA loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    return transformer


def parse_args():
    parser = argparse.ArgumentParser(description="Dual Stream Wan Model Training + Plucker Pose")
    
    # 路径
    parser.add_argument("--base_model_dir", type=str, required=True)
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--t5_path", type=str, default=None)
    parser.add_argument("--clip_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output_dual_stream")
    
    # 训练超参
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--allow_tf32", action="store_true")
    
    # LoRA 配置
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    # 精度与采样
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--resolution", type=int, default=720)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--weighting_scheme", type=str, default="logit_normal")
    parser.add_argument("--uniform_sampling", action="store_true")
    parser.add_argument("--train_sampling_steps", type=int, default=1000)
    
    # 交互层
    parser.add_argument("--interaction_interval", type=int, default=4)
    
    # 日志与检查点
    parser.add_argument("--checkpointing_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def main():
    args = parse_args()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # 分布式设置
    rank, world_size, local_rank, device = setup_distributed()
    assert world_size == 2, "This script requires exactly 2 GPUs"
    
    logger = setup_logging(rank)
    logger.info(f"Starting training on device {device}")
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    weight_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    

    # 加载共享组件
    logger.info("Loading shared components...")
    
    # 1. 加载 T5 (仅 Rank 0 加载并常驻 GPU)
    if rank == 0:
        logger.info("[Rank 0] Loading T5 Text Encoder to GPU...")
        tokenizer, text_encoder = load_text_encoder(args, device, weight_dtype)
        text_encoder.to(device) 
    else:
        tokenizer, text_encoder = None, None
    
    dist.barrier() # 确保 Rank 0 加载完，Rank 1 再开始
    gc.collect()
    torch.cuda.empty_cache()

    # 2. 串行加载 VAE 和 CLIP
    if rank == 0:
        logger.info("[Rank 0] Loading VAE + CLIP to GPU...")
        vae = load_vae(args, weight_dtype).to(device)
        clip_encoder = load_clip_encoder(args, weight_dtype).to(device)
    
    dist.barrier() 

    # Rank 1 再来
    if rank == 1:
        logger.info("[Rank 1] Loading VAE + CLIP to GPU...")
        vae = load_vae(args, weight_dtype).to(device)
        clip_encoder = load_clip_encoder(args, weight_dtype).to(device)

    dist.barrier()
    gc.collect()

    # 3. 串行加载 Transformer
    if rank == 0:
        logger.info("[Rank 0] Loading 14B Transformer to GPU...")
        my_transformer = load_transformer(args, weight_dtype).to(device)
    
    dist.barrier()

    if rank == 1:
        logger.info("[Rank 1] Loading 14B Transformer to GPU...")
        my_transformer = load_transformer(args, weight_dtype).to(device)

    dist.barrier()
    gc.collect()
    torch.cuda.empty_cache()
    

    # 加载模型
    logger.info("Loading transformers...")
    
    #测试模型
    wan_config = {
        "_class_name": "WanModel",
            "_diffusers_version": "0.30.0",
            "model_type": "i2v",        
            "dim": 1024,                # 缩减
            "eps": 1e-06,
            "ffn_dim": 4096,            # 缩减
            "freq_dim": 256,
            "in_dim": 36,               # 16+16+4
            "num_heads": 16,            # 缩减
            "num_layers": 8,            # 缩减
            "out_dim": 16,
            "text_len": 512
        }

    # 测试模型 from wan_config
   # transformer_rgb = WanTransformer3DModel(**wan_config).to(dtype=weight_dtype)
   # transformer_depth = WanTransformer3DModel(**wan_config).to(dtype=weight_dtype)
 #   my_transformer = WanTransformer3DModel(**wan_config).to(dtype=weight_dtype)

   # my_transformer = load_transformer(args, weight_dtype)
    
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q", "k", "v","ffn.0","ffn.2"],
        bias="none"
    )
    my_transformer = inject_adapter_in_model(lora_config, my_transformer)

    logger.info("Successfully initialized model.")

    if rank == 0:
        dual_model = DualStreamWanModel(
            rgb_model=my_transformer,
            depth_model=None,
            interaction_interval=args.interaction_interval
        )
    else:
        dual_model = DualStreamWanModel(
            rgb_model=None,
            depth_model=my_transformer,
            interaction_interval=args.interaction_interval
        )
    torch.cuda.empty_cache()
    
    dual_model.setup_for_rank(rank, device)
    dual_model.to(dtype=weight_dtype)
    
    fix_wan_precision_issues(dual_model)
    

    # 优化器和数据
    trainable_params = dual_model.get_trainable_params()
    print(f"[Rank {rank}] Trainable params count: {len(trainable_params)}", flush=True)
    print(f"[Rank {rank}] Trainable param total: {sum(p.numel() for p in trainable_params)}", flush=True)

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    
    # 数据集
   # dataset = DummyDualStreamDataset(
    #    length=10,
    #    num_frames=args.num_frames,
    #    height=128,
    #    width=256  
   # )
   
    dataset = DualStreamRealDataset(
        data_root="/root/autodl-tmp/Matrix-3D/data/dataset_train_round1",
        height=128,
        width=256,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        collate_fn=collate_fn_dual,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    
    # 噪声调度器
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=args.train_sampling_steps,
        shift=5.0
    )
    
    idx_sampling = DiscreteSampling(
        args.train_sampling_steps,
        uniform_sampling=args.uniform_sampling
    )
    
    # 混合精度
    scaler = GradScaler() if args.mixed_precision == "fp16" else None
    amp_context = torch.amp.autocast('cuda', dtype=weight_dtype) if args.mixed_precision != "no" else nullcontext()
    preprocess_fn = create_preprocess_fn(device, weight_dtype)
    

    # 训练
    logger.info("Starting training...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    global_step = 0

    if rank == 0:
        print("DEBUG: Before first comm")
    dist.barrier()
    if rank == 0:
        print("DEBUG: After barrier")
        
    print(f"[Rank {rank}] 正在执行数据读取压力测试...", flush=True)
    tmp_batch = next(iter(dataloader))
    print(f"[Rank {rank}] 压力测试通过，Batch 形状: "
          f"GT_RGB={tmp_batch['gt_rgb'].shape}, "
          f"Incomplete_RGB={tmp_batch['incomplete_rgb'].shape}, "
          f"Plucker={tmp_batch['plucker_rgb'].shape}", flush=True)

    torch.cuda.synchronize()
    print(f"[Rank {rank}] CUDA 同步完成", flush=True)
    
    dist.barrier()
    print(f"[Rank {rank}] 进入循环前的最后 Barrier 通过", flush=True)
    print(f"[Rank {rank}] 准备进入 Epoch 循环...", flush=True)
    
    for epoch in range(args.max_train_steps):
        print(f"[Rank {rank}] 进入 Epoch {epoch}, 准备获取 DataLoader 迭代器...", flush=True)
    
        dist.barrier()
        print(f"[Rank {rank}] Barrier 1 成功过关", flush=True)
    
        loader_iter = iter(dataloader)
        print(f"[Rank {rank}] 迭代器创建成功", flush=True)
    
        try:
            batch = next(loader_iter)
            print(f"[Rank {rank}] 成功拿到第一个 Batch", flush=True)
        except StopIteration:
            print(f"[Rank {rank}] DataLoader 为空！", flush=True)
            break
        i=0
        for step, batch in enumerate(dataloader):
            i=i+1
            if i <= 152:
                continue
            # 广播确
            batch = broadcast_batch(batch, rank, device)
            curr_bsz = batch['gt_rgb'].shape[0]
            optimizer.zero_grad()
            
            # 文本编码，只在 rank 0
            if rank == 0:
                with torch.no_grad():
                    prompt_ids = tokenizer(
                        batch['text'], padding="max_length",
                        max_length=512, truncation=True,
                        return_tensors="pt"
                    ).to(device)
                    context = text_encoder(
                        prompt_ids.input_ids.to(device)
                    )[0]
                    batch['context'] = context
            else:
                # 占位
                batch['context'] = torch.zeros(
                    curr_bsz, 512, 4096, device=device, dtype=weight_dtype
                )
            
            # 广播 context
            dist.broadcast(batch['context'], src=0)
            context = batch['context']
            
            # CLIP 和 VAE 编码
            clip_encoder.to(device, dtype=weight_dtype)
            vae.to(device, dtype=weight_dtype)
 
            with torch.no_grad():
                # CLIP 特征（用第一帧）
                rgb_first_frame = batch['gt_rgb'][:, :, 0:1, :, :]
                depth_first_frame = batch['gt_depth'][:, :, 0:1, :, :]
                
                clip_fea_rgb = clip_encoder([rgb_first_frame[0]]).to(weight_dtype)
                clip_fea_depth = clip_encoder([depth_first_frame[0]]).to(weight_dtype)
                
                # GT latent
                latents_gt_rgb = vae.encode(
                    batch['gt_rgb'].to(weight_dtype)
                ).latent_dist.sample()
                
                latents_gt_depth = vae.encode(
                    batch['gt_depth'].to(weight_dtype)
                ).latent_dist.sample()
                
                # masked latent
                latents_incomplete_rgb = vae.encode(
                    batch['incomplete_rgb'].to(weight_dtype)
                ).latent_dist.sample()
                
                latents_incomplete_depth = vae.encode(
                    batch['incomplete_depth'].to(weight_dtype)
                ).latent_dist.sample()

            clip_encoder.to("cpu")
            vae.to("cpu")
            clear_memory()
            

            # 扩散
            bsz = latents_gt_rgb.shape[0]
            
            indices = idx_sampling(bsz, device="cpu").long()
            timesteps = noise_scheduler.timesteps[indices].to(device)
            sigmas = noise_scheduler.sigmas[indices].to(device, dtype=weight_dtype)
            sigmas = sigmas.view(bsz, 1, 1, 1, 1)
            
            timesteps = sync_tensor(timesteps, rank)
            sigmas = sync_tensor(sigmas, rank)
            
            noise_rgb = torch.randn_like(latents_gt_rgb)
            noise_depth = torch.randn_like(latents_gt_depth)
            
            noise_rgb = sync_tensor(noise_rgb, rank)
            noise_depth = sync_tensor(noise_depth, rank)
            
            noisy_rgb = (1.0 - sigmas) * latents_gt_rgb + sigmas * noise_rgb
            noisy_depth = (1.0 - sigmas) * latents_gt_depth + sigmas * noise_depth
            
            # 条件：残缺图 latent + mask
            mask_rgb = F.interpolate(
                batch['mask_rgb'].to(device, dtype=weight_dtype),
                size=latents_gt_rgb.shape[2:]
            )
            mask_depth = F.interpolate(
                batch['mask_depth'].to(device, dtype=weight_dtype),
                size=latents_gt_depth.shape[2:]
            )
            
            y_rgb = torch.cat([mask_rgb, latents_incomplete_rgb * mask_rgb], dim=1)
            y_depth = torch.cat([mask_depth, latents_incomplete_depth * mask_depth], dim=1)
            
            f, h, w = noisy_rgb.shape[2], noisy_rgb.shape[3] // 2, noisy_rgb.shape[4] // 2
            seq_len = f * h * w
            

            plucker_rgb = batch['plucker_rgb'].to(device, dtype=weight_dtype)
            plucker_depth = batch['plucker_depth'].to(device, dtype=weight_dtype)
            

            # 前向传播
            with amp_context:
                pred = dual_model(
                    noisy_rgb, y_rgb, clip_fea_rgb,
                    noisy_depth, y_depth, clip_fea_depth,
                    timesteps, context, seq_len,
                    preprocess_fn,
                    plucker_rgb, plucker_depth    
                )
                
                # v-prediction
                if rank == 0:
                    target = noise_rgb - latents_gt_rgb
                else:
                    target = noise_depth - latents_gt_depth
                    
                weighting = compute_loss_weighting_for_sd3(
                    args.weighting_scheme,
                    sigmas.flatten()
                )

                mse_loss = custom_mse_loss(pred, target, weighting=weighting)
                
                # RGB流 Latent structure consistency loss 
                if rank == 0:
                    # 从v-prediction 还原 predicted clean latent
                    pred_latent = noisy_rgb - sigmas * pred
                    
                    # sigma-dependent weight
                    struct_weight = sigma_dependent_weight(sigmas, low=0.2, high=0.8)
                    
                    struct_loss = latent_structure_consistency_loss(
                        pred_latent, latents_gt_rgb, patch_size=2
                    )
                    # 20260211 删除 latent_global_consistency_loss
                    # global_loss = latent_global_consistency_loss(pred_latent)
                    
                    # loss = (mse_loss 
                    #         + 0.1 * struct_weight * struct_loss )
                            #+ 0.01 * struct_weight * global_loss)
                            
                            
                    # 20260211 添加 ssim 和 ray 损失
                    vae.to(device, dtype=weight_dtype)
                    vae.eval().requires_grad_(False)

                    decoded_pred = vae_decode_with_grad(vae, pred_latent)          # [B,3,F,H,W] (通常是 [-1,1] 或 [0,1])
                    decoded_gt   = batch['gt_rgb'].to(device, dtype=weight_dtype)

                    # 视VAE 输出范围决定是否映射；常见是 [-1,1] -> [0,1]
                    decoded_pred_01 = (decoded_pred * 0.5 + 0.5).clamp(0,1)
                    decoded_gt_01   = (decoded_gt   * 0.5 + 0.5).clamp(0,1)

                    loss_ssim = ssim_loss_video(decoded_pred_01, decoded_gt_01, window_size=11)

                    loss_ray  = ray_guided_photo_loss(
                        decoded_pred_01,
                        batch['plucker_rgb'].to(device, dtype=weight_dtype),
                        patch_size_img=8,
                        patch_size_plucker=8,
                        temperature=0.1,
                        top_k=8
                    )
                    loss = (mse_loss 
                            + 0.1 * struct_weight * struct_loss ) + 0.05 * loss_ssim + 0.05 * loss_ray
                else:
                    # Depth流 pose-guided consistency loss
                    pred_depth_latent = noisy_depth - sigmas * pred
                    print(pred_depth_latent.shape)
                    struct_weight = sigma_dependent_weight(sigmas, low=0.2, high=0.8)
                    
                    depth_consist_loss = pose_guided_depth_consistency_loss(
                        pred_depth_latent, latents_gt_depth,
                        batch['plucker_depth'].to(device, dtype=weight_dtype),
                        patch_size=2
                    )
                    
                    loss = (mse_loss
                            + 0.1 * struct_weight * depth_consist_loss)
                    
                    #20260211 添加 ssim 和 ray 损失
                    vae.to(device, dtype=weight_dtype)
                    vae.eval().requires_grad_(False)
                    
              
                    decoded_pred = vae_decode_with_grad(vae, pred_depth_latent)
                    decoded_gt   = batch['gt_depth'].to(device, dtype=weight_dtype)

                    decoded_pred_01 = (decoded_pred * 0.5 + 0.5).clamp(0,1)
                    decoded_gt_01   = (decoded_gt   * 0.5 + 0.5).clamp(0,1)

                    loss_ssim = ssim_loss_video(decoded_pred_01, decoded_gt_01, window_size=11)

                    loss_ray  = ray_guided_photo_loss(
                        decoded_pred_01,
                        batch['plucker_depth'].to(device, dtype=weight_dtype),
                        patch_size_img=8,
                        patch_size_plucker=8,
                        temperature=0.1,
                        top_k=8
                    )

                    loss = loss + 0.05 * loss_ssim + 0.05 * loss_ray
                

            # 反向传播
            if scaler:
                scaler.scale(loss).backward()
                sync_gradients_for_shared_params(dual_model)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(dual_model.get_trainable_params(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                sync_gradients_for_shared_params(dual_model)
                torch.nn.utils.clip_grad_norm_(dual_model.get_trainable_params(), 1.0)
                optimizer.step()
                
            lr_scheduler.step()
            # 日志
            global_step += 1
            
            if global_step % args.logging_steps == 0:
                log_msg = f"[Rank {rank}] Step {global_step} | Loss: {loss.item():.4f}"
                if rank == 0:
                    log_msg += f" | MSE: {mse_loss.item():.4f} | Struct: {struct_loss.item():.4f} | loss_ssim: {loss_ssim.item():.4f},ray: {loss_ray.item():.4f}"
                else:
                    log_msg += f" | MSE: {mse_loss.item():.4f} | DepthConsist: {depth_consist_loss.item():.4f} | loss_ssim: {loss_ssim.item():.4f},ray: {loss_ray.item():.4f}"
                print(log_msg, flush=True)
                
            if global_step % args.checkpointing_steps == 0:
                save_checkpoint(dual_model, optimizer, global_step, args.output_dir, rank)
            
    # 保存
    save_checkpoint(dual_model, optimizer, global_step, args.output_dir, rank)
    logger.info("Training completed!")
    
    cleanup_distributed()


def save_checkpoint(model, optimizer, step, output_dir, rank):
    """保存模型"""
    if rank == 0:
        # 保存 RGB 流的 LoRA 权重
        lora_state = get_peft_model_state_dict(model.local_model)
        torch.save(lora_state, os.path.join(output_dir, f"lora_rgb_step{step}.pt"))
        
        # 交互层
        torch.save(
            model.cross_attn_layers.state_dict(),
            os.path.join(output_dir, f"cross_attn_step{step}.pt")
        )
        torch.save(
            model.pose_cross_attn_layers.state_dict(),
            os.path.join(output_dir, f"pose_cross_attn_step{step}.pt")
        )
        
        # PluckerNet
        torch.save(
            model.plucker_net.state_dict(),
            os.path.join(output_dir, f"plucker_net_step{step}.pt")
        )
    else:
        # 保存 Depth 流的 LoRA 权重
        lora_state = get_peft_model_state_dict(model.local_model)
        torch.save(lora_state, os.path.join(output_dir, f"lora_depth_step{step}.pt"))
        
    dist.barrier()


if __name__ == "__main__":
    main()
