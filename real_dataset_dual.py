"""
真实双流数据集 - HM3D 多视角全景图补全

数据结构:
  data/
    {area}/
      {room}/
        input/
          panorama_0000.png                         # pose 0 完整全景图
          panorama_0000_depth.npy                   # pose 0 完整深度 (npy)
          pano0000_to_pano0001_rgb.png              # pose 0→1 投影的残缺 RGB
          pano0000_to_pano0001_depth_range.npy      # pose 0→1 投影的残缺 Depth (npy)
          pano0000_to_pano0001_mask.png             # pose 0→1 投影 mask (可选)
          pose_0000.json ~ pose_XXXX.json           # 相机位姿
          description.txt                           # 文本描述
        output/
          panorama_0000.png ~ panorama_XXXX.png             # GT 完整全景图
          panorama_0000_depth.npy ~ panorama_XXXX_depth.npy # GT 完整深度 (npy)

帧数处理:
  - Wan 要求帧数 % 4 == 1（即 5, 9, 13, ...）
  - < 5 帧的房间丢弃
  - > 5 帧时，保留首尾帧，中间随机采样，降到最近的 4n+1

Mask 生成:
  - 第 0 帧：全 0（完整，Wan 反转后不需要生成）
  - 其他帧：depth_range.npy 中 >0 的区域 → valid=1, 空洞=0
  - Wan 训练需要反转：1=需要生成, 0=已知 → 最终 mask = 1 - valid_mask

Plücker ray map:
  - 从 pose JSON (position + rotation_quaternion) 生成
  - ERP 全景图：水平 360°, 垂直 180°
  - 输出 [6, F, H, W]
"""
import glob
import json
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ============================================================
# Plücker Ray Map 工具函数
# ============================================================

def quaternion_to_rotation_matrix(q):
    """
    四元数 → 旋转矩阵

    Args:
        q: [w, x, y, z] world→camera 旋转四元数

    Returns:
        R: [3, 3] 旋转矩阵 (world→camera)
    """
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ])
    return R


def compute_erp_plucker_ray_map(position, rotation_quaternion, height, width):
    """
    从相机位姿生成 ERP 全景图的 Plücker ray map

    Args:
        position: [3] 相机位置 (相对第0帧)
        rotation_quaternion: [4] [w,x,y,z] world→camera 四元数
        height: 图像高度
        width: 图像宽度

    Returns:
        plucker: [6, H, W]  前3通道=direction, 后3通道=moment
    """
    v_coords, u_coords = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing='ij'
    )

    theta = (0.5 - (v_coords + 0.5) / height) * np.pi    # [-π/2, π/2]
    phi   = ((u_coords + 0.5) / width - 0.5) * 2 * np.pi # [-π, π]

    dir_cam_x = np.sin(phi) * np.cos(theta)
    dir_cam_y = -np.sin(theta)
    dir_cam_z = np.cos(phi) * np.cos(theta)
    dir_cam   = np.stack([dir_cam_x, dir_cam_y, dir_cam_z], axis=-1)  # [H,W,3]

    norm     = np.linalg.norm(dir_cam, axis=-1, keepdims=True) + 1e-8
    dir_cam  = dir_cam / norm

    R_wc     = quaternion_to_rotation_matrix(rotation_quaternion)
    R_cw     = R_wc.T
    dir_world = np.einsum('ij,hwj->hwi', R_cw, dir_cam)  # [H,W,3]

    norm      = np.linalg.norm(dir_world, axis=-1, keepdims=True) + 1e-8
    dir_world = dir_world / norm

    cam_origin       = np.array(position, dtype=np.float64)
    origin_expanded  = np.broadcast_to(cam_origin.reshape(1, 1, 3), dir_world.shape)
    moment           = np.cross(origin_expanded, dir_world)  # [H,W,3]

    plucker = np.concatenate([dir_world, moment], axis=-1).transpose(2, 0, 1)  # [6,H,W]
    return plucker.astype(np.float32)


def compute_plucker_for_frames(poses, height, width):
    """
    为所有帧计算 Plücker ray map

    Returns:
        plucker: [6, F, H, W]
    """
    frames = [
        compute_erp_plucker_ray_map(p['position'], p['rotation_quaternion'], height, width)
        for p in poses
    ]
    return np.stack(frames, axis=1)  # [6, F, H, W]


# ============================================================
# 深度 npy 工具函数
# ============================================================

def load_depth_npy(path, height, width):
    """
    加载深度 npy，resize，归一化到 [-1, 1]。
    NaN / Inf → 0（视为无效）。

    Returns:
        depth_3ch : [3, H, W] float32，三通道复制（与 RGB 流对齐）
        valid_mask: [H, W] float32，1=有效深度, 0=空洞
    """
    d = np.load(path).astype(np.float32)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    if d.shape[0] != height or d.shape[1] != width:
        d = cv2.resize(d, (width, height), interpolation=cv2.INTER_LINEAR)

    valid_mask = (d > 0).astype(np.float32)  # [H, W]

    d_max = d.max()
    if d_max > 0:
        d = d / d_max * 2.0 - 1.0   # 有效区域映射到 [-1, 1]
    else:
        d = np.full_like(d, -1.0)    # 全空洞帧

    depth_3ch = np.stack([d, d, d], axis=0).astype(np.float32)  # [3, H, W]
    return depth_3ch, valid_mask


# ============================================================
# 数据集扫描 & 帧采样
# ============================================================

def scan_data_root(data_root):
    """递归扫描，找到所有同时含 input/ 和 output/ 的目录"""
    rooms = []
    for dirpath, dirnames, _ in os.walk(data_root):
        if 'input' in dirnames and 'output' in dirnames:
            rooms.append(dirpath)
    return sorted(rooms)


def count_frames(input_dir):
    return len(glob.glob(os.path.join(input_dir, 'pose_*.json')))


def get_target_frame_count(total_frames):
    """返回最近的 4n+1 目标帧数，< 5 帧返回 -1"""
    if total_frames < 5:
        return -1
    n = (total_frames - 1) // 4
    target = 4 * n + 1
    return target if target >= 5 else -1


def select_frames(total_frames, target_frames):
    """保留首尾，中间随机采样到 target_frames"""
    if total_frames == target_frames:
        return list(range(total_frames))
    middle_pool = list(range(1, total_frames - 1))
    num_middle  = target_frames - 2
    selected    = sorted(random.sample(middle_pool, num_middle))
    return [0] + selected + [total_frames - 1]


# ============================================================
# 主数据集类
# ============================================================

class DualStreamRealDataset(Dataset):
    """
    HM3D 双流真实数据集（深度全走 npy）

    返回:
        gt_rgb           [3, F, H, W]  GT 完整全景图，归一化到 [-1,1]
        gt_depth         [3, F, H, W]  GT 完整深度，归一化到 [-1,1]
        incomplete_rgb   [3, F, H, W]  残缺全景图
        incomplete_depth [3, F, H, W]  残缺深度
        mask_rgb         [1, F, H, W]  Wan mask（1=需生成, 0=已知）
        mask_depth       [1, F, H, W]  同上
        plucker_rgb      [6, F, H, W]  Plücker ray map
        plucker_depth    [6, F, H, W]  同 plucker_rgb
        text             str           文本描述
    """

    def __init__(self, data_root, height=128, width=256):
        self.height = height
        self.width  = width

        all_rooms = scan_data_root(data_root)

        self.samples = []
        for room_path in all_rooms:
            input_dir = os.path.join(room_path, 'input')
            n_frames  = count_frames(input_dir)
            target    = get_target_frame_count(n_frames)
            if target == -1:
                continue
            self.samples.append({
                'room_path':     room_path,
                'total_frames':  n_frames,
                'target_frames': target,
            })

        print(f"[Dataset] Found {len(self.samples)} valid rooms "
              f"(discarded {len(all_rooms) - len(self.samples)} rooms with < 5 frames)")

    def __len__(self):
        return len(self.samples)

    def _load_rgb(self, path):
        """加载 PNG，resize，归一化到 [-1,1]，返回 [3,H,W] float32"""
        img = Image.open(path).convert('RGB')
        img = img.resize((self.width, self.height), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0  # [-1,1]
        return arr.transpose(2, 0, 1)  # [3,H,W]

    def __getitem__(self, idx):
        sample       = self.samples[idx]
        room_path    = sample['room_path']
        total_frames = sample['total_frames']
        target_frames= sample['target_frames']

        input_dir  = os.path.join(room_path, 'input')
        output_dir = os.path.join(room_path, 'output')

        # ---- 帧采样 ----
        frame_indices = select_frames(total_frames, target_frames)

        # ---- 加载 pose ----
        poses = []
        for fi in frame_indices:
            with open(os.path.join(input_dir, f'pose_{fi:04d}.json')) as f:
                poses.append(json.load(f))

        # ---- 加载文本描述 ----
        desc_path = os.path.join(input_dir, 'description.txt')
        text = open(desc_path).read().strip() if os.path.exists(desc_path) \
               else "A panoramic view of an indoor scene"

        # ---- 逐帧加载 ----
        gt_rgb_frames        = []
        gt_depth_frames      = []
        incomplete_rgb_frames  = []
        incomplete_depth_frames= []
        mask_frames          = []

        for fi in frame_indices:
            # ── GT RGB（output PNG）──
            gt_rgb_frames.append(
                self._load_rgb(os.path.join(output_dir, f'panorama_{fi:04d}.png'))
            )

            # ── GT Depth（output npy）──
            gt_depth_3ch, _ = load_depth_npy(
                os.path.join(output_dir, f'panorama_{fi:04d}_depth.npy'),
                self.height, self.width
            )
            gt_depth_frames.append(gt_depth_3ch)

            if fi == 0:
                # 第0帧：完整图，残缺图 = 完整图本身
                incomplete_rgb_frames.append(
                    self._load_rgb(os.path.join(input_dir, 'panorama_0000.png'))
                )
                inc_depth_3ch, _ = load_depth_npy(
                    os.path.join(input_dir, 'panorama_0000_depth.npy'),
                    self.height, self.width
                )
                incomplete_depth_frames.append(inc_depth_3ch)
                # mask 全0（全部已知，不需要生成）
                mask_frames.append(np.zeros((self.height, self.width), dtype=np.float32))

            else:
                # 其他帧：投影残缺图
                incomplete_rgb_frames.append(
                    self._load_rgb(os.path.join(input_dir, f'pano0000_to_pano{fi:04d}_rgb.png'))
                )
                inc_depth_3ch, valid_mask = load_depth_npy(
                    os.path.join(input_dir, f'pano0000_to_pano{fi:04d}_depth_range.npy'),
                    self.height, self.width
                )
                incomplete_depth_frames.append(inc_depth_3ch)
                # Wan mask：1=空洞（需要生成），0=已知
                mask_frames.append(1.0 - valid_mask)

        # ---- Stack → [C, F, H, W] ----
        def stack_chw(frames):
            # list of [C,H,W] → [C,F,H,W]
            return np.stack(frames, axis=1)  # [C,F,H,W] 直接 stack 在 axis=1

        gt_rgb        = np.stack(gt_rgb_frames, axis=0).transpose(1, 0, 2, 3)        # [3,F,H,W]
        gt_depth      = np.stack(gt_depth_frames, axis=0).transpose(1, 0, 2, 3)
        incomplete_rgb   = np.stack(incomplete_rgb_frames, axis=0).transpose(1, 0, 2, 3)
        incomplete_depth = np.stack(incomplete_depth_frames, axis=0).transpose(1, 0, 2, 3)

        mask = np.stack(mask_frames, axis=0)[np.newaxis, ...]  # [1,F,H,W]

        # ---- Plücker ray map ----
        plucker = compute_plucker_for_frames(poses, self.height, self.width)  # [6,F,H,W]

        print(f"[{idx}] gt_rgb:{gt_rgb.shape} gt_depth:{gt_depth.shape} "
              f"inc_rgb:{incomplete_rgb.shape} inc_depth:{incomplete_depth.shape} "
              f"mask:{mask.shape} plucker:{plucker.shape}")

        return {
            'gt_rgb':           torch.from_numpy(gt_rgb).float(),
            'gt_depth':         torch.from_numpy(gt_depth).float(),
            'incomplete_rgb':   torch.from_numpy(incomplete_rgb).float(),
            'incomplete_depth': torch.from_numpy(incomplete_depth).float(),
            'mask_rgb':         torch.from_numpy(mask).float(),
            'mask_depth':       torch.from_numpy(mask.copy()).float(),
            'plucker_rgb':      torch.from_numpy(plucker).float(),
            'plucker_depth':    torch.from_numpy(plucker.copy()).float(),
            'text':             text,
        }


# ============================================================
# Collate（支持不同帧数 padding）
# ============================================================

def collate_fn_dual(batch):
    max_frames = max(item['gt_rgb'].shape[1] for item in batch)

    def pad(tensor, target, dim=1):
        cur = tensor.shape[dim]
        if cur == target:
            return tensor
        pad_shape      = list(tensor.shape)
        pad_shape[dim] = target - cur
        return torch.cat([tensor, torch.zeros(pad_shape, dtype=tensor.dtype)], dim=dim)

    keys_tensor = ['gt_rgb', 'gt_depth', 'incomplete_rgb', 'incomplete_depth',
                   'mask_rgb', 'mask_depth', 'plucker_rgb', 'plucker_depth']
    result = {k: [] for k in keys_tensor}
    result['text']       = []
    result['num_frames'] = []

    for item in batch:
        n = item['gt_rgb'].shape[1]
        for k in keys_tensor:
            result[k].append(pad(item[k], max_frames))
        result['text'].append(item['text'])
        result['num_frames'].append(n)

    for k in keys_tensor:
        result[k] = torch.stack(result[k])

    return result


# ============================================================
# 测试入口
# ============================================================
if __name__ == '__main__':

    import glob
    import os
    
    def count_frames(input_dir):
        return len(glob.glob(os.path.join(input_dir, 'pose_*.json')))
    
    def scan_data_root(data_root):
        rooms = []
        for dirpath, dirnames, _ in os.walk(data_root):
            if 'input' in dirnames and 'output' in dirnames:
                rooms.append(dirpath)
        return sorted(rooms)
    
    data_root = "/root/autodl-tmp/Matrix-3D/data/dataset_train_round1"
    
    all_rooms = scan_data_root(data_root)
    print(f"Total rooms found: {len(all_rooms)}\n")
    
    broken = []
    for room_path in all_rooms:
        input_dir = os.path.join(room_path, 'input')
        n_frames = count_frames(input_dir)
        if n_frames < 5:
            broken.append((room_path, n_frames))
    
    print(f"Broken rooms (< 5 frames): {len(broken)}\n")
    
    # 按帧数分组统计
    from collections import Counter
    counter = Counter(n for _, n in broken)
    print("Frame count distribution:")
    for k in sorted(counter):
        print(f"  {k} frames: {counter[k]} rooms")
    
    print("\nBroken room details:")
    for room_path, n in broken:
        print(f"  [{n} frames] {room_path}")
        # 顺便看看 pose 文件实际叫什么
        poses = glob.glob(os.path.join(room_path, 'input', '*.json'))
        if poses:
            print(f"    json files: {[os.path.basename(p) for p in sorted(poses)[:3]]}")
        else:
            print(f"    NO json files found!")