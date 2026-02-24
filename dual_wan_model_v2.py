"""
双流 Wan 模型 + Plücker Pose 注入

设计思路：
1. 每个 GPU 持有完整的一个流模型（RGB 或 Depth）
2. 通过点对点通信交换中间特征
3. 交互层使用支持梯度反传的通信原语
4. PluckerNet 在两个 rank 上各复制一份，独立计算 pose 特征
5. Pose cross-attention 和跨流 cross-attention 交替注入
   - 第 1 个交互点 (e.g. layer 4)  → pose cross-attn
   - 第 2 个交互点 (e.g. layer 8)  → 跨流 cross-attn
   - 第 3 个交互点 (e.g. layer 12) → pose cross-attn
   - ...
"""
import torch
import torch.nn as nn
import torch.distributed as dist

from comm_utils import exchange_features, sync_tensor


def sinusoidal_embedding_1d(dim, position):
    """正弦位置编码"""
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    
    sinusoid = torch.outer(
        position, 
        torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


class PluckerNet(nn.Module):
    """
    小型 3D-CNN 将 Plücker ray map 编码为与 Wan 隐状态对齐的 pose 特征
    
    输入: [B, 6, F, H, W]  (6 = 3D direction + 3D moment)
    输出: [B, S, D]         (S = 目标序列长度, D = 模型维度)
    
    前向只在每次 forward 开头调用一次，各 pose cross-attn 注入点复用结果。
    """
    def __init__(self, in_channels=6, out_dim=3072, mid_channels=128):
        super().__init__()
        self.out_dim = out_dim
        
        # 3D CNN: [B, 6, F, H, W] -> [B, mid*2, F', H', W']
        self.conv_layers = nn.Sequential(
            # Block 1: 6 -> 64
            nn.Conv3d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            
            # Block 2: 64 -> 128, 空间下采样
            nn.Conv3d(64, mid_channels, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.SiLU(inplace=True),
            
            # Block 3: 128 -> 256
            nn.Conv3d(mid_channels, mid_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, mid_channels * 2),
            nn.SiLU(inplace=True),
        )
        
        # 投影到模型维度
        self.proj = nn.Linear(mid_channels * 2, out_dim)
        
        # 零初始化投影层，确保训练初期不破坏原有表示
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, plucker_map, target_seq_len):
        """
        Args:
            plucker_map: [B, 6, F, H, W] Plücker ray map
            target_seq_len: int, 需与 Wan 隐状态的序列长度 S 对齐
            
        Returns:
            pose_features: [B, S, D]
        """
        # 3D CNN 特征提取
        feat = self.conv_layers(plucker_map)       # [B, 256, F', H', W']
        
        # Flatten 时空维度: [B, 256, N]  where N = F'*H'*W'
        feat = feat.flatten(2)                      # [B, C, N]
        
        # Adaptive pooling 对齐到目标序列长度
        feat = nn.functional.adaptive_avg_pool1d(feat, target_seq_len)  # [B, C, S]
        
        # 转置 + 线性投影: [B, S, C] -> [B, S, D]
        feat = feat.transpose(1, 2)
        feat = self.proj(feat)
        
        return feat


class CrossStreamAttn(nn.Module):
    """
    跨流交叉注意力
    
    使用零初始化确保训练初期不破坏原有流的表示
    """
    def __init__(self, dim, num_heads=16):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim, eps=1e-6)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # 零初始化输出投影，保证初始状态下不影响原有流
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        
        # 可学习的缩放因子
        self.scale = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, other_x):
        """
        Args:
            x: 本流特征 [B, S, D]
            other_x: 对方流特征 [B, S, D]
        """
        residual = x
        q = self.norm_q(x)
        kv = self.norm_kv(other_x)
        out, _ = self.attn(q, kv, kv)
        return residual + self.scale * out


class PoseCrossAttn(nn.Module):
    """
    Pose 交叉注意力
    
    将 PluckerNet 编码的 pose 特征通过 cross-attention 注入到主流中。
    结构与 CrossStreamAttn 相同，零初始化 + 可学习缩放。
    """
    def __init__(self, dim, num_heads=16):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim, eps=1e-6)
        self.norm_kv = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        
        self.scale = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, pose_features):
        """
        Args:
            x: 主流特征 [B, S, D]
            pose_features: PluckerNet 输出 [B, S_pose, D]
        """
        residual = x
        q = self.norm_q(x)
        kv = self.norm_kv(pose_features)
        out, _ = self.attn(q, kv, kv)
        return residual + self.scale * out


class DualStreamWanModel(nn.Module):
    """
    双流 Wan 视频扩散模型 + Plücker Pose 注入
    
    交互策略（交替模式）：
    - 交互点由 interaction_interval 决定
    - 奇数次交互点 (第1, 3, 5, ...) → pose cross-attention
    - 偶数次交互点 (第2, 4, 6, ...) → 跨流 cross-attention
    - PluckerNet 在两个 rank 上各复制一份，独立计算
    """
    def __init__(self, rgb_model, depth_model, interaction_interval=4):
        super().__init__()
        self.rgb_model = rgb_model
        self.depth_model = depth_model
        
        # 从非 None 的模型获取配置
        ref_model = rgb_model if rgb_model is not None else depth_model
        self.num_layers = ref_model.num_layers
        self.interaction_interval = interaction_interval
        self.dim = ref_model.dim
        
        # 计算交互点总数及分配
        num_interaction_points = self.num_layers // interaction_interval
        num_pose_points = (num_interaction_points + 1) // 2   # 第 1,3,5,...
        num_cross_points = num_interaction_points // 2         # 第 2,4,6,...
        
        # Pose cross-attention 模块列表
        self.pose_cross_attn_layers = nn.ModuleList([
            PoseCrossAttn(self.dim) for _ in range(num_pose_points)
        ])
        
        # 跨流 cross-attention 模块列表
        self.cross_attn_layers = nn.ModuleList([
            CrossStreamAttn(self.dim) for _ in range(num_cross_points)
        ])
        
        # PluckerNet - 每个 rank 各持有一份（复制方案）
        self.plucker_net = PluckerNet(
            in_channels=6,
            out_dim=self.dim,
            mid_channels=128
        )
        
        self._device_setup_done = False
        
        print(f"[Model] layers={self.num_layers}, interval={interaction_interval}, "
              f"pose_attn={num_pose_points}, cross_stream={num_cross_points}")
        
    def setup_for_rank(self, rank, device):
        """根据 rank 设置模型"""
        self.rank = rank
        self.device = device
        
        if rank == 0:
            self.local_model = self.rgb_model.to(device)
            self.stream_name = "rgb"
            self.depth_model = None
        else:
            self.local_model = self.depth_model.to(device)
            self.stream_name = "depth"
            self.rgb_model = None
        
        print(f"rank {rank} is ready")
            
        # 所有新增模块移到对应设备
        self.cross_attn_layers.to(device)
        self.pose_cross_attn_layers.to(device)
        self.plucker_net.to(device)
        
        if hasattr(self.local_model, 'freqs'):
            self.local_model.freqs = self.local_model.freqs.to(device)
            
        self._device_setup_done = True
        print(f"[Rank {rank}] Setup complete: {self.stream_name} stream on {device}")
        
    def forward(
        self, 
        x_rgb, y_rgb, clip_rgb,
        x_depth, y_depth, clip_depth,
        t, context, seq_len, 
        preprocess_fn,
        plucker_rgb, plucker_depth
    ):
        """
        前向传播
        
        新增参数:
            plucker_rgb:   RGB 视角 Plücker ray map [B, 6, F, H, W]
            plucker_depth: Depth 视角 Plücker ray map [B, 6, F, H, W]
        """
        assert self._device_setup_done, "Must call setup_for_rank() before forward()"
        
        # 根据 rank 选择输入
        if self.rank == 0:
            x_local, y_local, clip_local = x_rgb, y_rgb, clip_rgb
            plucker_local = plucker_rgb
        else:
            x_local, y_local, clip_local = x_depth, y_depth, clip_depth
            plucker_local = plucker_depth

        # 预处理
        h, e, e0, ctx, lens, grids = preprocess_fn(
            self.local_model, x_local, t, context, seq_len, clip_local, y_local
        )
        if ctx.shape[1] % 8 != 0:
            pad_len = 8 - (ctx.shape[1] % 8)
            padding = torch.zeros(ctx.shape[0], pad_len, ctx.shape[2],
                                  device=ctx.device, dtype=ctx.dtype)
            ctx = torch.cat([ctx, padding], dim=1)

        h = h.to(dtype=self.local_model.dtype)
        e0 = e0.to(dtype=self.local_model.dtype)
        ctx = ctx.to(dtype=self.local_model.dtype)
        ctx_lens = torch.tensor([ctx.shape[1]], device=ctx.device, dtype=torch.long)

        # ---- PluckerNet: 计算 pose 特征（前向只算一次）----
        target_seq_len = h.shape[1]
        plucker_local = plucker_local.to(device=h.device, dtype=h.dtype)
        pose_features = self.plucker_net(plucker_local, target_seq_len)
        pose_features = pose_features.to(dtype=h.dtype)

        # ---- 逐层前向，交替 pose / cross-stream ----
        dist.barrier()
        pose_idx = 0
        cross_idx = 0
        interaction_count = 0
        
        for i in range(self.num_layers):

            h = h.contiguous()
            ctx = ctx.contiguous()
            h = self.local_model.blocks[i](
                x=h, 
                e=e0, 
                seq_lens=lens, 
                grid_sizes=grids,
                freqs=self.local_model.freqs, 
                context=ctx, 
                context_lens=ctx_lens,
                dtype=h.dtype
            )
            
            # 到达交互点
            if (i + 1) % self.interaction_interval == 0:
                if interaction_count % 2 == 0:
                    # 奇数位交互点 → Pose cross-attention
                    h = self.pose_cross_attn_layers[pose_idx](h, pose_features)
                    pose_idx += 1
                else:
                    # 偶数位交互点 → 跨流 cross-attention
                    h = self._cross_stream_interaction(h, cross_idx)
                    cross_idx += 1
                interaction_count += 1
                
        # 输出头
        out = self.local_model.head(h, e)
        out = self.local_model.unpatchify(out, grids)
        
        return torch.stack(out) if isinstance(out, list) else out
    
    def _cross_stream_interaction(self, h, inter_idx):
        """跨流交互 - 支持梯度反传的特征交换"""
        other_h = exchange_features(h, self.rank)
        other_h = other_h.to(device=h.device, dtype=h.dtype)
        h = self.cross_attn_layers[inter_idx](h, other_h)
        return h

    def get_trainable_params(self):
        """获取所有可训练参数"""
        params = []
        
        # 1. 本流 LoRA 参数
        for param in self.local_model.parameters():
            if param.requires_grad:
                params.append(param)
                
        # 2. 跨流 cross-attention 参数
        for param in self.cross_attn_layers.parameters():
            if param.requires_grad:
                params.append(param)
        
        # 3. Pose cross-attention 参数
        for param in self.pose_cross_attn_layers.parameters():
            if param.requires_grad:
                params.append(param)
        
        # 4. PluckerNet 参数
        for param in self.plucker_net.parameters():
            if param.requires_grad:
                params.append(param)
                
        return params


def create_preprocess_fn(device, weight_dtype):
    """
    预处理函数
    """
    def preprocess_fn(model, x, t, context, seq_len, clip_fea, y):
        if model.freqs.device != x.device:
            model.freqs = model.freqs.to(x.device)
            
        if y is not None:
            x_new = []
            for u, v in zip(x, y):
                mask = v[0:1, :, :, :]
                ref_latent = v[1:, :, :, :]
                mask_4ch = mask.repeat(4, 1, 1, 1)
                combined = torch.cat([u, mask_4ch, ref_latent], dim=0)
                x_new.append(combined)
            processed_x = x_new
        else:
            processed_x = [u for u in x]
            
        embedded_x = [model.patch_embedding(u.unsqueeze(0)) for u in processed_x]
        
        grids_cpu = torch.stack([
            torch.tensor(u.shape[2:], dtype=torch.long) for u in embedded_x
        ]).cpu()
        grid_sizes = grids_cpu.to(device=device)
        
        flattened_x = [u.flatten(2).transpose(1, 2) for u in embedded_x]
        
        seq_lens = torch.tensor(
            [u.size(1) for u in flattened_x], dtype=torch.long, device=device
        )
        
        max_len = max(u.size(1) for u in flattened_x)
        packed_x = torch.cat([
            torch.cat([u, u.new_zeros(1, max_len - u.size(1), u.size(2))], dim=1)
            for u in flattened_x
        ])
        
        t_emb = sinusoidal_embedding_1d(model.freq_dim, t)
        t_emb = t_emb.to(device=packed_x.device, dtype=packed_x.dtype)
        e = model.time_embedding(t_emb)
        e0 = model.time_projection(e).unflatten(1, (6, model.dim))
        
        context_list = [context[i] for i in range(context.size(0))]
        context_tensors = []
        for u in context_list:
            if u.size(0) < model.text_len:
                padding = u.new_zeros(model.text_len - u.size(0), u.size(1))
                u = torch.cat([u, padding], dim=0)
            else:
                u = u[:model.text_len]
            context_tensors.append(u)
            
        context_final = model.text_embedding(torch.stack(context_tensors))
        
        if clip_fea is not None:
            context_clip = model.img_emb(clip_fea)
            context_final = torch.cat([context_clip, context_final], dim=1)
            
        return packed_x, e, e0, context_final, seq_lens, grid_sizes
    
    return preprocess_fn
