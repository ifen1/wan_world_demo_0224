"""
通信工具：支持梯度反传的分布式通信原语
"""
import torch
import torch.distributed as dist


class ExchangeFeatures(torch.autograd.Function):
    """
    支持梯度反传的特征交换
    
    前向：rank 0 和 rank 1 互换特征
    反向：梯度也要互换回去
    """
    @staticmethod
    def forward(ctx, local_h, rank):
        ctx.rank = rank
        other_rank = 1 - rank
        
        # 创建接收缓冲区
        other_h = torch.zeros_like(local_h)
        
        # 点对点通信 - 比 all_gather 更高效
        # 使用同步版本确保数据完整
        if rank == 0:
            # Rank 0: 先发后收
            dist.send(local_h.contiguous(), dst=other_rank)
            dist.recv(other_h, src=other_rank)
        else:
            # Rank 1: 先收后发（避免死锁）
            dist.recv(other_h, src=other_rank)
            dist.send(local_h.contiguous(), dst=other_rank)
        
        return other_h
    
    @staticmethod
    def backward(ctx, grad_other_h):
        rank = ctx.rank
        other_rank = 1 - rank
        
        # 反向传播时，梯度也要交换
        grad_local = torch.zeros_like(grad_other_h)
        
        if rank == 0:
            dist.send(grad_other_h.contiguous(), dst=other_rank)
            dist.recv(grad_local, src=other_rank)
        else:
            dist.recv(grad_local, src=other_rank)
            dist.send(grad_other_h.contiguous(), dst=other_rank)
        
        return grad_local, None


class AllGatherWithGrad(torch.autograd.Function):
    """
    支持梯度反传的 all_gather
    用于需要收集所有 rank 特征的场景
    """
    @staticmethod
    def forward(ctx, tensor):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # 收集所有张量
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor.contiguous())
        
        ctx.rank = rank
        ctx.world_size = world_size
        
        return tuple(gathered)
    
    @staticmethod
    def backward(ctx, *grads):
        # 只返回对应 rank 的梯度
        return grads[ctx.rank]


def exchange_features(local_h, rank):
    """
    交换特征的便捷函数
    
    Args:
        local_h: 本地隐状态 [B, S, D]
        rank: 当前进程的 rank
        
    Returns:
        other_h: 对方的隐状态 [B, S, D]
    """
    return ExchangeFeatures.apply(local_h, rank)


def sync_tensor(tensor, rank):
    """
    确保两个 rank 的 tensor 完全一致（用于共享输入如 timesteps）
    以 rank 0 为准广播
    """
    if rank == 0:
        dist.broadcast(tensor, src=0)
    else:
        dist.broadcast(tensor, src=0)
    return tensor


def sync_gradients_for_shared_params(model):
    """
    同步共享参数的梯度
    
    需要同步的模块：
    - cross_attn_layers:      跨流交叉注意力
    - pose_cross_attn_layers:  Pose 交叉注意力
    - plucker_net:             PluckerNet（两个 rank 各持有一份，梯度需平均）
    """
    shared_keywords = ['cross_attn', 'plucker_net']
    for name, param in model.named_parameters():
        if param.grad is not None and any(kw in name for kw in shared_keywords):
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
