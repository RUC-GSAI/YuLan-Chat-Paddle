import paddle
import paddle.distributed as dist
import os

def main():
    # 初始化分布式环境
    dist.init_parallel_env()

    # 获取进程 ID 和总进程数
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 每个进程创建一个 Tensor，值等于 rank
    x = paddle.to_tensor([float(rank)], dtype='float32')
    print(f"[Before allreduce] Rank {rank}: {x.numpy()}")

    # allreduce sum
    dist.all_reduce(x, op=dist.ReduceOp.SUM)

    print(f"[After allreduce] Rank {rank}: {x.numpy()} (should be sum 0+1+...+{world_size-1})")

if __name__ == "__main__":
    main()

