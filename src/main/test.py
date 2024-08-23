import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank, world_size):
    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

    # Use only cuda:0 on each node, regardless of rank
    device = torch.device(f'cuda:0')

    # Create a simple tensor on each GPU and print it
    tensor = torch.ones(1).to(device) * rank
    print(f"Rank {rank} has tensor: {tensor.item()} on device {device}")

    # Synchronize across all GPUs
    dist.barrier()

    # All reduce (sum) across all nodes and GPUs
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Print the result after all-reduce (should be the sum of all ranks)
    print(f"Rank {rank} after all-reduce has tensor: {tensor.item()}")

    # Clean up the distributed environment
    dist.destroy_process_group()


def init_process(rank, world_size):
    run(rank, world_size)


if __name__ == "__main__":
    world_size = 2  # Total number of nodes
    mp.spawn(init_process, args=(world_size,), nprocs=1, join=True)
