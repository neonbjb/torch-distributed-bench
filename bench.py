from time import time

import torch
import torch.distributed as dist
from tqdm import tqdm


def bench_single_process(n, dev):
    t = torch.randn(1024 * 1024 * 1024 // 32, dtype=torch.float, device=dev)  # 1 GBit of data
    # warm up
    for _ in range(5):
        dist.all_reduce(t)

    start = time()
    for _ in tqdm(range(n), disable=dist.get_rank() != 0):
        dist.all_reduce(t)
    dist.barrier()
    elapsed = time() - start
    throughput = n * dist.get_world_size() * t.shape[0] * 32 / elapsed / 1024 / 1024 / 1024
    if dist.get_rank() == 0:
        print(n, dist.get_world_size(), t.shape[0], elapsed)
        print(f"Throughput: {throughput:.2f} GBit/s")


def bench(backend="gloo", iterations=100):
    dist.init_process_group(backend=backend)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bench_single_process(n=iterations, dev=dev)


if __name__ == '__main__':
    import fire
    fire.Fire(bench)