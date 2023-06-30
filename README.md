Simple utility for testing the throughput of torch.distributed connections.

## Usage

### GLOO on local CPU (for testing purposes)

```bash
torchrun --nnodes 1 --nproc-per-node 8 bench.py --iterations 1000
```

### NCCL on 8-gpu node

```bash
torchrun --nnodes 1 --nproc-per-node 8 bench.py --iterations 1000 --backend nccl
```

### 4 GPU nodes, 8 GPUs each (32 total)

```bash
torchrun --nnodes 4 --nproc-per-node 8 bench.py --iterations 1000 --backend nccl
```

Depending on how your set-up is configured, you may need to write your own bench() function to properly configure your
torch environment.