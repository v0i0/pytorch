import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

torch._dynamo.reset()

tensors = torch.load("captured_qkv.pt", map_location='cuda')
for l in tensors:
    for e in l:
        e.requires_grad_(False)

print(tensors[0][0].shape, tensors[0][0].dtype)
print(tensors[0][1].shape, tensors[0][1].dtype)
def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx
block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=tensors[0][0].shape[2], KV_LEN=tensors[0][1].shape[2])

def fn():
    os = []
    for q, k, v in tensors:
        o = torch.compile(flex_attention, mode='max-autotune-no-cudagraphs')(q, k, v, block_mask=block_mask, enable_gqa=True)
        os.append(o)
    return o

from triton.testing import do_bench

# print(do_bench(torch.compile(fn, mode='max-autotune-no-cudagraphs'), warmup=100, rep=100))

from torch.profiler import profile, ProfilerActivity, record_function

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(100):
        fn()

print(prof.key_averages())
