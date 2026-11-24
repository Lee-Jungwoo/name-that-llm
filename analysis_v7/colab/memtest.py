import sys, time, torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel
import transformers; transformers.logging.set_verbosity_error()
dev = torch.device("cuda:0")
def run(bs, T, ckpt, amp, n=6):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = AutoModel.from_pretrained("/content/models/deberta-v3-large").float()
    if ckpt: m.gradient_checkpointing_enable()
    m = m.to(dev).train(); head = nn.Linear(1024, 18).to(dev)
    opt = torch.optim.AdamW(list(m.parameters()) + list(head.parameters()), lr=1e-6, weight_decay=0.01)
    ids = torch.randint(5, 1000, (bs, T), device=dev); mask = torch.ones_like(ids); y = torch.randint(0, 18, (bs,), device=dev)
    try:
        t0 = None
        for i in range(n):
            if i == 1: torch.cuda.synchronize(); t0 = time.time()
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                h = m(input_ids=ids, attention_mask=mask).last_hidden_state
            z = h.float().mean(1)
            loss = F.cross_entropy(head(z), y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        print(f"bs={bs} T={T} ckpt={ckpt} amp={amp}: peak {torch.cuda.max_memory_allocated()/2**30:.2f}GB, {(time.time()-t0)/(n-1):.3f}s/micro-batch, loss finite={bool(torch.isfinite(loss))}", flush=True)
    except torch.OutOfMemoryError:
        print(f"bs={bs} T={T} ckpt={ckpt} amp={amp}: OOM (peak {torch.cuda.max_memory_allocated()/2**30:.2f}GB)", flush=True)
    del m, head, opt; import gc; gc.collect(); torch.cuda.empty_cache()
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)
cap8 = torch.cuda.get_device_capability(0)[0] >= 8
amp = torch.bfloat16 if cap8 else None
for bs, T, ckpt in [(4, 141, False), (4, 351, False), (4, 470, False), (4, 512, False), (4, 141, True), (4, 512, True), (8, 141, cap8 and False), (8, 512, False), (8, 512, True)]:
    run(bs, T, ckpt, amp)
