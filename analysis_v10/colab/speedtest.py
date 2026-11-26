# VM 안에서 실행: DeBERTa-v3-large의 T4 학습 속도·메모리를 실제 Open Pool 길이 분포로 잽니다 (v10 계획용).
# 1) 풀 전체를 DeBERTa 토크나이저로 토큰화해 길이 분포를 구하고, 노트북의 make_batches와 같은 방식(50배치 묶음 정렬)으로 배치를 만듭니다.
# 2) 무작위 배치 N개로 fp16 autocast + GradScaler(grad ckpt 끔/켬)와 fp32 + grad ckpt(v7 설정)의 마이크로 배치당 시간을 잽니다.
# 3) 가장 긴 배치로 최대 메모리를 잽니다. 4) train 2,699개 추론 시간(검증 손실 기록 비용)을 잽니다.
# 사용: cd /content/run && nohup python speedtest.py > /content/out/speedtest.log 2>&1 &
import gc, math, os, time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import transformers
from transformers import AutoModel, AutoTokenizer

transformers.logging.set_verbosity_error()
dev = torch.device("cuda:0")
MD = "/content/models/deberta-v3-large"
print(torch.cuda.get_device_name(0), torch.__version__, transformers.__version__, flush=True)
tok = AutoTokenizer.from_pretrained(MD)
pool = pd.read_parquet("/content/data/open-pool/ntl_open_pool.parquet", columns=["text", "model"])
train = pd.read_csv("/content/data/name-that-llm/train.csv")
t = time.time()
lens = []
for i in range(0, len(pool), 10000):
    lens += [len(x) for x in tok(pool.text.iloc[i:i + 10000].tolist(), truncation=True, max_length=512)["input_ids"]]
lens = np.array(lens)
tl = np.array([len(x) for x in tok(train.text.tolist(), truncation=True, max_length=512)["input_ids"]])
print(f"풀 {len(pool):,}행 토큰화 {time.time() - t:.0f}초 | 길이 평균 {lens.mean():.1f}, 중앙 {np.median(lens):.0f}, p99 {np.percentile(lens, 99):.0f}, "
      f"최대 {lens.max()}, 512에서 잘림 {(lens >= 512).mean():.4f} | train 평균 {tl.mean():.1f}, 최대 {tl.max()}", flush=True)


def make_batches(lengths, batch_size, seed=0):
    order = np.arange(len(lengths))
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    batches = []
    for i in range(0, len(order), batch_size * 50):
        chunk = order[i:i + batch_size * 50]
        chunk = chunk[np.argsort(lengths[chunk])]
        batches += [chunk[j:j + batch_size] for j in range(0, len(chunk), batch_size)]
    rng.shuffle(batches)
    return batches


BS = 8
batches = make_batches(lens, BS, seed=42)
widths = np.array([lens[b].max() for b in batches])
print(f"마이크로 배치 {len(batches):,}개 (bs {BS}), 폭 평균 {widths.mean():.1f}, 패딩 비율 {1 - lens.sum() / (widths * BS).sum():.3f}", flush=True)
rng = np.random.default_rng(0)
N = int(os.environ.get("N", "60"))
pick = rng.choice(len(batches), N, replace=False)
pw = widths[pick]
print(f"측정 배치 {N}개 폭 평균 {pw.mean():.1f} (전체 {widths.mean():.1f})", flush=True)


def build(ckpt):
    m = AutoModel.from_pretrained(MD).float()
    if ckpt:
        m.gradient_checkpointing_enable()
    return m.to(dev).train(), nn.Linear(1024, 18).to(dev)


def run(amp, ckpt, widths_list, label):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m, head = build(ckpt)
    opt = torch.optim.AdamW(list(m.parameters()) + list(head.parameters()), lr=1e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)
    times, bad = [], 0
    try:
        for i, T in enumerate([widths_list[0]] + list(widths_list)):
            ids = torch.randint(5, 120000, (BS, int(T)), device=dev); mask = torch.ones_like(ids); y = torch.randint(0, 18, (BS,), device=dev)
            torch.cuda.synchronize(); t0 = time.time()
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                h = m(input_ids=ids, attention_mask=mask).last_hidden_state
            loss = F.cross_entropy(head(h.float().mean(1)), y)
            if not torch.isfinite(loss):
                bad += 1
            scaler.scale(loss / 2).backward()
            if i % 2 == 1:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if i > 0:
                times.append(time.time() - t0)
        ms = float(np.mean(times))
        print(f"{label}: {ms:.3f}s/마이크로 배치 (폭 평균 {np.mean(widths_list):.0f}), 최대 메모리 {torch.cuda.max_memory_allocated() / 2**30:.2f}GB, "
              f"비정상 손실 {bad}, scaler {scaler.get_scale() if scaler.is_enabled() else '-'}", flush=True)
        return ms
    except torch.OutOfMemoryError:
        print(f"{label}: OOM (최대 {torch.cuda.max_memory_allocated() / 2**30:.2f}GB)", flush=True)
        return None
    finally:
        del m, head, opt
        gc.collect(); torch.cuda.empty_cache()


res = {}
res["fp16"] = run(torch.float16, False, pw, "fp16 autocast, grad ckpt 끔")
run(torch.float16, False, [512] * 4, "fp16, grad ckpt 끔, 폭 512 (최악)")
res["fp16_ckpt"] = run(torch.float16, True, pw, "fp16 autocast, grad ckpt 켬")
res["fp32_ckpt"] = run(None, True, pw[:20], "fp32, grad ckpt 켬 (v7 설정)")
n_micro = len(batches)
for k, v in res.items():
    if v:
        print(f"추정 Stage 1 ({k}): {n_micro:,} x {v:.3f}s = {n_micro * v / 3600:.2f}시간 (배치 폭 보정 전)", flush=True)

# 추론 시간 (train 2,699개, bs 32, fp16)
m, _ = build(False); m.eval()
ids_l = tok(train.text.tolist(), truncation=True, max_length=512)["input_ids"]
order = np.argsort([len(x) for x in ids_l])
torch.cuda.synchronize(); t0 = time.time()
with torch.no_grad():
    for i in range(0, len(order), 32):
        b = [ids_l[j] for j in order[i:i + 32]]; w = max(map(len, b))
        x = torch.zeros((len(b), w), dtype=torch.long); mk = torch.zeros_like(x)
        for r, s in enumerate(b):
            x[r, :len(s)] = torch.tensor(s); mk[r, :len(s)] = 1
        with torch.autocast("cuda", dtype=torch.float16):
            m(input_ids=x.to(dev), attention_mask=mk.to(dev)).last_hidden_state.float().mean().item()
torch.cuda.synchronize()
print(f"train 2,699개 추론 (fp16, bs 32): {time.time() - t0:.1f}초", flush=True)
print("== SPEEDTEST DONE ==", flush=True)
