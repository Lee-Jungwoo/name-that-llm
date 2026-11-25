# Stage 1 마이크로 배치의 폭(토큰 수) 분포: 지금까지 본 배치가 가장 긴 배치를 포함하는지 확인 (CPU만 사용)
import re, numpy as np, pandas as pd, time
from transformers import AutoTokenizer
t0 = time.time()
LABELS = ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra", "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash", "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"]
train = pd.read_csv("/content/data/name-that-llm/train.csv"); train["text"] = train.text.astype(str)
pool = pd.read_parquet("/content/data/open-pool/ntl_open_pool.parquet", columns=["text", "model", "match_weight", "prompt_group"])
pool = pool[pool.model.isin(LABELS)].reset_index(drop=True)
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
pool = pool[~pool.text.map(norm).isin(set(train.text.map(norm)))].reset_index(drop=True)
tail = pool.prompt_group.fillna("").astype(str).str.split(":").str[-1]
lp = pool[~tail.isin(set(train.prompt_id.astype(str))).values].reset_index(drop=True)
tok = AutoTokenizer.from_pretrained("/content/models/Qwen3-4B-Base", padding_side="right")
sep = len(tok("\n\n", add_special_tokens=False)["input_ids"])
texts = lp.text.tolist(); lens = []
for i in range(0, len(texts), 10000):
    enc = tok(texts[i:i + 10000], truncation=True, max_length=384, add_special_tokens=False)["input_ids"]
    lens += [2 * max(1, len(e)) + sep for e in enc]
lengths = np.array(lens)
def make_batches(lengths, batch_size, seed):
    order = np.arange(len(lengths)); rng = np.random.default_rng(seed); rng.shuffle(order); batches = []
    for i in range(0, len(order), batch_size * 50):
        chunk = order[i:i + batch_size * 50]; chunk = chunk[np.argsort(lengths[chunk])]
        batches += [chunk[j:j + batch_size] for j in range(0, len(chunk), batch_size)]
    rng.shuffle(batches); return batches
b = make_batches(lengths, 4, 42)
w = np.array([lengths[x].max() for x in b]); tot = np.array([lengths[x].max() * len(x) for x in b])
print("W rows", len(lengths), "sep", sep, "len pct 50/90/99/max", np.percentile(lengths, [50, 90, 99]).tolist(), lengths.max(), "| truncated(=max)", int((lengths == lengths.max()).sum()))
print("W batches", len(b), "width pct 50/90/99/max", np.percentile(w, [50, 90, 99]).tolist(), w.max(), "| batches at max width", int((w == w.max()).sum()))
print("W first 828 micro-batches: max width", w[:828].max(), "n at global max", int((w[:828] == w.max()).sum()), "| first index at max", int(np.argmax(w == w.max())))
print("W mean width", w.mean(), "mean width first 828", w[:828].mean(), f"| {time.time() - t0:.0f}s")
tr = [2 * max(1, len(e)) + sep for e in tok(train.text.tolist(), truncation=True, max_length=384, add_special_tokens=False)["input_ids"]]
print("W train len pct 50/90/99/max", np.percentile(tr, [50, 90, 99]).tolist(), max(tr))
