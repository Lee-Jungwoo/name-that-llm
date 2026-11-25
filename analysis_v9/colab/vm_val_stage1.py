# VM 안에서 실행: Stage 1 체크포인트마다 대회 train 2,699개의 검증 손실과 BA를 잽니다(학습이 다 끝난 뒤, GPU가 비었을 때).
# Stage 1은 train을 보지 않았으므로(출처 프롬프트 id가 같은 풀 86행도 뺌) train 전체가 Stage 1의 검증 세트입니다.
# 손실은 label smoothing과 라벨 가중치가 없는 평균 cross-entropy입니다. 모델, echo 입력, 풀링, 헤드는 실행 사본의 셀을 그대로 씁니다.
# 체크포인트: /content/ckpt_hist/ck.*.pt (VM에서 하드 링크로 남겨 둔 것)와 /content/up_ck/*.pt (서버에서 올린 것), 마지막에 Stage 1 가중치.
# 시간 제한(TBOX초) 안에서, 스텝이 고르게 퍼지도록 양 끝 -> 가운데 순서로 잽니다. 결과는 한 줄씩 /content/out/val_loss_stage1.csv에 씁니다.
# 사용: cd /content/run && nohup python vm_val_stage1.py > /content/out/val_stage1.log 2>&1 &
import glob, json, os, time

import numpy as np

T0 = time.time()
TBOX = float(os.environ.get("TBOX", "2100"))
MAXN = int(os.environ.get("MAXN", "12"))
nb = json.load(open(os.environ.get("RUN_NB", "/content/run/v9_lora.ipynb"), encoding="utf-8"))
codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
assert codes[-1].startswith("holdout_idx"), "실행 사본의 마지막 셀이 LoRA 실행 셀이 아닙니다."
G = {"__name__": "__main__"}
for s in codes[:-1]:  # 준비, 설정, 데이터, fold, sample_pool, LoRA 함수 정의까지 (LoRA 실행 셀은 뺌)
    exec(compile(s, "<run nb>", "exec"), G)
torch, CFG, train, y, K = G["torch"], G["CFG"], G["train"], G["y"], G["K"]
from sklearn.metrics import balanced_accuracy_score

device = G["pick_device"]()
model, tok = G["build_classifier"](G["find_base_model"](), device)
CFG["batch_size"] = int(os.environ.get("VAL_BS", "16"))  # predict_logits는 이 값의 2배로 묶음(추론만 하므로 크게)
texts = train.text.values

items = []  # (step, path, state를 꺼내는 키)
for p in sorted(glob.glob("/content/ckpt_hist/ck.*.pt") + glob.glob("/content/up_ck/*.pt")):
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        items.append((int(ck["step"]), p))
        del ck
    except Exception as e:
        print("읽기 실패:", p, repr(e), flush=True)
seen, uniq = set(), []
for st, p in sorted(items):
    if st not in seen:
        seen.add(st)
        uniq.append((st, p))
final = "/content/out/ntl_lora_v9_stage1.pt"
print(f"체크포인트 {len(uniq)}개 (스텝 {uniq[0][0] if uniq else '-'} ~ {uniq[-1][0] if uniq else '-'}), Stage 1 가중치 {os.path.exists(final)} | {time.time() - T0:.0f}초", flush=True)
if len(uniq) > MAXN:
    pick = sorted(set(np.linspace(0, len(uniq) - 1, MAXN).round().astype(int).tolist()))
    uniq = [uniq[i] for i in pick]


def bisect_order(n):
    """0..n-1을 양 끝, 가운데, 사분점 … 순서로. 중간에 끊겨도 점이 고르게 남습니다."""
    out, segs = [0, n - 1] if n > 1 else [0], [(0, n - 1)]
    while segs:
        nxt = []
        for a, b in segs:
            if b - a > 1:
                m = (a + b) // 2
                out.append(m)
                nxt += [(a, m), (m, b)]
        segs = nxt
    return list(dict.fromkeys(out))


order = [uniq[i] for i in bisect_order(len(uniq))] if uniq else []
if os.path.exists(final):
    order = [(None, final)] + order
out_path = "/content/out/val_loss_stage1.csv"
with open(out_path, "w") as f:
    f.write("step,val_loss,val_ba\n")
n_done = 0
for st, p in order:
    if time.time() - T0 > TBOX:
        print(f"시간 제한 {TBOX:.0f}초 도달: {n_done}/{len(order)}개에서 멈춤", flush=True)
        break
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if st is None:
        # Stage 1 가중치 파일에는 step이 없음: 예산에 걸리지 않고 끝났으면 전체 스텝 수
        st = int(os.environ.get("FINAL_STEP", "28869"))
    missing = model.load_state_dict(ck["state"], strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
    del ck
    t = time.time()
    logits = G["predict_logits"](model, tok, device, texts).astype(np.float64)
    lp = logits - logits.max(1, keepdims=True)
    lp = lp - np.log(np.exp(lp).sum(1, keepdims=True))
    loss, ba = float(-lp[np.arange(len(y)), y].mean()), float(balanced_accuracy_score(y, logits.argmax(1)))
    with open(out_path, "a") as f:
        f.write(f"{st},{loss:.6f},{ba:.6f}\n")
    n_done += 1
    print(f"step {st}: val_loss {loss:.4f}, val_ba {ba:.4f} | 추론 {time.time() - t:.0f}초, 누적 {time.time() - T0:.0f}초 | {os.path.basename(p)}", flush=True)
print("== VAL DONE ==", n_done, f"{time.time() - T0:.0f}초", flush=True)
