"""name_that_llm_v9.ipynb -> name_that_llm_v10.ipynb (DeBERTa Stage 1을 Open Pool 전체로 다시 학습).

v9에서 바뀌는 것은 DeBERTa 분기(4-2장)와 Colab 준비 셀뿐입니다. v9가 LoRA 분기에 한 것과 같은 변경입니다.
- Stage 1 데이터: 라벨당 2,500개 match_weight 비례 추출(45,000개) -> Open Pool 전체(230,950행, LoRA v9와 같은 행).
  출처의 프롬프트 id가 train prompt_id와 같은 풀 행(86개)은 빼고, 손실에 라벨 균형 가중치를 줍니다(평균 1).
- 정밀도: T4에서 fp32 -> fp16 autocast + GradScaler (deb_amp="auto": cc 8 이상은 bf16, T4는 fp16). "fp32"로 v7 동작.
- Stage 1 체크포인트와 이어 하기(가중치 fp16 + AdamW 상태 bf16 + 스케줄러·스케일러·위치). 시간 예산이나 NTL_STOP_AT에 닿으면
  체크포인트를 저장하고 실행을 멈춥니다(덜 학습된 Stage 1로 Stage 2를 하지 않음).
- 학습 중 검증 손실 기록: Stage 1은 train 2,699개 전체(Stage 1이 보지 않은 데이터), Stage 2 fold는 그 fold의 검증 지문.
  WORK_DIR/deb_curve.csv에 한 줄씩 씁니다.
- Stage 2: fold마다 OOF를 부분 파일로 저장하고, 다시 실행하면 끝난 fold와 전체 학습 모델은 건너뜀. 설정(2 epoch, lr 1e-5)은 v7 그대로.
- 토큰 id를 int32 배열로 보관하고 나눠서 토크나이즈(풀 전체에서 메모리 절약, 값은 같음).
- 파일 이름을 ntl_deb_v10_*로 바꾸고, 저장된 가중치의 deb_pool_mode가 다르면 멈춤(v7 가중치를 잘못 쓰지 않게).
n-gram(v8), LoRA(v9 가중치), 블렌드, 게이트, 중심화, Sinkhorn은 v9 그대로입니다.

사용: <python> analysis_v10/make_v10.py            # name_that_llm_v10.ipynb를 새로 만듦
      <python> analysis_v10/make_v10.py "<결과 줄>"  # 첫 셀 첫 줄에 제출 결과를 붙여서 만듦
"""
import ast, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT = sys.argv[1] if len(sys.argv) > 1 else "학습 전"
nb = json.load(open(ROOT / "name_that_llm_v9.ipynb", encoding="utf-8"))
cells = nb["cells"]


def get(i):
    return "".join(cells[i]["source"])


def put(i, text):
    cells[i]["source"] = text.splitlines(keepends=True)


def sub(i, old, new, count=1):
    s = get(i)
    assert s.count(old) == count, (i, old[:70], s.count(old))
    put(i, s.replace(old, new))


assert get(6).startswith("# Colab 전용 준비")
assert get(8).startswith("import gc, glob, json")
assert get(20).startswith("## 4-2. DeBERTa 분류기")
assert get(21).startswith("DEB_STAGE1_FILE")
assert get(22).startswith("deb_oof, deb_test = None, None")
assert get(24).startswith("import itertools")

# 0. 머리말
put(0, f"""> **v10 · v9 + DeBERTa Stage 1을 Open Pool 전체로 다시 학습** · {RESULT}
>
> v9에서 바뀐 것은 DeBERTa 분기(4-2장)뿐입니다. v9가 LoRA 분기에 한 것과 같은 변경입니다. Stage 1 데이터를 라벨당 2,500개 표본(45,000개)에서 **Open Pool 전체**(230,950행, `deb_pool_mode="full"`)로 늘리고 손실에 **라벨 균형 가중치**를 줍니다. 출처의 프롬프트 id가 train `prompt_id`와 같은 풀 행 86개는 뺍니다. T4에서는 fp32 대신 **fp16 autocast + GradScaler**로 학습합니다(`deb_amp`). Stage 2 설정은 v7 그대로입니다. n-gram(v8), LoRA(v9 가중치 `ntl-lora-weights-v9-fullpool`), 블렌드 가중치, 회사 게이트·중심화·Sinkhorn은 v9와 같고, Open Pool은 v9와 같은 고정판(231,036행)입니다. 노트북을 만든 스크립트는 `analysis_v10/make_v10.py`입니다.
""")

# 1. 제목
sub(1, "# Name That LLM: 짧은 지문으로 작성 모델 식별하기 (v9: v8 + LoRA Stage 1을 Open Pool 전체로)\n\n",
    "# Name That LLM: 짧은 지문으로 작성 모델 식별하기 (v10: v9 + DeBERTa Stage 1을 Open Pool 전체로)\n\n"
    "> v10 변경점: DeBERTa 분기의 Stage 1을 **Open Pool 전체**로 다시 학습합니다(아래 \"4-2. DeBERTa 분류기\"의 \"v10\" 참고). "
    "나머지는 v9와 같습니다.\n>\n")

# 2. 데이터 표
sub(2, "LoRA Stage 1은 전체(v9), DeBERTa Stage 1은 라벨당 2,500개 사용", "LoRA Stage 1은 전체(v9), DeBERTa Stage 1도 전체(v10) 사용")

# 3. 아키텍처(영문)
sub(3, "- Same two-stage schedule as branch B: Stage 1 on the Open Pool, then Stage 2 per fold",
    "- Same two-stage schedule as branch B: Stage 1 on the Open Pool (all of it since v10, label-balanced loss weights, same rows as branch B), then Stage 2 per fold")

# 5. 실행 방법
sub(5, "DeBERTa 결과 데이터셋(아래 표의 파일 3개)", "DeBERTa 결과 데이터셋 `ntl-deberta-v10-fullpool`(아래 표의 파일 3개)")
sub(5, "| DeBERTa | `ntl_deb_oof.npz`, `ntl_deb_full.pt`, `ntl_deb_base/` 폴더 | 약 0.9GB |",
    "| DeBERTa (v10) | `ntl_deb_v10_oof.npz`, `ntl_deb_v10_full.pt`, `ntl_deb_base/` 폴더 | 약 0.9GB |")
s5 = get(5)
a = s5.index("**Colab (v9: LoRA 학습용)**")
b = s5.index("**로컬 (동작 점검용)**")
put(5, s5[:a] + """**Colab (v10: DeBERTa 학습용)**

1. 아래 Colab 준비 셀이 DeBERTa-v3-large를 받습니다. Colab에서는 LoRA를 끕니다(`use_lora: false`). LoRA는 v9 가중치를 제출 커널에서만 붙입니다.
2. **Open Pool은 받지 않습니다.** v9와 같은 고정판(231,036행, md5 `10d1c13b…`)을 `/content/data/open-pool/ntl_open_pool.parquet`에 직접 올려 둡니다. 준비 셀이 md5를 확인하고 다르면 멈춥니다(원본 데이터셋은 11/25에 238,750행으로 갱신됨).
3. 대회 파일(`train.csv`, `test.csv`, `sample_submission.parquet`)을 `/content/data/name-that-llm/`에 둡니다.
4. Stage 1은 `deb_stage1_ckpt_min`(30)분마다 `/content/out/ntl_deb_v10_stage1_ckpt.pt`(약 2.6GB)를 저장합니다. VM이 끊기면 이 파일을 새 VM의 `/content/out/`에 되돌려 놓고 다시 실행하면 그 자리에서 이어 합니다. 환경 변수 `NTL_STOP_AT`(유닉스 시각)을 주면 그 시각에 체크포인트를 저장하고 멈춥니다(VM 수명 8시간 전에 계획적으로 넘기기 위함). Stage 1이 끝나면 `ntl_deb_v10_stage1.pt`, fold가 끝날 때마다 `ntl_deb_v10_oof_partial.npz`, 마지막에 `ntl_deb_v10_full.pt`와 `ntl_deb_v10_oof.npz`를 저장하고, 다시 실행하면 있는 것은 건너뜁니다.
5. 학습 손실과 검증 손실은 `/content/out/deb_curve.csv`에 기록됩니다.

""" + s5[b:])
sub(5, "name_that_llm_v9.ipynb", "name_that_llm_v10.ipynb")

# 6. Colab 준비 셀: Qwen 대신 DeBERTa, 풀은 고정판만
put(6, '''# Colab 전용 준비: DeBERTa 받기, 고정판 Open Pool 확인, 경로 지정 (Kaggle에서는 건너뜀)
import os
if os.path.exists("/content") and "KAGGLE_KERNEL_RUN_TYPE" not in os.environ:
    import hashlib
    from huggingface_hub import snapshot_download
    snapshot_download("microsoft/deberta-v3-large", local_dir="/content/models/deberta-v3-large",
                      allow_patterns=["config.json", "spm.model", "tokenizer_config.json", "pytorch_model.bin"])

    # v10: Open Pool은 v9와 같은 고정판(11/25 갱신 전, 231,036행)만 씁니다. 받지 않고, 올려 둔 파일의 md5를 확인합니다.
    pool_file = "/content/data/open-pool/ntl_open_pool.parquet"
    POOL_MD5 = "10d1c13b5a9f9a8c3c2cf64a1d4508b8"
    assert os.path.exists(pool_file), f"고정판 Open Pool을 {pool_file}에 올려 두세요 (Kaggle 비공개 데이터셋 ntl-open-pool-20261118과 같은 파일)."
    with open(pool_file, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    assert md5 == POOL_MD5, f"Open Pool md5 {md5}가 고정판({POOL_MD5})과 다릅니다."
    print(f"Open Pool: {os.path.getsize(pool_file) / 1e6:.0f}MB, md5 {md5} (고정판)")

    os.makedirs("/content/out", exist_ok=True)
    os.environ.update(
        NTL_INPUT="/content/data",
        NTL_DEB_MODEL="/content/models/deberta-v3-large",
        NTL_WORK="/content/out",
        # v10: Colab에서는 DeBERTa만 학습합니다. LoRA는 v9 가중치를 제출 커널에서만 붙입니다.
        NTL_CFG='{"use_lora": false, "ngram_train_weights": [12.0]}',
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",  # 메모리 조각화 완화 (torch import 전에 지정해야 함)
    )
    if os.path.exists("/content/out/ntl_deb_v10_stage1_ckpt.pt"):
        print("DeBERTa Stage 1 체크포인트가 있습니다. 그 자리에서 이어 합니다.")
    elif os.path.exists("/content/out/ntl_deb_v10_stage1.pt"):
        print("DeBERTa Stage 1 가중치가 있습니다. Stage 1을 건너뛰고 Stage 2부터 합니다.")
    else:
        print("v10: DeBERTa Stage 1을 Open Pool 전체로 처음부터 학습합니다(Colab 세션은 8시간에서 끊기므로 이어 하기가 필요할 수 있음).")
''')

# 8. CFG
sub(8, """    deb_pool_per_label=2500,     # Stage 1에 쓸 Open Pool 표본 수 (LoRA와 같은 seed라 같은 표본)
    deb_stage1_epochs=1, deb_stage1_lr=2e-5, deb_stage1_budget_h=2.5,
""", """    deb_pool_mode="full",        # v10: "full" = Open Pool 전체로 Stage 1 (LoRA v9와 같은 행). "sample" = v9까지처럼 라벨당 deb_pool_per_label개
    deb_pool_per_label=2500,     # deb_pool_mode="sample"일 때만 씀 (LoRA와 같은 seed라 같은 표본)
    deb_pool_balance=True,       # v10: Stage 1 손실에 라벨 균형 가중치 (풀 행 수 / 라벨 수) / (그 라벨의 풀 행 수). 평균 1
    deb_drop_train_prompt_ids=True,  # v10: 출처의 프롬프트 id가 train prompt_id와 같은 풀 행을 Stage 1에서 뺌
    deb_stage1_epochs=1, deb_stage1_lr=2e-5, deb_stage1_budget_h=14.0,  # v10: 풀 전체라 예산을 크게 잡음(v9까지 2.5). 실행 한 번 기준
    deb_stage1_ckpt_min=30,      # v10: Stage 1 체크포인트 간격(분, 약 2.6GB). 0이면 저장하지 않음
    deb_amp="auto",              # v10: "auto" = cc 8 이상 bf16, T4는 fp16 autocast + GradScaler. "fp32" = v9까지의 T4 동작
    deb_val_points=[30, 10],     # v10: 학습 중 검증 손실을 재는 횟수 [Stage 1, Stage 2 fold]. Stage 1은 train 전체, fold는 그 fold의 검증 지문
""")

# 20. DeBERTa 설명: v10 절
sub(20, "> **실행 기록 (2026-11-25, Colab T4):**", """### v10: Stage 1을 Open Pool 전체로

v9에서 LoRA 분기의 Stage 1을 풀 전체로 바꾸자 LoRA OOF가 0.5631에서 0.6072로 올랐습니다. DeBERTa 분기에도 같은 변경을 합니다. 풀 제작자의 기록(DeBERTa-v3-large 하나를 풀 전체로 학습해 Public LB 0.587)도 근거입니다.

| | v9까지 | v10 |
|---|---|---|
| Stage 1 데이터 | 라벨당 2,500개, `match_weight` 비례 추출 (45,000개) | **풀 전체** (`deb_pool_mode="full"`, 230,950행). LoRA v9와 같은 행 |
| Stage 1 손실 | cross-entropy (label smoothing 0.05) | 같은 손실에 **라벨 균형 가중치** (LoRA v9와 같은 식) |
| 정밀도 (T4) | fp32 + gradient checkpointing | **fp16 autocast + GradScaler** (`deb_amp="auto"`) |
| 체크포인트 | 없음 | `deb_stage1_ckpt_min`분마다, AdamW 상태 포함. 이어 하기 가능 |
| 검증 손실 | 기록하지 않음 | 학습 중 기록 (`deb_curve.csv`) |
| Stage 2 | 2 epoch, lr 1e-5 | 같음 |

- **fp16.** v7은 DeBERTa-v3의 fp16 NaN 사례를 피하려고 T4에서 fp32로 돌렸고 Stage 1(45,000개)에 123분이 걸렸습니다. 풀 전체는 fp32로 약 10.5시간입니다. 가중치는 fp32로 두고 연산만 fp16 autocast로 하며, GradScaler가 inf/NaN gradient를 낸 갱신을 건너뜁니다. 건너뛴 갱신 수를 로그에 남깁니다.
- **체크포인트.** 가중치는 fp16, AdamW의 1·2차 모멘트는 bf16으로 줄여 약 2.6GB입니다(fp32 그대로면 5.2GB). 이어 할 때 가중치가 fp16으로 한 번 반올림되는데, Stage 2는 v7부터 fp16으로 저장한 Stage 1 가중치에서 시작했으므로 같은 종류의 반올림입니다.
- **검증 손실.** Stage 1은 train을 보지 않으므로 train 2,699개 전체가 Stage 1의 검증 세트입니다. 손실은 label smoothing과 가중치가 없는 평균 cross-entropy(온도 1)입니다. 학습에는 쓰지 않고 기록만 합니다.

> **실행 기록 (2026-11-25, Colab T4):**""")

# 21. DeBERTa 함수
sub(21, '''DEB_STAGE1_FILE, DEB_FULL_FILE, DEB_OOF_FILE = "ntl_deb_stage1.pt", "ntl_deb_full.pt", "ntl_deb_oof.npz"
DEB_FOLD_FILES = [f"ntl_deb_fold{k}.pt" for k in range(CFG["n_folds"])]''',
    '''# v10: v7 가중치(ntl_deb_*.pt)와 섞이지 않도록 이름을 바꿈
DEB_STAGE1_FILE, DEB_FULL_FILE, DEB_OOF_FILE = "ntl_deb_v10_stage1.pt", "ntl_deb_v10_full.pt", "ntl_deb_v10_oof.npz"
DEB_STAGE1_CKPT, DEB_OOF_PARTIAL, DEB_CURVE_FILE = "ntl_deb_v10_stage1_ckpt.pt", "ntl_deb_v10_oof_partial.npz", "deb_curve.csv"
DEB_FOLD_FILES = [f"ntl_deb_v10_fold{k}.pt" for k in range(CFG["n_folds"])]''')
sub(21, '''    amp = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8 else None''',
    '''    # v10: T4(cc 7.5)에서도 fp16 autocast. 가중치는 fp32 그대로이고 학습에서는 GradScaler를 씁니다. "fp32"면 v9까지와 같음
    amp = None
    if device.type == "cuda" and CFG["deb_amp"] != "fp32":
        assert CFG["deb_amp"] == "auto", CFG["deb_amp"]
        amp = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16''')
sub(21, '''def deb_tokenize(tok, texts):
    return tok(list(texts), truncation=True, max_length=CFG["deb_max_len"])["input_ids"]''',
    '''def deb_tokenize(tok, texts, chunk=10000):
    """v10: 풀 전체(약 23만 개)를 다루므로 나눠서 토크나이즈하고 int32 배열로 보관합니다(값은 v9와 같음)."""
    texts, out = list(texts), []
    for i in range(0, len(texts), chunk):
        out += [np.asarray(s, dtype=np.int32) for s in
                tok(texts[i:i + chunk], truncation=True, max_length=CFG["deb_max_len"])["input_ids"]]
    return out''')
sub(21, '''        ids[row, :len(s)] = torch.tensor(s)
        mask[row, :len(s)] = 1''', '''        ids[row, :len(s)] = torch.from_numpy(s)
        mask[row, :len(s)] = 1''')

# deb_train_stage 전체를 바꿈
s21 = get(21)
a = s21.index("def deb_train_stage(")
b = s21.index("@torch.no_grad()\ndef deb_predict(")
s21 = s21[:a] + '''def deb_val_loss(model, tok, device, val_ids, val_y):
    """검증 손실(label smoothing·가중치 없는 평균 cross-entropy, 온도 1)과 BA. 학습 중 기록용(v10)."""
    logits = deb_predict(model, tok, device, None, ids_list=val_ids).astype(np.float64)
    lp = logits - logits.max(1, keepdims=True)
    lp -= np.log(np.exp(lp).sum(1, keepdims=True))
    return float(-lp[np.arange(len(val_y)), val_y].mean()), float(balanced_accuracy_score(val_y, logits.argmax(1)))


def deb_train_stage(model, tok, device, texts, labels, epochs, lr, budget_h, name, weights=None, ckpt_path=None, val=None,
                    val_points=10):
    """한 단계를 학습합니다. 끝까지 학습하면 True, 시간 예산이나 NTL_STOP_AT에 걸려 멈추면 False를 돌려줍니다.
    v10: weights = 행마다 손실 가중치(None이면 모두 1). ckpt_path를 주면 CFG["deb_stage1_ckpt_min"]분마다(그리고 멈출 때) 저장하고,
    파일이 있으면 이어 합니다. val = (지문, 라벨)이면 val_points번 검증 손실을 재서 WORK_DIR/deb_curve.csv에 씁니다."""
    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    body_params = [p for p in model.parameters() if id(p) not in head_ids]
    optim = torch.optim.AdamW([{"params": body_params, "lr": lr},
                               {"params": head_params, "lr": lr * CFG["deb_head_lr_mult"]}], weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=model.amp_dtype == torch.float16)
    ids_list = deb_tokenize(tok, texts)
    lengths = np.array([len(s) for s in ids_list])
    bs, accum = CFG["deb_batch_size"], CFG["deb_grad_accum"]
    n_steps = epochs * math.ceil(math.ceil(len(texts) / bs) / accum)  # 가중치 갱신 횟수
    warmup = max(1, int(n_steps * CFG["deb_warmup"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min(1, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / n_steps))))
    labels = torch.tensor(np.asarray(labels), dtype=torch.long)
    if weights is not None:
        weights = torch.tensor(np.asarray(weights, dtype=np.float32))
        assert len(weights) == len(labels)
    val_ids, val_y = (deb_tokenize(tok, val[0]), np.asarray(val[1])) if val is not None else (None, None)
    val_every = max(1, n_steps // max(1, val_points))
    curve = WORK_DIR / DEB_CURVE_FILE
    start, step, skipped, overflow, running = time.time(), 0, 0, 0, 0.0
    resume_epoch, resume_bi = 0, 0
    use_ckpt = ckpt_path is not None and CFG["deb_stage1_ckpt_min"] > 0
    stop_at = float(os.environ.get("NTL_STOP_AT", "inf"))

    def save_ckpt(epoch, bi):
        opt_sd = optim.state_dict()
        opt_sd["state"] = {k: {n: (v.to(torch.bfloat16) if n != "step" and v.is_floating_point() else v).cpu()
                               for n, v in st.items()} for k, st in opt_sd["state"].items()}
        torch.save({"state": deb_state(model), "optim": opt_sd, "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "bi": bi, "step": step, "skipped": skipped, "overflow": overflow, "running": float(running),
                    "n_rows": len(texts), "n_steps": n_steps}, str(ckpt_path) + ".tmp")
        os.replace(str(ckpt_path) + ".tmp", ckpt_path)
        del opt_sd
        gc.collect()

    def log_curve(val_loss=None, val_ba=None):
        new = not curve.exists()
        with open(curve, "a") as f:
            if new:
                f.write("stage,step,n_steps,train_loss,val_loss,val_ba,minutes,skipped,overflow\\n")
            f.write(f"{name},{step},{n_steps},{running:.5f},{'' if val_loss is None else f'{val_loss:.5f}'},"
                    f"{'' if val_ba is None else f'{val_ba:.5f}'},{(time.time() - start) / 60:.2f},{skipped},{overflow}\\n")

    def do_val():
        if val is None:
            log_curve()
            return
        vl, vb = deb_val_loss(model, tok, device, val_ids, val_y)
        model.train()
        log_curve(vl, vb)
        print(f"[{name}] val {step}/{n_steps}: loss {vl:.4f}, BA {vb:.4f} (train loss {running:.3f}) | {(time.time() - start) / 60:.1f}분", flush=True)

    if use_ckpt and Path(ckpt_path).exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ck["n_rows"] == len(texts) and ck["n_steps"] == n_steps, (
            f"{ckpt_path}는 행 {ck['n_rows']:,}개 / {ck['n_steps']}스텝 학습의 체크포인트인데 지금은 {len(texts):,}개 / {n_steps}스텝입니다.")
        model.load_state_dict(ck["state"])
        optim.load_state_dict(ck["optim"])  # 모멘트는 파라미터 dtype(fp32)으로 되돌아감
        sched.load_state_dict(ck["sched"])
        if ck["scaler"] and scaler.is_enabled():
            scaler.load_state_dict(ck["scaler"])
        resume_epoch, resume_bi = ck["epoch"], ck["bi"]
        step, skipped, overflow, running = ck["step"], ck["skipped"], ck.get("overflow", 0), ck["running"]
        del ck
        gc.collect()
        print(f"[{name}] 체크포인트에서 이어 하기: {step}/{n_steps} 스텝 (epoch {resume_epoch}, 마이크로 배치 {resume_bi}부터), "
              f"loss {running:.3f}", flush=True)
    model.train()
    if step == 0:
        do_val()  # 시작점 (Stage 2는 Stage 1 모델의 검증 손실)
    last_ck = time.time()
    log_every = max(1, n_steps // (50 if use_ckpt else 10))
    optim.zero_grad(set_to_none=True)
    for epoch in range(resume_epoch, epochs):
        batches = make_batches(lengths, bs, True, seed=CFG["seed"] + epoch)
        for bi, batch in enumerate(batches):
            if epoch == resume_epoch and bi < resume_bi:
                continue
            if bi % accum == 0 and (time.time() - start > budget_h * 3600 or time.time() > stop_at):
                # 누적 경계에서만 멈춥니다(gradient가 비어 있을 때). 체크포인트는 마이크로 배치 bi부터 이어 하도록 저장합니다.
                why = "시간 예산" if time.time() - start > budget_h * 3600 else "NTL_STOP_AT"
                if use_ckpt:
                    save_ckpt(epoch, bi)
                    print(f"[{name}] {why} 도달: {step}/{n_steps} 스텝에서 체크포인트를 저장하고 멈춤 | {(time.time() - start) / 60:.1f}분", flush=True)
                else:
                    print(f"[{name}] {why} 도달, {step}/{n_steps} 스텝에서 중단")
                model.eval()
                return False
            ids, mask = deb_collate(ids_list, batch, tok.pad_token_id, device)
            if weights is None:
                loss = F.cross_entropy(model(ids, mask), labels[batch].to(device), label_smoothing=0.05)
            else:  # 가중치 평균이 1이므로 배치 크기로만 나눕니다(LoRA v9와 같음).
                per_row = F.cross_entropy(model(ids, mask), labels[batch].to(device), label_smoothing=0.05, reduction="none")
                loss = (per_row * weights[batch].to(device)).mean()
            if torch.isfinite(loss):
                scaler.scale(loss / accum).backward()
                running = 0.98 * running + 0.02 * loss.item() if running else loss.item()
            else:
                skipped += 1
            if (bi + 1) % accum and bi + 1 < len(batches):
                continue
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale_before = scaler.get_scale() if scaler.is_enabled() else None
            scaler.step(optim)  # gradient에 inf/NaN이 있으면 갱신을 건너뛰고 scale을 줄임
            scaler.update()
            if scaler.is_enabled() and scaler.get_scale() < scale_before:
                overflow += 1
            sched.step()
            optim.zero_grad(set_to_none=True)
            step += 1
            if step % log_every == 0:
                print(f"[{name}] {step}/{n_steps} loss {running:.3f} | {(time.time() - start) / 60:.1f}분 | 건너뜀 {skipped}, "
                      f"fp16 overflow {overflow}" + (f", scale {scaler.get_scale():.0f}" if scaler.is_enabled() else ""), flush=True)
            if step % val_every == 0 or step == n_steps:
                do_val()
            if use_ckpt and time.time() - last_ck > CFG["deb_stage1_ckpt_min"] * 60 and bi + 1 < len(batches):
                save_ckpt(epoch, bi + 1)  # 가중치 갱신 직후라 누적 중간이 아님. 다음 실행은 마이크로 배치 bi + 1부터
                last_ck = time.time()
                print(f"[{name}] 체크포인트 저장: {step}/{n_steps} 스텝 | {(time.time() - start) / 60:.1f}분", flush=True)
    print(f"[{name}] 완료: {step}스텝, 비정상 손실로 건너뜀 {skipped}회, fp16 overflow로 건너뛴 갱신 {overflow}회, "
          f"{(time.time() - start) / 60:.1f}분", flush=True)
    model.eval()
    return True


''' + s21[b:]
put(21, s21)
sub(21, '''def deb_predict(model, tok, device, texts):
    model.eval()
    ids_list = deb_tokenize(tok, texts)''', '''def deb_predict(model, tok, device, texts, ids_list=None):
    model.eval()
    ids_list = deb_tokenize(tok, texts) if ids_list is None else ids_list''')
sub(21, '''    out = np.zeros((len(texts), K), dtype=np.float32)''', '''    out = np.zeros((len(ids_list), K), dtype=np.float32)''')
sub(21, '''    assert ckpt["cfg"]["deb_pooling"] == CFG["deb_pooling"], (
        f"{path}는 deb_pooling={ckpt['cfg']['deb_pooling']}로 학습됐는데 지금 설정은 {CFG['deb_pooling']}입니다.")''',
    '''    assert ckpt["cfg"]["deb_pooling"] == CFG["deb_pooling"], (
        f"{path}는 deb_pooling={ckpt['cfg']['deb_pooling']}로 학습됐는데 지금 설정은 {CFG['deb_pooling']}입니다.")
    saved_mode = ckpt["cfg"].get("deb_pool_mode", "sample")  # v7 가중치에는 이 값이 없음
    assert saved_mode == CFG["deb_pool_mode"], (
        f"{path}는 deb_pool_mode={saved_mode!r}로 학습된 가중치인데 지금 설정은 {CFG['deb_pool_mode']!r}입니다.")''')
# deb_cv: fold 이어 하기, 검증 손실, 전체 학습 모델 건너뛰기
sub(21, '''    stage1 = deb_state(model, half=False)  # CPU에 fp32로 보관 (약 1.7GB)
    oof, tests = np.zeros((len(train), K), dtype=np.float32), []
    for k, (a, b) in enumerate(cv_folds):
        model.load_state_dict(stage1)
        deb_train_stage(model, tok, device, train.text.values[a].tolist(), y[a], CFG["deb_stage2_epochs"],
                        CFG["deb_stage2_lr"], CFG["deb_stage2_budget_h"], f"deb-stage2-fold{k}")
        if CFG["deb_save_folds"]:
            save_deb(model, DEB_FOLD_FILES[k])
        oof[b] = deb_predict(model, tok, device, train.text.values[b])
        if CFG["deb_test"] == "folds":
            tests.append(deb_predict(model, tok, device, test.text.values))
        print(f"fold {k}: DeBERTa BA {balanced_accuracy_score(y[b], oof[b].argmax(1)):.4f} | {elapsed()}")
    np.savez(WORK_DIR / DEB_OOF_FILE, oof=oof, ids=train.id.astype(str).values.astype("U"))
    if CFG["deb_test"] == "folds":
        return oof, np.mean(tests, axis=0)
    model.load_state_dict(stage1)
    deb_train_stage(model, tok, device, train.text.tolist(), y, CFG["deb_stage2_epochs"],
                    CFG["deb_stage2_lr"], CFG["deb_stage2_budget_h"], "deb-stage2-full")
    save_deb(model, DEB_FULL_FILE)
    return oof, deb_predict(model, tok, device, test.text.values)''',
    '''    stage1 = deb_state(model, half=False)  # CPU에 fp32로 보관 (약 1.7GB)
    oof, tests, done = np.zeros((len(train), K), dtype=np.float32), [], []
    part = WORK_DIR / DEB_OOF_PARTIAL
    if part.exists() and CFG["deb_test"] == "full":  # v10: 이어 하기. 이 실행 폴더에 저장된 fold OOF는 다시 학습하지 않음
        _p = np.load(part)
        assert (_p["ids"] == train.id.astype(str).values).all()
        oof, done = _p["oof"].astype(np.float32), [int(v) for v in _p["done"]]
        print("이어 하기: 이미 끝난 fold", done)
    fold_of = np.zeros(len(train), dtype=int)
    for k, (_, b) in enumerate(cv_folds):
        fold_of[b] = k
    for k, (a, b) in enumerate(cv_folds):
        if k in done:
            print(f"fold {k}: (저장된 OOF) DeBERTa BA {balanced_accuracy_score(y[b], oof[b].argmax(1)):.4f}")
            continue
        model.load_state_dict(stage1)
        deb_train_stage(model, tok, device, train.text.values[a].tolist(), y[a], CFG["deb_stage2_epochs"],
                        CFG["deb_stage2_lr"], CFG["deb_stage2_budget_h"], f"deb-stage2-fold{k}",
                        val=(train.text.values[b].tolist(), y[b]), val_points=CFG["deb_val_points"][1])
        if CFG["deb_save_folds"]:
            save_deb(model, DEB_FOLD_FILES[k])
        oof[b] = deb_predict(model, tok, device, train.text.values[b])
        if CFG["deb_test"] == "folds":
            tests.append(deb_predict(model, tok, device, test.text.values))
        done.append(k)
        np.savez(str(part) + ".tmp.npz", oof=oof, done=np.array(done), ids=train.id.astype(str).values.astype("U"))
        os.replace(str(part) + ".tmp.npz", part)
        print(f"fold {k}: DeBERTa BA {balanced_accuracy_score(y[b], oof[b].argmax(1)):.4f} | {elapsed()}", flush=True)
    np.savez(WORK_DIR / DEB_OOF_FILE, oof=oof, ids=train.id.astype(str).values.astype("U"), y=y, fold=fold_of)
    if CFG["deb_test"] == "folds":
        return oof, np.mean(tests, axis=0)
    if (WORK_DIR / DEB_FULL_FILE).exists():  # v10: 이어 하기
        print("저장된 전체 학습 모델을 사용합니다:", WORK_DIR / DEB_FULL_FILE)
        load_deb(model, WORK_DIR / DEB_FULL_FILE)
    else:
        model.load_state_dict(stage1)
        deb_train_stage(model, tok, device, train.text.tolist(), y, CFG["deb_stage2_epochs"],
                        CFG["deb_stage2_lr"], CFG["deb_stage2_budget_h"], "deb-stage2-full", val_points=CFG["deb_val_points"][1])
        save_deb(model, DEB_FULL_FILE)
    return oof, deb_predict(model, tok, device, test.text.values)''')
s21 = get(21)
s21 += '''

def deb_stage1_pool():
    """v10: DeBERTa Stage 1 데이터와 행 가중치. LoRA의 lora_stage1_pool(v9)과 같은 규칙이고 deb_* 설정을 씁니다."""
    if pool is None:
        return None, None
    if CFG["deb_pool_mode"] == "full":
        dp = pool
    else:
        assert CFG["deb_pool_mode"] == "sample", CFG["deb_pool_mode"]
        dp = sample_pool(CFG["deb_pool_per_label"], seed=CFG["seed"])
    if CFG["deb_drop_train_prompt_ids"] and "prompt_group" in dp.columns:
        tail = dp.prompt_group.fillna("").astype(str).str.split(":").str[-1]
        drop = tail.isin(set(train.prompt_id.astype(str)))
        print(f"DeBERTa Stage 1: 출처의 프롬프트 id가 train prompt_id와 같은 풀 행 {int(drop.sum())}개를 뺍니다.")
        dp = dp[~drop.values]
    dp = dp.reset_index(drop=True)
    yl = dp.model.map(L2I).values
    counts = np.bincount(yl, minlength=K)
    w = None
    if CFG["deb_pool_balance"]:
        w = (len(yl) / (counts > 0).sum() / np.maximum(counts, 1))[yl]
        print(f"DeBERTa Stage 1 행 가중치(라벨 균형): 최소 {w.min():.3f} / 최대 {w.max():.3f} / 평균 {w.mean():.3f}")
    print(f"DeBERTa Stage 1 데이터 ({CFG['deb_pool_mode']}): {len(dp):,}행 | 라벨별 최소 {counts.min():,} / 최대 {counts.max():,}")
    return dp, w
'''
put(21, s21)

# 22. DeBERTa 실행 셀
sub(22, '''    else:
        assert has_weights(deb_dir), f"{deb_dir}에 사전학습 가중치가 없어 학습할 수 없습니다."''',
    '''    else:
        assert has_weights(deb_dir) or (WORK_DIR / DEB_STAGE1_CKPT).exists() or find_optional(DEB_STAGE1_CKPT) is not None, (
            f"{deb_dir}에 사전학습 가중치가 없어 학습할 수 없습니다.")''')
sub(22, '''        stage1_path = find_optional(DEB_STAGE1_FILE)
        if stage1_path is not None:
            print("저장된 DeBERTa Stage 1 가중치를 사용합니다:", stage1_path)
            load_deb(model, stage1_path)
        else:
            deb_pool = sample_pool(CFG["deb_pool_per_label"], seed=CFG["seed"])
            if deb_pool is not None:
                deb_train_stage(model, dtok, device, deb_pool.text.tolist(), deb_pool.model.map(L2I).values,
                                CFG["deb_stage1_epochs"], CFG["deb_stage1_lr"], CFG["deb_stage1_budget_h"], "deb-stage1")
                save_deb(model, DEB_STAGE1_FILE)
                del deb_pool
                gc.collect()''',
    '''        stage1_path = find_optional(DEB_STAGE1_FILE)
        if stage1_path is None and (WORK_DIR / DEB_STAGE1_FILE).exists():
            stage1_path = WORK_DIR / DEB_STAGE1_FILE
        if stage1_path is not None:
            print("저장된 DeBERTa Stage 1 가중치를 사용합니다:", stage1_path)
            load_deb(model, stage1_path)
        else:
            deb_pool, deb_w = deb_stage1_pool()
            if deb_pool is not None:
                ck_in = find_optional(DEB_STAGE1_CKPT)  # Kaggle에서 이어 할 때: 입력 데이터셋의 체크포인트를 실행 폴더로 복사
                if ck_in is not None and not (WORK_DIR / DEB_STAGE1_CKPT).exists():
                    import shutil
                    shutil.copy(ck_in, WORK_DIR / DEB_STAGE1_CKPT)
                finished = deb_train_stage(model, dtok, device, deb_pool.text.tolist(), deb_pool.model.map(L2I).values,
                                           CFG["deb_stage1_epochs"], CFG["deb_stage1_lr"], CFG["deb_stage1_budget_h"], "deb-stage1",
                                           weights=deb_w, ckpt_path=WORK_DIR / DEB_STAGE1_CKPT,
                                           val=(train.text.tolist(), y), val_points=CFG["deb_val_points"][0])
                if not finished and CFG["deb_stage1_ckpt_min"] > 0:
                    raise SystemExit("DeBERTa Stage 1이 끝나지 않았습니다. 체크포인트(" + DEB_STAGE1_CKPT + ")에서 이어 하세요.")
                save_deb(model, DEB_STAGE1_FILE)
                load_deb(model, WORK_DIR / DEB_STAGE1_FILE)  # 이어 하기와 같게: 모든 fold가 저장된(fp16) Stage 1 가중치에서 시작
                (WORK_DIR / DEB_STAGE1_CKPT).unlink(missing_ok=True)
                if device.type == "cuda":
                    print(f"DeBERTa Stage 1 GPU 메모리 최대 사용량: {torch.cuda.max_memory_allocated() / 2**30:.1f}GB")
                del deb_pool, deb_w
                gc.collect()''')

# 8. blend3 주석 (값은 그대로)
sub(8, "# 고정 가중치 (v7에서 Colab 실행의 DeBERTa OOF로 고름. v8, v9도 그대로.",
    "# 고정 가중치 (v7에서 Colab 실행의 DeBERTa OOF로 고름. v8, v9, v10도 그대로.")

# 24. 블렌드 저장 파일 이름
sub(24, 'np.savez(WORK_DIR / "oof_v9.npz"', 'np.savez(WORK_DIR / "oof_v10.npz"')

for c in cells:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
        c["outputs"] = []
        c["execution_count"] = None
dst = ROOT / "name_that_llm_v10.ipynb"
with open(dst, "w", encoding="utf-8") as f:
    f.write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("저장:", dst)
