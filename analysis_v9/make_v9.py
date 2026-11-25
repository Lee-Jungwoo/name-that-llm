"""name_that_llm_v8.ipynb -> name_that_llm_v9.ipynb (LoRA Stage 1을 Open Pool 전체로 다시 학습).

v8에서 바뀌는 것은 LoRA 분기(4장)와 Colab 준비 셀뿐입니다.
- Stage 1 데이터: 라벨당 2,500개 match_weight 비례 추출(45,000개) -> Open Pool 전체(약 231,000개).
  출처의 프롬프트 id가 train prompt_id와 같은 풀 행(86개)은 뺍니다.
- Stage 1 손실: 라벨 균형 가중치 (풀 행 수 / 라벨 수) / (그 라벨의 풀 행 수). 평균 1.
  match_weight는 손실에 쓰지 않습니다(데이터셋 설명: "Use for sampling, not as a loss weight unless you test it").
- Stage 1 체크포인트와 이어 하기(옵티마이저·스케줄러·스케일러 상태 포함), Stage 2는 fold 파일이 있으면 건너뜀.
- 토큰 id를 int32 배열로 보관하고 나눠서 토크나이즈(풀 전체에서 메모리 절약, 값은 같음).
- 가중치 파일 이름을 ntl_lora_v9_*.pt로 바꾸고, 저장된 가중치의 lora_pool_mode가 다르면 멈춤(v4 가중치를 잘못 쓰지 않게).
- LoRA OOF를 LoRA 셀 끝에서 바로 저장(ntl_lora_v9_oof.npz).
n-gram, DeBERTa(v7 가중치), 블렌드, 게이트, 중심화, Sinkhorn은 v8 그대로입니다.

사용: <python> analysis_v9/make_v9.py            # name_that_llm_v9.ipynb를 새로 만듦
      <python> analysis_v9/make_v9.py "<결과 줄>"  # 첫 셀 첫 줄에 제출 결과를 붙여서 만듦
"""
import ast, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT = sys.argv[1] if len(sys.argv) > 1 else "학습 전"
nb = json.load(open(ROOT / "name_that_llm_v8.ipynb", encoding="utf-8"))
cells = nb["cells"]


def get(i):
    return "".join(cells[i]["source"])


def put(i, text):
    cells[i]["source"] = text.splitlines(keepends=True)


def sub(i, old, new, count=1):
    s = get(i)
    assert s.count(old) == count, (i, old[:70], s.count(old))
    put(i, s.replace(old, new))


# 셀 위치가 v8과 같은지 먼저 확인
assert get(6).startswith("# Colab 전용 준비")
assert get(8).startswith("import gc, glob, json")
assert get(10).startswith("COMP = locate_competition_dir()")
assert get(17).startswith("## 4. LoRA 분류기")
assert get(18).startswith("import torch")
assert get(19).startswith("holdout_idx = cv_folds[0][1]")
assert get(24).startswith("import itertools")

# 0. 머리말
put(0, f"""> **v9 · v8 + LoRA Stage 1을 Open Pool 전체로 다시 학습** · {RESULT}
>
> v8에서 바뀐 것은 LoRA 분기(4장)뿐입니다. Stage 1 데이터를 라벨당 2,500개 표본(45,000개, 풀의 19%)에서 **Open Pool 전체**로 늘리고(`lora_pool_mode="full"`), 라벨마다 개수가 3,541 ~ 23,542개로 다르므로 손실에 **라벨 균형 가중치**를 줍니다(`lora_pool_balance`). `match_weight`는 손실에 쓰지 않습니다. 출처의 프롬프트 id가 train `prompt_id`와 같은 풀 행 86개는 뺍니다. n-gram(v8), DeBERTa(v7 가중치 `ntl-deberta-v7`), 회사 게이트·중심화·Sinkhorn은 v8과 같습니다.
>
> **블렌드 가중치(`blend3`)는 아직 v7 값입니다.** LoRA가 바뀌면 최적 조합도 바뀌므로, 학습이 끝나면 저장된 OOF(`ntl_lora_v9_oof.npz`)로 다시 고른 뒤 제출합니다. 노트북을 만든 스크립트는 `analysis_v9/make_v9.py`입니다.
""")

# 1. 제목과 변경점
sub(1, "# Name That LLM: 짧은 지문으로 작성 모델 식별하기 (v8: v7 + n-gram 풀 확대와 문체 블록)\n\n",
    "# Name That LLM: 짧은 지문으로 작성 모델 식별하기 (v9: v8 + LoRA Stage 1을 Open Pool 전체로)\n\n"
    "> v9 변경점: LoRA 분기의 Stage 1을 **Open Pool 전체**(약 231,000개)로 다시 학습합니다. 손실에는 라벨 균형 가중치를 줍니다"
    "(아래 \"4. LoRA 분류기\"의 \"v9\" 참고). 나머지는 v8과 같습니다.\n>\n")

# 2. 데이터 표
sub(2, "| `ntl_open_pool.parquet` (외부, 선택) | n-gram은 라벨당 최대 5,000개(v8), LoRA·DeBERTa Stage 1은 2,500개 사용 |",
    "| `ntl_open_pool.parquet` (외부, 선택) | n-gram은 라벨당 최대 5,000개(v8), LoRA Stage 1은 전체(v9), DeBERTa Stage 1은 라벨당 2,500개 사용 |")
sub(2, "| `text`, `model`, `match_weight` |", "| `text`, `model`, `match_weight`, `prompt_group` |")

# 3. 아키텍처 설명(영문): LoRA Stage 1 데이터
sub(3, "| Data | Open Pool, up to 2,500 per label (`match_weight` sampling), trained once |",
    "| Data | Open Pool: all of it since v9 (label-balanced loss weights, 86 rows whose source prompt id is a `train` prompt id removed); "
    "up to 2,500 per label (`match_weight` sampling) before. Trained once |")

# 5. 실행 방법
sub(5, "LoRA 가중치 데이터셋 `ntl-lora-weights-v4-echo-cv`", "LoRA 가중치 데이터셋(v9 학습 결과, 아래 표의 파일)")
sub(5, "v8은 n-gram 표본이 커져서 n-gram 구간이 v7(약 12분)보다 오래 걸립니다(로컬 CPU 기준 약 33분).",
    "n-gram 구간은 v8과 같이 약 33분입니다(로컬 CPU 기준).")
sub(5, "| LoRA | `ntl_lora_fold0.pt` ~ `ntl_lora_fold4.pt` (+ `ntl_lora_full.pt`) | 개당 약 25MB |",
    "| LoRA (v9) | `ntl_lora_v9_fold0.pt` ~ `ntl_lora_v9_fold4.pt`, `ntl_lora_v9_full.pt` | 개당 약 25MB |")
sub(5, """**Colab (DeBERTa 학습용)**

1. 아래 Colab 준비 셀이 Qwen3-4B-Base, DeBERTa-v3-large, Open Pool을 받습니다.
2. LoRA를 다시 학습하지 않도록 v4 가중치(`weights_v4/` 안의 `ntl_lora_fold*.pt`, `ntl_lora_full.pt`)를 `/content/data/weights_v4/`에 올려 둡니다. 없으면 LoRA를 처음부터 학습합니다.
3. 결과는 `/content/out`에 저장되고, 마지막 셀이 Google Drive의 `ntl/`로 복사합니다.""",
    """**Colab (v9: LoRA 학습용)**

1. 아래 Colab 준비 셀이 Qwen3-4B-Base와 Open Pool을 받습니다. Colab에서는 DeBERTa를 끕니다(`use_deberta: false`). DeBERTa는 v7 가중치를 제출 커널에서만 붙입니다.
2. 대회 파일(`train.csv`, `test.csv`, `sample_submission.parquet`)을 `/content/data/name-that-llm/`에 둡니다.
3. Stage 1은 `stage1_ckpt_min`(10)분마다 `/content/out/ntl_lora_v9_stage1_ckpt.pt`를 저장합니다. VM이 끊기면 이 파일을 새 VM의 `/content/out/`에 되돌려 놓고 다시 실행하면 그 자리에서 이어 합니다. Stage 1이 끝난 뒤에는 `ntl_lora_v9_stage1.pt`를, fold마다 `ntl_lora_v9_fold{k}.pt`를 저장하고, 다시 실행하면 있는 파일은 건너뜁니다.
4. 결과는 `/content/out`에 저장되고, 마지막 셀이 Google Drive의 `ntl/`로 복사합니다. 제출 커널에 붙일 파일은 위 표의 LoRA 파일들이고, 블렌드 가중치를 다시 고를 때는 `ntl_lora_v9_oof.npz`를 씁니다.""")
sub(5, "name_that_llm_v8.ipynb", "name_that_llm_v9.ipynb")

# 6. Colab 준비 셀: DeBERTa 끄기, v9 Stage 1 설정
sub(6, '    snapshot_download("microsoft/deberta-v3-large", local_dir="/content/models/deberta-v3-large")  # v7, 약 0.9GB\n', "")
sub(6, '        NTL_DEB_MODEL="/content/models/deberta-v3-large",\n', "")
sub(6, """        # n-gram 가중치 후보는 v3부터 매번 12가 뽑혀서, Colab에서는 12만 돌려 약 18분을 아낌.
        NTL_CFG='{"batch_size": 4, "grad_accum": 2, "ngram_train_weights": [12.0]}',""",
    """        # n-gram 가중치 후보는 v3부터 매번 12가 뽑혀서, Colab에서는 12만 돌려 약 18분을 아낌.
        # v9: Colab에서는 LoRA만 학습합니다. DeBERTa는 v7 가중치를 제출 커널에서만 붙입니다(여기서 켜면 처음부터 학습함).
        NTL_CFG='{"batch_size": 4, "grad_accum": 2, "ngram_train_weights": [12.0], "use_deberta": false}',""")
sub(6, """    import glob as _glob
    if not _glob.glob("/content/data/**/ntl_lora_fold*.pt", recursive=True):
        print("경고: /content/data에 LoRA fold 가중치가 없습니다. LoRA를 처음부터 학습합니다(수 시간).")""",
    """    if os.path.exists("/content/out/ntl_lora_v9_stage1_ckpt.pt"):
        print("Stage 1 체크포인트가 있습니다. 그 자리에서 이어 합니다.")
    elif os.path.exists("/content/out/ntl_lora_v9_stage1.pt"):
        print("Stage 1 가중치가 있습니다. Stage 1을 건너뛰고 Stage 2부터 합니다.")
    else:
        print("v9: LoRA Stage 1을 Open Pool 전체로 처음부터 학습합니다(L4 약 3시간, T4는 그 몇 배로 추정).")""")

# 8. CFG
sub(8, """    lora_pool_per_label=2500,    # Stage 1 표본 수. v8에서도 그대로(LoRA는 다시 학습하지 않음)
    stage1_epochs=1, stage1_lr=2e-4, stage1_budget_h=3.0,
""", """    lora_pool_mode="full",       # v9: "full" = Open Pool 전체로 Stage 1. "sample" = v8까지처럼 라벨당 lora_pool_per_label개를 match_weight 비례로 추출
    lora_pool_per_label=2500,    # lora_pool_mode="sample"일 때만 씀
    lora_pool_balance=True,      # v9: Stage 1 손실에 라벨 균형 가중치 (풀 행 수 / 라벨 수) / (그 라벨의 풀 행 수)를 줌. 평균 1. match_weight는 손실에 쓰지 않음
    lora_drop_train_prompt_ids=True,  # v9: 출처의 프롬프트 id(prompt_group의 마지막 부분)가 train prompt_id와 같은 풀 행을 Stage 1에서 뺌
    stage1_epochs=1, stage1_lr=2e-4, stage1_budget_h=14.0,  # v9: 풀 전체라 예산을 크게 잡음(v8까지 3.0). 예산은 실행 한 번 기준이고 이어 하면 다시 셈
    stage1_ckpt_min=10,          # v9: Stage 1 체크포인트 간격(분). 0이면 저장하지 않음
""")

# 10. 풀 읽기: prompt_group 열이 있으면 같이 읽음(LoRA Stage 1의 프롬프트 id 제외용)
sub(10, """    pool = pd.read_parquet(pool_path, columns=["text", "model", "match_weight"])""",
    """    import pyarrow.parquet as pq
    pool_cols = ["text", "model", "match_weight"] + [c for c in ["prompt_group"] if c in pq.ParquetFile(pool_path).schema_arrow.names]
    pool = pd.read_parquet(pool_path, columns=pool_cols)  # v9: prompt_group은 LoRA Stage 1에서만 씀""")

# 17. LoRA 설명: v9 절 추가
sub(17, "### v4: echo 입력\n", """### v9: Stage 1을 Open Pool 전체로

v8까지 Stage 1은 라벨당 2,500개를 `match_weight`에 비례해 뽑은 45,000개(풀의 19%)로 1 epoch만 학습했습니다. 이 크기는 실험으로 고른 값이 아니라 v2에서 공개 노트북의 기본값을 가져온 것이고, 그 뒤 바꿔 본 적이 없습니다. n-gram은 같은 풀에서 표본을 2,500개에서 5,000개로 늘리자 OOF BA가 0.4693에서 0.5012로 올랐습니다(v8). 데이터셋 설명에는 제작자가 DeBERTa-v3-large를 풀 전체(가중치 없이 약 193,000개)로 학습해 모델 하나로 Public LB 0.587을 냈다는 기록이 있습니다(우리 v7의 3분기 앙상블은 0.556).

| | v8까지 | v9 |
|---|---|---|
| Stage 1 데이터 | 라벨당 2,500개, `match_weight` 비례 비복원 추출 (45,000개) | **풀 전체** (`lora_pool_mode="full"`). 출처의 프롬프트 id가 train `prompt_id`와 같은 86행은 뺌 |
| 라벨별 개수 | 모두 2,500 | 3,541(gemini-3.7-flash) ~ 23,542(gemini-3.8-flash) |
| Stage 1 손실 | cross-entropy (label smoothing 0.05) | 같은 손실에 **라벨 균형 가중치**를 곱함 |
| 체크포인트 | 없음 | `stage1_ckpt_min`분마다. 이어 하기 가능 |

- **라벨 균형 가중치.** 라벨 \\(L\\)의 풀 행에 \\(w_L = (N / 18) / N_L\\)를 줍니다(\\(N\\)은 풀 행 수, \\(N_L\\)은 라벨 \\(L\\)의 행 수). 모든 라벨의 가중치 합이 \\(N / 18\\)로 같아지고 행 평균은 1입니다. 마이크로 배치의 손실은 \\(\\frac{1}{B}\\sum_i w_{y_i}\\,\\mathrm{CE}_i\\)입니다. 가중치 평균이 1이라 가중치 합으로 다시 나누지 않습니다. 가장 작은 라벨(gemini-3.7-flash)의 가중치는 약 3.6, 가장 큰 라벨(gemini-3.8-flash)은 약 0.55입니다.
- **`match_weight`를 손실에 쓰지 않는 이유.** 데이터셋 설명에 "추출용 가중치이고, 시험해 보지 않았다면 손실 가중치로 쓰지 말라"고 적혀 있습니다. 라벨마다 Kish 유효 표본 비율 \\((\\sum w)^2 / (n \\sum w^2)\\)이 0.31 ~ 0.68이라, 손실에 곱하면 풀 전체를 넣어도 실제로는 절반 정도만 쓰는 셈이기도 합니다. 대회 지문의 표기에 맞추는 일은 train으로 하는 Stage 2에 맡깁니다.
- **프롬프트 id가 같은 86행.** 출처(`bingli283/ntl-gen-pilot-v1`)가 train 프롬프트를 다시 생성한 것으로 보이는 행입니다(`analysis_pool/logs/leak_check.log`, 같은 번호의 train 프롬프트와 가장 가까움). Stage 1이 검증 fold의 프롬프트를 미리 보지 않도록 뺍니다. 주제만 가까운 풀 행(약 7%)은 v8까지의 표본에도 같은 비율로 있었으므로 그대로 둡니다.
- **이어 하기.** Stage 1 체크포인트에는 학습 파라미터, AdamW·스케줄러·GradScaler 상태, 몇 번째 마이크로 배치까지 썼는지가 들어갑니다. 배치 순서는 seed로 정해지므로 같은 순서로 이어 갑니다. 저장은 가중치 갱신 직후에만 하므로 누적 중간에서 끊기지 않습니다. 풀 행 수나 스텝 수가 다르면 이어 하지 않고 멈춥니다.
- **가중치 파일.** `ntl_lora_v9_stage1.pt`, `ntl_lora_v9_fold{k}.pt`, `ntl_lora_v9_full.pt`로 이름을 바꿨습니다. 저장된 가중치의 `lora_pool_mode`가 지금 설정과 다르면 불러오지 않고 멈춥니다. v4 가중치는 이 값이 없어 `"sample"`로 읽히므로 v9 설정에서는 쓸 수 없습니다.
- **메모리.** 토큰 id를 지문마다 int32 배열로 보관하고 1만 개씩 나눠 토크나이즈합니다. 값은 v8과 같습니다.
- **예상 시간 (추정).** v3의 Stage 1(45,000개, echo 없음)이 L4에서 31분이었습니다. 풀 전체는 5.1배, echo로 길이가 2배라 L4에서 수 시간, T4에서는 그 몇 배가 걸릴 수 있습니다. 첫 로그의 스텝당 시간으로 다시 추정합니다.

### v4: echo 입력
""")
sub(17, "- 가중치는 `ntl_lora_stage1.pt`와 `ntl_lora_fold0.pt` ~ `ntl_lora_fold4.pt`로 저장됩니다.",
    "- 가중치는 `ntl_lora_stage1.pt`와 `ntl_lora_fold0.pt` ~ `ntl_lora_fold4.pt`로 저장됩니다(v9부터는 이름에 `v9_`가 붙음).")

# 18. LoRA 코드
sub(18, '''def tokenize_all(tok, texts):
    """토큰 id 목록과, 풀링을 시작할 위치(echo면 두 번째 사본의 시작, 아니면 0)를 돌려줍니다."""
    enc = tok(list(texts), truncation=True, max_length=CFG["max_len"], add_special_tokens=False)["input_ids"]
    enc = [ids if ids else [tok.pad_token_id] for ids in enc]
    if not CFG["echo_input"]:
        return enc, np.zeros(len(enc), dtype=int)
    sep = tok(ECHO_SEP, add_special_tokens=False)["input_ids"]
    return [ids + sep + ids for ids in enc], np.array([len(ids) + len(sep) for ids in enc])''',
    '''def tokenize_all(tok, texts, chunk=10000):
    """토큰 id 배열(int32) 목록과, 풀링을 시작할 위치(echo면 두 번째 사본의 시작, 아니면 0)를 돌려줍니다.
    v9: 풀 전체(약 23만 개)를 다루므로 나눠서 토크나이즈하고 int32 배열로 보관합니다(값은 v8과 같음)."""
    texts = list(texts)
    sep = tok(ECHO_SEP, add_special_tokens=False)["input_ids"] if CFG["echo_input"] else []
    out, starts = [], []
    for i in range(0, len(texts), chunk):
        enc = tok(texts[i:i + chunk], truncation=True, max_length=CFG["max_len"], add_special_tokens=False)["input_ids"]
        for ids in enc:
            ids = ids if ids else [tok.pad_token_id]
            if CFG["echo_input"]:
                out.append(np.asarray(ids + sep + ids, dtype=np.int32))
                starts.append(len(ids) + len(sep))
            else:
                out.append(np.asarray(ids, dtype=np.int32))
                starts.append(0)
    return out, np.array(starts, dtype=int)''')
sub(18, "        ids[row, :len(s)] = torch.tensor(s)\n", "        ids[row, :len(s)] = torch.from_numpy(s)\n")

sub(18, '''def train_stage(model, tok, device, texts, labels, epochs, lr, budget_h, name):
    params = [p for p in model.parameters() if p.requires_grad]''',
    '''def train_stage(model, tok, device, texts, labels, epochs, lr, budget_h, name, weights=None, ckpt_path=None):
    """weights: 행마다 손실 가중치(None이면 모두 1). ckpt_path: 주면 CFG["stage1_ckpt_min"]분마다 저장하고, 파일이 있으면 이어 합니다(v9)."""
    params = [p for p in model.parameters() if p.requires_grad]''')
sub(18, '''    labels = torch.tensor(np.array(labels), dtype=torch.long)
    start, step, skipped, running = time.time(), 0, 0, 0.0
    model.train()
    optim.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        batches = make_batches(lengths, CFG["batch_size"], True, seed=CFG["seed"] + epoch)
        for bi, batch in enumerate(batches):
            if time.time() - start > budget_h * 3600:''',
    '''    labels = torch.tensor(np.array(labels), dtype=torch.long)
    if weights is not None:
        weights = torch.tensor(np.asarray(weights, dtype=np.float32))
        assert len(weights) == len(labels)
    start, step, skipped, running = time.time(), 0, 0, 0.0
    resume_epoch, resume_bi = 0, 0
    use_ckpt = ckpt_path is not None and CFG["stage1_ckpt_min"] > 0
    if use_ckpt and Path(ckpt_path).exists():
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ck["n_rows"] == len(texts) and ck["n_steps"] == n_steps, (
            f"{ckpt_path}는 행 {ck['n_rows']:,}개 / {ck['n_steps']}스텝 학습의 체크포인트인데 지금은 {len(texts):,}개 / {n_steps}스텝입니다.")
        missing = model.load_state_dict(ck["state"], strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
        optim.load_state_dict(ck["optim"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        resume_epoch, resume_bi = ck["epoch"], ck["bi"]
        step, skipped, running = ck["step"], ck["skipped"], ck["running"]
        print(f"[{name}] 체크포인트에서 이어 하기: {step}/{n_steps} 스텝 (epoch {resume_epoch}, 마이크로 배치 {resume_bi}부터), "
              f"loss {running:.3f}", flush=True)
    last_ck = time.time()
    log_every = max(1, n_steps // (50 if use_ckpt else 10))
    model.train()
    optim.zero_grad(set_to_none=True)
    for epoch in range(resume_epoch, epochs):
        batches = make_batches(lengths, CFG["batch_size"], True, seed=CFG["seed"] + epoch)
        for bi, batch in enumerate(batches):
            if epoch == resume_epoch and bi < resume_bi:
                continue
            if time.time() - start > budget_h * 3600:''')
sub(18, '''            loss = F.cross_entropy(model(ids, mask, pool), labels[batch].to(device), label_smoothing=0.05)
''', '''            if weights is None:
                loss = F.cross_entropy(model(ids, mask, pool), labels[batch].to(device), label_smoothing=0.05)
            else:  # v9: 가중치 평균이 1이므로 배치 크기로만 나눕니다(가중치 합으로 다시 나누지 않음).
                per_row = F.cross_entropy(model(ids, mask, pool), labels[batch].to(device), label_smoothing=0.05, reduction="none")
                loss = (per_row * weights[batch].to(device)).mean()
''')
sub(18, '''            step += 1
            if step % max(1, n_steps // 10) == 0:
                print(f"[{name}] {step}/{n_steps} loss {running:.3f} | {(time.time() - start) / 60:.1f}분")
    print(f"[{name}] 완료: {step}스텝, 비정상 손실로 건너뜀 {skipped}회, {(time.time() - start) / 60:.1f}분")''',
    '''            step += 1
            if step % log_every == 0:
                print(f"[{name}] {step}/{n_steps} loss {running:.3f} | {(time.time() - start) / 60:.1f}분", flush=True)
            if use_ckpt and time.time() - last_ck > CFG["stage1_ckpt_min"] * 60 and bi + 1 < len(batches):
                # 가중치 갱신 직후에만 저장하므로 누적 중간에서 끊기지 않습니다. 다음 실행은 마이크로 배치 bi + 1부터 시작합니다.
                torch.save({"state": trainable_state(model), "optim": optim.state_dict(), "sched": sched.state_dict(),
                            "scaler": scaler.state_dict(), "epoch": epoch, "bi": bi + 1, "step": step, "skipped": skipped,
                            "running": float(running), "n_rows": len(texts), "n_steps": n_steps}, str(ckpt_path) + ".tmp")
                os.replace(str(ckpt_path) + ".tmp", ckpt_path)
                last_ck = time.time()
                print(f"[{name}] 체크포인트 저장: {step}/{n_steps} 스텝 | {(time.time() - start) / 60:.1f}분", flush=True)
    print(f"[{name}] 완료: {step}스텝, 비정상 손실로 건너뜀 {skipped}회, {(time.time() - start) / 60:.1f}분")''')

sub(18, '''FOLD_FILES = [f"ntl_lora_fold{k}.pt" for k in range(CFG["n_folds"])]
FULL_FILE = "ntl_lora_full.pt"''',
    '''# v9: v4 가중치(ntl_lora_fold*.pt)와 섞이지 않도록 이름을 바꿈
FOLD_FILES = [f"ntl_lora_v9_fold{k}.pt" for k in range(CFG["n_folds"])]
FULL_FILE = "ntl_lora_v9_full.pt"
STAGE1_FILE = "ntl_lora_v9_stage1.pt"
STAGE1_CKPT = "ntl_lora_v9_stage1_ckpt.pt"
LORA_OOF_FILE = "ntl_lora_v9_oof.npz"''')
sub(18, '''    assert saved_echo == CFG["echo_input"], (
        f"{path}는 echo_input={saved_echo}로 학습된 가중치인데 지금 설정은 {CFG['echo_input']}입니다.")''',
    '''    assert saved_echo == CFG["echo_input"], (
        f"{path}는 echo_input={saved_echo}로 학습된 가중치인데 지금 설정은 {CFG['echo_input']}입니다.")
    saved_mode = ckpt["cfg"].get("lora_pool_mode", "sample")
    assert saved_mode == CFG["lora_pool_mode"], (
        f"{path}는 lora_pool_mode={saved_mode!r}로 학습된 가중치인데 지금 설정은 {CFG['lora_pool_mode']!r}입니다.")''')
sub(18, '''    torch.save({"state": stage1_state, "cfg": CFG, "labels": LABELS}, WORK_DIR / "ntl_lora_stage1.pt")''',
    '''    torch.save({"state": stage1_state, "cfg": CFG, "labels": LABELS}, WORK_DIR / STAGE1_FILE)''')
sub(18, '''    for k, (a, b) in enumerate(cv_folds):
        model.load_state_dict(stage1_state, strict=False)
        train_stage(model, tok, device, train.text.values[a].tolist(), y[a],
                    CFG["stage2_epochs"], CFG["stage2_lr"], CFG["stage2_budget_h"], f"stage2-fold{k}")
        torch.save({"state": trainable_state(model), "cfg": CFG, "labels": LABELS}, WORK_DIR / FOLD_FILES[k])''',
    '''    for k, (a, b) in enumerate(cv_folds):
        if (WORK_DIR / FOLD_FILES[k]).exists():  # v9: 이어 하기. 이 실행 폴더에 이미 저장된 fold는 다시 학습하지 않음
            print(f"fold {k}: 저장된 가중치 {WORK_DIR / FOLD_FILES[k]}를 사용합니다.")
            load_lora(model, WORK_DIR / FOLD_FILES[k])
        else:
            model.load_state_dict(stage1_state, strict=False)
            train_stage(model, tok, device, train.text.values[a].tolist(), y[a],
                        CFG["stage2_epochs"], CFG["stage2_lr"], CFG["stage2_budget_h"], f"stage2-fold{k}")
            torch.save({"state": trainable_state(model), "cfg": CFG, "labels": LABELS}, WORK_DIR / FOLD_FILES[k])''')
sub(18, '''        load_lora(model, WORK_DIR / "ntl_lora_stage1.pt")''', '''        load_lora(model, WORK_DIR / STAGE1_FILE)''')
s18 = get(18)
s18 += '''

def lora_stage1_pool():
    """v9: Stage 1 데이터와 행 가중치. lora_pool_mode="full"이면 풀 전체, "sample"이면 v8까지와 같은 추출."""
    if pool is None:
        return None, None
    if CFG["lora_pool_mode"] == "full":
        lp = pool
    else:
        assert CFG["lora_pool_mode"] == "sample", CFG["lora_pool_mode"]
        lp = sample_pool(CFG["lora_pool_per_label"], seed=CFG["seed"])
    if CFG["lora_drop_train_prompt_ids"] and "prompt_group" in lp.columns:
        tail = lp.prompt_group.fillna("").astype(str).str.split(":").str[-1]
        drop = tail.isin(set(train.prompt_id.astype(str)))
        print(f"Stage 1: 출처의 프롬프트 id가 train prompt_id와 같은 풀 행 {int(drop.sum())}개를 뺍니다.")
        lp = lp[~drop.values]
    lp = lp.reset_index(drop=True)
    yl = lp.model.map(L2I).values
    counts = np.bincount(yl, minlength=K)
    w = None
    if CFG["lora_pool_balance"]:
        w = (len(yl) / (counts > 0).sum() / np.maximum(counts, 1))[yl]
        sums = np.bincount(yl, weights=w, minlength=K)
        print(f"Stage 1 행 가중치(라벨 균형): 최소 {w.min():.3f} / 최대 {w.max():.3f} / 평균 {w.mean():.3f} | "
              f"라벨별 가중치 합 {sums.min():,.1f} ~ {sums.max():,.1f}")
    print(f"Stage 1 데이터 ({CFG['lora_pool_mode']}): {len(lp):,}행 | 라벨별 최소 {counts.min():,} / 최대 {counts.max():,} | "
          + ", ".join(f"{LABELS[k]} {c:,}" for k, c in enumerate(counts)))
    return lp, w
'''
put(18, s18)

# 19. LoRA 실행 셀
sub(19, '''        lora_pool = sample_pool(CFG["lora_pool_per_label"], seed=CFG["seed"])
        if lora_pool is not None:
            train_stage(model, tok, device, lora_pool.text.tolist(), lora_pool.model.map(L2I).values,
                        CFG["stage1_epochs"], CFG["stage1_lr"], CFG["stage1_budget_h"], "stage1")
            del lora_pool
            gc.collect()
        lora_oof, lora_test = cv_after_stage1(model, tok, device)''',
    '''        stage1_saved = find_optional(STAGE1_FILE)
        if stage1_saved is None and (WORK_DIR / STAGE1_FILE).exists():
            stage1_saved = WORK_DIR / STAGE1_FILE
        if stage1_saved is not None:
            print("저장된 Stage 1 가중치를 사용합니다:", stage1_saved)
            load_lora(model, stage1_saved)
        else:
            lora_pool, lora_w = lora_stage1_pool()
            if lora_pool is not None:
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                train_stage(model, tok, device, lora_pool.text.tolist(), lora_pool.model.map(L2I).values,
                            CFG["stage1_epochs"], CFG["stage1_lr"], CFG["stage1_budget_h"], "stage1",
                            weights=lora_w, ckpt_path=WORK_DIR / STAGE1_CKPT)
                torch.save({"state": trainable_state(model), "cfg": CFG, "labels": LABELS}, WORK_DIR / STAGE1_FILE)
                (WORK_DIR / STAGE1_CKPT).unlink(missing_ok=True)  # Stage 1 가중치를 저장했으므로 체크포인트는 지움
                if device.type == "cuda":
                    print(f"Stage 1 GPU 메모리 최대 사용량: {torch.cuda.max_memory_allocated() / 2**30:.1f}GB")
                del lora_pool, lora_w
                gc.collect()
        lora_oof, lora_test = cv_after_stage1(model, tok, device)''')
sub(19, '''    print(f"LoRA OOF BA: {balanced_accuracy_score(y, lora_oof.argmax(1)):.4f} "
          f"(fold 0: {balanced_accuracy_score(y[holdout_idx], lora_oof[holdout_idx].argmax(1)):.4f}) | {elapsed()}")''',
    '''    print(f"LoRA OOF BA: {balanced_accuracy_score(y, lora_oof.argmax(1)):.4f} "
          f"(fold 0: {balanced_accuracy_score(y[holdout_idx], lora_oof[holdout_idx].argmax(1)):.4f}) | {elapsed()}")
    # v9: 블렌드 가중치를 따로 다시 고를 수 있게 LoRA 결과를 바로 저장합니다.
    fold_of = np.zeros(len(train), dtype=int)
    for k, (_, b) in enumerate(cv_folds):
        fold_of[b] = k
    np.savez(WORK_DIR / LORA_OOF_FILE, oof=lora_oof, test=lora_test, y=y, fold=fold_of,
             ids=train.id.astype(str).values.astype("U"), test_ids=test.id.astype(str).values.astype("U"))
    print("저장:", WORK_DIR / LORA_OOF_FILE)''')

# 24. 블렌드 저장 파일 이름
sub(24, 'np.savez(WORK_DIR / "oof_v8.npz"', 'np.savez(WORK_DIR / "oof_v9.npz"')

for c in cells:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
        c["outputs"] = []
        c["execution_count"] = None
dst = ROOT / "name_that_llm_v9.ipynb"
with open(dst, "w", encoding="utf-8") as f:
    f.write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("저장:", dst)
