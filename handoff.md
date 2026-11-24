# Handoff: v5~v7 실험을 서버에서 이어서 하기

작성 2026-11-24. v5~v7 실험은 서버에서 끝까지 진행합니다(Colab 학습, Kaggle 제출 포함). 이 문서는 지금 상태와 바로 다음에 할 일을 정리한 것입니다.

## 1. 지금 상태

| 항목 | 상태 |
|---|---|
| v5 노트북 | [name_that_llm_v5.ipynb](name_that_llm_v5.ipynb). v4 full에서 n-gram 분기에 문체 피처 27개(α = 0.2)만 추가 |
| v5 Kaggle 커널 | `blkbeansoymilk/name-that-llm-v5-stylo` **버전 1 실행 중** (2026-11-24 22:18 KST쯤 푸시, v4 기준 약 33분) |
| v5 제출 | **아직 안 함** |
| EXPERIMENTS.md | **아직 v5 내용 없음.** 따로 파일을 만들지 않고 기존 [EXPERIMENTS.md](EXPERIMENTS.md)에 "실험 5"로 추가하기로 함 |
| v6 | [name_that_llm_v6.ipynb](name_that_llm_v6.ipynb). v4 + 블렌드 뒤 **회사 게이트**(회사 안 분포를 회사 전용 LR 쪽으로 옮김). EXPERIMENTS.md 실험 6에 기록됨. v4 OOF로 잰 nested BA 0.5749 → 0.5864. **미제출** |
| v7 | [name_that_llm_v7.ipynb](name_that_llm_v7.ipynb). v6 + **DeBERTa-v3-large 세 번째 분기**. **아직 실행 안 함.** DeBERTa는 Colab L4에서 학습하고 결과를 Kaggle 데이터셋으로 올려 제출 커널에서는 추론만 함 |

> [!NOTE]
> v6과 v7은 다른 세션에서 만들어 이 커밋 전에 main에 올라온 것입니다. **v5는 v4에서 따로 갈라진 실험이라 v6·v7에는 문체 블록이 없습니다.** v5가 좋으면 v6·v7의 n-gram 셀에도 같은 블록을 넣을지 정해야 합니다. 회사 게이트의 `gate_alpha`와 문체 블록의 `stylo_alpha`는 이름만 비슷하고 서로 다른 값입니다.

## 2. v5에서 바꾼 것

n-gram 분기(4뷰 TF-IDF + LinearSVC)의 행렬 옆에 문체 피처 블록을 이어 붙였습니다. LoRA 분기, 블렌드(`w` = 0.6), 중심화, Sinkhorn은 v4와 같습니다.

- **피처**: [analysis_intra_org/features.py](analysis_intra_org/features.py)의 `feats` 27개(문장 길이 평균·표준편차, 단어 길이, MATTR, 기능어 비율, 100단어당 구두점, 축약형 등). 노트북 안의 `stylo_feats`와 값이 같음을 확인함(최대 차이 0.0).
- **스케일링**: `log1p` → fold의 train 지문으로 z-score 표준화(풀은 쓰지 않음) → ±4에서 자름 → `α / √27` 곱하기. 표준화한 27차원 행 벡터의 L2 노름 기댓값이 √27 ≈ 5.2이므로, 곱한 뒤의 L2 노름은 약 α입니다. TF-IDF 블록의 행 L2 노름은 √(1.0² + 0.7² + 0.5² + 0.5²) ≈ 1.4입니다.
- **설정**: `CFG["stylo_alpha"] = 0.2`(0이면 v4와 같음), `CFG["lora_test"] = "full"`.
- **LoRA**: 데이터셋 `ntl-lora-weights-v4-echo-cv`의 fold 5개와 full 가중치를 그대로 씁니다(재학습 없음).

### α를 0.2로 고른 근거 (train만, 풀 없음, 5-fold)

[v5_stylo/stylo_tfidf.py](v5_stylo/stylo_tfidf.py)로 측정했습니다.

| 설정 | OOF BA |
|---|:-:|
| 기준: 4뷰 TF-IDF | 0.3704 |
| + 연속 문체 27개, α = 0.15 / **0.2** / 0.3 / 0.4 / 1.0 | 0.3845 / **0.3864** / 0.3826 / 0.3786 / 0.3697 |
| + 구조 피처 10개만 / 구두점·어휘 17개만 (α = 0.3) | 0.3801 / 0.3741 |
| + 분위수 구간 one-hot (α = 0.3) | 0.3749 |

## 3. 비교 기준: Open Pool이 바뀌었음

> [!IMPORTANT]
> Open Pool이 v4 실행(11/17, 226,191개) 뒤에 갱신되어 지금은 **231,036개**입니다. 같은 v4 n-gram 코드를 새 풀로 다시 돌리면 OOF가 0.4745에서 **0.4641**로 떨어집니다. 그래서 v5는 v4 로그 숫자가 아니라 **새 풀 기준선**과 비교해야 합니다.

| 설정 | 풀 | n-gram OOF | 블렌드 OOF (w = 0.6) | fold 0 블렌드 |
|---|---|:-:|:-:|:-:|
| v4 (11/17 Kaggle 로그) | 226,191 | 0.4745 | 0.5749 | 0.5593 |
| v4 n-gram, 새 풀 (α = 0, 로컬 CPU) | 231,036 | **0.4641** | **0.5679** | 0.5463 |
| v5 (α = 0.2) | 231,036 | 커널 로그에서 | 커널 로그에서 | |

- 새 풀 기준선의 OOF와 로그: [v5_stylo/ngram_oof_alpha0_pool231k.npz](v5_stylo/ngram_oof_alpha0_pool231k.npz), [v5_stylo/baseline_alpha0_run.log](v5_stylo/baseline_alpha0_run.log). train 가중치 1 / 4 / 12 → 0.4604 / 0.4630 / 0.4641
- 블렌드는 [v5_stylo/blend.py](v5_stylo/blend.py)로 계산합니다. n-gram OOF npz와 v4 LoRA OOF([analysis_v4/oof_v4.npz](analysis_v4/oof_v4.npz))로 노트북과 같은 온도 선택과 블렌드를 합니다. v4 원래 n-gram OOF를 넣으면 0.4745 / 0.5749 / 0.5593이 그대로 재현됩니다.
- 새 풀 기준선 블렌드의 `w`별 OOF: 0.5 → 0.5593, 0.6 → 0.5679, 0.7 → 0.5753, 0.8 → 0.5764, 0.9 → 0.5775. 새 풀에서는 최고점이 `w` = 0.9 쪽으로 옮겨 갔습니다.

## 4. 바로 다음에 할 일

### 4.1 v5 마무리

```bash
# 1) 커널 상태
kaggle kernels status blkbeansoymilk/name-that-llm-v5-stylo

# 2) 끝나면 로그와 submission.parquet 받기
kaggle kernels output blkbeansoymilk/name-that-llm-v5-stylo -p out/v5

# 3) 제출 (코드 대회라 커널 버전을 지정해 제출)
kaggle competitions submit -c name-that-llm -k blkbeansoymilk/name-that-llm-v5-stylo -v 1 \
  -f submission.parquet -m "v5: v4 full + stylometric 27 (alpha=0.2) in n-gram branch, OOF blend BA <로그 값>"

# 4) 점수 확인
kaggle competitions submissions -c name-that-llm | head
```

- 로그에서 볼 줄: `train 가중치 ...: OOF BA`, `문체 피처 27개, alpha=0.2`, `LoRA 가중치별 OOF BA`, `최종 검증 BA`.
- 커널이 실패하면 로그 끝의 에러를 봅니다. 노트북에서 v4와 다른 곳은 3장 n-gram 코드 셀과 `CFG`뿐입니다. 커널을 다시 만들려면 [v5_stylo/kernel/kernel-metadata.json](v5_stylo/kernel/kernel-metadata.json)이 있는 폴더에 노트북을 `name-that-llm-v5-stylo.ipynb`라는 이름으로 복사한 뒤 `kaggle kernels push -p <폴더>`를 실행합니다.
- 노트북을 고쳤다면 [v5_stylo/make_v5.py](v5_stylo/make_v5.py)(v4 → v5 변환 스크립트)도 같이 고치거나, 노트북을 직접 고칩니다.

### 4.2 EXPERIMENTS.md에 "실험 5" 추가

- 실험 6(v6) **앞에** 넣습니다. 기존 형식(요약 → 변한 점 → 실험 세팅 → 결과 → 해석)을 따릅니다. "한눈에 보기" 표와 "정리" 표에도 v5 행을 추가합니다.
- 수치는 **제출 커널(Kaggle T4) 로그 기준**입니다(문서 머리의 NOTE).
- **풀 갱신 때문에 기준선이 바뀐 점**(3장)을 꼭 적습니다. v5의 효과는 새 풀 기준선(n-gram 0.4641, 블렌드 0.5679)과 비교한 값입니다.

### 4.3 v6, v7

1. **v6 제출**: 노트북은 있고 아직 제출하지 않았습니다. 실험 6의 수치(0.5864)는 **옛 풀**로 만든 v4 OOF로 잰 값입니다. Kaggle에서 돌리면 새 풀(231,036개) 때문에 n-gram이 달라지니, 커널 로그 수치로 다시 기록합니다. v6 노트북의 `lora_test`는 `"folds"`입니다(v4 LB는 full이 0.002 높았음).
2. **v7**: DeBERTa를 Colab L4에서 학습 → 결과를 Kaggle 데이터셋으로 업로드 → 제출 커널에서 추론만 합니다. 자세한 순서는 v7 노트북 머리말과 4-2장을 봅니다.
3. **v5와 합치기**: v5가 새 풀 기준선(블렌드 0.5679)보다 나으면, v6·v7의 n-gram 셀에 문체 블록을 넣는 것을 검토합니다. v4 → v5 변경은 [v5_stylo/make_v5.py](v5_stylo/make_v5.py)의 3장(셀 16)과 `CFG` 부분입니다.

### 4.4 그 밖의 후보 (확정 아님)

1. **α 다시 고르기**: 풀을 넣은 실제 분기에서 `stylo_alpha` ∈ {0.1, 0.2, 0.3}을 OOF로 비교합니다(CPU만, 1회 약 15분). 풀을 넣으면 TF-IDF 쪽이 강해져 최적값이 달라질 수 있습니다.
2. **`n_words`, `n_sents` 빼기**: 풀과 train의 모델별 길이 분포가 다르면 SVC가 이 두 피처를 지름길로 배울 수 있습니다. 먼저 분포를 비교합니다.
3. **블렌드 가중치 다시 보기**: 새 풀에서는 `w` = 0.6이 OOF 최고(0.9)보다 0.01쯤 낮습니다.
4. **헷갈리는 쌍 손실**: [analysis_intra_org/experiment_plan.md](analysis_intra_org/experiment_plan.md). 0단계(쌍별 bias, GPU 불필요) → 1단계(LoRA Stage 2에 쌍 손실 + KL 규제, Colab L4)입니다.

## 5. 서버 준비

- **Kaggle 인증**: `~/.kaggle/credentials.json` (로그인은 사용자가 직접 함)
- **대회 데이터** (`data/`는 git 제외): `kaggle competitions download -c name-that-llm -p data && unzip -o data/name-that-llm.zip -d data`
- **Open Pool**: `kaggle datasets download crtproduction/name-that-llm-open-pool -f ntl_open_pool.parquet -p data/open-pool`
- **로컬 n-gram 실행**: 노트북 코드 셀을 이어 붙여 실행하고, `NTL_INPUT=<data 폴더> NTL_CFG='{"use_lora": false}'`를 줍니다. LoRA 셀이 `torch`를 import하므로 torch가 없으면 LoRA 셀 앞에서 끊습니다.
- **LoRA 가중치**: `*.pt`는 git 제외입니다. 데이터셋 `blkbeansoymilk/ntl-lora-weights-v4-echo-cv`(fold 5개 + full)에서 받습니다.
- **Colab**: [README.md](README.md)의 실행 방법과 [colab_setup.py](colab_setup.py)를 봅니다. 긴 학습은 VM 안에서 `nohup`으로 돌리고, 결과 파일을 `/content/out`에 남깁니다. 랩탑 연결이 끊겨도 끝까지 돌게 하기 위해서입니다.

## 6. 저장소 규칙

- 브랜치를 만들지 않고 main에 바로 커밋하고 푸시합니다.
- 실험 기록은 [EXPERIMENTS.md](EXPERIMENTS.md) 한 파일에 이어서 씁니다.
