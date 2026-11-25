# Handoff: v5~v7 실험을 서버에서 이어서 하기

작성 2026-11-24. v5~v7 실험은 서버에서 끝까지 진행합니다(Colab 학습, Kaggle 제출 포함). 이 문서는 지금 상태와 바로 다음에 할 일을 정리한 것입니다.

## 1. 지금 상태

| 항목 | 상태 |
|---|---|
| v5 노트북 | [name_that_llm_v5.ipynb](name_that_llm_v5.ipynb). v4 full에서 n-gram 분기에 문체 피처 27개(α = 0.2)만 추가 |
| v5 Kaggle 커널 | `blkbeansoymilk/name-that-llm-v5-stylo` **버전 1 완료** (2026-11-24 22:18 KST쯤 푸시, 34.6분) |
| v5 제출 | **제출함** (2026-11-24). Public LB **0.550** (v4 full 0.545보다 +0.005) |
| EXPERIMENTS.md | **"실험 5"로 기록함** (커밋 `ffad936`). 따로 파일을 만들지 않고 기존 [EXPERIMENTS.md](EXPERIMENTS.md)에 추가함 |
| v6 | [name_that_llm_v6.ipynb](name_that_llm_v6.ipynb). v4 full + 블렌드 뒤 **회사 게이트**("5-1. 회사 게이트" 셀 하나, 문체 블록 없음). Kaggle 커널 `blkbeansoymilk/name-that-llm-v6-gate` **버전 1 완료**(2026-11-25, 32.5분, 새 풀 231,036개). **제출함**, Public LB **0.552**(v4 full 0.545보다 +0.007, v5 0.550보다 +0.002). 커널의 블렌드 OOF는 0.5719 → 0.5786이고, 새 풀 nested로는 +0.0044(95% 구간이 0을 포함)입니다. 옛 v4 OOF로 잰 nested는 0.5749 → 0.5864였습니다. **EXPERIMENTS.md 실험 6을 제출 결과로 갱신함** |
| v7 | [name_that_llm_v7.ipynb](name_that_llm_v7.ipynb). v6 + **DeBERTa-v3-large 세 번째 분기**(5장 블렌드 3분기, 5-1장 게이트 입력에 DeBERTa 추가). DeBERTa는 **Colab T4에서 학습함**(2026-11-25, 198분, OOF BA 0.5053). Kaggle 데이터셋 `blkbeansoymilk/ntl-deberta-v7`, 커널 `blkbeansoymilk/name-that-llm-v7-deberta` **버전 1 완료**(19.0분). **제출함**, Public LB **0.556**(v6 0.552보다 +0.004, v4 full 0.545보다 +0.011). 커널의 3분기 블렌드 OOF는 0.5868(같은 커널의 v6 가중치 0.5719, 가중치 nested로는 0.5827), 게이트 뒤 0.5882입니다. **EXPERIMENTS.md 실험 7에 기록함**. 진행 중인 Colab 세션이나 커널은 없습니다 |
| v8 | [name_that_llm_v8.ipynb](name_that_llm_v8.ipynb). v7에서 **n-gram 분기만** 바꿈: v5의 문체 피처 27개(α = 0.2) + n-gram용 Open Pool 표본 라벨당 2,500 → **5,000개**(라벨 균형 가중치, 88,107행). LoRA·DeBERTa는 다시 학습하지 않았고 `blend3`·`gate_alpha`도 v7 그대로입니다. Kaggle 커널 `blkbeansoymilk/name-that-llm-v8-pool` **버전 1 완료**(2026-11-25, 33.9분). **제출함**, Public LB **0.563**(v7 0.556보다 +0.007, 지금까지 최고). 커널의 n-gram OOF는 0.4693 → 0.5016, 3분기 블렌드 0.5868 → 0.5982, 게이트 뒤 0.5882 → 0.6023입니다(로컬 예상 0.5012 / 0.5982 / 0.6020). OOF로는 블렌드 이득 +0.0115 중 풀 크기가 +0.0096, 문체 블록이 +0.0018이고 LB로는 둘을 나눌 수 없습니다. 근거 실험은 `analysis_pool/`, 노트북 생성·점검·커널 로그는 `analysis_v8/`. **EXPERIMENTS.md 실험 8에 기록함** |

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
> Open Pool이 v4 실행(11/17, 226,191개) 뒤에 갱신되어 지금은 **231,036개**입니다. 같은 v4 n-gram 코드를 새 풀로 다시 돌리면 OOF가 0.4745에서 **0.4690**으로 떨어집니다(커널과 같은 fold 분할 기준). 그래서 v5는 v4 로그 숫자가 아니라 **새 풀 기준선**과 비교해야 합니다.

| 설정 | 풀 | n-gram OOF | 블렌드 OOF (w = 0.6) | fold 0 블렌드 |
|---|---|:-:|:-:|:-:|
| v4 (11/17 Kaggle 로그) | 226,191 | 0.4745 | 0.5749 | 0.5593 |
| v4 n-gram, 새 풀 (α = 0, 로컬 CPU, 커널 fold 분할) | 231,036 | **0.4690** | **0.5716** | 0.5537 |
| v5 (α = 0.2, Kaggle 커널 로그) | 231,036 | 0.4749 | 0.5734 | 0.5593 |

- 새 풀 기준선의 OOF와 로그: [v5_stylo/ngram_oof_alpha0_kernelfold.npz](v5_stylo/ngram_oof_alpha0_kernelfold.npz), [v5_stylo/alpha_sweep_run.log](v5_stylo/alpha_sweep_run.log). train 가중치는 12만 쟀습니다. 같은 방법으로 α = 0.2를 돌리면 n-gram 0.4756, 블렌드 0.5738로 커널 수치가 거의 재현됩니다([v5_stylo/ngram_oof_alpha0.2_kernelfold.npz](v5_stylo/ngram_oof_alpha0.2_kernelfold.npz)).
- **fold 분할 주의**: `StratifiedGroupKFold(shuffle=True, random_state=42)`의 분할이 로컬(scikit-learn 1.9.1)과 제출 커널에서 다릅니다(scikit-learn 버전 차이로 보임, 추정). 로컬에서 OOF를 비교할 때는 [analysis_v4/oof_v4.npz](analysis_v4/oof_v4.npz)에 저장된 `fold`를 써야 합니다([v5_stylo/alpha_sweep.py](v5_stylo/alpha_sweep.py)가 그렇게 합니다). 노트북 첫 출력 "프롬프트 그룹 KFold BA"가 0.3145이면 커널과 같은 분할이고, 로컬 분할에서는 0.3115가 나옵니다.
- 처음 잰 기준선(n-gram 0.4641, 블렌드 0.5679, fold 0 0.5463, train 가중치 1 / 4 / 12 → 0.4604 / 0.4630 / 0.4641)은 **커널과 다른 로컬 fold 분할로 잰 값이라 비교에 쓰지 않습니다.** 파일은 [v5_stylo/ngram_oof_alpha0_pool231k.npz](v5_stylo/ngram_oof_alpha0_pool231k.npz), [v5_stylo/baseline_alpha0_run.log](v5_stylo/baseline_alpha0_run.log)로 남아 있습니다.
- 블렌드는 [v5_stylo/blend.py](v5_stylo/blend.py)로 계산합니다. n-gram OOF npz와 v4 LoRA OOF([analysis_v4/oof_v4.npz](analysis_v4/oof_v4.npz))로 노트북과 같은 온도 선택과 블렌드를 합니다. v4 원래 n-gram OOF를 넣으면 0.4745 / 0.5749 / 0.5593이 그대로 재현됩니다.
- 새 풀 기준선 블렌드의 `w`별 OOF: 0.5 → 0.5649, 0.6 → 0.5716, 0.7 → 0.5742, 0.8 → 0.5775, 0.9 → 0.5731. 최고점은 옛 풀의 v4와 같은 `w` = 0.8입니다.

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
- **풀 갱신 때문에 기준선이 바뀐 점**(3장)을 꼭 적습니다. v5의 효과는 새 풀 기준선(n-gram 0.4690, 블렌드 0.5716)과 비교한 값입니다.

### 4.3 v6: 회사 게이트 제출

> [!NOTE]
> **2026-11-25 상태: 아래 순서는 모두 끝났습니다.** 이 절의 나머지는 제출 전에 쓴 내용 그대로입니다.
> - 커널 `name-that-llm-v6-gate` 버전 1(`lora_test="full"`, 32.5분)을 제출했고 Public LB는 **0.552**입니다(제출 ref 58845044).
> - 커널 로그: 풀 231,036개, n-gram OOF 0.4693, LoRA OOF 0.5631, 블렌드(`w` = 0.6) 0.5719 → 게이트 0.5786(+0.0067, fold 5개 모두 오름).
> - `gate_alpha`를 새 OOF로 다시 확인했습니다. nested 변화는 +0.0044(95% 구간 [−0.002, +0.011])로 옛 풀(+0.0115)의 절반쯤이고, 어떤 alpha를 써도 차이가 작아서 값은 그대로 뒀습니다.
> - 새 풀 OOF와 로그는 [analysis_v6/newpool/](analysis_v6/newpool/)에 있습니다(`oof_v6.npz`, `kernel.log`, `gate2.log`, `check_gate.log`). 노트북의 게이트 셀 뒤 셀이 `oof_v6.npz`를 저장합니다.
> - `gate2.py`와 `check_gate.py`는 OOF 경로를 세 번째 인자로 받습니다: `python analysis_v6/gate2.py . - analysis_v6/newpool/oof_v6.npz`. fold는 npz의 `fold`를 씁니다.
> - 자세한 수치와 해석은 [EXPERIMENTS.md](EXPERIMENTS.md) 실험 6에 있습니다.

v6 세션에서 받은 내용입니다. 재학습은 필요 없습니다. v4 노트북에 "5-1. 회사 게이트" 셀 하나만 더했고, LoRA는 저장된 가중치로 추론만 합니다.

**게이트가 하는 일.** 회사(GPT·Gemini·Claude·GLM)마다 회사 안 모델을 맞히는 로지스틱 회귀(C = 1)를 둡니다. 입력은 n-gram과 LoRA 각각의 회사 열 log 확률입니다. 블렌드 1순위가 그 회사 안인 지문만, 회사 확률 합은 그대로 두고 회사 안 분포를 비율 `gate_alpha`(gpt 0.75, gemini 0.25, claude 0.5, glm 1.0)만큼 LR 쪽으로 옮깁니다.

**현재 상태**
- 완료: 노트북, [analysis_v6/](analysis_v6/)(report.md, 스크립트 4개, `out/*.log`), EXPERIMENTS.md 실험 6. 모두 커밋 `1b6a68a`
- 진행 중인 Kaggle 커널이나 Colab 작업은 없습니다. v6 세션에 main에 안 올린 변경도 없습니다.
- 검증한 범위:
  - 게이트 셀만 v4 OOF로 실행해 OOF BA 0.5882를 재현했습니다.
  - `use_lora=false`로 노트북 전체를 CPU에서 끝까지 실행했습니다(이때 게이트는 건너뜀).
  - **LoRA와 게이트를 함께 켠 실행은 아직 없습니다.** 첫 Kaggle 실행이 곧 첫 통합 실행입니다.

**순서**
1. 커널을 새로 만듭니다. 이름 제안은 `name-that-llm-v6-gate`이고, 아직 없는 커널입니다. [v5_stylo/kernel/kernel-metadata.json](v5_stylo/kernel/kernel-metadata.json)을 복사해 `id`, `title`, `code_file`만 바꾸면 됩니다. 입력은 v4와 같습니다: 대회 `name-that-llm`, 데이터셋 `crtproduction/name-that-llm-open-pool`, `blkbeansoymilk/ntl-lora-weights-v4-echo-cv`, 모델 `qwen-lm/qwen-3/Transformers/4b-base/1`, GPU T4, Internet Off.
2. `kaggle kernels push -p <폴더>`로 전체를 실행합니다. 실행 시간은 v4와 비슷한 약 33분입니다(추정).
3. 로그의 "5-1" 출력을 확인합니다. 회사별 alpha 표와 `블렌드 OOF BA: v4 … -> 게이트 …` 한 줄입니다.
4. 4.1장의 v5와 같은 방식으로 제출합니다.

**설정**
- `lora_test`: v6 노트북의 기본값은 `"folds"`입니다. v4에서 LB가 0.002 높았던 full로 내려면 `"full"`로 바꿉니다. `ntl_lora_full.pt`가 데이터셋에 있어서 불러오기만 합니다.
- `use_gate=False`로 두면 v4와 같아집니다.

**수치를 읽을 때**
- 실험 6의 수치는 모두 [analysis_v4/oof_v4.npz](analysis_v4/oof_v4.npz)(옛 풀 226,191개 시점)로 계산했습니다.
  - 0.5864(+0.0115)는 nested 값입니다. fold 하나를 빼고 나머지 4개로 alpha를 골라 뺀 fold에 적용했습니다. 프롬프트 bootstrap 95% 구간은 +0.003 ~ +0.020입니다.
  - 노트북이 출력하는 0.5882는 alpha를 같은 OOF로 고른 값이라 낙관적입니다.
  - 둘 다 Sinkhorn 전 값이고, 게이트 + Sinkhorn도 0.5882입니다.
- **새 풀에서의 영향(v6 세션 의견, 추정)**:
  - 게이트의 LR은 커널 안에서 그 커널의 OOF로 다시 학습하므로, 새 풀로 바뀐 n-gram에 맞춰집니다. 옛 OOF에서 고정한 값은 `gate_alpha`와 C = 1뿐입니다.
  - 이득은 회사마다 n-gram과 LoRA를 다르게 다시 섞는 데서 나왔습니다. 그래서 n-gram이 약해져도 이득이 사라지지는 않겠지만 크기는 달라질 수 있습니다.
- **alpha를 다시 확인하는 방법**:
  1. 새 풀 커널에서 `y, fold, eval_lp, ng_oof_lp, lora_lp, lora_oof, T_LORA, labels, prompt_id`를 `oof_v4.npz`와 같은 키로 저장합니다.
  2. [analysis_v6/gate2.py](analysis_v6/gate2.py)와 [analysis_v6/check_gate.py](analysis_v6/check_gate.py)를 다시 돌립니다. 지금은 `analysis_v4/oof_v4.npz` 경로가 고정이라 경로를 바꿔야 합니다.
  3. nested ΔBA가 비슷하면 alpha를 그대로 둡니다. 많이 다르면 `CFG["gate_alpha"]`를 새 OOF로 다시 고릅니다.
- v4와 같은 조건으로 비교하고 싶다면, 커널 입력에서 Open Pool 데이터셋의 옛 버전을 고르는 방법도 있습니다.
- 게이트는 OOF와 테스트 로짓만 있으면 되는 후처리입니다. v5나 v7 블렌드 위에도 같은 셀을 얹을 수 있으며, 그때는 alpha를 그 버전의 OOF로 다시 고릅니다.

### 4.4 v7: DeBERTa 분기

> [!NOTE]
> **2026-11-25 상태: 끝났습니다.** 아래 내용은 실행 전에 쓴 계획이고, 실제 결과는 [EXPERIMENTS.md](EXPERIMENTS.md) 실험 7과 [analysis_v7/](analysis_v7/)에 있습니다.
> - **결과**: DeBERTa OOF BA 0.5053. `blend3`은 lora 0.6 / deb 0.2 / ng 0.2로 고정했습니다. 제출 커널의 블렌드 OOF는 0.5868(v6 가중치 0.5719, nested 0.5827), 게이트 뒤 0.5882, Public LB 0.556입니다.
> - **Colab에서 알게 된 것**:
>   - L4와 A100은 다섯 시간 동안 한 번도 배정되지 않았고 T4만 잡혔습니다. T4는 fp32로 돌고 `deb_grad_ckpt=True`가 필요합니다(없으면 마이크로 배치 8에서 OOM, 켜면 최대 8.1GB). Stage 1 123분, Stage 2 fold당 약 11.5분, 전체 198분입니다. bf16 경로(L4, A100)는 아직 실행해 보지 못했습니다.
>   - 아래 `jupyter nbconvert --execute` 명령은 셀 출력을 로그로 내보내지 않습니다. `papermill <노트북> <출력> --log-output --request-save-on-cell-execute`로 실행하면 진행이 로그에 보입니다.
>   - Colab CLI의 `upload`·`download`는 100MB쯤부터 실패합니다. 큰 파일은 40MB 조각으로 나눠 옮기고 md5를 확인합니다.
>   - T4 VM이 실행 중에 두 번 서버 쪽에서 사라졌습니다(원인 미확인). 1분마다 세션 커널에 짧은 `colab exec`를 보내고, 결과 파일을 생기는 대로 서버로 받아 둡니다.
>   - 그래서 Colab에서는 **DeBERTa만 학습하는 실행 사본**을 돌렸습니다([analysis_v7/colab/make_deb_run.py](analysis_v7/colab/make_deb_run.py). n-gram·LoRA 셀 제외, Stage 1 체크포인트, fold마다 OOF 저장, fold는 `analysis_v4/oof_v4.npz`에서 읽음). 블렌드 가중치는 서버에서 [analysis_v7/blend3_local.py](analysis_v7/blend3_local.py)로 고르고, 제출 커널이 전체를 다시 계산했습니다.

v7 세션에서 받은 내용입니다. 노트북은 커밋 `4f2251a`에 있습니다. **코드 셀의 문법(`ast.parse`)만 확인했습니다.** DeBERTa의 학습, 저장, 추론 경로는 한 줄도 실행되지 않았습니다. Colab 세션, Kaggle 데이터셋, 커널도 만들지 않았습니다.

> [!IMPORTANT]
> **v7 결과는 새 풀(231,036개) 기준입니다. v4(0.5749)나 v6 nested(0.5864) 같은 문서 수치와 직접 비교하지 마세요.** 둘 다 옛 풀 기준입니다.
> - 공정한 비교 기준은 같은 실행 안에서 출력되는 `(v6 가중치 lora=0.6, deb=0.0, ng=0.4: OOF BA …)` 줄입니다. 새 풀로 만든 v6 블렌드에 해당합니다.
> - 게이트도 같은 실행의 "블렌드 → 게이트" 출력으로 비교합니다.
> - LoRA는 v4 가중치로 OOF만 다시 계산하므로 풀 변경의 영향을 받지 않습니다. n-gram과 DeBERTa Stage 1은 새 풀에서 표본을 뽑습니다.

**1) Colab L4 입력 준비**
- 대회 파일(`train.csv`, `test.csv`, `sample_submission.parquet`)을 `/content/data` 아래에 둡니다. `locate_competition_dir`가 `name-that-llm` 폴더나 `sample_submission.parquet`을 찾습니다.
- LoRA v4 가중치(`ntl_lora_fold0~4.pt`, `ntl_lora_full.pt`)를 `/content/data/weights_v4/`에 둡니다. Kaggle 데이터셋 `ntl-lora-weights-v4-echo-cv`에 있습니다. **없으면 LoRA를 처음부터 학습해서 수 시간이 걸립니다.**
- 노트북 cell 6이 Qwen3-4B-Base, `microsoft/deberta-v3-large`, Open Pool을 받고 `NTL_DEB_MODEL` 등 환경 변수를 설정합니다. n-gram 가중치 후보도 `[12.0]`만 쓰도록 덮어씁니다.

**2) 짧은 통과 점검 (먼저 권장)**
- `NTL_CFG`에 `"deb_stage1_budget_h": 0.02, "deb_stage2_budget_h": 0.02`를 추가합니다. 각 단계가 약 1분 만에 끊기고 끝까지 갑니다.
- 끝나면 `/content/out`에 `ntl_deb_oof.npz`, `ntl_deb_full.pt`, `ntl_deb_base/`가 생기는지 확인합니다.

**3) 본 실행**: 예산 덮어쓰기를 빼고 VM 안에서 `nohup`으로 돌립니다.

```bash
nohup jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 name_that_llm_v7.ipynb \
  --output /content/out/v7_executed.ipynb > /content/out/v7.log 2>&1 &
```

- cell 7(`drive.mount`)은 대화형 인증이라 비대화형 실행에서 멈출 수 있습니다. 미리 마운트하거나 그 셀을 건너뜁니다.
- 마지막 셀이 `/content/out` 전체를 Drive의 `ntl/`로 복사합니다. DeBERTa 파일만 약 1.8GB이니 Drive 용량을 확인합니다.

**4) 결과 파일 (`/content/out`)**

| 파일 | 내용 | 제출에 필요 |
|---|---|:-:|
| `ntl_deb_oof.npz` | DeBERTa OOF 로짓 [2699, 18]과 id | ✅ |
| `ntl_deb_full.pt` | train 전체로 Stage 2를 학습한 모델(fp16, 약 0.9GB) | ✅ |
| `ntl_deb_base/` | 토크나이저와 config | ✅ |
| `ntl_deb_stage1.pt` | Stage 1 가중치(약 0.9GB). 재실험 때 붙이면 Stage 1을 건너뜀 | |
| `oof_v7.npz` | 세 분기의 OOF 로짓 | |
| `submission.parquet` | Colab 실행의 제출 파일(테스트 3행) | |

**5) Kaggle 데이터셋 만들기**: 제안 이름은 `ntl-deberta-v7`이고, 아직 정하지 않았습니다. 위 표에서 ✅인 세 개를 올립니다.

**6) 제출 커널**: 제안 이름은 `name-that-llm-v7-deberta`입니다.
- 입력: `name-that-llm`, `crtproduction/name-that-llm-open-pool`, `qwen-lm/qwen-3`(4b-base), `ntl-lora-weights-v4-echo-cv`, `ntl-deberta-v7`. GPU T4, Internet Off
- 저장된 OOF가 입력에 있으면 DeBERTa는 테스트 추론만 합니다. T4에서는 fp32로 돕니다.
- **제출 전에** Colab 로그의 "선택한 가중치"를 `CFG["blend3"] = dict(ng=…, lora=…, deb=…)`로 고정합니다. `None`이면 커널 안에서 OOF로 다시 고르는데, n-gram OOF가 실행마다 미세하게 다를 수 있습니다.
- 커널의 n-gram 가중치 후보는 기본값 `[1, 4, 12]`라 약 27분이 걸립니다. 줄이려면 `[12.0]`으로 바꿉니다.
- 제출은 4.1장과 같은 방식입니다.

**7) 예상 시간 (측정값 없음, 추정)**

| 단계 | 시간 |
|---|---|
| Stage 1 (Open Pool 45,000개, 1 epoch, L4) | 30~90분, 예산 상한 2.5시간 |
| Stage 2 (fold 5개 + full) | 각 3~8분 |
| LoRA OOF 재계산 + n-gram | 약 30분 |
| Colab 합계 | 약 1.5~3시간 |
| Kaggle 추론 | DeBERTa 약 4,000개에 수 분, 나머지는 v6과 같음 |

**주의할 점**
- **GPU 메모리(확인 안 함)**: 마이크로 배치 8 × 512토큰, bf16 autocast, fp32 가중치와 AdamW로 24GB 안에 들 것으로 봅니다(추정). OOM이 나면 `deb_grad_ckpt=True`로 하거나 `deb_batch_size=4`, `deb_grad_accum=4`로 바꿉니다.
- **CPU RAM**: fold마다 Stage 1 가중치를 fp32로 CPU에 보관해 약 1.7GB를 씁니다.
- **fp16은 쓰지 않습니다**: DeBERTa-v3에서 NaN 사례가 있어서입니다.
- **토크나이저 패키지**: 첫 로드 때 토크나이저 변환에 `sentencepiece`와 `protobuf`가 필요합니다.
- **`deb_test="folds"`**: Kaggle에서 쓰려면 학습 때 `deb_save_folds=True`로 fold 가중치 5개(약 4.5GB)를 저장해야 합니다. 기본값은 `"full"`입니다.
- **낙관적인 점수**: 게이트 alpha는 v4 OOF에서 고른 값 그대로이고, 셀이 alpha별 OOF BA를 출력합니다. 블렌드 선택 점수도 같은 OOF로 골라 낙관적이니, 비교에는 셀이 함께 출력하는 nested BA를 씁니다.

### 4.5 v5와 합치기

v5가 새 풀 기준선(블렌드 0.5716)보다 나으면, v6·v7의 n-gram 셀에 문체 블록을 넣는 것을 검토합니다. v4 → v5 변경은 [v5_stylo/make_v5.py](v5_stylo/make_v5.py)의 3장(셀 16)과 `CFG` 부분입니다. 게이트를 v5 위에 얹으면 `gate_alpha`는 v5 OOF로 다시 고릅니다.

### 4.6 그 밖의 후보 (확정 아님)

1. **α 다시 고르기**: 풀을 넣은 실제 분기에서 `stylo_alpha` ∈ {0.1, 0.2, 0.3}을 OOF로 비교합니다(CPU만, 1회 약 15분). 풀을 넣으면 TF-IDF 쪽이 강해져 최적값이 달라질 수 있습니다.
2. **`n_words`, `n_sents` 빼기**: 풀과 train의 모델별 길이 분포가 다르면 SVC가 이 두 피처를 지름길로 배울 수 있습니다. 먼저 분포를 비교합니다.
3. **블렌드 가중치 다시 보기**: 새 풀에서는 `w` = 0.6이 OOF 최고(0.8)보다 0.003~0.006 낮습니다(기준선 0.5716 대 0.5775, v5 0.5734 대 0.5768).
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
