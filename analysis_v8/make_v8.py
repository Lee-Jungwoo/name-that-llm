"""name_that_llm_v7.ipynb -> name_that_llm_v8.ipynb (n-gram 분기만 바꿈: 문체 블록 + 풀 표본 라벨당 5,000개, 라벨 균형 가중치).

n-gram 코드 셀은 v5 노트북의 셀(문체 블록이 이미 들어 있음)을 가져와 v8 변경만 더합니다. v7의 n-gram 셀이 v5 셀에서
문체 블록만 뺀 것과 같은지 먼저 확인합니다. 나머지 셀은 v7 그대로이고, 설명 셀과 CFG만 v8에 맞게 고칩니다.

사용: .venv/bin/python analysis_v8/make_v8.py            # name_that_llm_v8.ipynb를 새로 만듦
      .venv/bin/python analysis_v8/make_v8.py "<결과 줄>"  # 첫 셀 첫 줄에 제출 결과를 붙여서 만듦
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULT = sys.argv[1] if len(sys.argv) > 1 else "제출 전"
nb = json.load(open(ROOT / "name_that_llm_v7.ipynb", encoding="utf-8"))
v5 = json.load(open(ROOT / "name_that_llm_v5.ipynb", encoding="utf-8"))
cells = nb["cells"]


def get(i):
    return "".join(cells[i]["source"])


def put(i, text):
    cells[i]["source"] = text.splitlines(keepends=True)


def sub(i, old, new, count=1):
    s = get(i)
    assert s.count(old) == count, (i, old[:60], s.count(old))
    put(i, s.replace(old, new))


def rep(s, old, new):
    assert s.count(old) == 1, (old[:60], s.count(old))
    return s.replace(old, new)


# 0. 머리말
put(0, f"""> **v8 · v7 + n-gram 분기 강화 (문체 피처 27개, Open Pool 표본 라벨당 5,000개 + 라벨 균형 가중치)** · {RESULT}
>
> v7에서 바뀐 것은 n-gram 분기(3장)뿐입니다. ① v5의 문체 블록(27개, `stylo_alpha` = 0.2)을 넣고, ② n-gram용 Open Pool 표본을 라벨당 2,500개에서 5,000개로 늘리면서(`ngram_pool_per_label`) 풀 행에 라벨 균형 가중치를 줍니다(`ngram_pool_balance`). LoRA(v4 가중치 `ntl-lora-weights-v4-echo-cv`)와 DeBERTa(`ntl-deberta-v7`)는 다시 학습하지 않고, 블렌드 가중치(lora 0.6 / deb 0.2 / ng 0.2)·회사 게이트·중심화·Sinkhorn도 v7과 같습니다.
>
> 근거는 로컬 CPU에서 커널 fold로 잰 OOF입니다(`analysis_pool/`, `analysis_v7/whatif_ngram.py`): n-gram 0.4690 → 0.5012, 3분기 블렌드 0.5864 → 0.5982, 게이트 뒤 0.5882 → 0.6020. 두 변경을 한 번에 넣었고, OOF로는 이득의 대부분이 풀 크기에서 나옵니다. 제출 커널은 `name-that-llm-v8-pool`이고, 노트북을 만든 스크립트와 점검은 `analysis_v8/`, 기록은 EXPERIMENTS.md 실험 8에 있습니다.
""")

# 1. 제목과 변경점
sub(1, "# Name That LLM: 짧은 지문으로 작성 모델 식별하기 (v7: v6 + DeBERTa 분기)\n\n",
    "# Name That LLM: 짧은 지문으로 작성 모델 식별하기 (v8: v7 + n-gram 풀 확대와 문체 블록)\n\n"
    "> v8 변경점: n-gram 분기에 v5의 **문체 피처 27개**를 넣고, n-gram용 **Open Pool 표본을 라벨당 5,000개**로 늘립니다"
    "(풀 행에는 라벨 균형 가중치). 신경망 분기와 블렌드는 v7과 같습니다(아래 \"3. n-gram 모델\" 참고).\n>\n")

# 2. 데이터 표
sub(2, "| `ntl_open_pool.parquet` (외부, 선택) | 라벨당 최대 2,500개 사용 |",
    "| `ntl_open_pool.parquet` (외부, 선택) | n-gram은 라벨당 최대 5,000개(v8), LoRA·DeBERTa Stage 1은 2,500개 사용 |")

# 3. 아키텍처 설명(영문): 문체 블록(v5와 같은 문장)과 풀 표본
sub(3, "   TF-IDF per view, weighted, hstack                       |\n"
       "   X [N, <=510k] (sparse)                         tap hidden states at depths",
    "   TF-IDF per view, weighted, hstack                       |\n"
    "   + 27 stylometric stats (v5/v8, norm ~0.2)               |\n"
    "   X [N, <=510k] (sparse)                         tap hidden states at depths")
sub(3, "They are fit on an unlabeled Open Pool sample (up to 2,500 passages per label), never on validation prompts.",
    "They are fit on an unlabeled Open Pool sample (up to 5,000 passages per label since v8; 2,500 before), never on validation prompts.")
sub(3, "| A2 | Concatenate views | `X[n,p] = hstack_v x_v[n,p]` | 4 x `[N, P_v]` | `[N, P]`, `P = sum_v P_v <= 510,000` | none |",
    "| A2 | Concatenate views | `X[n,p] = hstack_v x_v[n,p]` | 4 x `[N, P_v]` | `[N, P]`, `P = sum_v P_v <= 510,000` | none |\n"
    "| A2' | Stylometric block (v5, back in v8) | `z[n,j] = clip((log1p(f[n,j]) - mu[j]) / sd[j], -4, 4) * 0.2 / sqrt(27)`; `mu`, `sd` from the `train` rows of the fold; `X <- hstack(X, z)` | 27 stats per passage | `[N, P + 27]` | none |")
sub(3, "- Parameters: `W` is `[18, P]`, `b` is `[18]`, so up to about 9.2M weights.",
    "- Parameters: `W` is `[18, P]`, `b` is `[18]`, so up to about 9.2M weights.\n"
    "- Row L2 norms: the four TF-IDF views give `sqrt(1.0^2 + 0.7^2 + 0.5^2 + 0.5^2) ~ 1.4`. A standardized 27-dim block has an expected L2 norm of `sqrt(27) ~ 5.2`, so it is scaled by `alpha / sqrt(27)` to an L2 norm of about `alpha = 0.2`. A larger block would be cheaper for the L2-penalized SVC to use and would crowd out the TF-IDF columns.")
sub(3, "Open Pool rows get sample weight 1.0. `train` rows get weight `w_train`, chosen from `{1, 4, 12}` by out-of-fold balanced accuracy. "
       "The Open Pool sample (45,000 rows) is about 20 times larger than the `train` fold (about 2,160 rows), so with equal weights the pool dominates the decision boundary.",
    "`train` rows get weight `w_train = 12` (it won against `{1, 4}` by out-of-fold balanced accuracy in every run since v3, so only 12 is run now). "
    "The Open Pool sample (88,107 rows in v8; 45,000 before) is about 40 times larger than the `train` fold (about 2,160 rows), so with equal weights the pool dominates the decision boundary.\n"
    "- Pool row weights (v8): a pool row of label `L` gets `(N_pool / 18) / N_pool[L]`, so every label has the same total pool weight and the mean pool weight is 1. "
    "Two labels have fewer than 5,000 pool passages (`gemini-3.7-flash` 3,541, `qwen3.8-max` 4,566) and get weights 1.38 and 1.07; the other 16 get 0.98.")

# 4. 파이프라인 표
sub(4, "| 3. n-gram 모델 | 4뷰 TF-IDF + LinearSVC를 학습해 OOF 점수와 test 점수를 만듭니다. train 행 가중치는 후보 중 OOF가 가장 좋은 값을 씁니다.",
    "| 3. n-gram 모델 | 4뷰 TF-IDF + 문체 피처 27개 + LinearSVC를 학습해 OOF 점수와 test 점수를 만듭니다. "
    "Open Pool 표본은 라벨당 최대 5,000개이고 풀 행에 라벨 균형 가중치를 줍니다(v8). train 행 가중치는 12입니다.")

# 5. 실행 방법
sub(5, "3. **Save Version**으로 전체 실행한 뒤 Submit합니다. 제출 시 숨겨진 테스트(약 4,000개)로 노트북이 다시 실행됩니다.",
    "3. **Save Version**으로 전체 실행한 뒤 Submit합니다. 제출 시 숨겨진 테스트(약 4,000개)로 노트북이 다시 실행됩니다. "
    "v8은 n-gram 표본이 커져서 n-gram 구간이 v7(약 12분)보다 오래 걸립니다(로컬 CPU 기준 약 33분).")
sub(5, "name_that_llm_v7.ipynb", "name_that_llm_v8.ipynb")

# 8. CFG
sub(8, "    ngram_pool_per_label=2500,   # 라벨당 Open Pool 표본 수\n"
       "    ngram_train_weights=[12.0],  # 진짜 train 지문의 샘플 가중치 후보 (OOF로 선택, 풀 행은 1.0). v3부터 매번 12가 뽑혀 12만 둠(blend3도 12 기준). 다시 비교하려면 [1.0, 4.0, 12.0]\n"
       "    svc_c=0.1,\n",
    "    ngram_pool_per_label=5000,   # v8: 라벨당 Open Pool 표본 수 (v7까지 2500). 풀이 이보다 작은 라벨은 있는 만큼만 뽑힘\n"
    "    ngram_pool_balance=True,     # v8: 풀 행에 라벨 균형 가중치 (풀 행 수 / 라벨 수) / (그 라벨의 풀 행 수)를 줌. 평균은 1. False면 모든 풀 행이 1.0\n"
    "    ngram_train_weights=[12.0],  # 진짜 train 지문의 샘플 가중치 후보 (OOF로 선택, 풀 행은 평균 1.0). v3부터 매번 12가 뽑혀 12만 둠(blend3도 12 기준). 다시 비교하려면 [1.0, 4.0, 12.0]\n"
    "    svc_c=0.1,\n"
    "    stylo_alpha=0.2,             # v5의 문체 블록(v8에서 v7 위에 넣음): 블록의 행 L2 노름(TF-IDF 블록은 약 1.4). 0이면 블록 없음\n")
sub(8, "    echo_input=True,\n"
       "    lora_test=\"full\",            # 테스트 로짓: \"folds\" = fold 모델 5개 평균, \"full\" = train 전체로 Stage 2를 한 번 더 학습한 모델             # Qwen에 '지문 + 빈 줄 + 지문'을 넣고 두 번째 사본만 풀링\n"
       "    lora_pool_per_label=2500,\n",
    "    echo_input=True,             # Qwen에 '지문 + 빈 줄 + 지문'을 넣고 두 번째 사본만 풀링\n"
    "    lora_test=\"full\",            # 테스트 로짓: \"folds\" = fold 모델 5개 평균, \"full\" = train 전체로 Stage 2를 한 번 더 학습한 모델\n"
    "    lora_pool_per_label=2500,    # Stage 1 표본 수. v8에서도 그대로(LoRA는 다시 학습하지 않음)\n")
sub(8, "    blend3=dict(ng=0.2, lora=0.6, deb=0.2),  # 고정 가중치 (Colab 실행의 DeBERTa OOF로 고름: OOF BA 0.5864, nested 0.5823). None이면 OOF BA가 가장 높은 조합(0.1 간격)",
    "    blend3=dict(ng=0.2, lora=0.6, deb=0.2),  # 고정 가중치 (v7에서 Colab 실행의 DeBERTa OOF로 고름. v8도 그대로). None이면 OOF BA가 가장 높은 조합(0.1 간격)")
sub(8, "def find_optional(name):\n"
       "    hits = sorted(glob.glob(str(INPUT_ROOT / f\"**/{name}\"), recursive=True))\n"
       "    return Path(hits[0]) if hits else None",
    "def find_optional(name):\n"
    "    hits = sorted(glob.glob(str(INPUT_ROOT / f\"**/{name}\"), recursive=True))\n"
    "    return Path(hits[0]) if hits else None\n"
    "\n\n"
    "def peak_rss_gb():\n"
    "    \"\"\"지금까지 이 프로세스가 쓴 메모리의 최댓값(GB). /proc이 없는 환경(macOS 등)에서는 None.\"\"\"\n"
    "    try:\n"
    "        with open(\"/proc/self/status\") as f:\n"
    "            for line in f:\n"
    "                if line.startswith(\"VmHWM\"):\n"
    "                    return int(line.split()[1]) / 2**20\n"
    "    except OSError:\n"
    "        pass\n"
    "    return None")

# 15. n-gram 설명
put(15, """## 3. n-gram 모델

주제 단어에 끌려가지 않도록, 같은 지문을 4가지로 바꿔 봅니다.

- **char**: 문자 3~5-gram. 문장부호와 철자 습관을 잡습니다.
- **word**: 단어와 문장부호의 1~2-gram.
- **shape**: 단어를 `x`/`X`로 치환한 글자 형태. 주제 정보가 거의 사라집니다.
- **func**: 기능어만 남기고 나머지는 `_`로 치환. 문장 구조만 남습니다.
- **문체 피처 (v5, v8에서 다시 넣음)**: 지문 전체를 세어야 나오는 통계 27개입니다. 문장 길이 평균과 표준편차, 단어 길이, 긴 단어 비율, 어휘 다양도(MATTR), 기능어 비율, 100단어당 구두점(—, ;, :, 따옴표 종류 등)과 축약형·인칭 대명사 수 등입니다. 정의는 `analysis_intra_org/features.py`의 `feats`와 같습니다. n-gram 창(최대 6글자, 3단어)으로는 문장 길이 같은 지문 전체 통계를 볼 수 없어서 따로 넣습니다.

**문체 블록을 붙이는 방법.** 피처마다 `log1p`로 큰 값을 누르고, fold의 train 지문으로 구한 평균과 표준편차로 표준화한 뒤 ±4에서 자릅니다. 그다음 `α / √27`을 곱합니다. 표준화한 27차원 벡터의 L2 노름은 평균 √27 ≈ 5.2이므로, 곱한 뒤의 L2 노름은 약 α입니다. TF-IDF 블록의 L2 노름은 √(1.0² + 0.7² + 0.5² + 0.5²) ≈ 1.4입니다. 문체 블록이 이보다 크면 L2 규제가 걸린 SVC가 27개 열에 먼저 기대게 되므로, α = 0.2(TF-IDF의 약 1/7)로 작게 붙입니다. 표준화 통계를 풀이 아니라 train에서 구하는 이유는 풀이 모델마다 주제가 쏠려 있어서입니다. 코드는 v5와 같습니다. v6·v7은 v4에서 갈라져 나와 이 블록이 없었습니다. 풀 2,500개에서 이 블록의 효과는 n-gram OOF 0.4690 → 0.4756이고, 3분기 블렌드에서는 0.5864 → 0.5886으로 오차 범위 안입니다(로컬 계산).

벡터라이저는 라벨 없이 Open Pool에만 맞춰서, 검증 프롬프트의 정보가 들어가지 않게 했습니다. Open Pool의 `match_weight`는 풀의 문장부호 분포를 실제 지문에 맞추는 가중치입니다. 이 값은 train 전체로 계산됐기 때문에, CV 점수가 약간 높게 나올 수 있습니다.

**풀 표본 크기 (v8).** n-gram용 표본을 라벨당 2,500개에서 **5,000개**로 늘립니다(`CFG["ngram_pool_per_label"]`). 표본은 `match_weight`에 비례해 뽑고, TF-IDF 어휘도 그 표본에 다시 맞춥니다. gemini-3.7-flash(3,541개)와 qwen3.8-max(4,566개)는 풀 전체가 5,000개보다 적어서 표본은 88,107행입니다. LoRA와 DeBERTa의 Stage 1 표본(라벨당 2,500개)은 그대로입니다(다시 학습하지 않음).

근거는 로컬 CPU에서 제출 커널과 같은 fold 분할로 잰 OOF입니다(`analysis_pool/`, 문체 블록 포함, train 가중치 12. 3분기 열은 `analysis_v7/whatif_ngram.py`).

| 라벨당 표본 | 풀 행 | n-gram OOF BA | 2분기 블렌드 (lora 0.6 / ng 0.4) | 3분기 블렌드 (lora 0.6 / deb 0.2 / ng 0.2) |
|---|:-:|:-:|:-:|:-:|
| 2,500 (v5와 같은 n-gram) | 45,000 | 0.4756 | 0.5738 | 0.5886 |
| 3,500 | 63,000 | 0.4923 | 0.5768 | - |
| 5,000 | 88,107 | 0.5008 | 0.5890 | 0.5960 |
| **5,000 + 라벨 균형 (v8)** | 88,107 | **0.5012** | **0.5905** | **0.5982** |
| 7,500 | 128,107 | 0.5105 | 0.5894 | 0.5979 |
| 7,500 + 라벨 균형 | 128,107 | 0.5134 | 0.5905 | 0.5942 |

- **이 표는 낙관적일 수 있습니다.** 5,000개는 이 OOF를 보고 고른 값입니다. 2,500개 대비 차이의 95% 구간(프롬프트 단위 bootstrap)은 n-gram +0.0256 [+0.011, +0.041], 2분기 블렌드 +0.0167 [+0.005, +0.027]입니다.
- **DeBERTa가 있는 블렌드에서는 이득이 줄어듭니다.** 3분기 고정 가중치에서는 +0.0096 [+0.002, +0.017]입니다. n-gram이 더한 정보의 일부를 DeBERTa가 이미 갖고 있기 때문으로 보입니다(추정).
- **7,500개를 쓰지 않은 이유.** n-gram 단독은 더 높지만 블렌드에서는 5,000개와 같고, fit 시간과 메모리가 약 1.5배입니다.
- **누수 점검.** 풀에는 train 프롬프트와 주제가 가까운 지문이 있습니다. fold마다 그 fold의 검증 프롬프트와 가까운 풀 행을 빼고 다시 재도(풀 행의 약 1.7%, 넓게 잡으면 약 12%) 2,500 → 5,000의 n-gram 차이는 +0.0252, +0.0278로, 빼지 않았을 때(+0.0252)와 같았습니다. 표본을 늘려 생긴 이득이 주제 누수 때문은 아니라는 뜻입니다.
- **풀이 작은 라벨이 밀립니다.** 라벨 균형 가중치 없이 5,000개를 뽑으면 gemini-3.7-flash의 n-gram recall이 0.420 → 0.260으로 떨어집니다(다른 Gemini 모델의 표본만 늘어나기 때문). 균형 가중치를 주면 0.367까지 돌아옵니다.

**풀 행 가중치 (v8, `CFG["ngram_pool_balance"]`).** 라벨 \\(L\\)의 풀 행에 (풀 행 수 / 18) / (\\(L\\)의 풀 행 수)를 줍니다. 모든 라벨의 풀 가중치 합이 같아지고 평균은 1입니다. 5,000개를 채운 16개 라벨은 0.979, qwen3.8-max는 1.072, gemini-3.7-flash는 1.382입니다. 전체 BA는 거의 그대로이고(n-gram 0.5008 → 0.5012, 3분기 블렌드 0.5960 → 0.5982, 차이의 구간은 0을 포함), 한 라벨이 밀리는 것을 줄이려고 켭니다.

**train 행 가중치.** 학습 행의 대부분(88,107개)이 Open Pool이고, 실제 대회 지문인 train은 fold당 약 2,160개입니다. 가중치가 같으면 경계가 풀 분포에 맞춰집니다. 그래서 train 행에만 더 큰 가중치를 줍니다. 후보 `CFG["ngram_train_weights"]`를 모두 5-fold로 돌리고 OOF 점수가 가장 높은 값을 쓰는데, v3부터 매번 12가 뽑혀서 12만 둡니다. 풀이 커지면 train의 가중치 몫이 0.42에서 0.27로 줄지만, 몫을 맞추려고 가중치를 23.5로 올린 실행은 나아지지 않았습니다(n-gram 0.5008 → 0.4979). Open Pool이 없으면 모든 행이 train이라 가중치가 의미가 없어서 1.0만 씁니다.

**메모리.** 라벨당 5,000개에서 풀 행렬의 0이 아닌 원소는 약 1.9억 개입니다. `fit_svc`는 fold마다 [train 행 + 풀 행]에 문체 블록을 붙인 행렬을 새로 조립하므로, 조립에 쓴 사본과 float32 행렬을 fit 전에 놓아 줍니다(값은 그대로이고 최대 메모리만 줄어듭니다). 로컬 CPU에서 이 셀까지의 프로세스 최대 메모리는 8.3GB였고, 놓아 주지 않으면 약 11GB입니다(추정). 셀 끝에서 최대 메모리를 출력하고, 풀 행렬은 Qwen과 DeBERTa를 올리기 전에 지웁니다.

**로컬 점검.** 이 셀을 로컬 CPU에서 커널 fold로 그대로 실행하면 표본·행렬·가중치와 5개 fold의 OOF 점수가 풀 크기 실험의 값과 비트 단위로 같습니다(n-gram OOF 0.5012, `analysis_v8/verify_v8.py`, 로그는 `analysis_v8/logs/`).
""")

# 16. n-gram 코드: v5 셀(문체 블록 포함)에서 출발
ng7, ng5 = get(16), "".join(v5["cells"][16]["source"])
assert ng7.startswith("FUNCTION_WORDS") and ng5.startswith("FUNCTION_WORDS")
# v7 셀 = v5 셀에서 문체 블록만 뺀 것인지 확인: 두 셀의 줄 차이가 모두 문체 관련 줄이어야 합니다.
import difflib
for line in difflib.unified_diff(ng7.splitlines(), ng5.splitlines(), lineterm="", n=0):
    if line.startswith("-") and not line.startswith("---"):
        assert line[1:].strip() in (
            "Xs, ys, ws = [X_train[train_idx]], [y[train_idx]], [np.full(len(train_idx), train_weight)]",
            "Xs.append(X_pool); ys.append(y_pool); ws.append(np.ones(len(y_pool)))",
            "oof[b] = fit_svc(a, train_weight).decision_function(X_train[b])",
            "ng_test = fit_svc(np.arange(len(train)), ngram_w).decision_function(X_test)",
            "del X_pool, ng_pool"), line
s = ng5
s = rep(s, "# v5: 문체 피처 27개 (analysis_intra_org/features.py의 feats와 같은 정의)",
        "# v5: 문체 피처 27개 (analysis_intra_org/features.py의 feats와 같은 정의). v8에서 v7 위에 그대로 넣음")
s = rep(s, "if use_stylo:\n"
           "    print(f\"문체 피처 {S_train.shape[1]}개, alpha={CFG['stylo_alpha']} | {elapsed()}\")\n",
        "if use_stylo:\n"
        "    print(f\"문체 피처 {S_train.shape[1]}개, alpha={CFG['stylo_alpha']} | {elapsed()}\")\n"
        "\n"
        "# v8: 풀 행 가중치. 라벨당 5,000개를 뽑으면 풀이 그보다 작은 라벨(gemini-3.7-flash, qwen3.8-max)은 표본이 적어서\n"
        "# 라벨마다 풀 가중치 합이 달라집니다. 라벨 L의 풀 행에 (풀 행 수 / 라벨 수) / (L의 풀 행 수)를 줘서 합을 같게 맞춥니다.\n"
        "# 평균이 1이라 train 행 가중치(12)와의 비율은 그대로입니다.\n"
        "w_pool = None\n"
        "if X_pool is not None:\n"
        "    pool_counts = np.bincount(y_pool, minlength=K)\n"
        "    if CFG[\"ngram_pool_balance\"]:\n"
        "        w_pool = (len(y_pool) / (pool_counts > 0).sum() / np.maximum(pool_counts, 1))[y_pool]\n"
        "    else:\n"
        "        w_pool = np.ones(len(y_pool))\n"
        "    short = {LABELS[k]: int(c) for k, c in enumerate(pool_counts) if c < CFG[\"ngram_pool_per_label\"]}\n"
        "    print(f\"풀 표본 {len(y_pool):,}행 (라벨당 최대 {CFG['ngram_pool_per_label']:,}개) | 라벨별 {pool_counts.tolist()}\")\n"
        "    print(f\"  {CFG['ngram_pool_per_label']:,}개에 못 미치는 라벨: {short if short else '없음'}\")\n"
        "    print(f\"풀 행 가중치 (라벨 균형 {'켬' if CFG['ngram_pool_balance'] else '끔'}): 최소 {w_pool.min():.4f} / 최대 {w_pool.max():.4f} \"\n"
        "          f\"/ 평균 {w_pool.mean():.4f} | 라벨별 가중치 합 최소 {np.bincount(y_pool, weights=w_pool, minlength=K).min():,.1f} \"\n"
        "          f\"/ 최대 {np.bincount(y_pool, weights=w_pool, minlength=K).max():,.1f}\")\n")
s = rep(s, "        Xs.append(with_stylo(X_pool, S_pool, stats)); ys.append(y_pool); ws.append(np.ones(len(y_pool)))\n"
           "    model = LinearSVC(C=CFG[\"svc_c\"])\n"
           "    model.fit(sparse.vstack(Xs).tocsr(), np.concatenate(ys), sample_weight=np.concatenate(ws))\n",
        "        Xs.append(with_stylo(X_pool, S_pool, stats)); ys.append(y_pool); ws.append(w_pool)\n"
        "    X_fit = sparse.vstack(Xs).tocsr()\n"
        "    del Xs  # v8: 조립에 쓴 사본(풀 5,000개에서 약 1.6GB)을 fit 전에 놓아 줍니다.\n"
        "    X_fit = X_fit.astype(np.float64)  # LinearSVC가 안에서 하는 변환을 미리 해서 float32 행렬도 놓아 줍니다(값은 같음).\n"
        "    model = LinearSVC(C=CFG[\"svc_c\"])\n"
        "    model.fit(X_fit, np.concatenate(ys), sample_weight=np.concatenate(ws))\n"
        "    del X_fit\n")
s = rep(s, "del oof_by_weight\n"
           "del X_pool, ng_pool, S_pool\n"
           "gc.collect()",
        "del oof_by_weight\n"
        "del X_pool, ng_pool, S_pool, w_pool  # Qwen과 DeBERTa를 올리기 전에 풀 행렬을 지웁니다.\n"
        "gc.collect()\n"
        "if peak_rss_gb() is not None:\n"
        "    print(f\"n-gram 구간까지 프로세스 최대 메모리(RSS): {peak_rss_gb():.1f}GB | {elapsed()}\")")
put(16, s)

# 23. 블렌딩 설명
sub(23, "- DeBERTa가 없으면 v6과 같이 LoRA 가중치 `blend_w`(0.6)로 두 분기만 섞습니다.",
    "- **v8:** `blend3`은 v7 값을 그대로 둡니다. n-gram이 강해지면 OOF 최고 조합이 lora 0.4 / deb 0.3 / ng 0.3 쪽으로 옮겨 가지만(로컬 계산 0.5997), "
    "고정값(0.5982)과의 차이가 0.0015(지문 4개)입니다. 셀이 출력하는 상위 5개 조합으로 다시 확인합니다.\n"
    "- DeBERTa가 없으면 v6과 같이 LoRA 가중치 `blend_w`(0.6)로 두 분기만 섞습니다.")

# 24. 블렌딩 코드: 저장 파일 이름
sub(24, 'np.savez(WORK_DIR / "oof_v7.npz"', 'np.savez(WORK_DIR / "oof_v8.npz"')

# 출력 지우기
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None
dst = ROOT / "name_that_llm_v8.ipynb"
with open(dst, "w", encoding="utf-8") as f:
    f.write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print("저장:", dst)
