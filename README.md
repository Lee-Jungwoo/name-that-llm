# Name That LLM

짧은 영어 지문을 입력받아 작성한 모델을 18개 LLM 중 하나로 분류하는 Name That LLM 과제의 실험 노트북입니다.
평가 지표는 클래스별 recall의 평균인 balanced accuracy이며, 무작위 추측 수준은 1/18 ≈ 0.056입니다.
테스트 프롬프트는 학습 데이터에 등장하지 않으므로, 주제보다 문체의 차이를 학습하는 것이 핵심입니다.

## 접근

두 분기를 각각 학습하고, 온도 보정한 로그 확률을 블렌딩합니다.

- **n-gram 분기**: 문자, 단어, 글자 형태(`x`/`X`), 기능어의 4가지 TF-IDF 뷰를 연결해 LinearSVC로 분류합니다.
- **LoRA 분기**: Hugging Face Transformers로 Qwen3-4B-Base의 앞쪽 18개 층을 불러옵니다.
  원래 가중치는 고정하고, 어텐션의 q/k/v/o 투영층에 PyTorch로 직접 구현한 rank-16 LoRA를 붙입니다.
- 깊이 2, 4, 6, 9, 12, 15, 18의 은닉 상태를 masked mean pooling하고,
  깊이별 LayerNorm 후 연결해 선형 분류기에 넣습니다.
- Open Pool 학습(Stage 1) 후 대회 train 데이터로 미세 조정(Stage 2)합니다.
- 두 분기의 로그 확률을 가중합한 뒤, 테스트 주제 클러스터의 크기가 2~18인 경우에만
  클러스터 평균 점수의 0.5배를 빼는 프롬프트 중심화를 적용합니다.
- 기본 설정에서 GPU 또는 Qwen3 가중치가 없으면 LoRA를 건너뛰고 n-gram만 사용합니다.

## 평가 방식

- train의 `prompt_id`를 그룹으로 삼아 `StratifiedGroupKFold` 5-fold를 구성합니다(seed 42).
  같은 프롬프트의 지문이 학습과 검증 폴드에 나뉘지 않도록 합니다.
- n-gram은 각 폴드의 OOF 점수를 만들고, LoRA는 fold 0의 프롬프트를 홀드아웃으로 제외합니다.
- n-gram 온도는 전체 OOF의 NLL, LoRA 온도는 홀드아웃의 NLL로 선택합니다.
  블렌드 가중치는 같은 홀드아웃의 balanced accuracy로 선택합니다.
- Open Pool에서는 정규화한 텍스트가 train과 일치하는 지문을 제외합니다.
  풀이 있으면 TF-IDF를 풀에 맞추고, 없으면 train 전체에 맞춥니다.
- Open Pool의 `match_weight`는 train 전체로 계산된 값이므로 CV가 다소 높게 나올 수 있다고
  노트북에 명시되어 있습니다. 블렌드 점수도 가중치 선택에 쓴 홀드아웃의 결과입니다.

## 결과

아래 수치는 `name_that_llm_v2.ipynb`에 저장된 Colab 학습 실행 출력의 balanced accuracy입니다.
제출 커널(Kaggle T4)에서는 LoRA 출력이 조금 달라 LoRA 0.5222, 블렌드 0.5444였습니다.
모든 버전의 제출 기준 수치는 [EXPERIMENTS.md](EXPERIMENTS.md)에 있습니다.

- n-gram 5-fold OOF: **0.4667** (`## 3. n-gram 모델`의 코드 셀 출력)
- n-gram fold 0: **0.4593** (같은 코드 셀의 fold 0 출력)
- LoRA fold 0 홀드아웃: **0.5204** (`## 4. LoRA 분류기`의 학습 실행 셀 출력)
- 블렌드 fold 0 홀드아웃: **0.5426**, n=540, LoRA 가중치 **0.5** (`## 5. 블렌딩`의 코드 셀 출력)

OOF 전체와 fold 0 점수는 평가 범위가 다릅니다. 위 점수는 리더보드나 독립 테스트 성능이 아닙니다.
저장된 테스트 3행에서는 중심화 대상이 0개였으며, 위 블렌드 검증 점수에는 테스트 중심화가 포함되지 않습니다.

## 실행 방법

### Colab

1. 준비 스크립트의 안내에 따라 Colab CLI 세션에서 먼저 실행합니다.
   `colab exec -s ntl -f colab_setup.py --timeout 0`
2. 스크립트는 Qwen3-4B-Base를 내려받고 `/content/input/name-that-llm`과 `/content/out`을 만듭니다.
   대회 파일 `train.csv`, `test.csv`, `sample_submission.parquet`는 별도로 준비합니다.
3. 노트북을 같은 커널에서 순서대로 실행합니다. 단, 첫 Colab 준비 셀은 Open Pool을 받고
   `NTL_INPUT=/content/data`, `NTL_CFG='{"batch_size": 8}'` 등으로 설정을 다시 지정합니다.
   이 셀을 실행한다면 대회 파일도 `/content/data/name-that-llm` 아래에 둡니다.
4. 스크립트만 사용하는 Open Pool 없는 설정과 노트북 첫 셀의 Open Pool 사용 설정은 서로 다릅니다.
   노트북의 배치 8 설정은 L4(22GB)의 메모리 부족을 피하기 위한 값입니다.

### Kaggle

1. Add Input에 대회 `name-that-llm`, 데이터셋 `crtproduction/name-that-llm-open-pool`,
   모델 `qwen-lm/qwen-3`(Transformers, `4b-base`)를 추가합니다.
2. GPU T4, Internet Off로 설정하고 Save Version으로 전체 실행한 뒤 제출합니다.
3. 결과는 `submission.parquet`와 `ntl_lora.pt`로 저장됩니다.
   저장한 `ntl_lora.pt`를 입력에 추가하면 재학습 없이 불러옵니다.

경로와 설정은 다음 환경 변수로 지정합니다. 로컬 실행에는 `pyarrow`가 필요합니다.

- `NTL_INPUT`: 데이터 탐색 루트입니다(기본 `/kaggle/input`, 준비 스크립트는 `/content/input`).
- `NTL_BASE_MODEL`: Qwen3 모델 경로를 직접 지정합니다.
- `NTL_WORK`: 출력 폴더입니다(기본 `.`, Colab 준비 시 `/content/out`).
- `NTL_CFG`: `CFG` 일부를 JSON으로 덮어씁니다. 예: `{"use_lora": false}`.

## 파일 구성

버전별 노트북입니다. 각 노트북은 데이터 처리, 검증, 두 분기 학습, 블렌딩과 제출 파일 생성을 담고, 첫 셀에 버전 설명이 있습니다.

| 노트북 | 버전 | Public LB |
|---|---|:-:|
| `name_that_llm_v1.ipynb` | LoRA(train만) + n-gram. Open Pool을 쓰지 않음 | 0.451 |
| `name_that_llm_v2.ipynb` | LoRA(Open Pool Stage 1 + Stage 2) + n-gram | 0.507 |
| `name_that_llm_v3.ipynb` | n-gram train 가중치 12, Sinkhorn, 블렌드 가중치 0.6 고정 | 0.536 |
| `name_that_llm_v4.ipynb` | Qwen echo 입력 + LoRA 5-fold | 0.545 |

- `EXPERIMENTS.md`: 버전별 실험 기록입니다.
- `colab_setup.py`: Colab CLI 세션의 모델 다운로드, 폴더 생성과 환경 변수 설정을 수행합니다.
- `README.md`: 접근, 저장된 결과와 실행 방법을 정리합니다.
