# Name That LLM: 대회 준비 과정 전체 정리

v1부터 v9까지 이 대회를 위해 준비해 온 과정을 정리한 문서의 목차입니다. 내용은 모두 아래 링크의 하위 문서에 있습니다. 2026-11-26 기준으로 v8(Public LB 0.563)까지 제출했고, v9는 학습 중입니다.

## v1~v9 공통

| 문서 | 다루는 것 |
|---|---|
| [1. 사용 데이터셋](01_datasets.md) | 대회 데이터, 18개 라벨, Open Pool과 갱신 이력, 채택하지 않은 외부 데이터, 전처리 |
| [2. 학습 fold와 검증 fold 분할](02_fold_split.md) | `prompt_id` 그룹 5-fold, 홀드아웃과 OOF, 누수 주의점, 환경에 따라 달라지는 분할 |
| [3. 모델 구조](03_model_architecture.md) | n-gram 분기, Qwen3 LoRA 분기, 블렌딩, 프롬프트 중심화 |
| [4. 실험 방법](04_experiment_method.md) | 지표, 학습·제출 환경, 설정 체계, 검증 수치를 재고 비교하는 규칙, 제출과 기록 절차 |

## 버전별 변화

| 문서 | 다루는 것 |
|---|---|
| [5. v1 → v9 변경과 개선 분석](05_version_history.md) | 버전마다 바뀐 것, 검증 BA와 Public LB의 변화, 개선 기여 분해, 효과가 없었던 시도, 현재 상태 |
| [7. 버전별 모델 구조 다이어그램](07_architecture_by_version.md) | v1~v9 구조를 버전마다 그림 한 장으로, 직전 버전 대비 바뀐 곳 표시 |

## 분석과 계획

| 문서 | 다루는 것 |
|---|---|
| [6. 회사별 혼동 행렬 (v4, v7, v8)](06_company_confusion.md) | 회사 내부 혼동 행렬을 OOF 2,699개 기준으로 다시 그린 것, 회사 단위 적중률, 헷갈린 쌍, 모델별 recall |
| [외부 데이터 수집 계획](외부데이터_수집.md) | GLM과 Claude를 가르기 위한 추가 학습 데이터 수집 계획 (11/25 작성, 아직 실행 기록 없음) |
| [8. 아직 헷갈리는 모델 쌍: 지문의 차이](08_hard_pairs.md) | v8 OOF(v9의 LoRA 외 부분)에서 가장 헷갈리는 8쌍의 문체·표지 차이, 설명 가능한 분류기, v8이 틀린 지문의 특징(길이 의존 포함) |

## 원본 기록

| 문서 | 다루는 것 |
|---|---|
| [README.md](../README.md) | 접근 요약과 실행 방법 |
| [EXPERIMENTS.md](../EXPERIMENTS.md) | 실험 1~8의 상세 기록 (제출 커널 로그 기준 수치) |
| [analysis_v4/report.md](../analysis_v4/report.md) | v1~v4 비교, 모델별 recall, 혼동 쌍 |
| [analysis_intra_org/report.md](../analysis_intra_org/report.md) | 같은 회사 모델 간 차이 분석 |
| [analysis_intra_org/experiment_plan.md](../analysis_intra_org/experiment_plan.md) | 헷갈리는 쌍 손실 실험 계획 |
| [analysis_v6/report.md](../analysis_v6/report.md) | 회사 게이트 비교 실험 |
| [analysis_pool/](../analysis_pool/) | n-gram 풀 표본 크기 실험과 누수 점검 (v8의 근거) |
| [analysis_v7/](../analysis_v7/), [analysis_v8/](../analysis_v8/), [analysis_v9/](../analysis_v9/) | 버전별 Colab 실행 도구, 노트북 생성·점검 스크립트, 커널 로그, 저장된 OOF |
| [stylometric_features.md](../stylometric_features.md) | 문체 통계 피처 후보 |
| [name_that_llm_high_scorers_report.md](../name_that_llm_high_scorers_report.md) | 상위권 참가자의 구조와 학습 전략 조사 |

## 노트북

[v1](../name_that_llm_v1.ipynb) · [v2](../name_that_llm_v2.ipynb) · [v3](../name_that_llm_v3.ipynb) · [v4](../name_that_llm_v4.ipynb) · [v5](../name_that_llm_v5.ipynb) · [v6](../name_that_llm_v6.ipynb) · [v7](../name_that_llm_v7.ipynb) · [v8](../name_that_llm_v8.ipynb) · [v9](../name_that_llm_v9.ipynb) (학습 중)
