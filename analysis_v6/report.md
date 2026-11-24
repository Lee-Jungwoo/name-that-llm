# v6: 회사 안 혼동을 고치는 두 방식 비교

같은 회사 모델끼리 헷갈리는 오답을 줄이려고, 특화 분류기를 v4 예측에 넣는 방식을 비교했습니다. 모든 수치는 v4 블렌드 OOF(`analysis_v4/oof_v4.npz`, 지문 2,699개)에서 GPU 없이 계산했습니다. 결론은 세 가지입니다.

1. **채택: 회사 게이트.** 블렌드 1순위가 GPT·Gemini·Claude·GLM 중 한 회사이면, 그 회사의 확률 합(질량)은 그대로 두고 회사 안 분포만 회사 전용 분류기 쪽으로 옮깁니다. nested OOF BA는 0.5749에서 **0.5864(+0.0115)** 로 올랐습니다. 프롬프트 단위 bootstrap 95% 구간은 +0.003\~+0.020이고, fold 5개 모두에서 올랐습니다.
2. **V40A 방식(모든 지문에 잔차를 더하고 회사 질량 보존)은 이득이 없었습니다.** 같은 특화 분류기를 쓰고도 nested ΔBA가 +0.0004입니다.
3. **특화 분류기에 새 정보를 넣어도 나아지지 않았습니다.** Open Pool로 학습한 회사 전용 n-gram SVC는 회사 안 정확도가 v4 블렌드보다 낮았습니다. 이 SVC를 입력에 더하면 nested ΔBA가 +0.0052로 오히려 줄었습니다. 게이트의 이득은 이미 있는 n-gram·LoRA 점수를 **회사마다 다르게 다시 조합한 것**에서 나옵니다.

## 출발점

| 양 | 값 | 계산 |
|---|:-:|---|
| v4 OOF BA | 0.5749 | 블렌드 `eval_lp`의 argmax |
| 같은 회사 안 오답 비율 | 59% (681 / 1,147) | 예측과 정답의 회사가 같은 오답 |
| 회사 단위 정확도 | 0.832 | 회사별 `logsumexp` 질량의 argmax |
| 정답 회사를 알려 줄 때 BA | 0.694 | 정답 회사 열만 남기고 argmax (상한) |
| 딱딱한 2단계 분류 BA | 0.567 | 회사 질량 argmax로 회사를 고른 뒤 그 안에서 argmax |

딱딱한 2단계 분류가 v4보다 낮은 이유는 회사 단위 recall이 GLM 0.44, Kimi 0.34, Qwen 0.66으로 낮기 때문입니다. 회사 간 오답 중 가장 많은 것은 claude-fable-5와 GLM 두 모델 사이(네 방향 합 88개)입니다.

## 비교한 것

**특화 분류기(입력)**: 회사 \(F\)의 정답 지문만으로 학습해 \(F\) 안의 모델을 맞히는 분류기입니다. 모두 같은 `fold`로 fold 밖에서 학습했습니다.

| 이름 | 입력 | 학습 데이터 |
|---|---|---|
| stack | n-gram, LoRA 각각의 \(F\) 열 log 확률 → 로지스틱 회귀 | train OOF |
| text | 문자 3\~5-gram + 단어 1\~2-gram TF-IDF → 로지스틱 회귀 | train만 |
| stack+svc | stack 입력 + 회사 전용 4뷰 n-gram LinearSVC 점수 | SVC는 Open Pool(라벨당 2,500개, HF 미러) + train(가중치 12) |

**넣는 방식**: 회사 질량 \(M_F = \log\sum_{k\in F} e^{\ell_k}\), 회사 안 log 분포 \(\ell_k - M_F\), 특화 분류기 log 확률 \(\log s_F(k)\)입니다.

| 이름 | 바꾸는 지문 | 식 |
|---|---|---|
| gate | 1순위가 \(F\) 안인 지문만 | \(M_F + \log\operatorname{softmax}\big((1-\alpha)(\ell_k - M_F) + \alpha \log s_F(k)\big)\) |
| v40a | 모든 지문 | \(\tilde\ell_k = \ell_k + \alpha r_k\), \(\tilde\ell_k - \log\sum_{j\in F} e^{\tilde\ell_j} + M_F\). \(r_k\)는 중심화한 \(\log s_F(k)\) |
| free | 모든 지문 | \(\ell_k + \alpha r_k\) (질량 보존 없음) |

**묶음**: 회사(GPT 5, Gemini 4, Claude 5, GLM 2)와, 혼동 기준 묶음({fable, glm-5.3, glm-5.3-flash}, GPT 5개, Gemini flash 3개, 나머지 Claude 4개)입니다.

**\(\alpha\) 선택**: 묶음마다 후보 중에서 고릅니다. "best"는 OOF 전체로 고른 값(낙관적)이고, "nested"는 fold 하나를 빼고 나머지 4개 fold로 고른 \(\alpha\)를 뺀 fold에 적용한 값입니다.

## 결과

### 방식 × 입력 × 묶음 ([out/inject.log](out/inject.log))

| 묶음 | 입력 | gate nested | v40a nested | free nested |
|---|---|:-:|:-:|:-:|
| 회사 | stack | **+0.0111** | +0.0004 | −0.0011 |
| 회사 | text | +0.0011 | −0.0004 | −0.0037 |
| 혼동 | stack | +0.0052 | +0.0033 | −0.0011 |
| 혼동 | text | +0.0033 | +0.0004 | +0.0004 |

- 회사로 묶은 gate가 가장 좋습니다. 혼동 기준 묶음에서 fable을 GLM과 묶어도 나아지지 않았습니다.
- train만으로 학습한 text 분류기는 회사 안 정확도가 0.35\~0.54로, v4 블렌드(0.45\~0.82)보다 낮습니다.

### 회사 전용 SVC를 더한 경우 ([out/famsvc.log](out/famsvc.log), [out/gate2.log](out/gate2.log))

| 회사 | 회사 안 정확도: v4 블렌드 | v4 n-gram | 회사 전용 SVC | stack (C = 1) |
|---|:-:|:-:|:-:|:-:|
| GPT | 0.624 | 0.543 | 0.553 | 0.631 |
| Gemini | 0.648 | 0.605 | 0.595 | 0.630 |
| Claude | 0.726 | 0.619 | 0.649 | 0.720 |
| GLM | 0.573 | 0.550 | 0.533 | **0.657** |

| gate 설정 | best ΔBA | nested ΔBA | nested 95% 구간 | 고침 / 망침 |
|---|:-:|:-:|:-:|:-:|
| stack, C = 1 | +0.0133 | **+0.0115** | [+0.0033, +0.0200] | 91 / 60 |
| stack+svc, C = 1 | +0.0115 | +0.0052 | [−0.0044, +0.0148] | 88 / 74 |
| stack, C = 0.1 | +0.0107 | +0.0078 | [−0.0011, +0.0170] | 89 / 68 |

- 회사 전용 SVC는 18클래스 n-gram과 회사 안 정확도가 ±0.03 차이로 비슷합니다. 회사만 따로 학습해도 n-gram에서 새 신호가 나오지 않았습니다. [analysis_intra_org/report.md](../analysis_intra_org/report.md)의 "쌍 전용 분류기가 v4보다 낮다"는 결과와 같은 방향입니다.
- 이득이 가장 큰 회사는 GLM입니다(회사 안 정확도 0.573 → 0.657). v4 블렌드 가중치 0.6은 GLM 쌍에서 최적이 아닌 것으로 보입니다(추정).

### 채택한 설정의 세부 ([out/check_gate.log](out/check_gate.log))

설정은 stack, C = 1, \(\alpha\) = GPT 0.75, Gemini 0.25, Claude 0.5, GLM 1.0입니다(OOF 전체로 선택).

| fold | 0 | 1 | 2 | 3 | 4 |
|---|:-:|:-:|:-:|:-:|:-:|
| v4 BA | 0.5593 | 0.6037 | 0.5889 | 0.5598 | 0.5630 |
| 게이트 BA | 0.5852 | 0.6130 | 0.6037 | 0.5653 | 0.5741 |

- Sinkhorn(\(\rho\) = 0.35)을 더해도 0.5882로 같습니다. v4에 Sinkhorn만 더하면 0.5764입니다.
- recall이 많이 오른 모델은 glm-5.3-flash(+0.073), gpt-5.6-sol·terra(+0.060), claude-opus-4-8(+0.033)입니다. 많이 떨어진 모델은 gpt-5.5(−0.027), gpt-5.6-luna(−0.013)입니다.
- 회사 질량은 보존하지만, 회사 안 분포가 고르게 퍼지면 1순위가 회사 밖으로 넘어갈 수 있습니다. 그래서 qwen3.8-max(+0.007)와 kimi-k3(+0.013)도 조금 바뀌었습니다.

## 새 풀 재확인 (2026-11-25)

위 수치는 모두 옛 Open Pool(226,191개) 시점의 v4 OOF로 계산했습니다. v6 제출 커널(`name-that-llm-v6-gate` 버전 1, 풀 231,036개, Public LB 0.552)이 저장한 OOF([newpool/oof_v6.npz](newpool/oof_v6.npz))로 같은 스크립트를 다시 돌렸습니다. 설정은 stack, C = 1이고 fold는 npz의 `fold`(커널의 분할)입니다. 로그는 [newpool/gate2.log](newpool/gate2.log), [newpool/check_gate.log](newpool/check_gate.log), 커널 로그는 [newpool/kernel.log](newpool/kernel.log)입니다.

| 항목 | 옛 풀 (v4 OOF) | 새 풀 (v6 커널 OOF) |
|---|:-:|:-:|
| 게이트 전 블렌드 BA | 0.5749 | 0.5719 |
| 고정 \(\alpha\) (0.75 / 0.25 / 0.5 / 1.0) | 0.5882 (+0.0133) | 0.5786 (+0.0067) |
| 고정 \(\alpha\)의 95% 구간, 고침 / 망침 | [+0.0048, +0.0226], 94 / 58 | [−0.0011, +0.0148], 71 / 53 |
| 그 OOF에서 고른 최고 \(\alpha\) | 0.5882 (+0.0133), 고정값과 같음 | 0.5823 (+0.0104), 0.75 / 1.0 / 0.25 / 0.75 |
| nested \(\alpha\) | 0.5864 (+0.0115) | 0.5764 (+0.0044) |
| nested의 95% 구간, 고침 / 망침 | [+0.0033, +0.0200], 91 / 60 | [−0.0019, +0.0111], 66 / 54 |
| 회사 안 정확도, 블렌드 → stack: GPT | 0.624 → 0.631 | 0.631 → 0.632 |
| Gemini | 0.648 → 0.630 | 0.635 → 0.643 |
| Claude | 0.726 → 0.720 | 0.732 → 0.724 |
| GLM | 0.573 → 0.657 | 0.600 → 0.627 |
| 한 회사만 고정 \(\alpha\)로 게이트: GPT / Gemini / Claude / GLM | +0.0041 / +0.0004 / +0.0018 / +0.0070 | +0.0041 / 0 / −0.0004 / +0.0022 |

- 고정 \(\alpha\) 게이트는 새 풀에서도 fold 5개 모두에서 올랐습니다(0.5537 → 0.5648, 0.5870 → 0.5963, 0.5907 → 0.5963, 0.5653 → 0.5690, 0.5630 → 0.5667). 로컬에서 다시 계산한 예측은 커널의 `gated_lp`와 2,699개 모두 같습니다.
- 이득은 옛 풀의 절반쯤이고, 고정 \(\alpha\)와 nested의 구간이 모두 0을 포함합니다. 새 풀에서 이득을 내는 회사는 GPT와 GLM입니다.
- recall은 gpt-5.6-terra(+0.067), glm-5.3-flash(+0.060), gpt-5.6-sol(+0.033)이 올랐고, glm-5.3(−0.053), gpt-5.5(−0.020)가 떨어졌습니다. 옛 풀에서 glm-5.3은 −0.007이었습니다.
- \(\alpha\)는 다시 고르지 않았습니다. 새 OOF의 최고 조합(+0.0104)이 nested에서는 +0.0044로 줄고, 모든 회사에 같은 \(\alpha\)를 써도 0.5 / 0.75 / 1.0에서 +0.0059 / +0.0085 / +0.0063으로 비슷합니다.
- 옛 풀의 고정 \(\alpha\) 구간은 `check_gate.py`에 이번에 더한 출력으로 계산했습니다. 옛 풀의 nested 구간 [+0.0033, +0.0200]은 stack+svc를 함께 돌렸을 때의 값이고, famsvc npz 없이 돌리면 난수 순서가 달라져 [+0.0022, +0.0200]이 나옵니다.

```bash
python analysis_v6/gate2.py . - analysis_v6/newpool/oof_v6.npz        # 세 번째 인자가 OOF 경로, "-"는 famsvc npz 없음
python analysis_v6/check_gate.py . - analysis_v6/newpool/oof_v6.npz
```

## 한계

- \(\alpha\)와 LR 정규화 강도 \(C\)를 v4 OOF로 골랐습니다. 노트북이 출력하는 게이트 OOF BA(0.5882)는 낙관적이고, 기대치는 nested +0.0115 쪽입니다.
- 특화 LR은 OOF 로짓으로 학습하고 테스트 로짓에 적용합니다. 테스트 로짓은 n-gram이 train 전체 모델, LoRA가 fold 5개 평균(또는 전체 학습 모델)이라 OOF와 스케일이 조금 다를 수 있습니다.
- Open Pool은 Kaggle 데이터셋(`crtproduction`) 대신 HF 미러(`HLDKNotFound/name-that-llm-dataset`)의 49,050개를 썼습니다.

## 재현

```bash
# 임시 환경: scikit-learn, scipy, pandas, pyarrow
python analysis_v6/inject.py .                                    # 방식 × 입력 × 묶음 (train.csv + oof_v4.npz)
python analysis_v6/famsvc.py . <HF parquet 폴더> famsvc_oof.npz   # 회사 전용 SVC OOF (약 3분)
python analysis_v6/gate2.py . famsvc_oof.npz                      # stack vs stack+svc, bootstrap
python analysis_v6/check_gate.py . famsvc_oof.npz                 # 채택 설정의 fold별·모델별 변화
```
