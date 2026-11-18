# 문체를 기반으로 한 피처 후보



## 분석 조건

- 데이터: `data/train.csv`, 지문 2,699개, 모델 18개(모델당 150개 안팎)
- 순위 기준: 상호정보량(MI), 즉 피처 하나만으로 18개 모델을 얼마나 구분하는지
- 단위: 개수 피처는 모두 **100단어당 개수**로 정규화했습니다(`100 * 개수 / 단어 수`).
- 출현율: 해당 기호가 1번 이상 나오는 지문의 비율입니다.
- 높은 쪽 / 낮은 쪽: 비고에 적힌 기준(출현율, 100단어당 개수 등)으로 값이 높게 나온 모델과 낮게 나온 모델입니다.

---

## 1순위: 대시

| 피처 | 정의 | MI | 높은 쪽 | 낮은 쪽 | 비고 |
|---|---|---|---|---|---|
| `em_dash` | `—` 개수 / 100단어 | **0.297** (1위) | Claude 63\~83%, GLM 79\~82%, Kimi 52% | GPT 9\~37%, Qwen 3%, Gemini 0\~1% | 출현율 기준 |
| `unspaced_em_dash` | 앞뒤에 공백 없이 단어에 붙여 쓴 em dash 개수 (예: `word—word`) / 100단어 | 0.223 (2위) | Claude 48\~83%, GLM 77\~79%, Kimi 52% | GPT 9\~36%, Qwen 3%, Gemini 0\~1% | 출현율 기준. 대부분의 모델은 em dash를 이렇게 붙여 씀 |
| `spaced_em_dash` | 앞뒤에 공백을 두고 띄어 쓴 em dash 개수 (예: `word — word`) / 100단어 | 0.074 | Claude opus-4-6 27%, opus-5 25% | Claude fable-5, opus-4-8, sonnet-5 0% | 출현율 기준. **같은 Claude끼리도 구분됩니다.** GLM도 2\~3%뿐입니다 |

## 2순위: 따옴표와 아포스트로피

큰따옴표와 아포스트로피는 **곧은 것과 곱은 것을 나눠 세야** 신호가 생깁니다.

| 피처 | 정의 | MI | 높은 쪽 | 낮은 쪽 | 비고 |
|---|---|---|---|---|---|
| `straight_apos` | `'` 개수 / 100단어 | 0.171 | Claude, GLM 약 1개 | GPT 0\~0.15개 | 100단어당 개수 기준 |
| `curly_apos` | `’` 개수 / 100단어 | 0.155 | GPT 42\~53%, Kimi 32%, Qwen 29%, Gemini 6\~31% | Claude, GLM 0% | 출현율 기준 |
| `curly_dq` | `“`와 `”` 개수 / 100단어 | 0.050 | GPT 7\~15%, Kimi 9% | Claude, Gemini, GLM 0\~2% | 출현율 기준 |
| `straight_dq` | `"` 개수 / 100단어 | 0.018 | Claude, GLM 7\~12% | GPT 0\~1%, Gemini 0\~3% | 출현율 기준 |

> **추가 아이디어:** `straight_apos`와 `curly_apos`를 묶어 "아포스트로피 스타일(곧은 / 곱은 / 섞임 / 없음)"이라는 범주형 피처를 하나 만들어도 됩니다.

## 3순위: 그 밖의 구두점

| 피처 | 정의 | MI | 높은 쪽 | 낮은 쪽 | 비고 |
|---|---|---|---|---|---|
| `comma_per_sent` | 쉼표 개수 / 문장 수 | 0.121 |  |  |  |
| `semicolon` | 세미콜론(`;`) 개수 / 100단어 | 0.100 | GPT 43\~62%, Kimi 61% | Gemini 19\~28% | 출현율 기준. 모델마다 다름: Claude 15\~54% |
| `colon` | 콜론(`:`) 개수 / 100단어 | 0.082 | Kimi 0.98, GPT 0.76\~0.89 | Gemini 0.16\~0.18 | 100단어당 개수 기준 |
| `comma` | `,` 개수 / 100단어 | 0.073 |  |  |  |
| `hyphen_compound` | 하이픈(`-`)으로 이어 붙인 단어 개수 (예: `well-known`, `long-term`) / 100단어 | 0.022 |  |  | 약함 |

## 4순위: 길이와 구조

문장은 [정규식](#부록-정규식) `(?<=[.!?])\s+`로 나누고, 단어는 [정규식](#부록-정규식) `[A-Za-z’']+`로 셉니다.

| 피처 | 정의 | MI | 높은 쪽 | 낮은 쪽 | 비고 |
|---|---|---|---|---|---|
| `word_len_mean` | 평균 단어 길이(글자 수) | 0.121 |  |  | 상위권 |
| `long_sent_ratio` | 30단어 이상인 문장의 비율 | 0.073 |  |  |  |
| `long_word_ratio` | 9글자 이상인 단어의 비율 | 0.067 |  |  |  |
| `sent_len_min` | 가장 짧은 문장의 단어 수 | 0.064 |  |  |  |
| `sent_len_mean` | 평균 문장 길이(단어 수) | 0.050 | Claude opus-4-6 27.5 | Gemini 3.1 20.3, GPT-6-astra 21.2 | 평균 단어 수 기준 |
| `sent_len_std` | 문장 길이의 표준편차 | 0.045 | Claude 6.0\~8.6 | Gemini 5.0\~5.2 | Claude가 문장 길이를 더 들쭉날쭉하게 씀 |
| `sent_len_max` | 가장 긴 문장의 단어 수 | 0.037 |  |  |  |
| `n_words` | 지문의 단어 수 | 0.032 |  |  | 약함 |
| `short_sent_ratio` | 8단어 이하인 문장의 비율 | 0.029 |  |  | 약함 |

## 5순위: 어휘와 수사 습관

| 피처 | 정의 | MI | 높은 쪽 | 낮은 쪽 | 비고 |
|---|---|---|---|---|---|
| `func_ratio` | 전체 단어 중 기능어의 비율. 기능어는 the, of, and처럼 뜻보다 문법 역할을 하는 단어입니다. 목록은 [부록](#부록-기능어-목록) 참고 | 0.047 |  |  |  |
| `ttr` | 서로 다른 단어 수 / 전체 단어 수(어휘 다양도) | 0.028 |  |  |  |
| `first_person` | 1인칭 단수/복수 대명사(i, me, my, we, us, our) 개수 / 100단어 | 0.022 |  |  | 약함 |
| `not_x_but_y` | "X가 아니라 Y" 표현 개수 (예: `not a line but a breath`, `not only A but also B`) / 100단어 | 0.016 | Qwen 0.44, GPT 0.31\~0.41 | Gemini 0.14\~0.16 | 100단어당 개수 기준 |
| `sent_start_and_but` | And, But, Yet, So로 시작하는 문장의 비율 | 0.014 |  |  | 약함 |

---

## 추천 세트 (21개)

```
em_dash, spaced_em_dash, unspaced_em_dash,
curly_apos, straight_apos, curly_dq, straight_dq,
semicolon, colon, comma, comma_per_sent,
word_len_mean, long_word_ratio,
sent_len_mean, sent_len_std, sent_len_min, sent_len_max, long_sent_ratio,
func_ratio, ttr, not_x_but_y
```

## 의미 없었던 피처

train에서 MI가 0에 가깝거나, 모든 지문에서 값이 같았던 피처입니다.

MI가 0이라는 건 그 피처 값을 알아도 어떤 모델이 쓴 글인지 전혀 좁혀지지 않는다는 뜻입니다. 모든 모델이 비슷한 값을 내거나(예: 18개 모델 모두 `!`를 거의 안 씀), 모든 지문의 값이 같으면(예: 모두 1문단) 모델마다 다른 점이 없으니 구분하는 데 쓸 정보가 없습니다.

| 피처 | 정의 | MI | 이유 |
|---|---|---|---|
| `n_paras` | 문단 수 | 0.000 | 모든 지문이 1문단이라 값이 같음 |
| `has_trailing_ws` | 지문 끝에 공백이 있는지 | 0.006 | 모든 지문의 끝에 공백이 없어서 값이 같음 |
| `en_dash` | `–` 개수 / 100단어 | 0.000 | 거의 나오지 않음 |
| `double_hyphen` | `--` 개수 / 100단어 | 0.000 | 한 번도 나오지 않음 |
| `spaced_hyphen_dash` | ` - `(앞뒤 공백) 개수 / 100단어 | 0.000 | 한 번도 나오지 않음 |
| `question` | `?` 개수 / 100단어 | 0.003 | MI가 0에 가까움 |
| `exclaim` | `!` 개수 / 100단어 | 0.000 | MI가 0에 가까움 |
| `paren` | `(` 개수 / 100단어 | 0.000 | MI가 0에 가까움 |
| `ellipsis` | `…`와 `...` 개수 / 100단어 | 0.000 | MI가 0에 가까움 |
| `slash` | `/` 개수 / 100단어 | 0.000 | MI가 0에 가까움 |
| `asterisk` | `*` 개수 / 100단어 | 0.001 | MI가 0에 가까움 |
| `digit` | 숫자 글자 수 / 100단어 | 0.000 | MI가 0에 가까움 |
| `non_ascii` | 대시·따옴표를 뺀 비ASCII 문자 수 / 100단어 | 0.000 | MI가 0에 가까움 |
| `ends_with_period` | 지문이 마침표로 끝나는지 | 0.000 | 97\~100% 지문이 마침표로 끝나서 차이가 없음 |
| `second_person` | `you, your` 개수 / 100단어 | 0.000 | MI가 0에 가까움 |

---

## 부록: 기능어 목록

`func_ratio` 계산에 쓴 기능어 89개입니다. v4 노트북의 `FUNCTION_WORDS`와 같습니다.

```
a, an, the, and, or, but, if, while, although, because, as, so,
than, that, which, who, whom, whose, this, these, those, it, its, they,
them, their, he, him, his, she, her, we, us, our, you, your,
i, me, my, of, in, on, at, by, for, with, from, to,
into, over, under, about, between, through, during, not, no, nor, is, are,
was, were, be, been, being, has, have, had, do, does, did, will,
would, can, could, may, might, must, shall, should, very, more, most, such,
only, own, same, just, also
```

---

## 부록: 정규식

**정규식**(regular expression, regex)은 "특정 모양의 글 조각"을 짧은 기호로 적는 방법입니다. 

컴퓨터가 글에서 특정 패턴을 찾거나, 그 패턴을 기준으로 글을 자를 때 씁니다.

자주 쓰는 기호는 다음과 같습니다.

| 기호 | 뜻 | 예 |
|---|---|---|
| `[...]` | 괄호 안의 글자 중 하나 | `[.!?]`는 `.`, `!`, `?` 중 하나 |
| `A-Z`, `a-z` | 알파벳 범위 | `[A-Za-z]`는 영어 대문자 또는 소문자 하나 |
| `+` | 바로 앞의 것이 1번 이상 이어짐 | `[A-Za-z]+`는 알파벳이 1글자 이상 이어진 것 |
| `\s` | 공백 문자(띄어쓰기, 줄바꿈 등) | `\s+`는 공백 1개 이상 |
| `(?<=...)` | 바로 앞에 `...`가 있는 위치 | `(?<=a)b`는 `a` 바로 뒤에 오는 `b` |

이 문서에서 쓴 정규식은 두 개입니다.

| 정규식 | 하는 일 | 예 |
|---|---|---|
| `(?<=[.!?])\s+` | 마침표, 느낌표, 물음표 **바로 뒤의 공백**을 찾습니다. 이 공백에서 잘라 문장을 나눕니다 | `It rained. We left!`는 `It rained.`와 `We left!` 두 문장이 됩니다 |
| `[A-Za-z’']+` | 알파벳과 아포스트로피가 **이어진 덩어리 하나**를 찾습니다. 이 덩어리 하나를 단어 하나로 셉니다 | `Don’t stop, 2 cats`에서는 `Don’t`, `stop`, `cats` 세 단어를 셉니다. 숫자와 쉼표는 빠집니다 |

> 문장 나누기는 단순한 규칙이라 `Dr. Kim`처럼 약어 뒤에 마침표가 있으면 문장이 잘못 잘릴 수 있습니다.

