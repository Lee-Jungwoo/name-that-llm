"""name_that_llm_v4.ipynb -> name_that_llm_v5.ipynb (n-gram 분기에 문체 피처 블록 추가)."""
import json, sys

src, dst = sys.argv[1], sys.argv[2]
nb = json.load(open(src))
cells = nb["cells"]


def get(i):
    return "".join(cells[i]["source"])


def put(i, text):
    cells[i]["source"] = text.splitlines(keepends=True)


def sub(i, old, new, count=1):
    s = get(i)
    assert s.count(old) == count, (i, old[:60], s.count(old))
    put(i, s.replace(old, new))


# 0. 머리말
put(0, """> **v5 · v4 full + n-gram 분기에 문체 피처 27개(α = 0.2)** · `EXPERIMENTS.md`의 실험 5
>
> v4(`EXPERIMENTS.md`의 실험 3, 4)에서 n-gram 분기만 바꿨습니다. LoRA는 v4 가중치(`ntl-lora-weights-v4-echo-cv`)를 그대로 쓰고, 테스트 로짓은 train 전체로 학습한 모델 1개(full)입니다.
>
> 제출 커널은 `name-that-llm-v5-stylo`입니다.
""")

# 1. 제목과 변경점
sub(1, "# Name That LLM: 짧은 지문으로 작성 모델 식별하기 (v4: Qwen echo 입력)\n\n"
       "> v4 변경점: LoRA 분류기에 지문을 **두 번 이어서** 넣고 두 번째 사본만 풀링합니다(4장 참고). 나머지는 v3과 같습니다.",
    "# Name That LLM: 짧은 지문으로 작성 모델 식별하기 (v5: 문체 피처 블록)\n\n"
    "> v5 변경점: n-gram 분기의 TF-IDF 행렬 옆에 **문체 피처 27개**(문장 길이, 단어 길이, 구두점 빈도 등)를 작은 가중치로 이어 붙입니다(3장 참고). "
    "LoRA 분기는 v4와 같고(지문을 두 번 이어서 넣고 두 번째 사본만 풀링), 저장된 v4 가중치를 그대로 씁니다.")

# 3. 아키텍처 설명(영문)에 문체 블록 반영
sub(3, "   TF-IDF per view, weighted, hstack                       |\n"
       "   X [N, <=510k] (sparse)                         tap hidden states at depths",
    "   TF-IDF per view, weighted, hstack                       |\n"
    "   + 27 stylometric stats (v5, norm ~0.2)                  |\n"
    "   X [N, <=510k] (sparse)                         tap hidden states at depths")
sub(3, "| A2 | Concatenate views | `X[n,p] = hstack_v x_v[n,p]` | 4 x `[N, P_v]` | `[N, P]`, `P = sum_v P_v <= 510,000` | none |",
    "| A2 | Concatenate views | `X[n,p] = hstack_v x_v[n,p]` | 4 x `[N, P_v]` | `[N, P]`, `P = sum_v P_v <= 510,000` | none |\n"
    "| A2' | Stylometric block (v5) | `z[n,j] = clip((log1p(f[n,j]) - mu[j]) / sd[j], -4, 4) * 0.2 / sqrt(27)`; `mu`, `sd` from the `train` rows of the fold; `X <- hstack(X, z)` | 27 stats per passage | `[N, P + 27]` | none |")
sub(3, "- Parameters: `W` is `[18, P]`, `b` is `[18]`, so up to about 9.2M weights.",
    "- Parameters: `W` is `[18, P]`, `b` is `[18]`, so up to about 9.2M weights.\n"
    "- Row L2 norms: the four TF-IDF views give `sqrt(1.0^2 + 0.7^2 + 0.5^2 + 0.5^2) ~ 1.4`. A standardized 27-dim block has an expected L2 norm of `sqrt(27) ~ 5.2`, so it is scaled by `alpha / sqrt(27)` to an L2 norm of about `alpha = 0.2`. A larger block would be cheaper for the L2-penalized SVC to use and would crowd out the TF-IDF columns.")

# 4. 파이프라인 표
sub(4, "| 3. n-gram 모델 | 4뷰 TF-IDF + LinearSVC를 학습해 OOF 점수와 test 점수를 만듭니다.",
    "| 3. n-gram 모델 | 4뷰 TF-IDF + 문체 피처 27개(v5) + LinearSVC를 학습해 OOF 점수와 test 점수를 만듭니다.")

# 5. 실행 방법의 파일 이름
sub(5, "name_that_llm_v4.ipynb", "name_that_llm_v5.ipynb")

# 8. CFG
sub(8, "    svc_c=0.1,\n",
    "    svc_c=0.1,\n"
    "    stylo_alpha=0.2,             # v5: 문체 블록의 행 L2 노름(TF-IDF 블록은 약 1.4). 0이면 v4와 같음\n")
sub(8, '    lora_test="folds",           # 테스트 로짓: "folds" = fold 모델 5개 평균, "full" = train 전체로 Stage 2를 한 번 더 학습한 모델             # Qwen에 \'지문 + 빈 줄 + 지문\'을 넣고 두 번째 사본만 풀링',
    '    lora_test="full",            # 테스트 로짓: "folds" = fold 모델 5개 평균, "full" = train 전체로 Stage 2를 한 번 더 학습한 모델')
sub(8, "    echo_input=True,\n", "    echo_input=True,             # Qwen에 '지문 + 빈 줄 + 지문'을 넣고 두 번째 사본만 풀링\n")

# 15. n-gram 설명
sub(15, "- **func**: 기능어만 남기고 나머지는 `_`로 치환. 문장 구조만 남습니다.\n",
    "- **func**: 기능어만 남기고 나머지는 `_`로 치환. 문장 구조만 남습니다.\n"
    "- **문체 피처 (v5)**: 지문 전체를 세어야 나오는 통계 27개입니다. 문장 길이 평균과 표준편차, 단어 길이, 긴 단어 비율, 어휘 다양도(MATTR), 기능어 비율, 100단어당 구두점(—, ;, :, 따옴표 종류 등)과 축약형·인칭 대명사 수 등입니다. "
    "정의는 `analysis_intra_org/features.py`의 `feats`와 같습니다. n-gram 창(최대 6글자, 3단어)으로는 문장 길이 같은 지문 전체 통계를 볼 수 없어서 따로 넣습니다.\n\n"
    "**문체 블록을 붙이는 방법 (v5).** 피처마다 `log1p`로 큰 값을 누르고, fold의 train 지문으로 구한 평균과 표준편차로 표준화한 뒤 ±4에서 자릅니다. "
    "그다음 `α / √27`을 곱합니다. 표준화한 27차원 벡터의 L2 노름은 평균 √27 ≈ 5.2이므로, 곱한 뒤의 L2 노름은 약 α입니다. "
    "TF-IDF 블록의 L2 노름은 √(1.0² + 0.7² + 0.5² + 0.5²) ≈ 1.4입니다. 문체 블록이 이보다 크면 L2 규제가 걸린 SVC가 27개 열에 먼저 기대게 되므로, "
    "α = 0.2(TF-IDF의 약 1/7)로 작게 붙입니다. train만으로 한 5-fold 비교에서 α = 0.2가 가장 좋았습니다(0.3704 → 0.3864). "
    "표준화 통계를 풀이 아니라 train에서 구하는 이유는 풀이 모델마다 주제가 쏠려 있어서입니다.\n")

# 16. n-gram 코드
s = get(16)
old_after_views = '''def sample_pool(n_per_label, seed=0):'''
stylo_code = '''# v5: 문체 피처 27개 (analysis_intra_org/features.py의 feats와 같은 정의)
STYLO_FUNC = set("""a an the and or but if while although because as so than that which who whom whose this these those
it its they them their he him his she her we us our you your i me my of in on at by for with from to into over
under about between through during not no nor is are was were be been being have has had do does did will would
can could should may might must""".split())
SENT_RE = re.compile(r"(?<=[.!?])\\s+")
WORD_RE = re.compile(r"[A-Za-z’']+")
CONTR_RE = re.compile(r"\\b\\w+(?:n['’]t|['’](?:s|re|ll|ve|d|m))\\b", re.I)
NOT_BUT_RE = re.compile(r"\\bnot\\b[^.;:!?]{1,60}?\\bbut\\b", re.I)
LY_RE = re.compile(r"\\b\\w{3,}ly\\b", re.I)


def stylo_feats(text):
    words = WORD_RE.findall(text)
    nw = max(len(words), 1)
    sents = [s for s in SENT_RE.split(text.strip()) if s]
    sl = np.array([len(WORD_RE.findall(s)) for s in sents]) if sents else np.array([nw])
    lw = [w.lower() for w in words]
    per = lambda c: 100.0 * c / nw
    win = 40
    mattr = (np.mean([len(set(lw[i:i + win])) / win for i in range(0, nw - win + 1, 5)])
             if nw > win else len(set(lw)) / nw)
    first_words = {len(WORD_RE.findall(" ".join(sents[:i]))) for i in range(len(sents))}
    caps = sum(1 for i, w in enumerate(words) if w[0].isupper() and i not in first_words and w not in ("I",))
    return [
        nw, len(sents), sl.mean(), sl.std(),                                    # 단어 수, 문장 수, 문장 길이 평균/표준편차
        np.mean([len(w) for w in words]) if words else 0,                       # 평균 단어 길이
        np.mean([len(w) >= 9 for w in words]) if words else 0,                  # 9글자 이상 단어 비율
        mattr,                                                                  # 어휘 다양도(MATTR, 창 40단어)
        np.mean([w in STYLO_FUNC for w in lw]) if lw else 0,                    # 기능어 비율
        per(text.count("—")), per(text.count(" — ")), per(text.count("–")),     # 이하 100단어당 개수
        per(text.count(";")), per(text.count(":")), per(text.count(",")),
        per(text.count("(")), per(text.count("?")),
        per(text.count("’")), per(text.count("'")),
        per(text.count("“") + text.count("”")), per(text.count('"')),
        per(len(CONTR_RE.findall(text))),                                       # 축약형
        per(sum(w in ("i", "me", "my", "we", "us", "our") for w in lw)),        # 1인칭
        per(sum(w in ("you", "your") for w in lw)),                             # 2인칭
        per(len(NOT_BUT_RE.findall(text))),                                     # not ... but
        per(len(LY_RE.findall(text))),                                          # -ly 단어
        np.mean([bool(re.match(r"(And|But|Yet|So)\\b", s)) for s in sents]) if sents else 0,
        caps / nw,                                                              # 문장 첫 단어가 아닌 대문자 단어 비율
    ]


def stylo_raw(texts):
    """지문마다 27개 피처 -> log1p. 표준화는 fold마다 fit_svc에서 합니다."""
    return np.log1p(np.clip(np.array([stylo_feats(t) for t in texts], dtype=np.float64), 0, None)).astype(np.float32)


def stylo_block(S, stats):
    mu, sd = stats
    Z = np.clip((S - mu) / sd, -4, 4) * (CFG["stylo_alpha"] / np.sqrt(S.shape[1]))
    return sparse.csr_matrix(Z.astype(np.float32))


def with_stylo(X, S, stats):
    return X if stats is None else sparse.hstack([X, stylo_block(S, stats)]).tocsr()


'''
assert s.count(old_after_views) == 1
s = s.replace(old_after_views, stylo_code + old_after_views)

old_feat_print = 'X_test = vectorizer.transform(test.text)\nprint("특성 수:", X_train.shape[1], "|", elapsed())\n'
new_feat_print = '''X_test = vectorizer.transform(test.text)
print("특성 수:", X_train.shape[1], "|", elapsed())

use_stylo = CFG["stylo_alpha"] > 0
S_train = stylo_raw(train.text) if use_stylo else None
S_test = stylo_raw(test.text) if use_stylo else None
S_pool = stylo_raw(ng_pool.text) if use_stylo and ng_pool is not None else None
if use_stylo:
    print(f"문체 피처 {S_train.shape[1]}개, alpha={CFG['stylo_alpha']} | {elapsed()}")
'''
assert s.count(old_feat_print) == 1
s = s.replace(old_feat_print, new_feat_print)

old_fit = '''def fit_svc(train_idx, train_weight):
    Xs, ys, ws = [X_train[train_idx]], [y[train_idx]], [np.full(len(train_idx), train_weight)]
    if X_pool is not None:
        Xs.append(X_pool); ys.append(y_pool); ws.append(np.ones(len(y_pool)))
    model = LinearSVC(C=CFG["svc_c"])
    model.fit(sparse.vstack(Xs).tocsr(), np.concatenate(ys), sample_weight=np.concatenate(ws))
    return model


def ngram_oof(train_weight):
    oof = np.zeros((len(train), K))
    for a, b in cv_folds:
        oof[b] = fit_svc(a, train_weight).decision_function(X_train[b])
    return oof
'''
new_fit = '''def fit_svc(train_idx, train_weight):
    # 문체 블록의 표준화 통계는 이 fold의 train 지문으로만 구합니다(풀은 모델마다 주제가 쏠려 있음).
    stats = None
    if use_stylo:
        St = S_train[train_idx]
        stats = (St.mean(0), St.std(0) + 1e-6)
    Xs = [with_stylo(X_train[train_idx], S_train[train_idx] if use_stylo else None, stats)]
    ys, ws = [y[train_idx]], [np.full(len(train_idx), train_weight)]
    if X_pool is not None:
        Xs.append(with_stylo(X_pool, S_pool, stats)); ys.append(y_pool); ws.append(np.ones(len(y_pool)))
    model = LinearSVC(C=CFG["svc_c"])
    model.fit(sparse.vstack(Xs).tocsr(), np.concatenate(ys), sample_weight=np.concatenate(ws))
    model.stylo_stats = stats
    return model


def svc_scores(model, X, S):
    return model.decision_function(with_stylo(X, S, model.stylo_stats))


def ngram_oof(train_weight):
    oof = np.zeros((len(train), K))
    for a, b in cv_folds:
        oof[b] = svc_scores(fit_svc(a, train_weight), X_train[b], S_train[b] if use_stylo else None)
    return oof
'''
assert s.count(old_fit) == 1
s = s.replace(old_fit, new_fit)

old_test = 'ng_test = fit_svc(np.arange(len(train)), ngram_w).decision_function(X_test)\ndel oof_by_weight\ndel X_pool, ng_pool\n'
new_test = 'ng_test = svc_scores(fit_svc(np.arange(len(train)), ngram_w), X_test, S_test)\ndel oof_by_weight\ndel X_pool, ng_pool, S_pool\n'
assert s.count(old_test) == 1
s = s.replace(old_test, new_test)
put(16, s)

# 출력 지우기
for c in cells:
    if c["cell_type"] == "code":
        c["outputs"] = []
        c["execution_count"] = None
json.dump(nb, open(dst, "w"), ensure_ascii=False, indent=1)
print("저장:", dst)
