"""train만으로: v4 4뷰 TF-IDF + LinearSVC에 문체 피처를 붙이는 방식 비교."""
import re, sys, time
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.svm import LinearSVC

ROOT = sys.argv[1] if len(sys.argv) > 1 else str(__import__("pathlib").Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT + "/analysis_intra_org")
from features import feats  # 27개 문체 피처

train = pd.read_csv(ROOT + "/data/train.csv")
train["text"] = train.text.astype(str)
LABELS = sorted(train.model.unique())
y = train.model.map({m: i for i, m in enumerate(LABELS)}).values
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(train, y, train.prompt_id.values))

FUNCTION_WORDS = set("""a an the and or but if while although because as so than that which who whom whose this these
those it its they them their he him his she her we us our you your i me my of in on at by for with from to into over
under about between through during not no nor is are was were be been being has have had do does did will would can
could may might must shall should very more most such only own same just also""".split())
to_shape = lambda s: re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", s)
to_function = lambda s: " ".join(w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "_"
                                 for w in re.findall(r"\w+|[^\w\s]", s))
SENT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-z’']+")


def to_rhythm(s):
    """문장마다 '길이 구간 + 끝 부호 + 문장 안 구두점 수' 토큰 하나."""
    toks = []
    for sent in [x for x in SENT_RE.split(s.strip()) if x]:
        n = len(WORD_RE.findall(sent))
        b = next(i for i, t in enumerate([5, 10, 15, 20, 25, 30, 40, 10**9]) if n <= t)
        p = sum(sent.count(c) for c in ",;:—")
        toks.append(f"L{b}p{min(p, 4)}{'q' if sent.endswith('?') else ''}")
    return " ".join(toks) or "EMPTY"


VIEWS = [
    ("char", None, dict(analyzer="char", ngram_range=(3, 5), max_features=250_000), 1.0),
    ("word", None, dict(analyzer="word", token_pattern=r"\w+|[^\w\s]", ngram_range=(1, 2), max_features=150_000), 0.7),
    ("shape", to_shape, dict(analyzer="char", ngram_range=(2, 6), max_features=60_000), 0.5),
    ("func", to_function, dict(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 3), max_features=50_000), 0.5),
]
t0 = time.time()
mats = {}
for name, fn, opts, w in VIEWS:
    txt = [fn(t) for t in train.text] if fn else train.text.tolist()
    mats[name] = TfidfVectorizer(lowercase=False, sublinear_tf=True, min_df=2, dtype=np.float32, **opts).fit_transform(txt) * w
rhythm = TfidfVectorizer(lowercase=False, sublinear_tf=True, min_df=2, token_pattern=r"\S+", ngram_range=(1, 3),
                         dtype=np.float32).fit_transform([to_rhythm(t) for t in train.text])
BASE = sp.hstack(list(mats.values())).tocsr()
F = pd.DataFrame([feats(t) for t in train.text])
F = F.loc[:, F.std() > 0]
print(f"피처 준비 {time.time() - t0:.0f}s, TF-IDF {BASE.shape[1]:,}개, 문체 {F.shape[1]}개", flush=True)


def dense_block(a, b, alpha):
    """fold 학습 지문으로 log1p + 표준화. 행 노름이 대략 alpha가 되도록 sqrt(피처 수)로 나눔."""
    X = np.log1p(F.clip(lower=0).values)
    mu, sd = X[a].mean(0), X[a].std(0) + 1e-6
    Z = np.clip((X - mu) / sd, -4, 4) * alpha / np.sqrt(X.shape[1])
    return sp.csr_matrix(Z.astype(np.float32))


def binned_block(a, b, alpha, n_bins=6):
    """fold 학습 지문의 분위수로 구간을 나눠 one-hot. 비선형(예: 0이냐 아니냐)을 선형 SVC가 쓸 수 있게."""
    kb = KBinsDiscretizer(n_bins=n_bins, encode="onehot", strategy="quantile", subsample=None)
    import warnings; warnings.filterwarnings("ignore")
    kb.fit(F.values[a])
    return (kb.transform(F.values) * (alpha / np.sqrt(F.shape[1]))).tocsr().astype(np.float32)


def cv(make_extra=None, C=0.1):
    pred = np.zeros(len(y), dtype=int)
    for a, b in folds:
        X = BASE if make_extra is None else sp.hstack([BASE, make_extra(a, b)]).tocsr()
        pred[b] = LinearSVC(C=C).fit(X[a], y[a]).predict(X[b])
    fold_ba = [balanced_accuracy_score(y[b], pred[b]) for _, b in folds]
    return balanced_accuracy_score(y, pred), fold_ba


res = []
def run(name, fn=None):
    ba, fb = cv(fn)
    res.append((name, ba, fb))
    print(f"{name:<34} OOF BA {ba:.4f}  fold {np.round(fb, 3).tolist()}  ({time.time() - t0:.0f}s)", flush=True)


run("기준: 4뷰 TF-IDF")
for al in (0.3, 0.6, 1.0):
    run(f"+ 연속 문체 27개 (alpha={al})", lambda a, b, al=al: dense_block(a, b, al))
for al in (0.3, 0.6, 1.0):
    run(f"+ 구간 one-hot 문체 (alpha={al})", lambda a, b, al=al: binned_block(a, b, al))
for w in (0.3, 0.5):
    run(f"+ 문장 리듬 뷰 (w={w})", lambda a, b, w=w: rhythm * w)
run("+ 구간 one-hot(0.6) + 리듬(0.3)", lambda a, b: sp.hstack([binned_block(a, b, 0.6), rhythm * 0.3]))

# 참고: 문체 피처만으로
pred = np.zeros(len(y), dtype=int)
for a, b in folds:
    Z = binned_block(a, b, 1.0)
    pred[b] = LinearSVC(C=1.0).fit(Z[a], y[a]).predict(Z[b])
print(f"{'참고: 구간 one-hot 문체만':<34} OOF BA {balanced_accuracy_score(y, pred):.4f}")
