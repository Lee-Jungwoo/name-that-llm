"""n-gram 분기의 학습 행에 Response Corpus(short 뷰)를 더했을 때의 OOF를 냅니다 (CPU만).

v5_stylo/alpha_sweep.py와 같은 방식으로 v5 노트북의 코드 셀(설정, 데이터, 검증, n-gram)을 그대로 실행하고,
fold는 제출 커널의 분할(analysis_v4/oof_v4.npz)로 바꿉니다. 기준선(out/v5/ngram_oof_alpha0.2.npz)과 다른 점은
LinearSVC 학습 행에 corpus 행이 더 들어가는 것 하나뿐입니다.

- Open Pool 표본(라벨당 2,500개)과 TF-IDF 어휘는 기준선 그대로입니다(벡터라이저는 풀 표본에만 맞춤).
- corpus 행은 벡터라이저의 transform만 거칩니다. 문체 블록의 표준화 통계도 기준선처럼 fold의 train 지문으로만 구합니다.
- stylo_alpha=0.2, train 가중치 12, C=0.1은 그대로입니다.

실행마다 부모에서 fork한 프로세스 하나를 쓰고, 한 번에 하나만 돌립니다(메모리: 부모 약 3.5GB + 실행 하나 약 5GB,
5-fold 한 번에 15~17분). 남은 메모리가 7GB보다 적으면 실행과 fold를 시작하지 않고 기다립니다.

사용법: .venv/bin/python -I -W ignore analysis_corpus/corpus_oof.py [실행 이름 ...]
  check          corpus 없이 fold 0만 학습해 기준선 OOF의 fold 0 점수와 같은지 확인(파이프라인 점검)
  all_w1 등      RUNS에 있는 corpus 변형
  poolextra_w1   대조군: corpus 대신 Open Pool에서 라벨별로 같은 개수를 더 뽑아 넣음
"""
import json, multiprocessing as mp, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")

from common import LABELS, MARKDOWN_RE, OUT, REFUSAL_RE, ROOT, load_corpus, nb_cell

OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NTL_INPUT", str(ROOT / "data"))
os.environ["NTL_CFG"] = json.dumps({"use_lora": False, "ngram_train_weights": [12.0], "stylo_alpha": 0.2})

G = {"__name__": "ntl"}
for src in (nb_cell("import gc, glob"), nb_cell("COMP = locate_competition_dir()"), nb_cell("cv_folds = "),
            nb_cell("FUNCTION_WORDS").split("weight_grid = ")[0]):
    exec(src, G)
np, sparse, y, CFG = G["np"], G["sparse"], G["y"], G["CFG"]
assert G["LABELS"] == LABELS and G["pool_path"] is not None and "response-corpus" not in str(G["pool_path"])

v4 = np.load(ROOT / "analysis_v4" / "oof_v4.npz", allow_pickle=True)
assert (v4["y"] == y).all() and (v4["prompt_id"] == G["train"].prompt_id.values).all()
G["cv_folds"] = [(np.where(v4["fold"] != k)[0], np.where(v4["fold"] == k)[0]) for k in range(CFG["n_folds"])]
print("fold: 커널 분할(oof_v4.npz) 사용", flush=True)

# corpus: train이나 Open Pool 표본과 정규화 텍스트가 같은 셀, 같은 라벨 안 중복은 뺍니다.
norm = G["norm"]
corpus = load_corpus()
seen = set(G["train"].text.map(norm)) | set(G["test"].text.map(norm)) | set(G["ng_pool"].text.map(norm))
corpus = corpus[~corpus.norm.isin(seen)].drop_duplicates(["model", "norm"]).reset_index(drop=True)
cross = corpus.groupby("norm").model.transform("nunique")
corpus = corpus[cross == 1].reset_index(drop=True)  # 서로 다른 라벨에 같은 텍스트가 있으면 둘 다 뺌
corpus["plain"] = (~corpus.text.str.contains("\n") & ~corpus.text.map(lambda s: bool(MARKDOWN_RE.search(s)))
                   & ~corpus.text.map(lambda s: bool(REFUSAL_RE.match(s))))
X_ext = G["vectorizer"].transform(corpus.text)
S_ext = G["stylo_raw"](corpus.text)
y_ext = corpus.model.map(G["L2I"]).values
print(f"corpus: {len(corpus):,}행 (중복 제거 후), X_ext {X_ext.shape} | {G['elapsed']()}", flush=True)


def capped(mask, cap, seed=0):
    """mask 안에서 라벨마다 최대 cap개를 무작위로 고릅니다."""
    rng = np.random.default_rng(seed)
    keep = []
    for k in range(len(LABELS)):
        idx = np.where(mask & (y_ext == k))[0]
        keep.append(idx if len(idx) <= cap else rng.choice(idx, cap, replace=False))
    return np.sort(np.concatenate(keep))


ALL = np.ones(len(corpus), dtype=bool)
# (이름, corpus 행 인덱스, corpus 행의 샘플 가중치). 풀 행은 1, train 행은 12입니다.
RUNS = {
    "check": (np.array([], dtype=int), 1.0),
    "all_w1": (np.where(ALL)[0], 1.0),
    "all_w0.3": (np.where(ALL)[0], 0.3),
    "cap800_w1": (capped(ALL, 800), 1.0),
    "plain_w1": (np.where(corpus.plain.values)[0], 1.0),  # 한 문단, 마크다운 없음, 거절 아님 (train과 같은 겉모습)
}


# 대조군: corpus 대신 Open Pool에서 라벨별로 같은 개수를 더 뽑아 넣습니다(기준선 표본 45,000개와 겹치지 않게).
# "행을 더 넣는 것 자체"의 효과와 "corpus라서" 생기는 효과를 나눠 보기 위한 것입니다. 이름을 줄 때만 준비합니다.
if "poolextra_w1" in sys.argv[1:]:
    rest = G["pool"][~G["pool"].text.isin(set(G["ng_pool"].text))]
    rng = np.random.default_rng(1)
    picks = []
    for label, g in rest.groupby("model"):
        w = g.match_weight.fillna(0).to_numpy(dtype=float)
        k = min(int((y_ext == G["L2I"][label]).sum()), int((w > 0).sum()))
        picks.append(g.index.to_numpy()[rng.choice(len(g), k, replace=False, p=w / w.sum())])
    extra = rest.loc[np.concatenate(picks)]
    X_px, S_px, y_px = G["vectorizer"].transform(extra.text), G["stylo_raw"](extra.text), extra.model.map(G["L2I"]).values
    print(f"대조군: Open Pool 추가 표본 {len(extra):,}행 | {G['elapsed']()}", flush=True)


def run(name):
    ext_idx, ext_w = RUNS.get(name, (None, 1.0))
    if name == "poolextra_w1":
        ext_idx, Xe, Se, ye = np.arange(len(y_px)), X_px, S_px, y_px
    else:
        Xe, Se, ye = X_ext[ext_idx], S_ext[ext_idx], y_ext[ext_idx]
    X_train, S_train, X_pool, S_pool, y_pool = (G[k] for k in ("X_train", "S_train", "X_pool", "S_pool", "y_pool"))
    with_stylo = G["with_stylo"]
    n_iter = []

    def fit_svc(train_idx, train_weight):
        # 노트북의 fit_svc와 같고, 풀 행 뒤에 corpus 행을 더하는 것만 다릅니다.
        # 메모리: 노트북은 float32 조각들을 쥔 채로 fit에 넘기고 scikit-learn이 float64로 한 번 더 복사합니다.
        # 여기서는 float64 행렬을 먼저 만들고 조각을 지운 뒤 fit합니다(값은 같음, check로 확인). corpus 행이 길어서
        # 이렇게 하지 않으면 실행 하나가 약 8GB를 씁니다.
        while mem_available_gb() < 7:
            time.sleep(20)
        St = S_train[train_idx]
        stats = (St.mean(0), St.std(0) + 1e-6)
        Xs = [with_stylo(X_train[train_idx], S_train[train_idx], stats), with_stylo(X_pool, S_pool, stats)]
        ys, ws = [y[train_idx], y_pool], [np.full(len(train_idx), train_weight), np.ones(len(y_pool))]
        if len(ext_idx):
            Xs.append(with_stylo(Xe, Se, stats)); ys.append(ye); ws.append(np.full(len(ye), ext_w))
        X_fit = sparse.vstack(Xs, format="csr", dtype=np.float64)
        del Xs
        model = G["LinearSVC"](C=CFG["svc_c"])
        model.fit(X_fit, np.concatenate(ys), sample_weight=np.concatenate(ws))
        del X_fit
        model.stylo_stats = stats
        n_iter.append(int(np.max(model.n_iter_)))
        return model

    BA = G["balanced_accuracy_score"]
    if name == "check":
        a, b = G["cv_folds"][0]
        got = G["svc_scores"](fit_svc(a, 12.0), X_train[b], S_train[b])
        ref = np.load(ROOT / "out" / "v5" / "ngram_oof_alpha0.2.npz")["oof_ng"][b]
        print(f"check: fold 0 BA {BA(y[b], got.argmax(1)):.4f} (기준선 {BA(y[b], ref.argmax(1)):.4f}), "
              f"점수 최대 차이 {np.abs(got - ref).max():.2e}, argmax가 다른 행 {int((got.argmax(1) != ref.argmax(1)).sum())} "
              f"| {G['elapsed']()}", flush=True)
        return
    oof = np.zeros((len(y), len(LABELS)))
    for a, b in G["cv_folds"]:
        oof[b] = G["svc_scores"](fit_svc(a, 12.0), X_train[b], S_train[b])
    folds = [round(float(BA(y[b], oof[b].argmax(1))), 3) for _, b in G["cv_folds"]]
    ids = np.arange(len(ye)) if name == "poolextra_w1" else corpus.cell_id.values[ext_idx]
    np.savez_compressed(OUT / f"ngram_oof_{name}.npz", oof_ng=oof, ext_cell_id=ids, ext_weight=ext_w)
    per_label = np.bincount(ye, minlength=len(LABELS)).tolist()
    print(f"{name}: 추가 {len(ext_idx):,}행 x 가중치 {ext_w} (라벨별 {per_label}) -> n-gram OOF BA {BA(y, oof.argmax(1)):.4f} "
          f"(fold별 {folds}) | liblinear 반복 수 {n_iter} | {G['elapsed']()}", flush=True)


def mem_available_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 2**20
    return float("inf")


if __name__ == "__main__":
    names = sys.argv[1:] or ["check"]
    ctx = mp.get_context("fork")
    failed = []
    for name in names:
        while mem_available_gb() < 7:
            time.sleep(20)
        p = ctx.Process(target=run, args=(name,))
        p.start(); p.join()
        if p.exitcode != 0:
            failed.append(name)
            print(f"{name}: 실패 (exitcode {p.exitcode})", flush=True)
    sys.exit(1 if failed else 0)
