"""n-gram 분기의 Open Pool 표본 크기(ngram_pool_per_label)를 키웠을 때의 OOF를 냅니다 (CPU만).

노트북이 ngram_pool_per_label을 키우면 하는 일을 그대로 합니다: sample_pool(n)으로 match_weight에 비례해 뽑고,
그 표본에 TF-IDF 어휘를 다시 맞춥니다. 나머지(문체 27개, stylo_alpha 0.2, C 0.1, 커널 fold)는 v5와 같습니다.

두 단계로 나눠 한 번에 프로세스 하나만 돌립니다.
  prepare <n>     v5 노트북의 설정·데이터·n-gram 셀(벡터화까지)을 ngram_pool_per_label=n으로 실행하고
                  행렬을 out/pool/cache_n<n>/에 저장합니다. 표본이 train에서 얼마나 멀어졌는지도 같이 잽니다.
  fit <이름>      저장한 행렬로 5-fold OOF를 냅니다. 이름: n<n>[_bal][_tw][_lk|_lk10]
                    _bal  풀 행에 라벨별 가중치를 줘서 모든 라벨의 풀 가중치 합을 같게 함
                    _tw   train 가중치를 12 x (풀 행 수 / 45,000)으로 키워 train의 가중치 몫을 기준선과 같게 함
                    _lk   누수 통제: fold마다, 그 fold의 검증 프롬프트 중 하나와 내용어 TF-IDF 코사인이 0.184
                          (같은 프롬프트 train 지문의 1% 분위) 이상인 풀 행을 학습에서 뺌 (leak_check.py가 먼저 필요)
                    _lk10 같은 방식으로 코사인 0.10 이상을 뺌 (훨씬 넓게: fold마다 풀 행의 약 15~20%)
  selftest        메모리 절약 행렬 조립이 노트북 방식(hstack -> vstack -> float64)과 비트 단위로 같은지 확인

메모리: 풀 행렬은 디스크에 두고 mmap으로 읽고, fold마다 float64 CSR을 직접 조립합니다(노트북 방식의 중간 복사본 없음).
fit의 최대 사용량은 약 1.5GB + 28바이트 x (학습 행렬의 0 아닌 원소 수)입니다. 시작 전과 fold마다
MemAvailable이 예상 사용량 + 3GB 이상인지 확인하고, 모자라면 시작하지 않습니다.

사용법: .venv/bin/python -I -W ignore analysis_pool/pool_oof.py prepare 5000
        .venv/bin/python -I -W ignore analysis_pool/pool_oof.py fit n5000_tw
"""
import gc, json, os, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "analysis_corpus"))
import matplotlib
matplotlib.use("Agg")
import numpy as np
import scipy.sparse as sparse

from common import LABELS, ROOT, STYLO_NAMES, nb_cell

OUT = ROOT / "out" / "pool"
MARGIN_GB = 3.0
BASE_ROWS = 45_000  # 기준선 표본(라벨당 2,500개)의 행 수


def mem_available_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 2**20
    return float("inf")


class PeakMemory(threading.Thread):
    """1초마다 /proc/self/status를 읽어 RssAnon(파일 매핑 제외)과 VmRSS의 최댓값을 기록합니다."""

    def __init__(self):
        super().__init__(daemon=True)
        self.anon = self.rss = 0.0
        self.start()

    def run(self):
        while True:
            for line in open("/proc/self/status"):
                if line.startswith("RssAnon"):
                    self.anon = max(self.anon, int(line.split()[1]) / 2**20)
                elif line.startswith("VmRSS"):
                    self.rss = max(self.rss, int(line.split()[1]) / 2**20)
            time.sleep(1)


def require_memory(need_gb, what, wait_min=0):
    """MemAvailable >= need_gb + 3GB일 때만 진행합니다. wait_min 동안 기다려도 안 되면 종료합니다."""
    deadline = time.time() + wait_min * 60
    while True:
        avail = mem_available_gb()
        if avail >= need_gb + MARGIN_GB:
            return avail
        if time.time() >= deadline:
            print(f"건너뜀: {what}의 예상 최대 메모리 {need_gb:.1f}GB + 여유 {MARGIN_GB:.0f}GB > 남은 메모리 {avail:.1f}GB", flush=True)
            sys.exit(3)
        time.sleep(20)


def run_notebook(cells, cfg):
    os.environ.setdefault("NTL_INPUT", str(ROOT / "data"))
    os.environ["NTL_CFG"] = json.dumps({"use_lora": False, "ngram_train_weights": [12.0], "stylo_alpha": 0.2, **cfg})
    G = {"__name__": "ntl"}
    for src in cells:
        exec(src, G)
    return G


def stack_blocks(parts, n_tfidf):
    """vstack([hstack([X, Z]) for X, Z in parts])를 float64 CSR로 직접 만듭니다.

    parts의 원소는 (X, Z) 또는 (X, Z, rows)입니다. X는 (data, indices, indptr) 세 배열(mmap 가능), Z는 문체 블록
    CSR이고, rows를 주면 X에서 그 행들만 씁니다(Z는 이미 그 행들에 대한 것). 행 안의 순서는 X의 원소, 그다음 Z의
    원소입니다. 노트북 방식(hstack -> vstack)의 결과는 행 안이 열 번호 순으로 정렬돼 있으므로, X의 행 안이 정렬돼
    있어야 그 결과와 같아집니다(fit_transform이 낸 풀 행렬은 정렬돼 있지 않아서 prepare에서 정렬해 저장합니다).
    중간 복사본을 만들지 않으려고 행 묶음 단위로 채웁니다.
    """
    parts = [(p[0], p[1], p[2] if len(p) > 2 else np.arange(p[1].shape[0])) for p in parts]
    lens = [np.asarray(xp, dtype=np.int64)[rows + 1] - np.asarray(xp, dtype=np.int64)[rows] for (_, _, xp), _, rows in parts]
    nnz = int(sum(lx.sum() + z.nnz for lx, (_, z, _) in zip(lens, parts)))
    n_rows = sum(z.shape[0] for _, z, _ in parts)
    assert nnz < 2**31
    data, indices = np.empty(nnz, np.float64), np.empty(nnz, np.int32)
    indptr = np.zeros(n_rows + 1, np.int32)
    off, r0 = 0, 0
    for ((xd, xi, xp), z, rows), lx in zip(parts, lens):
        n = z.shape[0]
        assert len(rows) == n
        xp = np.asarray(xp, dtype=np.int64); zp = z.indptr.astype(np.int64)
        lz = np.diff(zp)
        starts = off + np.concatenate([[0], np.cumsum(lx + lz)])
        indptr[r0 + 1:r0 + n + 1] = starts[1:]
        for c0 in range(0, n, 2000):
            c1 = min(c0 + 2000, n)
            l = lx[c0:c1]
            within = np.arange(l.sum()) - np.repeat(np.cumsum(l) - l, l)
            src = np.repeat(xp[rows[c0:c1]], l) + within
            dest = np.repeat(starts[c0:c1], l) + within
            if len(src) and src[-1] - src[0] + 1 == len(src):  # 연속한 행이면 한 번에 읽음
                data[dest] = xd[src[0]:src[-1] + 1]; indices[dest] = xi[src[0]:src[-1] + 1]
            else:
                data[dest] = xd[src]; indices[dest] = xi[src]
            rz = np.repeat(np.arange(c0, c1), lz[c0:c1])
            dest = starts[rz] + lx[rz] + (np.arange(zp[c0], zp[c1]) - zp[rz])
            data[dest] = z.data[zp[c0]:zp[c1]]; indices[dest] = z.indices[zp[c0]:zp[c1]] + n_tfidf
        off, r0 = int(starts[-1]), r0 + n
    return sparse.csr_matrix((data, indices, indptr), shape=(n_rows, n_tfidf + parts[0][1].shape[1]), copy=False)


def csr_parts(X):
    return X.data, X.indices, X.indptr


def drift(S_train, y_train, S_pool, y_pool):
    """표본이 train에서 얼마나 먼지: 문체 27개로 train과 구별되는 AUC, 라벨 평균의 train과의 거리."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    idx = np.random.default_rng(0).choice(len(S_pool), min(len(S_pool), 8000), replace=False)
    X = np.vstack([S_train, S_pool[idx]]); t = np.r_[np.zeros(len(S_train)), np.ones(len(idx))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    p = cross_val_predict(clf, X, t, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    raw_t, raw_p = np.expm1(S_train), np.expm1(S_pool)
    sd = raw_t.std(0) + 1e-9
    dist = [np.mean(np.abs(raw_p[y_pool == k].mean(0) - raw_t[y_train == k].mean(0)) / sd) for k in range(len(LABELS))]
    return float(roc_auc_score(t, p)), float(np.mean(dist)), float(np.max(dist))


def prepare(n):
    cache = OUT / f"cache_n{n}"
    if (cache / "meta.json").exists():
        print(f"prepare n={n}: 이미 있음 ({cache})"); return
    pool_sizes = [3541, 4566, 9880, 10513, 10516, 11168, 11841, 12158, 12450, 12745, 12773, 13248, 14315, 14720, 15719,
                  18447, 18894, 23542]  # 라벨별 Open Pool 행 수(메모리 예상에만 씀)
    rows_est = sum(min(n, s) for s in pool_sizes)
    # 실측 최대 메모리(GB): 45,000행 2.4 / 63,000행 4.0 / 88,107행 5.5 / 128,107행 7.0.
    # 잰 범위 안에서는 바로 위 실측값 + 0.5GB, 그 밖은 마지막 실측값에서 전체 평균 기울기(행당 55KB)로 늘립니다.
    above = [(r, g) for r, g in PREP_MEASURED if r >= rows_est]
    need = above[0][1] + 0.5 if above else PREP_MEASURED[-1][1] + (rows_est - PREP_MEASURED[-1][0]) * PREP_KB_PER_ROW / 2**20
    avail = require_memory(need, f"prepare n={n} (약 {rows_est:,}행)")
    print(f"prepare n={n}: 약 {rows_est:,}행, 예상 최대 메모리 {need:.1f}GB, 남은 메모리 {avail:.1f}GB", flush=True)
    peak, t0 = PeakMemory(), time.time()
    G = run_notebook([nb_cell("import gc, glob"), nb_cell("COMP = locate_competition_dir()"),
                      nb_cell("FUNCTION_WORDS").split("weight_grid = ")[0]], {"ngram_pool_per_label": n})
    ng, Xp, Xt = G["ng_pool"], G["X_pool"], G["X_train"]
    assert Xp.dtype == np.float32 and Xp.indices.dtype == np.int32 and G["LABELS"] == LABELS
    Xp.sort_indices()  # 값은 그대로, 행 안의 원소 순서만 열 번호 순으로 (stack_blocks 설명 참고)
    assert Xt.has_sorted_indices
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "Xp_data.npy", Xp.data); np.save(cache / "Xp_indices.npy", Xp.indices)
    np.save(cache / "Xp_indptr.npy", Xp.indptr)
    sparse.save_npz(cache / "X_train.npz", Xt, compressed=False)
    np.save(cache / "S_train.npy", G["S_train"]); np.save(cache / "S_pool.npy", G["S_pool"])
    np.save(cache / "y_pool.npy", G["y_pool"])
    counts = np.bincount(G["y_pool"], minlength=len(LABELS))
    mw = ng.match_weight.to_numpy(dtype=float)
    auc, dist_mean, dist_max = drift(G["S_train"], G["y"], G["S_pool"], G["y_pool"])
    meta = dict(n=n, rows=int(len(ng)), nnz_pool=int(Xp.nnz), nnz_train=int(Xt.nnz), n_features=int(Xp.shape[1]),
                counts=counts.tolist(), match_weight_mean=float(mw.mean()), match_weight_median=float(np.median(mw)),
                match_weight_by_label=[float(mw[G["y_pool"] == k].mean()) for k in range(len(LABELS))],
                share_used=[float(c / s) for c, s in zip(counts, G["pool"].groupby("model").size().reindex(LABELS))],
                drift_auc=auc, drift_label_dist_mean=dist_mean, drift_label_dist_max=dist_max,
                prepare_min=(time.time() - t0) / 60, prepare_peak_anon_gb=peak.anon)
    (cache / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"prepare n={n}: 풀 표본 {meta['rows']:,}행 (라벨별 {meta['counts']}), 특성 {meta['n_features']:,}개, "
          f"풀 행렬 nnz {meta['nnz_pool'] / 1e6:.0f}M | match_weight 평균 {meta['match_weight_mean']:.3f} "
          f"중앙값 {meta['match_weight_median']:.3f} | train 대 표본 AUC {auc:.3f}, 라벨 평균 거리 {dist_mean:.3f} (최대 {dist_max:.3f}) "
          f"| {meta['prepare_min']:.1f}분, 최대 메모리 {peak.anon:.1f}GB", flush=True)


def load_cache(n):
    cache = OUT / f"cache_n{n}"
    meta = json.loads((cache / "meta.json").read_text())
    Xp = (np.load(cache / "Xp_data.npy", mmap_mode="r"), np.load(cache / "Xp_indices.npy", mmap_mode="r"),
          np.load(cache / "Xp_indptr.npy"))
    return meta, Xp, sparse.load_npz(cache / "X_train.npz").tocsr(), np.load(cache / "S_train.npy"), \
        np.load(cache / "S_pool.npy"), np.load(cache / "y_pool.npy")


def notebook_defs():
    """RNG 상태를 alpha_sweep.py와 맞추려고 설정·데이터·검증 셀을 실행합니다(검증 셀의 LinearSVC 10번이 전역 난수를 씀)."""
    G = run_notebook([nb_cell("import gc, glob"), nb_cell("COMP = locate_competition_dir()"), nb_cell("cv_folds = ")], {})
    del G["pool"]; gc.collect()
    ngram = nb_cell("FUNCTION_WORDS")
    exec(ngram[ngram.index("def stylo_block"):ngram.index("def sample_pool")], G)  # stylo_block, with_stylo
    v4 = np.load(ROOT / "analysis_v4" / "oof_v4.npz", allow_pickle=True)
    assert (v4["y"] == G["y"]).all() and (v4["prompt_id"] == G["train"].prompt_id.values).all()
    G["cv_folds"] = [(np.where(v4["fold"] != k)[0], np.where(v4["fold"] == k)[0]) for k in range(G["CFG"]["n_folds"])]
    return G


def selftest():
    G = notebook_defs()
    meta, Xp, Xt, St, Sp, yp = load_cache(2500)
    m = 400
    Xp_small = sparse.csr_matrix((np.array(Xp[0][:Xp[2][m]]), np.array(Xp[1][:Xp[2][m]]), Xp[2][:m + 1]), shape=(m, meta["n_features"]))
    a = np.arange(0, 300)
    stats = (St[a].mean(0), St[a].std(0) + 1e-6)
    ref = sparse.vstack([G["with_stylo"](Xt[a], St[a], stats), G["with_stylo"](Xp_small, Sp[:m], stats)]).tocsr().astype(np.float64)
    got = stack_blocks([(csr_parts(Xt[a]), G["stylo_block"](St[a], stats)), (csr_parts(Xp_small), G["stylo_block"](Sp[:m], stats))],
                       meta["n_features"])
    same = (ref.shape == got.shape and np.array_equal(ref.indptr, got.indptr) and np.array_equal(ref.indices, got.indices)
            and np.array_equal(ref.data, got.data))
    print(f"selftest: 조립한 행렬이 노트북 방식과 비트 단위로 같은가: {same} (행 {got.shape[0]}, nnz {got.nnz:,})", flush=True)
    sys.exit(0 if same else 1)


def fit(name):
    spec = name.split("_")
    n, balanced, scale_tw = int(spec[0][1:]), "bal" in spec[1:], "tw" in spec[1:]
    leak_tau = "lk" if "lk" in spec[1:] else 0.10 if "lk10" in spec[1:] else None
    if not (OUT / f"cache_n{n}" / "meta.json").exists():
        print(f"건너뜀: fit {name} (prepare {n}의 결과가 없음)", flush=True); sys.exit(3)
    meta = json.loads((OUT / f"cache_n{n}" / "meta.json").read_text())
    nnz_fit = meta["nnz_pool"] + meta["nnz_train"] * 0.8 + (meta["rows"] + 2160) * 27
    need = 1.5 + 28 * nnz_fit / 2**30
    avail = require_memory(need, f"fit {name}")
    train_weight = 12.0 * meta["rows"] / BASE_ROWS if scale_tw else 12.0
    print(f"fit {name}: 풀 {meta['rows']:,}행, train 가중치 {train_weight:.2f}, 라벨 균형 가중치 {balanced} | "
          f"예상 최대 메모리 {need:.1f}GB, 남은 메모리 {avail:.1f}GB", flush=True)
    peak, t0 = PeakMemory(), time.time()
    G = notebook_defs()
    y, K = G["y"], len(LABELS)
    _, Xp, Xt, St, Sp, yp = load_cache(n)
    counts = np.bincount(yp, minlength=K)
    w_pool = (len(yp) / K / counts)[yp] if balanced else np.ones(len(yp))
    BA = G["balanced_accuracy_score"]
    oof, n_iter, fold_min, dropped = np.zeros((len(y), K)), [], [], []
    if leak_tau is not None:
        leak = np.load(OUT / f"cache_n{n}" / "leak.npz", allow_pickle=True)
        leak_tau = float(leak["tau1"]) if leak_tau == "lk" else leak_tau
        prompt_col = {p: i for i, p in enumerate(leak["prompt_ids"])}
    for a, b in G["cv_folds"]:
        require_memory(need, f"fit {name}의 다음 fold", wait_min=30)
        t1 = time.time()
        stats = (St[a].mean(0), St[a].std(0) + 1e-6)  # 문체 표준화 통계는 fold의 train 지문으로만 (노트북과 같음)
        if leak_tau is None:
            keep = np.arange(len(yp))
        else:  # 이 fold의 검증 프롬프트와 주제가 가까운 풀 행을 뺌
            cols = sorted({prompt_col[p] for p in G["train"].prompt_id.values[b]})
            keep = np.where(leak["sim_all"][:, cols].max(1) < leak_tau)[0]
        dropped.append(len(yp) - len(keep))
        X_fit = stack_blocks([(csr_parts(Xt[a]), G["stylo_block"](St[a], stats)), (Xp, G["stylo_block"](Sp[keep], stats), keep)],
                             meta["n_features"])
        model = G["LinearSVC"](C=G["CFG"]["svc_c"])
        model.fit(X_fit, np.concatenate([y[a], yp[keep]]),
                  sample_weight=np.concatenate([np.full(len(a), train_weight), w_pool[keep]]))
        del X_fit; gc.collect()
        oof[b] = model.decision_function(G["with_stylo"](Xt[b], St[b], stats))
        n_iter.append(int(np.max(model.n_iter_))); fold_min.append((time.time() - t1) / 60)
        del model
    ba = BA(y, oof.argmax(1))
    folds = [round(float(BA(y[b], oof[b].argmax(1))), 3) for _, b in G["cv_folds"]]
    train_share = train_weight * len(y) / (train_weight * len(y) + len(yp))
    np.savez_compressed(OUT / f"ngram_oof_{name}.npz", oof_ng=oof, n=n, rows=meta["rows"], train_weight=train_weight,
                        balanced=balanced, fit_min=sum(fold_min), fold_min=np.array(fold_min), peak_anon_gb=peak.anon,
                        n_iter=np.array(n_iter), dropped=np.array(dropped), leak_tau=-1.0 if leak_tau is None else leak_tau)
    if leak_tau is not None:
        print(f"{name}: 누수 통제(코사인 >= {leak_tau:.3f}) fold별로 뺀 풀 행 {dropped} (평균 {np.mean(dropped) / len(yp):.3f})", flush=True)
    print(f"{name}: n-gram OOF BA {ba:.4f} (fold별 {folds}) | train 가중치 몫 {train_share:.3f} | liblinear 반복 수 {n_iter} | "
          f"fit {sum(fold_min):.1f}분 (fold당 {np.mean(fold_min):.1f}분), 전체 {(time.time() - t0) / 60:.1f}분, "
          f"최대 메모리 {peak.anon:.1f}GB", flush=True)
    if name == "n2500":
        ref = np.load(ROOT / "out" / "v5" / "ngram_oof_alpha0.2.npz")["oof_ng"]
        print(f"n2500 대 기준선(out/v5/ngram_oof_alpha0.2.npz): 점수 최대 차이 {np.abs(oof - ref).max():.2e}, "
              f"argmax가 다른 행 {int((oof.argmax(1) != ref.argmax(1)).sum())}", flush=True)


PREP_MEASURED = [(45_000, 2.4), (63_000, 4.0), (88_107, 5.5), (128_107, 7.0)]  # prepare의 (행 수, 실측 최대 메모리 GB)
PREP_KB_PER_ROW = float(os.environ.get("NTL_PREP_KB_PER_ROW", "55"))  # 잰 범위 밖에서 행 하나당 더하는 메모리(KB)

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "prepare":
        prepare(int(args[0]))
    elif cmd == "fit":
        fit(args[0])
    elif cmd == "selftest":
        selftest()
