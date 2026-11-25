"""v8 노트북을 커널에 올리기 전에 로컬 CPU에서 점검합니다 (GPU·제출 없음).

  parse        모든 코드 셀이 파싱되는지, v7 대비 바뀐 셀이 어느 것인지 출력
  ngram        노트북의 설정·데이터·검증·n-gram 셀을 그대로 실행하고(라벨당 5,000개, 라벨 균형 가중치, 문체 블록),
               풀 크기 실험이 저장한 값과 비교합니다.
                 - 벡터화 직후: 표본·행렬·문체 피처·풀 행 가중치를 out/pool/cache_n5000과 비교
                 - fold마다: OOF 점수를 out/pool/ngram_oof_n5000_bal.npz(n-gram OOF 0.5012)와 비교
               그다음 저장된 LoRA·DeBERTa OOF로 5장 이후 셀(블렌드, 게이트, 중심화, Sinkhorn, 제출 파일)까지 실행합니다.
  downstream [npz] [n_test]
               n-gram을 다시 학습하지 않고, 저장된 n-gram OOF(기본 out/pool/ngram_oof_n5000_bal.npz)로 5장 이후 셀만 실행합니다.
               n_test가 3이면 실제 test.csv 3행, 그보다 크면 fold 0의 train 지문을 가짜 테스트로 넣어 중심화와 Sinkhorn이
               실제로 도는지 봅니다(이때 테스트 로짓은 그 지문들의 OOF 로짓).

fold는 로컬 scikit-learn(1.9.1)의 분할이 커널과 달라서 analysis_v4/oof_v4.npz의 fold(= Kaggle 커널의 분할)로 바꿉니다.
검증 셀은 그대로 실행합니다. LinearSVC에 random_state가 없어서, 앞에서 몇 번 fit했는지가 뒤 fit의 시드를 정하기 때문입니다
(analysis_pool/pool_oof.py와 같은 순서).

메모리(ngram): 약 1.5GB + 풀 행렬 float32(8바이트 x nnz) + fit 중 float64 행렬과 liblinear 사본(28바이트 x nnz) = 약 8.5GB.
시작 전에 MemAvailable이 예상치 + 3GB 이상인지 확인하고, 실행 중 MemAvailable이 2GB 아래로 내려가면 스스로 끝냅니다.

사용: .venv/bin/python -W ignore analysis_v8/verify_v8.py ngram > analysis_v8/logs/verify_ngram_local.log 2>&1
      .venv/bin/python -W ignore analysis_v8/verify_v8.py downstream
      .venv/bin/python -W ignore analysis_v8/verify_v8.py downstream out/pool/ngram_oof_n5000_bal.npz 540
"""
import ast, gc, json, os, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "name_that_llm_v8.ipynb"
WORK = ROOT / "out" / "v8" / "local"
EST_GB, MARGIN_GB, ABORT_GB = 8.5, 3.0, 2.0


def code_cells(path=NB):
    nb = json.load(open(path, encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def find(cells, prefix):
    return next(s for s in cells if s.startswith(prefix))


def mem_available_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 2**20
    return float("inf")


class MemoryWatch(threading.Thread):
    """1초마다 RssAnon 최댓값과 MemAvailable 최솟값을 기록하고, MemAvailable이 ABORT_GB 아래면 프로세스를 끝냅니다."""

    def __init__(self):
        super().__init__(daemon=True)
        self.anon, self.min_avail = 0.0, float("inf")
        self.start()

    def run(self):
        while True:
            for line in open("/proc/self/status"):
                if line.startswith("RssAnon"):
                    self.anon = max(self.anon, int(line.split()[1]) / 2**20)
            avail = mem_available_gb()
            self.min_avail = min(self.min_avail, avail)
            if avail < ABORT_GB:
                print(f"중단: MemAvailable {avail:.1f}GB < {ABORT_GB}GB (이 프로세스 RssAnon {self.anon:.1f}GB)", flush=True)
                os._exit(4)
            time.sleep(1)


def parse():
    v8, v7 = code_cells(), code_cells(ROOT / "name_that_llm_v7.ipynb")
    for s in v8:
        ast.parse(s)
    changed = [s.split("\n", 1)[0][:50] for s, t in zip(v8, v7) if s != t]
    print(f"코드 셀 {len(v8)}개 모두 파싱됨 | v7과 다른 코드 셀 {len(changed)}개: {changed}")
    assert len(v8) == len(v7)


def setup(cells):
    """설정, 데이터, 문장부호 그림, 검증 셀을 노트북 순서대로 실행하고 fold를 커널 분할로 바꿉니다."""
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    os.environ.setdefault("NTL_INPUT", str(ROOT / "data"))
    WORK.mkdir(parents=True, exist_ok=True)
    os.environ["NTL_WORK"] = str(WORK)
    os.environ.setdefault("NTL_CFG", "{}")
    G = {"__name__": "ntl"}
    for prefix in ("import gc, glob", "COMP = locate_competition_dir()", "plt.rcParams.update", "cv_folds = "):
        exec(find(cells, prefix), G)
    cfg = G["CFG"]
    print("CFG:", {k: cfg[k] for k in ("ngram_pool_per_label", "ngram_pool_balance", "ngram_train_weights", "stylo_alpha",
                                       "svc_c", "blend3", "gate_alpha", "center_strength", "sinkhorn_rho")})
    v4 = np.load(ROOT / "analysis_v4" / "oof_v4.npz", allow_pickle=True)
    assert (v4["y"] == G["y"]).all() and (v4["prompt_id"] == G["train"].prompt_id.values).all()
    same = all((v4["fold"][b] == k).all() for k, (_, b) in enumerate(G["cv_folds"]))
    G["cv_folds"] = [(np.where(v4["fold"] != k)[0], np.where(v4["fold"] == k)[0]) for k in range(cfg["n_folds"])]
    print(f"fold: 커널 분할(oof_v4.npz)로 바꿈. 로컬 StratifiedGroupKFold 분할과 같은가: {same}")
    return G, v4


def downstream(G, v4, cells, n_test):
    """저장된 LoRA·DeBERTa OOF를 넣고 5장 이후 셀을 실행합니다. G에는 oof_ng와 ng_test(3행)가 있어야 합니다."""
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score as bas
    y, train = G["y"], G["train"]
    deb = np.load(ROOT / "out" / "v7" / "colab" / "ntl_deb_oof.npz")
    assert (deb["ids"] == train.id.astype(str).values).all()
    G["lora_oof"], G["deb_oof"] = v4["lora_oof"], deb["oof"]
    G["holdout_idx"] = G["cv_folds"][0][1]
    if n_test <= 3:
        # 테스트 로짓은 로컬에 없으므로(GPU 없음) LoRA·DeBERTa는 OOF 앞 3행을 자리 채움으로 넣습니다. 테스트 쪽 출력은 의미 없음.
        G["lora_test"], G["deb_test"] = G["lora_oof"][:3], G["deb_oof"][:3]
        print("\n== 5장 이후: 실제 test.csv 3행 (LoRA·DeBERTa 테스트 로짓은 자리 채움) ==")
    else:
        rows = G["holdout_idx"][:n_test]
        G["test"] = train.iloc[rows][["id", "text"]].reset_index(drop=True)
        G["ng_test"], G["lora_test"], G["deb_test"] = G["oof_ng"][rows], G["lora_oof"][rows], G["deb_oof"][rows]
        print(f"\n== 5장 이후: fold 0의 train 지문 {len(rows)}개를 가짜 테스트로 (테스트 로짓 = 그 지문의 OOF 로짓) ==")
    for name, arr in (("n-gram", G["oof_ng"]), ("LoRA", G["lora_oof"]), ("DeBERTa", G["deb_oof"])):
        print(f"{name:>8} OOF BA {bas(y, arr.argmax(1)):.4f} | fold별 {[round(float(bas(y[b], arr[b].argmax(1))), 4) for _, b in G['cv_folds']]}")
    for prefix in ("import itertools", "from scipy.special import log_softmax", "pred_eval = ", "def topic_clusters",
                   "def sinkhorn_balance", "submission = pd.DataFrame"):
        exec(find(cells, prefix), G)
    if n_test > 3:
        yt = y[rows]
        print(f"가짜 테스트 BA: 게이트 뒤 {bas(yt, G['test_lp'].argmax(1)):.4f} -> 중심화 + Sinkhorn 뒤 "
              f"{bas(yt, G['final_lp'].argmax(1)):.4f} (특화 LR이 이 지문들도 보고 학습했으므로 점수 자체는 참고용)")
    saved = np.load(WORK / "oof_v8.npz")
    print(f"저장 확인: {(WORK / 'oof_v8.npz').relative_to(ROOT)} 키 {sorted(saved.files)}, submission.parquet {G['submission'].shape}")


def run_downstream(ng_path, n_test):
    import numpy as np
    cells = code_cells()
    G, v4 = setup(cells)
    G["oof_ng"] = np.load(ROOT / ng_path)["oof_ng"]
    G["ng_test"] = G["oof_ng"][:3]  # 자리 채움
    print(f"n-gram OOF: {ng_path}")
    downstream(G, v4, cells, n_test)


def run_ngram():
    avail = mem_available_gb()
    if avail < EST_GB + MARGIN_GB:
        print(f"시작하지 않음: 예상 최대 메모리 {EST_GB}GB + 여유 {MARGIN_GB}GB > MemAvailable {avail:.1f}GB")
        sys.exit(3)
    print(f"예상 최대 메모리 {EST_GB}GB, MemAvailable {avail:.1f}GB", flush=True)
    watch, t0 = MemoryWatch(), time.time()
    import numpy as np
    import scipy.sparse as sparse
    cells = code_cells()
    G, v4 = setup(cells)
    y, K = G["y"], G["K"]
    ngram = find(cells, "FUNCTION_WORDS")
    part1, part2 = ngram.split("weight_grid = ")
    exec(part1, G)  # 표본 추출, 벡터화, 문체 피처, 풀 행 가중치, 함수 정의
    print(f"-- 벡터화까지 {(time.time() - t0) / 60:.1f}분, 최대 메모리(RssAnon) {watch.anon:.1f}GB", flush=True)

    # 1) 풀 크기 실험이 저장한 행렬과 비교 (실험은 풀 행렬의 행 안을 열 번호 순으로 정렬해 저장했습니다. 값은 같음)
    cache = ROOT / "out" / "pool" / "cache_n5000"
    yp = np.load(cache / "y_pool.npy")
    Xp = G["X_pool"].copy(); Xp.sort_indices()
    checks = {
        "y_pool": np.array_equal(G["y_pool"], yp),
        "S_pool": np.array_equal(G["S_pool"], np.load(cache / "S_pool.npy")),
        "S_train": np.array_equal(G["S_train"], np.load(cache / "S_train.npy")),
        "X_pool.indptr": np.array_equal(Xp.indptr, np.load(cache / "Xp_indptr.npy")),
        "X_pool.indices": np.array_equal(Xp.indices, np.load(cache / "Xp_indices.npy", mmap_mode="r")),
        "X_pool.data": np.array_equal(Xp.data, np.load(cache / "Xp_data.npy", mmap_mode="r")),
    }
    del Xp; gc.collect()
    Xt = sparse.load_npz(cache / "X_train.npz").tocsr()
    checks["X_train"] = (Xt.shape == G["X_train"].shape and np.array_equal(Xt.indptr, G["X_train"].indptr)
                         and np.array_equal(Xt.indices, G["X_train"].indices) and np.array_equal(Xt.data, G["X_train"].data))
    counts = np.bincount(yp, minlength=K)
    w_ref = (len(yp) / K / counts)[yp]  # analysis_pool/pool_oof.py의 _bal 가중치
    checks["w_pool"] = np.array_equal(G["w_pool"], w_ref)
    print(f"-- 실험 캐시(cache_n5000)와 비트 단위로 같은가: {checks}", flush=True)
    print(f"   풀 행렬 {G['X_pool'].shape}, nnz {G['X_pool'].nnz:,}, dtype {G['X_pool'].dtype}/{G['X_pool'].indices.dtype} | "
          f"풀 가중치 합 {G['w_pool'].sum():.3f} (행 수 {len(yp):,})", flush=True)

    # 2) fold마다 OOF 점수를 실험 결과와 비교: 노트북의 svc_scores를 감싸 호출될 때마다 기록합니다(노트북 코드는 그대로).
    ref = np.load(ROOT / "out" / "pool" / "ngram_oof_n5000_bal.npz")["oof_ng"]
    BA = G["balanced_accuracy_score"]
    inner, calls, t_fold = G["svc_scores"], [], [time.time()]

    def logged_scores(model, X, S):
        out = inner(model, X, S)
        k = len(calls); calls.append(k)
        if k < len(G["cv_folds"]):
            b = G["cv_folds"][k][1]
            print(f"-- fold {k}: 실험 OOF와 점수 최대 차이 {np.abs(out - ref[b]).max():.2e}, argmax가 다른 행 "
                  f"{int((out.argmax(1) != ref[b].argmax(1)).sum())} / {len(b)} | BA {BA(y[b], out.argmax(1)):.4f} "
                  f"(실험 {BA(y[b], ref[b].argmax(1)):.4f}) | liblinear 반복 {int(np.max(model.n_iter_))} | "
                  f"{(time.time() - t_fold[-1]) / 60:.1f}분, 최대 메모리 {watch.anon:.1f}GB, MemAvailable 최소 {watch.min_avail:.1f}GB",
                  flush=True)
        else:
            print(f"-- train 전체 fit(테스트용): {(time.time() - t_fold[-1]) / 60:.1f}분, 최대 메모리 {watch.anon:.1f}GB", flush=True)
        t_fold.append(time.time())
        return out

    G["svc_scores"] = logged_scores
    exec("weight_grid = " + part2, G)  # 5-fold OOF, train 전체 fit, 풀 행렬 삭제
    oof = G["oof_ng"]
    print(f"-- 전체: n-gram OOF BA {BA(y, oof.argmax(1)):.4f} (실험 n5000_bal {BA(y, ref.argmax(1)):.4f}) | 점수 최대 차이 "
          f"{np.abs(oof - ref).max():.2e}, argmax가 다른 행 {int((oof.argmax(1) != ref.argmax(1)).sum())} / {len(y)} | "
          f"n-gram 구간 {(time.time() - t0) / 60:.1f}분, 최대 메모리(RssAnon) {watch.anon:.1f}GB", flush=True)
    assert "X_pool" not in G and "ng_pool" not in G and "S_pool" not in G, "풀 행렬이 지워지지 않았습니다."
    np.savez_compressed(WORK / "ngram_oof_v8_local.npz", oof_ng=oof, ng_test=G["ng_test"])
    downstream(G, v4, cells, 3)
    # 커널 분할의 지문 확인: 검증 셀의 작은 모델을 커널 fold로 다시 돌리면 커널 로그의 0.3145가 나와야 합니다.
    print(f"\n커널 fold로 다시 잰 프롬프트 그룹 KFold BA: {G['quick_cv'](G['cv_folds']):.4f} (커널 로그 0.3145)")
    print(f"총 {(time.time() - t0) / 60:.1f}분, 최대 메모리(RssAnon) {watch.anon:.1f}GB, MemAvailable 최소 {watch.min_avail:.1f}GB")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "parse"
    if cmd == "parse":
        parse()
    elif cmd == "ngram":
        run_ngram()
    elif cmd == "downstream":
        run_downstream(sys.argv[2] if len(sys.argv) > 2 else "out/pool/ngram_oof_n5000_bal.npz",
                       int(sys.argv[3]) if len(sys.argv) > 3 else 3)
