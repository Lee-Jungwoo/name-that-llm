"""풀을 넣은 실제 n-gram 분기에서 stylo_alpha와 피처 구성을 OOF로 비교합니다 (CPU만, train 가중치 12 고정).

v5 노트북의 코드 셀(설정, 데이터, 검증, n-gram)을 그대로 실행한 뒤 fit 부분만 설정을 바꿔 가며 다시 돌립니다.
결과 OOF는 out/v5/ngram_oof_<이름>.npz에 저장하고, v5_stylo/blend.py로 블렌드 점수를 냅니다.

실행마다 부모에서 새로 fork한 프로세스 하나를 씁니다. 노트북의 LinearSVC에는 random_state가 없어서 liblinear가
전역 NumPy 난수 상태에서 시드를 받는데, 이렇게 하면 시드를 따로 주지 않은 실행은 모두 같은 난수 상태에서 출발합니다.
이름 끝에 _s<숫자>가 붙은 실행은 그 숫자로 np.random.seed를 다시 걸어 시드에 따른 흔들림을 잽니다.

메모리: 부모 약 2.2GB + 실행 하나당 약 4.7GB. 동시에 도는 수는 NTL_SWEEP_PROCS(기본 2)이고,
남은 메모리가 7GB보다 적으면 새 실행을 시작하지 않고 기다립니다.

사용법: NTL_INPUT=data python v5_stylo/alpha_sweep.py [실행 이름 ...]
"""
import json, multiprocessing as mp, os, sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import nbformat

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "v5"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NTL_INPUT", str(ROOT / "data"))
os.environ["NTL_CFG"] = json.dumps({"use_lora": False, "ngram_train_weights": [12.0]})

nb = nbformat.read(ROOT / "name_that_llm_v5.ipynb", as_version=4)
cells = [c.source for c in nb.cells if c.cell_type == "code"]
setup = next(c for c in cells if c.startswith("import gc, glob"))
data = next(c for c in cells if c.startswith("COMP = locate_competition_dir()"))
cv = next(c for c in cells if c.startswith("cv_folds = "))
ngram = next(c for c in cells if c.startswith("FUNCTION_WORDS"))
ngram = ngram.split("weight_grid = ")[0]  # 벡터화와 함수 정의까지만 (fit은 아래에서)
G = {"__name__": "ntl"}
for src in (setup, data, cv, ngram):
    exec(src, G)

np, y, CFG = G["np"], G["y"], G["CFG"]

# StratifiedGroupKFold(shuffle=True)의 분할이 환경마다 다릅니다(scikit-learn 버전 차이로 보임). 로컬(1.9.1)의 분할은
# 제출 커널의 분할과 전혀 다른 프롬프트 묶음입니다. 커널 수치와 비교할 수 있도록 analysis_v4/oof_v4.npz에 저장된 커널의 fold를 씁니다.
# NTL_SWEEP_FOLDS=local이면 로컬 분할을 그대로 씁니다(이때 결과 파일 이름에 _localfold가 붙습니다).
LOCAL_FOLDS = os.environ.get("NTL_SWEEP_FOLDS", "kernel") == "local"
if not LOCAL_FOLDS:
    v4 = np.load(ROOT / "analysis_v4" / "oof_v4.npz", allow_pickle=True)
    assert (v4["y"] == y).all() and (v4["prompt_id"] == G["train"].prompt_id.values).all()
    same = all((v4["fold"][b] == k).all() for k, (_, b) in enumerate(G["cv_folds"]))
    G["cv_folds"] = [(np.where(v4["fold"] != k)[0], np.where(v4["fold"] == k)[0]) for k in range(CFG["n_folds"])]
    print(f"fold: 커널 분할(oof_v4.npz) 사용. 로컬 StratifiedGroupKFold 분할과 같은가: {same}")
SUFFIX = "_localfold" if LOCAL_FOLDS else ""
STYLO_NAMES = ["n_words", "n_sents", "sent_len_mean", "sent_len_std", "word_len_mean", "long_word_ratio", "mattr",
               "func_ratio", "em_dash", "spaced_em_dash", "en_dash", "semicolon", "colon", "comma", "paren", "question",
               "curly_apos", "straight_apos", "curly_quote", "straight_quote", "contraction", "first_person",
               "second_person", "not_but", "ly_adverb", "conj_start", "mid_caps"]
assert G["S_train"].shape[1] == len(STYLO_NAMES)
np.savez_compressed(OUT / "stylo_raw.npz", S_train=G["S_train"], S_pool=G["S_pool"], y=y, y_pool=G["y_pool"],
                    names=np.array(STYLO_NAMES))

# (이름, alpha, 뺄 피처, 시드). 시드가 None이면 부모에서 물려받은 난수 상태를 그대로 씁니다.
RUNS = [
    ("alpha0", 0.0, [], None),
    ("alpha0.1", 0.1, [], None),
    ("alpha0.2", 0.2, [], None),
    ("alpha0.3", 0.3, [], None),
    ("alpha0.4", 0.4, [], None),
    ("alpha0.2_nolen", 0.2, ["n_words", "n_sents"], None),
]
for seed in (1, 2, 3):
    RUNS += [(f"alpha0_s{seed}", 0.0, [], seed), (f"alpha0.2_s{seed}", 0.2, [], seed)]


def run(spec):
    name, alpha, drop, seed = spec
    if seed is not None:
        np.random.seed(seed)
    CFG["stylo_alpha"] = alpha  # 0이면 문체 블록이 모두 0이라 v4와 같습니다
    keep = [i for i, n in enumerate(STYLO_NAMES) if n not in drop]
    for k in ("S_train", "S_pool", "S_test"):
        G[k] = G[k][:, keep]
    n_iter = []
    fit_svc = G["fit_svc"]

    def fit_logged(train_idx, train_weight):
        model = fit_svc(train_idx, train_weight)
        n_iter.append(int(np.max(model.n_iter_)))
        return model

    G["fit_svc"] = fit_logged
    oof = G["ngram_oof"](12.0)
    ba = G["balanced_accuracy_score"](y, oof.argmax(1))
    folds = [round(float(G["balanced_accuracy_score"](y[b], oof[b].argmax(1))), 3) for _, b in G["cv_folds"]]
    np.savez_compressed(OUT / f"ngram_oof_{name}{SUFFIX}.npz", oof_ng=oof)
    print(f"{name}{SUFFIX}: n-gram OOF BA {ba:.4f} (fold별 {folds}) | liblinear 반복 수 {n_iter} | {G['elapsed']()}", flush=True)


def mem_available_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 2**20
    return float("inf")


if __name__ == "__main__":
    only = sys.argv[1:]
    known = {r[0]: r for r in RUNS}
    pending = [known[n] for n in only] if only else list(RUNS)
    procs = int(os.environ.get("NTL_SWEEP_PROCS", "2"))
    ctx = mp.get_context("fork")
    active, failed, last_start = [], [], 0.0
    while pending or active:
        # 방금 시작한 실행의 메모리가 잡힐 때까지 90초는 다음 실행을 시작하지 않습니다.
        if pending and len(active) < procs and time.time() - last_start > 90 and mem_available_gb() > 7:
            spec = pending.pop(0)
            p = ctx.Process(target=run, args=(spec,))
            p.start()
            active.append((spec[0], p))
            last_start = time.time()
        time.sleep(2)
        for name, p in list(active):
            if not p.is_alive():
                p.join()
                active.remove((name, p))
                if p.exitcode != 0:
                    failed.append(name)
                    print(f"{name}: 실패 (exitcode {p.exitcode})", flush=True)
    sys.exit(1 if failed else 0)
