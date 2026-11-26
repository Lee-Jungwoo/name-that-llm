"""저장소 노트북(v10) -> Colab용 'DeBERTa만' 실행 사본.
n-gram 학습과 DeBERTa 셀 뒤의 모든 셀은 뺍니다(블렌드·게이트는 서버에서 저장된 OOF로 계산). LoRA 셀은 함수 정의
(make_batches 등)와 holdout_idx 때문에 남기고 use_lora=false로 건너뜁니다.
fold는 Kaggle 커널의 분할(analysis_v4/oof_v4.npz의 fold)을 그대로 씁니다.
사용: make_deb_run.py <repo nb> <out nb> '<NTL_CFG 추가 JSON>' [<work dir>]   (추가 JSON은 점검용 예산 덮어쓰기 등)
      make_deb_run.py <repo nb> <out nb> '<추가 JSON>' - kaggle              (Kaggle 커널용: Colab을 못 쓸 때)"""
import ast, json, sys
src, dst, extra = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
WORK = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else None
KAGGLE = len(sys.argv) > 5 and sys.argv[5] == "kaggle"
nb = json.load(open(src, encoding="utf-8"))
BASE = {"use_lora": False, "ngram_train_weights": [12.0], "deb_grad_ckpt": True}


def rep(s, old, new, count=1):
    assert s.count(old) == count, (s.count(old), old[:60])
    return s.replace(old, new)


cells = []
for c in nb["cells"]:
    s = "".join(c["source"])
    if c["cell_type"] != "code":
        continue
    if "drive.mount" in s or s.startswith("plt.rcParams.update"):
        continue
    if s.startswith("# Colab 전용 준비"):
        s = rep(s, """NTL_CFG='{"use_lora": false, "ngram_train_weights": [12.0]}'""", "NTL_CFG=" + repr(json.dumps({**BASE, **extra})))
        if WORK:
            s = rep(s, 'NTL_WORK="/content/out",', f'NTL_WORK={WORK!r},')
            s = rep(s, 'os.makedirs("/content/out", exist_ok=True)', f'os.makedirs({WORK!r}, exist_ok=True)')
    elif s.startswith("cv_folds = list(") and KAGGLE:
        pass  # Kaggle의 분할이 기준 (로그의 "프롬프트 그룹 KFold BA: 0.3145"로 확인)
    elif s.startswith("cv_folds = list("):
        s = rep(s, 'print(f"프롬프트 그룹 KFold BA: {quick_cv(cv_folds):.4f}")', '''# 실행 사본 전용: fold는 Kaggle 커널의 분할(oof_v4.npz의 fold)을 그대로 씁니다. scikit-learn 버전이 달라도 같은 분할이 됩니다.
_z = np.load(os.environ.get("NTL_FOLD_REF", "/content/run/oof_v4.npz"), allow_pickle=True)
assert (_z["y"] == y).all() and (_z["prompt_id"] == groups).all()
_f = np.zeros(len(train), dtype=int)
for _k, (_, _b) in enumerate(cv_folds):
    _f[_b] = _k
print("StratifiedGroupKFold가 Kaggle 커널 분할과 같은가:", bool((_f == _z["fold"]).all()), "| scikit-learn", __import__("sklearn").__version__)
cv_folds = [(np.where(_z["fold"] != k)[0], np.where(_z["fold"] == k)[0]) for k in range(CFG["n_folds"])]
print(f"프롬프트 그룹 KFold BA: {quick_cv(cv_folds):.4f}")''')
    elif s.startswith("FUNCTION_WORDS = set("):
        s = s.split("class MultiViewVectorizer")[0]  # sample_pool까지만 (n-gram 학습은 뺌)
        assert "def sample_pool" in s
    elif s.startswith("deb_oof, deb_test = None, None"):
        s = rep(s, 'print(f"DeBERTa 폴더={deb_dir}, 실행={run_deb}")',
                'print(f"DeBERTa 폴더={deb_dir}, 실행={run_deb}")\n'
                'if device.type == "cuda":\n'
                '    print("GPU:", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0), "| torch", torch.__version__,\n'
                '          "| transformers", __import__("transformers").__version__, "| deb_amp", CFG["deb_amp"], "| grad ckpt", CFG["deb_grad_ckpt"])')
        s += '\nprint("== DEB RUN DONE ==", elapsed())\n'
        c["source"] = s.splitlines(keepends=True)
        cells.append(c)
        break
    c["source"] = s.splitlines(keepends=True)
    cells.append(c)
assert "".join(cells[-1]["source"]).rstrip().endswith('print("== DEB RUN DONE ==", elapsed())')
if KAGGLE:
    first = {"cell_type": "code", "metadata": {}, "source": (
        "# Kaggle 실행 사본 전용: Colab 준비 셀이 하던 설정 덮어쓰기를 여기서 합니다(학습 설정을 Colab 실행과 같게).\n"
        "import os\n"
        f"os.environ[\"NTL_CFG\"] = {json.dumps({**BASE, **extra})!r}\n"
        "os.environ[\"PYTORCH_CUDA_ALLOC_CONF\"] = \"expandable_segments:True\"\n").splitlines(keepends=True)}
    cells.insert(0, first)
nb["cells"] = cells
for c in cells:
    ast.parse("".join(c["source"]))
    c["outputs"], c["execution_count"] = [], None
json.dump(nb, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(dst, len(cells), "cells", {**BASE, **extra}, "work", WORK, "kaggle" if KAGGLE else "colab")
