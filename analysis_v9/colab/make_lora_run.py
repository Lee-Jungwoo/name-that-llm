"""저장소 노트북 -> Colab용 'LoRA만' 실행 사본.
n-gram 학습(호스트 RAM 약 11GB)과 LoRA 셀 뒤의 모든 셀은 뺍니다(블렌드·게이트는 서버에서 저장된 OOF로 계산).
fold는 Kaggle 커널의 분할(analysis_v4/oof_v4.npz의 fold)을 그대로 씁니다.
사용: make_lora_run.py <repo nb> <out nb> '<NTL_CFG 추가 JSON>'   (추가 JSON은 점검용 예산 덮어쓰기 등)"""
import ast, json, sys
src, dst, extra = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
nb = json.load(open(src, encoding="utf-8"))


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
        base = {"batch_size": 4, "grad_accum": 2, "ngram_train_weights": [12.0], "use_deberta": False}
        s = rep(s, """NTL_CFG='{"batch_size": 4, "grad_accum": 2, "ngram_train_weights": [12.0], "use_deberta": false}'""",
                "NTL_CFG=" + repr(json.dumps({**base, **extra})))
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
    elif s.startswith("holdout_idx = cv_folds[0][1]"):
        s = rep(s, "print(f\"device={device}, base={base_dir}, LoRA 실행={run_lora}\")",
                "print(f\"device={device}, base={base_dir}, LoRA 실행={run_lora}\")\n"
                "if device.type == \"cuda\":\n"
                "    print(\"GPU:\", torch.cuda.get_device_name(0), \"| capability\", torch.cuda.get_device_capability(0), \"| torch\", torch.__version__,\n"
                "          \"| transformers\", __import__(\"transformers\").__version__, \"| 정밀도\", \"bf16\" if torch.cuda.get_device_capability(0)[0] >= 8 else \"fp16 + GradScaler\")")
        s += '\nprint("== LORA RUN DONE ==", elapsed())\n'
        c["source"] = s.splitlines(keepends=True)
        cells.append(c)
        break
    c["source"] = s.splitlines(keepends=True)
    cells.append(c)
assert "".join(cells[-1]["source"]).rstrip().endswith('print("== LORA RUN DONE ==", elapsed())')
nb["cells"] = cells
for c in cells:
    ast.parse("".join(c["source"]))
    c["outputs"], c["execution_count"] = [], None
json.dump(nb, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(dst, len(cells), "cells", extra)
