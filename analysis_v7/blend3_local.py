"""저장된 OOF 세 개로 v7 노트북의 5장(3분기 블렌드)과 5-1장(회사 게이트) 셀을 그대로 실행합니다 (CPU, 재학습 없음).

Colab에서는 DeBERTa만 학습했기 때문에(VM이 자주 회수되어 n-gram·LoRA는 뺌), 블렌드 가중치는 여기서 고릅니다.
  - n-gram OOF : v5_stylo/ngram_oof_alpha0_kernelfold.npz (새 풀 231,036개, train 가중치 12, 커널 fold)
  - LoRA OOF   : analysis_v4/oof_v4.npz 의 lora_oof (v4 가중치, 커널 fold. 커널 로그의 0.5631과 같음)
  - DeBERTa OOF: Colab 실행의 ntl_deb_oof.npz
fold는 scikit-learn 버전에 따라 달라지므로 oof_v4.npz의 fold(= Kaggle 커널의 분할)를 씁니다.
테스트 로짓은 없으므로 OOF 앞 3행을 자리 채움으로 넣습니다(테스트 쪽 출력은 의미 없음).

사용: .venv/bin/python analysis_v7/blend3_local.py out/v7/colab/ntl_deb_oof.npz '{"blend3": null}'   # 가중치를 OOF로 다시 고름(nested BA 출력)
      .venv/bin/python analysis_v7/blend3_local.py out/v7/colab/ntl_deb_oof.npz                       # 노트북에 고정된 blend3 사용
      .venv/bin/python analysis_v7/blend3_local.py none      # 비교 기준: 같은 OOF로 v6(2분기 + 게이트)
"""
import json, os, sys, tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["MPLBACKEND"] = "Agg"
os.environ["NTL_INPUT"] = os.path.join(ROOT, "data")
os.environ["NTL_WORK"] = os.environ.get("NTL_WORK") or tempfile.mkdtemp()
os.environ["NTL_CFG"] = sys.argv[2] if len(sys.argv) > 2 else "{}"

nb = json.load(open(os.path.join(ROOT, "name_that_llm_v7.ipynb")))
code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
find = lambda key: next(s for s in code if key in s)

g = {}
exec(find("T0 = time.time()"), g)                                                             # CFG, LABELS
exec(find("COMP = locate_competition_dir()").replace('find_optional("ntl_open_pool.parquet")', "None"), g)  # train, y
y, train = g["y"], g["train"]

v4 = np.load(os.path.join(ROOT, "analysis_v4/oof_v4.npz"), allow_pickle=True)
assert (v4["y"] == y).all() and (v4["prompt_id"] == train.prompt_id.values).all()
fold = v4["fold"]
g["cv_folds"] = [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(g["CFG"]["n_folds"])]
g["holdout_idx"] = g["cv_folds"][0][1]

NO_DEB = sys.argv[1] == "none"  # "none"이면 DeBERTa 없이 v6(2분기, blend_w=0.6 + 게이트)을 같은 OOF로 계산합니다(비교 기준).
deb = {"oof": None} if NO_DEB else np.load(sys.argv[1])
assert NO_DEB or (deb["ids"] == train.id.astype(str).values).all(), "DeBERTa OOF의 지문 순서가 train과 다릅니다."
g["oof_ng"] = np.load(os.path.join(ROOT, "v5_stylo/ngram_oof_alpha0_kernelfold.npz"))["oof_ng"]
g["lora_oof"], g["deb_oof"] = v4["lora_oof"], deb["oof"]
g["ng_test"], g["lora_test"] = g["oof_ng"][:3], g["lora_oof"][:3]
g["deb_test"] = None if NO_DEB else g["deb_oof"][:3]
g["test"] = g["test"].iloc[:3]

from sklearn.metrics import balanced_accuracy_score as bas
for name, arr in (("n-gram", g["oof_ng"]), ("LoRA", g["lora_oof"]), ("DeBERTa", g["deb_oof"])):
    if arr is None:
        continue
    print(f"{name:>8} OOF BA {bas(y, arr.argmax(1)):.4f} | fold별 {[round(float(bas(y[b], arr[b].argmax(1))), 4) for _, b in g['cv_folds']]}")

exec(find("def log_softmax_np"), g)   # 5장: 온도, 3분기 가중치 표, nested BA
exec(find("FAMILIES = "), g)          # 5-1장: 회사 게이트
exec(find("def sinkhorn_balance").split("if CFG[\"use_sinkhorn\"] and len(test)")[0], g)  # Sinkhorn을 OOF에 적용해 보는 표
