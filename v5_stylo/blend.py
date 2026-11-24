"""n-gram OOF(npz) + v4 LoRA OOF -> 노트북과 같은 방식의 블렌드 OOF BA."""
import sys, numpy as np
from sklearn.metrics import balanced_accuracy_score as BA
d = np.load(str(__import__("pathlib").Path(__file__).resolve().parent.parent / "analysis_v4" / "oof_v4.npz"), allow_pickle=True)
ng = np.load(sys.argv[1])["oof_ng"]; y = d["y"]; lora = d["lora_oof"]
def lsm(x, t=1.0):
    x = x / t; x = x - x.max(1, keepdims=True); return x - np.log(np.exp(x).sum(1, keepdims=True))
def best_t(s, grid=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3)):
    return grid[int(np.argmin([-lsm(s, t)[np.arange(len(y)), y].mean() for t in grid]))]
tn, tl = best_t(ng), best_t(lora)
ngl, ll = lsm(ng, tn), lsm(lora, tl)
print("n-gram OOF", round(BA(y, ng.argmax(1)), 4), "| T_NG", tn, "T_LORA", tl, "| LoRA OOF", round(BA(y, lora.argmax(1)), 4))
print("w별:", {w / 10: round(BA(y, (w / 10 * ll + (1 - w / 10) * ngl).argmax(1)), 4) for w in range(11)})
fold0 = d["fold"] == 0
e = 0.6 * ll + 0.4 * ngl
print("블렌드 w=0.6 OOF", round(BA(y, e.argmax(1)), 4), "| fold0", round(BA(y[fold0], e[fold0].argmax(1)), 4))
