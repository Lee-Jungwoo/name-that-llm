"""풀 표본(라벨당 2,500개)과 train의 모델별 n_words, n_sents 분포 비교 (handoff 4.6-2).

alpha_sweep.py가 저장한 out/v5/stylo_raw.npz(27개 피처의 log1p 값)를 읽습니다.
사용법: python v5_stylo/length_check.py
"""
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score as BA
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

root = Path(__file__).resolve().parent.parent
d = np.load(f"{root}/out/v5/stylo_raw.npz", allow_pickle=True)
v4 = np.load(f"{root}/analysis_v4/oof_v4.npz", allow_pickle=True)
labels = [str(x) for x in v4["labels"]]
assert (v4["y"] == d["y"]).all()
names = [str(x) for x in d["names"]]
St, Sp, y, yp = d["S_train"].astype(np.float64), d["S_pool"].astype(np.float64), d["y"], d["y_pool"]
print("train", St.shape, "pool", Sp.shape, "pool 라벨당", np.bincount(yp).min(), "~", np.bincount(yp).max())


def eta2(x, g):
    """모델(집단) 평균이 설명하는 분산 비율."""
    m = x.mean()
    between = sum((g == k).sum() * (x[g == k].mean() - m) ** 2 for k in np.unique(g))
    return between / ((x - m) ** 2).sum()


for f in ("n_words", "n_sents", "sent_len_mean"):
    j = names.index(f)
    rt, rp = np.expm1(St[:, j]), np.expm1(Sp[:, j])
    mt = np.array([np.median(rt[y == k]) for k in range(18)])
    mp = np.array([np.median(rp[yp == k]) for k in range(18)])
    it = np.array([np.subtract(*np.percentile(rt[y == k], [75, 25])) for k in range(18)])
    ip = np.array([np.subtract(*np.percentile(rp[yp == k], [75, 25])) for k in range(18)])
    print(f"\n== {f} ==")
    print(f"전체 중앙값: train {np.median(rt):.1f} / pool {np.median(rp):.1f}")
    print(f"모델별 중앙값 범위: train {mt.min():.1f}~{mt.max():.1f} (폭 {mt.max()-mt.min():.1f}, std {mt.std():.2f})"
          f" / pool {mp.min():.1f}~{mp.max():.1f} (폭 {mp.max()-mp.min():.1f}, std {mp.std():.2f})")
    print(f"모델 안 IQR 평균: train {it.mean():.1f} / pool {ip.mean():.1f}")
    print(f"eta^2(log1p 값, 모델): train {eta2(St[:, j], y):.3f} / pool {eta2(Sp[:, j], yp):.3f}")
    r = np.corrcoef(mt, mp)[0, 1]
    rk = np.corrcoef(np.argsort(np.argsort(mt)), np.argsort(np.argsort(mp)))[0, 1]
    print(f"모델별 중앙값의 train-pool 상관: pearson {r:.3f}, spearman {rk:.3f}")
    print(f"{'model':28s} train  pool   pool-train")
    for k in np.argsort(-(mp - mt)):
        print(f"{labels[k]:28s} {mt[k]:6.1f} {mp[k]:6.1f} {mp[k]-mt[k]:+7.1f}")

# 길이 두 피처만으로 모델을 얼마나 맞히나
J = [names.index("n_words"), names.index("n_sents")]
clf = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=2000))
fold = v4["fold"]
pt = cross_val_predict(clf(), St[:, J], y, cv=[(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(5)])
pp = cross_val_predict(clf(), Sp[:, J], yp, cv=StratifiedKFold(5, shuffle=True, random_state=0))
m = clf().fit(Sp[:, J], yp)
print("\n== n_words + n_sents 두 개만으로 18개 모델 맞히기 (무작위 0.0556) ==")
print(f"train 안 (프롬프트 5-fold): {BA(y, pt):.4f}")
print(f"pool 안 (5-fold)          : {BA(yp, pp):.4f}")
print(f"pool로 학습 -> train 예측 : {BA(y, m.predict(St[:, J])):.4f}")
m2 = clf().fit(St[:, J], y)
print(f"train으로 학습 -> pool 예측: {BA(yp, m2.predict(Sp[:, J])):.4f}")

# v5 스케일링(train 통계로 z-score, ±4 자름)에서 풀이 어디에 놓이나
mu, sd = St.mean(0), St.std(0) + 1e-6
Zp, Zt = (Sp - mu) / sd, (St - mu) / sd
print("\n== train 통계로 표준화했을 때 풀의 위치 ==")
print(f"{'feature':16s} pool z평균  pool z표준편차  |z|>4 비율(pool)  |z|>4 비율(train)")
for j, n in enumerate(names):
    print(f"{n:16s} {Zp[:, j].mean():+8.2f} {Zp[:, j].std():10.2f} {np.mean(np.abs(Zp[:, j]) > 4):14.4f} {np.mean(np.abs(Zt[:, j]) > 4):14.4f}")

# 모델별 평균(z, ±4 자름)이 train과 풀에서 얼마나 다른가. GPT 5개와 나머지를 나눠 봅니다.
Ct, Cp = np.clip(Zt, -4, 4), np.clip(Zp, -4, 4)
mt = np.array([Ct[y == k].mean(0) for k in range(18)])
mp = np.array([Cp[yp == k].mean(0) for k in range(18)])
se = np.sqrt(np.array([Ct[y == k].var(0) / (y == k).sum() for k in range(18)])
             + np.array([Cp[yp == k].var(0) / (yp == k).sum() for k in range(18)]))
D = mp - mt
gpt = np.array([l.startswith("gpt") for l in labels])
print("\n== 모델별 평균 z의 이동 D = pool - train (이동이 큰 피처 순) ==")
print(f"{'feature':16s} 모델 간 std(train) std(pool) | D의 RMS  |D|/SE>3인 모델 수 | GPT 5개 평균 D / 나머지 평균 D | 모델별 평균의 상관")
for j in np.argsort(-np.sqrt((D ** 2).mean(0))):
    print(f"{names[j]:16s} {mt[:, j].std():6.2f} {mp[:, j].std():6.2f} | {np.sqrt((D[:, j] ** 2).mean()):5.2f} "
          f"{int((np.abs(D[:, j]) / se[:, j] > 3).sum()):3d} | {D[gpt, j].mean():+5.2f} / {D[~gpt, j].mean():+5.2f} | "
          f"{np.corrcoef(mt[:, j], mp[:, j])[0, 1]:.2f}")
