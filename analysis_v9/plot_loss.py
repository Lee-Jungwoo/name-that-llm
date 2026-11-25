"""v9 학습 로그(out/v9/v9.log)에서 train loss 곡선을 읽어 analysis_v9/loss_v9.png로 저장합니다 (CPU만, 가벼움).

왼쪽: Stage 1 train loss(지수이동평균 0.98/0.02, label smoothing 0.05, 클래스 가중) 대 스텝. 위쪽 축에 전체 28,869스텝 대비 %를 같이 보입니다.
오른쪽: Stage 2의 fold0~fold4, full 곡선. 로그에 아직 없으면 오른쪽 패널은 그리지 않습니다.
analysis_v9/val_loss_stage1.csv(열: step,val_loss, 선택 val_ba)가 있으면 Stage 1 패널에 "val loss (train.parquet)" 점으로 겹쳐 그립니다.

로그는 캐리지 리턴이나 깨진 바이트가 섞여 있을 수 있어 errors="replace"로 읽습니다.
VM이 끊겨 재개하면 이전 로그가 out/v9/v9_run<번호>.log로 보관되므로, v9_run*.log를 번호 순으로 읽은 뒤 v9.log를 읽어 합칩니다.
같은 스텝이 여러 번 나오면 나중 파일(같은 파일 안에서는 나중 줄)의 값이 이깁니다. "체크포인트에서 이어 하기: N/..." 줄은 재개 지점으로 Stage 1 패널에 점선으로 표시합니다.
로그마다 분(minute)이 0부터 다시 시작하므로 x축에는 스텝만 씁니다. (v9_run2_failed_newpool.log 같은 이름은 번호 형식이 아니라 읽지 않습니다.)

사용법: .venv/bin/python analysis_v9/plot_loss.py [로그 경로 ...]  (생략하면 out/v9의 v9_run*.log와 v9.log)
"""
import csv, re, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
LOGDIR = ROOT / "out" / "v9"
VAL_CSV = ROOT / "analysis_v9" / "val_loss_stage1.csv"
OUT_PNG = ROOT / "analysis_v9" / "loss_v9.png"
RESUME = re.compile(r"\[stage1\] 체크포인트에서 이어 하기: (\d+)/")
LINE = re.compile(r"\[(stage1|stage2-fold\d+|stage2-full)\] (\d+)/(\d+) loss ([0-9.]+) \| ([0-9.]+)분")


def log_files():
    runs = sorted((int(m.group(1)), p) for p in LOGDIR.glob("v9_run*.log") if (m := re.fullmatch(r"v9_run(\d+)\.log", p.name)))
    return [p for _, p in runs] + [LOGDIR / "v9.log"]


def read_log(paths):
    """{이름: {스텝: (loss, 분)}}, {이름: 전체 스텝 수}, [재개 스텝]. 같은 스텝은 나중 값이 이깁니다."""
    runs, totals, resumes = {}, {}, []
    text = "\n".join(Path(p).read_bytes().decode("utf-8", errors="replace") for p in paths if Path(p).exists())
    for chunk in re.split(r"[\r\n]+", text):
        r = RESUME.search(chunk)
        if r:
            resumes.append(int(r.group(1)))
        m = LINE.search(chunk)
        if not m:
            continue
        name, step, n, loss, minutes = m.group(1), int(m.group(2)), int(m.group(3)), float(m.group(4)), float(m.group(5))
        runs.setdefault(name, {})[step] = (loss, minutes)
        totals[name] = n
    return runs, totals, resumes


def read_val(path):
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = [(int(float(r["step"])), float(r["val_loss"])) for r in csv.DictReader(f) if r.get("val_loss")]
    return sorted(rows) or None


def pick_font():
    """한글 글꼴이 있으면 쓰고 true, 없으면 false(영문 라벨)."""
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["NanumGothic", "NanumBarunGothic", "Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic", "AppleGothic", "Noto Serif CJK KR"]:
        if name in have:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def main():
    ko = pick_font()
    T = (lambda k, e: k) if ko else (lambda k, e: e)
    runs, totals, resumes = read_log([Path(a) for a in sys.argv[1:]] or log_files())
    s1 = runs.get("stage1", {})
    s2 = {k: v for k, v in runs.items() if k != "stage1"}
    val = read_val(VAL_CSV)
    ncol = 2 if s2 else 1
    fig, axes = plt.subplots(1, ncol, figsize=(7.5 * ncol, 4.8), squeeze=False)
    ax = axes[0][0]
    n1 = totals.get("stage1", 28869)
    if s1:
        xs = sorted(s1)
        ax.plot(xs, [s1[x][0] for x in xs], color="#2a6fb0", lw=1.6, label=T("train loss (지수이동평균)", "train loss (EMA)"))
    if val:
        ax.plot([v[0] for v in val], [v[1] for v in val], "o", color="#d9622b", ms=5, label="val loss (train.parquet)")
    for i, r in enumerate(sorted(set(resumes))):
        ax.axvline(r, color="gray", ls="--", lw=0.8)
        ax.text(r, 0.98, T("재개", "resume"), transform=ax.get_xaxis_transform(), rotation=90, ha="right", va="top", color="gray", fontsize=9)
    ax.set_xlim(0, n1)
    ax.set_xlabel(T("스텝", "step"))
    ax.set_ylabel("loss")
    ax.set_title("Stage 1" + (f" ({len(s1)} pts, last {max(s1)}/{n1})" if s1 else ""))
    ax.grid(alpha=0.3)
    ax.legend()
    top = ax.secondary_xaxis("top", functions=(lambda x: x / n1 * 100, lambda p: p * n1 / 100))
    top.set_xlabel(T(f"진행률 (% of {n1:,})", f"progress (% of {n1:,})"))
    if s2:
        ax2 = axes[0][1]
        for name in sorted(s2):
            xs = sorted(s2[name])
            ax2.plot(xs, [s2[name][x][0] for x in xs], lw=1.5, label=name.replace("stage2-", ""))
        ax2.set_xlabel(T("스텝", "step"))
        ax2.set_ylabel("loss")
        ax2.set_title("Stage 2")
        ax2.grid(alpha=0.3)
        ax2.legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print(f"저장: {OUT_PNG} | stage1 {len(s1)}점" + (f" ({s1[min(s1)][0]:.3f} -> {s1[max(s1)][0]:.3f})" if s1 else "")
          + f" | stage2 {sorted(s2)} | val {len(val) if val else 0}점 | 한글 글꼴 {'있음' if ko else '없음(영문)'}")


if __name__ == "__main__":
    main()
