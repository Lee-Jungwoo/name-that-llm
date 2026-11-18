"""v4 모델 구조 그림. 사용법: python plot_architecture.py figures/"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SURFACE, INK, MUTED, EDGE = "#fcfcfb", "#0b0b0b", "#52514e", "#b9b8b3"
A_FILL, A_EDGE = "#eef4fd", "#86b6ef"
B_FILL, B_EDGE = "#fdf0ea", "#f2a582"
FONT = ["Apple SD Gothic Neo", "AppleGothic", "DejaVu Sans"]
plt.rcParams.update({"font.family": FONT, "text.color": INK})

W, H = 100, 128
fig = plt.figure(figsize=(10, 12.8))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)  # 위에서 아래로
ax.axis("off")
fig.patch.set_facecolor(SURFACE)


def panel(x0, y0, x1, y1, fill, edge, dashed=False):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.6",
                                fc=fill, ec=edge, lw=1.3, ls=(0, (4, 3)) if dashed else "-", zorder=1))


def box(cx, cy, title, sub=None, w=36, h=7.2, edge=EDGE, lw=1.2):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                fc="#ffffff", ec=edge, lw=lw, zorder=3))
    if sub:
        ax.text(cx, cy - 1.3, title, ha="center", va="center", fontsize=11, fontweight="bold", zorder=4)
        ax.text(cx, cy + 1.75, sub, ha="center", va="center", fontsize=8.6, color=MUTED, zorder=4)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=11, fontweight="bold", zorder=4)
    return (cx, cy, w, h)


def arrow(points, dashed=False, color=MUTED):
    """직각 경로. 마지막 구간에 화살촉."""
    xs, ys = zip(*points)
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=color, lw=1.2,
            ls=(0, (4, 3)) if dashed else "-", zorder=2, solid_capstyle="butt")
    ax.annotate("", xy=points[-1], xytext=points[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, mutation_scale=11,
                                shrinkA=0, shrinkB=0), zorder=2)


def header(x, y, name, desc, train):
    ax.text(x, y, name, fontsize=12.5, fontweight="bold", va="top")
    ax.text(x, y + 3.2, desc, fontsize=9, color=MUTED, va="top")
    ax.text(x, y + 5.9, train, fontsize=9, color=MUTED, va="top")


# 제목
ax.text(4, 3.5, "v4 모델 구조", fontsize=16, fontweight="bold", va="top")
ax.text(4, 7.6, "지문 하나를 두 가지 방식으로 점수 매긴 뒤 섞어 18개 모델 중 하나를 고릅니다.",
        fontsize=9.5, color=MUTED, va="top")

# 입력
inp = box(50, 16, "입력", "영어 지문 1개 (단어 36~274개)", w=34, edge=INK, lw=1.5)

# 패널
AX, BX = 26, 74                 # 두 열의 중심
PT, PB = 26, 93                 # 패널 위·아래
panel(3, PT, 49, PB, A_FILL, A_EDGE)
panel(51, PT, 97, PB, B_FILL, B_EDGE)
header(5.5, PT + 2.2, "A. n-gram", "CPU · 표면 습관 (문장부호, 철자, 리듬)", "학습: Open Pool 45,000개 + train (train 가중치 12)")
header(53.5, PT + 2.2, "B. LoRA", "GPU · 사전학습 언어 모델의 문체 표현",
       "학습: Stage 1 Open Pool 45,000개 → Stage 2 train (fold마다)")

top, bottom = PT + 15, PB - 6.5
b_ys = [top + i * (bottom - top) / 4 for i in range(5)]
a_ys = [top + i * (bottom - top) / 3 for i in range(4)]

a = [box(AX, a_ys[0], "4가지 보기로 다시 쓰기", "char 3-5 · word 1-2 · shape 2-6 · func 1-3", w=40),
     box(AX, a_ys[1], "TF-IDF", "보기별 가중치 1.0 / 0.7 / 0.5 / 0.5, 최대 51만 차원", w=40),
     box(AX, a_ys[2], "LinearSVC", "one-vs-rest, C = 0.1", w=40),
     box(AX, a_ys[3], "온도 조정 → log 확률", "[N, 18]", w=40)]
b = [box(BX, b_ys[0], "echo 입력 (v4 추가)", "지문 + 빈 줄 + 지문, 사본당 최대 384토큰", w=40),
     box(BX, b_ys[1], "Qwen3-4B-Base 앞 18층 (고정)", "q/k/v/o에 LoRA r = 16", w=40),
     box(BX, b_ys[2], "깊이 7곳 평균 풀링", "층 2·4·6·9·12·15·18, 두 번째 사본만, 층별 LayerNorm", w=40),
     box(BX, b_ys[3], "이어 붙이기 → 선형 헤드", "7 × 2,560 = 17,920 → 18", w=40),
     box(BX, b_ys[4], "온도 조정 → log 확률", "[N, 18]", w=40)]
for chain in (a, b):
    for (x0, y0, _, h0), (x1, y1, _, h1) in zip(chain, chain[1:]):
        arrow([(x0, y0 + h0 / 2), (x1, y1 - h1 / 2)])

# 입력 → 두 패널 (패널 위 테두리까지)
split_y = 22.5
arrow([(50, inp[1] + inp[3] / 2), (50, split_y), (AX, split_y), (AX, PT)])
arrow([(50, split_y), (BX, split_y), (BX, PT)])

# 블렌드
blend = box(50, 102, "블렌드", "0.6 × LoRA + 0.4 × n-gram", w=34, edge=INK, lw=1.5)
merge_y = 97
for x, (_, y, _, h) in ((AX, a[-1]), (BX, b[-1])):
    ax.plot([x, x, 50], [y + h / 2, merge_y, merge_y], color=MUTED, lw=1.2, zorder=2)
arrow([(50, merge_y), (50, blend[1] - blend[3] / 2)])

# 테스트 전용 후처리
panel(42, 108.5, 97, 119.5, SURFACE, EDGE, dashed=True)
ax.text(44, 109.8, "테스트에서만", fontsize=9, color=MUTED, va="top")
center = box(57, 115, "프롬프트 중심화", "비슷한 주제끼리 묶어 평균의 0.5배를 뺌", w=26, h=6.8)
sink = box(83.5, 115, "클래스 비율 보정 (Sinkhorn)", "예측 비율을 균등 쪽으로 35% 이동", w=24, h=6.8)
arrow([(center[0] + center[2] / 2, 115), (sink[0] - sink[2] / 2, 115)])
arrow([(50, blend[1] + blend[3] / 2), (50, 106.5), (center[0], 106.5), (center[0], 115 - 3.4)])

# 출력
out = box(20, 115, "argmax → 모델 이름", "18개 중 1개", w=28, h=6.8, edge=INK, lw=1.5)
arrow([(sink[0], 115 + 3.4), (sink[0], 124), (out[0], 124), (out[0], 115 + 3.4)])
arrow([(50, 106.5), (out[0], 106.5), (out[0], 115 - 3.4)], dashed=True)
ax.text(out[0] + 1.2, 107.6, "validation은 바로 argmax", fontsize=8.6, color=MUTED, va="top")

out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "figures")
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "architecture_v4.png", dpi=200, facecolor=SURFACE)
print("저장:", out_dir / "architecture_v4.png")
