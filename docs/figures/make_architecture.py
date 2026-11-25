"""docs/07_architecture_by_version.md의 버전별 구조 그림(arch_v1.png ~ arch_v8.png)을 그립니다.

그림은 위에서 아래로 흐릅니다(좌표는 아래에서 위로 쌓은 뒤 y축을 뒤집음). 블록 색은 종류를 뜻하고(입력 · 특징 추출 · 풀링/연결 · 분류기 · 확률 · 결합 · 후처리 · 출력),
직전 버전(계보 기준) 대비 바뀐 블록은 빨간 테두리와 NEW 표시를 붙입니다. 데이터셋 관련 내용은 그림에 넣지 않습니다.

사용: python docs/figures/make_architecture.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent

# 한글 폰트: Pretendard(보통·굵게)가 있으면 쓰고, 없으면 Apple SD Gothic Neo
_pretendard = [d / f"Pretendard-{w}.otf" for d in (Path.home() / "Library/Fonts", Path("/Library/Fonts"))
               for w in ("Regular", "Bold")]
_fonts = [p for p in _pretendard if p.exists()] or [Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")]
for path in _fonts:
    font_manager.fontManager.addfont(str(path))
plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(_fonts[0])).get_name()

# 블록 종류별 색 (채움, 테두리). Transformer 논문 그림의 색 배치를 따릅니다.
COLORS = {
    "inp": "#f9d3d6",   # 입력 · 토큰화 (분홍)
    "feat": "#fcdcb4",  # 특징 추출: TF-IDF, 트랜스포머 층 (주황)
    "pool": "#f2f5c6",  # 풀링 · 연결 · 정규화 (연노랑)
    "clf": "#c9e7f5",   # 분류기 · 선형 헤드 (하늘)
    "prob": "#d3eedd",  # 온도 · log-softmax (연두)
    "comb": "#e3dcf3",  # 분기 결합: 블렌드, 게이트 (연보라)
    "post": "#ffffff",  # 테스트 후처리 (흰색, 점선 묶음)
    "out": "#bfe6cd",   # 최종 출력 (초록)
}
EDGE, NEW = "#333333", "#e03131"
BOX_W, COL_W, GAP, LINE_H, PAD = 3.3, 4.8, 0.42, 0.27, 0.3
SCALE = 0.9  # 데이터 1단위 = 0.9인치


def box_h(text):
    return 0.34 + LINE_H * (text.count("\n") + 1)


def draw_box(ax, cx, y, text, kind, new=False, w=BOX_W, fs=10.5):
    h = box_h(text)
    ax.add_patch(FancyBboxPatch((cx - w / 2, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=COLORS[kind], ec=NEW if new else EDGE, lw=2.6 if new else 1.2, zorder=3))
    ax.text(cx, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=4, linespacing=1.35)
    if new:
        tag(ax, cx + w / 2 - 0.05, y + 0.02)  # y축을 뒤집으므로 화면에서는 상자 위쪽
    return h


def tag(ax, x, y):
    ax.text(x, y, "NEW", ha="center", va="center", fontsize=7.5, color="white", fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.15", fc=NEW, ec=NEW))


def arrow(ax, x1, y1, x2, y2, style="-|>", ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=2,
                arrowprops=dict(arrowstyle=style, lw=1.4, color="#222", shrinkA=0, shrinkB=0,
                                mutation_scale=13, linestyle=ls))


def group(ax, x0, y0, x1, y1, label, dashed=False, new=False, label_pos="top"):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.25",
                                fc="#ffffff" if dashed else "#f0f0f0", ec=NEW if new else "#555555",
                                lw=2.4 if new else 1.3, ls=(0, (4, 3)) if dashed else "-", zorder=1))
    if label_pos == "top":  # 화면 위쪽(= y0) 왼편. 가운데는 들어오는 화살표 자리
        ax.text(x0 + 0.18, y0 + 0.27, label, ha="left", va="center", fontsize=10.5, fontweight="bold", zorder=4)
    else:
        ax.text(x1 + 0.15, (y0 + y1) / 2, label, ha="left", va="center", fontsize=10, color="#555", zorder=4)
    if new:
        tag(ax, x1 - 0.1, y0)


# ---- 분기 정의 ---------------------------------------------------------------------------------------------

def ngram_branch(stylo=None):
    """stylo: None(없음), 'old'(직전에도 있음), 'new'(이 버전에서 생김)."""
    boxes = [
        dict(t="4가지 뷰로 변환\nchar · word · shape · func", k="inp"),
        dict(t="뷰별 TF-IDF\n× 뷰 가중치 1.0 / 0.7 / 0.5 / 0.5", k="feat"),
        dict(t="가로로 연결 (희소 행렬)", k="pool", new=stylo == "new"),
        dict(t="LinearSVC (C = 0.1)\n클래스별 마진 18개", k="clf"),
    ]
    if stylo:
        boxes[2]["side"] = dict(t="문체 피처 27개\n표준화 → × α / √27", k="inp", new=stylo == "new")
    return dict(title="n-gram 분기 (CPU)", boxes=boxes)


def lora_branch(echo=None):
    boxes = []
    if echo:
        boxes.append(dict(t="echo 입력\n지문 + 빈 줄 + 지문", k="inp", new=echo == "new"))
    boxes += [
        dict(t="Qwen3 토크나이저\n최대 384토큰", k="inp"),
        dict(t="Qwen3-4B-Base 디코더 층\n가중치 고정 + LoRA (q·k·v·o)", k="feat", rep="×18"),
        dict(t=("깊이 7곳 평균 풀링\n두 번째 사본 토큰만" if echo else "깊이 7곳 평균 풀링\n패딩 뺀 모든 토큰"),
             k="pool", new=echo == "new"),
        dict(t="깊이별 LayerNorm → 연결 (17,920)", k="pool"),
        dict(t="Dropout → Linear → 로짓 18개", k="clf"),
    ]
    return dict(title="LoRA 분기 (GPU)", boxes=boxes)


def deberta_branch(new=False):
    return dict(title="DeBERTa 분기 (GPU)", new=new, boxes=[
        dict(t="DeBERTa 토크나이저\n최대 512토큰", k="inp"),
        dict(t="DeBERTa-v3-large 인코더 층\n전체 학습", k="feat", rep="×24"),
        dict(t="마지막 층 평균 풀링", k="pool"),
        dict(t="Dropout → Linear → 로짓 18개", k="clf"),
    ])


TEMP = dict(t="온도 T로 나눔 → log-softmax", k="prob")

BLEND2_BEST = dict(t="log 확률 가중합\nw · LoRA + (1 − w) · n-gram,  w = 검증 최고값", k="comb")
BLEND2_FIX = dict(t="log 확률 가중합\n0.6 · LoRA + 0.4 · n-gram (고정)", k="comb")
BLEND3 = dict(t="log 확률 가중합 (3분기)\n0.6 · LoRA + 0.2 · DeBERTa + 0.2 · n-gram", k="comb")
GATE2 = dict(t="회사 게이트\n1순위 회사 안 분포를 회사 전용 로지스틱 회귀 쪽으로 옮김\n(입력: n-gram · LoRA의 회사 열 log 확률)", k="comb")
GATE3 = dict(t="회사 게이트\n1순위 회사 안 분포를 회사 전용 로지스틱 회귀 쪽으로 옮김\n(입력: n-gram · LoRA · DeBERTa의 회사 열 log 확률)", k="comb")
CENTER = dict(t="프롬프트 중심화\n주제 묶음 평균 log 확률의 0.5배를 뺌", k="post")
SINKHORN = dict(t="Sinkhorn 클래스 비율 보정 (ρ = 0.35)", k="post")


def new(d):
    return {**d, "new": True}


VERSIONS = {
    1: dict(branches=[ngram_branch(), lora_branch()], top=[BLEND2_BEST], post=[CENTER]),
    2: dict(branches=[ngram_branch(), lora_branch()], top=[BLEND2_BEST], post=[CENTER]),
    3: dict(branches=[ngram_branch(), lora_branch()], top=[new(BLEND2_FIX)], post=[CENTER, new(SINKHORN)]),
    4: dict(branches=[ngram_branch(), lora_branch("new")], top=[BLEND2_FIX], post=[CENTER, SINKHORN]),
    5: dict(branches=[ngram_branch("new"), lora_branch("old")], top=[BLEND2_FIX], post=[CENTER, SINKHORN]),
    6: dict(branches=[ngram_branch(), lora_branch("old")], top=[BLEND2_FIX, new(GATE2)], post=[CENTER, SINKHORN]),
    7: dict(branches=[ngram_branch(), lora_branch("old"), deberta_branch(new=True)],
            top=[new(BLEND3), new(GATE3)], post=[CENTER, SINKHORN]),
    8: dict(branches=[ngram_branch("new"), lora_branch("old"), deberta_branch()],
            top=[BLEND3, GATE3], post=[CENTER, SINKHORN]),
}


# ---- 그리기 ------------------------------------------------------------------------------------------------

def draw(version, spec):
    branches = spec["branches"]
    has_side = any("side" in b for br in branches for b in br["boxes"])
    left = BOX_W + 0.9 if has_side else 0.4           # 옆 입력 상자 자리
    xs = [left + PAD + COL_W / 2 + i * (COL_W + 0.5) for i in range(len(branches))]
    right = xs[-1] + COL_W / 2 + PAD + 0.9               # ×N 표기 자리

    fig, ax = plt.subplots(figsize=(right * SCALE, 1))  # 높이는 뒤에서 맞춤
    y_in = 0.35
    ax.text((xs[0] + xs[-1]) / 2, y_in, "입력: 영어 지문 1개", ha="center", va="center", fontsize=12)

    # 분기: 흐름 순서대로 y를 늘려 가며 쌓음. 맨 위 '온도' 상자는 모든 분기가 같은 높이에 둡니다.
    y_bus, y0 = y_in + 0.45, y_in + 0.85
    ax.plot([xs[0], xs[-1]], [y_bus, y_bus], color="#222", lw=1.4, zorder=2)
    ax.plot([(xs[0] + xs[-1]) / 2] * 2, [y_in + 0.2, y_bus], color="#222", lw=1.4, zorder=2)
    stacks = []
    for br in branches:
        hs = [box_h(b["t"]) for b in br["boxes"]]
        stacks.append(sum(hs) + GAP * (len(hs) - 1))
    body = max(stacks)
    TITLE = 0.45  # 분기 이름 자리
    y_temp = y0 + PAD + TITLE + body + GAP + 0.1
    y_top_group = y_temp + box_h(TEMP["t"]) + PAD

    tops = []
    for x, br, stack in zip(xs, branches, stacks):
        group(ax, x - COL_W / 2, y0, x + COL_W / 2, y_top_group, br["title"], new=br.get("new", False))
        arrow(ax, x, y_bus, x, y0 + PAD + TITLE)
        y = y0 + PAD + TITLE
        n = len(br["boxes"])
        gap = GAP + (body - stack) / n  # 짧은 분기는 간격을 고르게 넓힘
        for i, b in enumerate(br["boxes"]):
            h = draw_box(ax, x, y, b["t"], b["k"], b.get("new", False))
            if "rep" in b:
                ax.text(x + BOX_W / 2 + 0.12, y + h / 2, b["rep"], ha="left", va="center", fontsize=12)
            if "side" in b:
                s = b["side"]
                sx = x - COL_W / 2 - 0.45 - BOX_W / 2
                sh = box_h(s["t"])
                draw_box(ax, sx, y + h / 2 - sh / 2, s["t"], s["k"], s.get("new", False))
                ax.text((sx + BOX_W / 2 + x - BOX_W / 2) / 2, y + h / 2 + 0.1, "연결", ha="center", va="top", fontsize=8.5, color="#555")
                arrow(ax, sx + BOX_W / 2, y + h / 2, x - BOX_W / 2, y + h / 2)
            y_next = y + h + gap
            if i < n - 1:
                arrow(ax, x, y + h, x, y_next)
            y = y_next
        arrow(ax, x, y - gap, x, y_temp)
        draw_box(ax, x, y_temp, TEMP["t"], TEMP["k"])
        tops.append(y_temp + box_h(TEMP["t"]))

    # 결합 단계: 분기 전체 폭의 상자
    wide = (xs[-1] - xs[0]) + BOX_W if len(xs) > 1 else BOX_W
    cx = (xs[0] + xs[-1]) / 2
    y = y_top_group + 0.55
    for x, t in zip(xs, tops):
        arrow(ax, x, y_top_group, x, y)
    for i, b in enumerate(spec["top"]):
        h = draw_box(ax, cx, y, b["t"], b["k"], b.get("new", False), w=wide)
        arrow(ax, cx, y + h, cx, y + h + GAP + (0.25 if i == len(spec["top"]) - 1 else 0))
        y += h + GAP

    # 테스트 후처리 묶음
    y += 0.25
    g0 = y
    y += PAD
    for i, b in enumerate(spec["post"]):
        h = draw_box(ax, cx, y, b["t"], b["k"], b.get("new", False), w=wide - 0.8)
        if i < len(spec["post"]) - 1:
            arrow(ax, cx, y + h, cx, y + h + GAP)
        y += h + GAP
    g1 = y - GAP + PAD
    group(ax, cx - wide / 2 - 0.15, g0, cx + wide / 2 + 0.15, g1, "테스트에만\n적용", dashed=True, label_pos="side")
    ax.plot([cx, cx], [y - GAP, g1], color="#222", lw=1.4, zorder=2)
    arrow(ax, cx, g1, cx, g1 + 0.45)
    h = draw_box(ax, cx, g1 + 0.45, "argmax", "out", w=2.2, fs=11)
    y_end = g1 + 0.45 + h
    arrow(ax, cx, y_end, cx, y_end + 0.45)
    ax.text(cx, y_end + 0.75, "예측: 18개 모델 중 하나", ha="center", va="center", fontsize=12)

    total_h = y_end + 1.1
    ax.set_xlim(0, right + 0.4)
    ax.set_ylim(0, total_h)
    ax.invert_yaxis()  # 위에서 아래로 흐르게
    ax.set_aspect("equal")
    ax.axis("off")
    fig.set_size_inches((right + 0.4) * SCALE, total_h * SCALE)
    fig.savefig(OUT / f"arch_v{version}.png", dpi=130, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    for v, spec in VERSIONS.items():
        draw(v, spec)
        print(f"arch_v{v}.png")
