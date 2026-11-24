"""같은 기업 모델 비교용 데이터 로딩과 문체 피처.

데이터
- data/train_recovered.csv: 대회 train 2,699행 중 지문을 찾은 2,625행. oof_v4.npz의 (prompt_id, model)과 1:1로 맞춘 것.
- data/pool_hf.parquet: Open Pool 지문 49,050개(모델당 2,725개).
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LABELS = ["gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
          "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
          "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
          "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3"]
ORGS = {"GPT": LABELS[0:5], "Gemini": LABELS[5:9], "Claude": LABELS[9:14], "GLM": LABELS[15:17]}

FUNC = set("""a an the and or but if while although because as so than that which who whom whose this these those
it its they them their he him his she her we us our you your i me my of in on at by for with from to into over
under about between through during not no nor is are was were be been being have has had do does did will would
can could should may might must""".split())

SENT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-z’']+")
CONTR_RE = re.compile(r"\b\w+(?:n['’]t|['’](?:s|re|ll|ve|d|m))\b", re.I)
NOT_BUT_RE = re.compile(r"\bnot\b[^.;:!?]{1,60}?\bbut\b", re.I)
LY_RE = re.compile(r"\b\w{3,}ly\b", re.I)

FEATURES = {
    # 길이와 구조
    "n_words": "단어 수",
    "n_sents": "문장 수",
    "sent_len_mean": "평균 문장 길이(단어)",
    "sent_len_std": "문장 길이 표준편차",
    "word_len_mean": "평균 단어 길이(글자)",
    "long_word_ratio": "9글자 이상 단어 비율",
    "mattr": "어휘 다양도(MATTR, 창 40단어)",
    "func_ratio": "기능어 비율",
    # 구두점(100단어당)
    "em_dash": "— / 100단어",
    "spaced_em_dash": "띄어 쓴 — / 100단어",
    "en_dash": "– / 100단어",
    "semicolon": "; / 100단어",
    "colon": ": / 100단어",
    "comma": ", / 100단어",
    "paren": "( / 100단어",
    "question": "? / 100단어",
    "curly_apos": "’ / 100단어",
    "straight_apos": "' / 100단어",
    "curly_dq": "“” / 100단어",
    "straight_dq": "\" / 100단어",
    # 어휘 습관(100단어당)
    "contraction": "축약형(don't, it's 등) / 100단어",
    "first_person": "i/me/my/we/us/our / 100단어",
    "second_person": "you/your / 100단어",
    "not_but": "not … but / 100단어",
    "ly_adverb": "-ly 단어 / 100단어",
    "and_but_start": "And/But/Yet/So로 시작하는 문장 비율",
    "capital_ratio": "문장 첫 단어가 아닌 대문자 단어 비율",
}


def feats(text):
    words = WORD_RE.findall(text)
    nw = max(len(words), 1)
    sents = [s for s in SENT_RE.split(text.strip()) if s]
    sl = np.array([len(WORD_RE.findall(s)) for s in sents]) if sents else np.array([nw])
    lw = [w.lower() for w in words]
    per = lambda c: 100.0 * c / nw
    win = 40
    if nw > win:
        mattr = np.mean([len(set(lw[i:i + win])) / win for i in range(0, nw - win + 1, 5)])
    else:
        mattr = len(set(lw)) / nw
    first_words = {len(WORD_RE.findall(" ".join(sents[:i]))) for i in range(len(sents))}
    caps = sum(1 for i, w in enumerate(words) if w[0].isupper() and i not in first_words and w not in ("I",))
    return {
        "n_words": nw,
        "n_sents": len(sents),
        "sent_len_mean": sl.mean(),
        "sent_len_std": sl.std(),
        "word_len_mean": np.mean([len(w) for w in words]) if words else 0,
        "long_word_ratio": np.mean([len(w) >= 9 for w in words]) if words else 0,
        "mattr": mattr,
        "func_ratio": np.mean([w in FUNC for w in lw]) if lw else 0,
        "em_dash": per(text.count("—")),
        "spaced_em_dash": per(text.count(" — ")),
        "en_dash": per(text.count("–")),
        "semicolon": per(text.count(";")),
        "colon": per(text.count(":")),
        "comma": per(text.count(",")),
        "paren": per(text.count("(")),
        "question": per(text.count("?")),
        "curly_apos": per(text.count("’")),
        "straight_apos": per(text.count("'")),
        "curly_dq": per(text.count("“") + text.count("”")),
        "straight_dq": per(text.count('"')),
        "contraction": per(len(CONTR_RE.findall(text))),
        "first_person": per(sum(w in ("i", "me", "my", "we", "us", "our") for w in lw)),
        "second_person": per(sum(w in ("you", "your") for w in lw)),
        "not_but": per(len(NOT_BUT_RE.findall(text))),
        "ly_adverb": per(len(LY_RE.findall(text))),
        "and_but_start": np.mean([bool(re.match(r"(And|But|Yet|So)\b", s)) for s in sents]) if sents else 0,
        "capital_ratio": caps / nw,
    }


def load_train():
    """대회 train(지문 2,625개) + v4 OOF 예측. 행 순서는 oof_v4.npz와 같다(지문이 없는 74행은 text가 NaN)."""
    df = pd.read_csv(DATA / "train_recovered.csv")
    d = np.load(ROOT / "analysis_v4" / "oof_v4.npz", allow_pickle=True)
    assert list(d["labels"]) == LABELS
    assert (df.model.map(LABELS.index).values == d["y"]).all()
    df["y"] = d["y"]
    df["fold"] = d["fold"]
    df["pred"] = d["eval_lp"].argmax(1)
    df["pred_model"] = [LABELS[i] for i in df.pred]
    return df, d


def load_pool():
    return pd.read_parquet(DATA / "pool_hf.parquet").reset_index(drop=True)


def pool_features():
    """Open Pool 피처(계산이 1분쯤 걸려 data/에 캐시)."""
    cache = DATA / "pool_feats.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    pool = load_pool()
    P = pd.concat([pool[["model"]], feature_frame(pool.text)], axis=1)
    P.to_parquet(cache)
    return P


def feature_frame(texts):
    return pd.DataFrame([feats(t) for t in texts])
