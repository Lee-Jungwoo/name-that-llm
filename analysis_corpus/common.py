"""외부 데이터셋 olegzholobov/name-that-llm-response-corpus 검토용 공통 코드.

내려받은 파일(data/response-corpus/)은 데이터로만 읽습니다. 이 폴더의 스크립트는 `python -I`로 실행합니다.
"""
import re
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "response-corpus"
OUT = ROOT / "out" / "corpus"

_nb = nbformat.read(ROOT / "name_that_llm_v5.ipynb", as_version=4)
NB_CELLS = [c.source for c in _nb.cells if c.cell_type == "code"]


def nb_cell(prefix):
    return next(c for c in NB_CELLS if c.startswith(prefix))


# LABELS와 문체 피처 정의는 노트북에서 그대로 가져옵니다(복사본을 따로 두지 않음).
_setup = nb_cell("import gc, glob")
LABELS = eval(re.search(r"LABELS = (\[.*?\])", _setup, re.S).group(1))
L2I = {name: i for i, name in enumerate(LABELS)}
_ngram = nb_cell("FUNCTION_WORDS")
_G = {"np": np, "re": re}
exec(_ngram[_ngram.index("# v5: 문체 피처"):_ngram.index("def stylo_block")], _G)
stylo_feats, stylo_raw = _G["stylo_feats"], _G["stylo_raw"]
STYLO_NAMES = ["n_words", "n_sents", "sent_len_mean", "sent_len_std", "word_len_mean", "long_word_ratio", "mattr",
               "func_ratio", "em_dash", "spaced_em_dash", "en_dash", "semicolon", "colon", "comma", "paren", "question",
               "curly_apos", "straight_apos", "curly_quote", "straight_quote", "contraction", "first_person",
               "second_person", "not_but", "ly_adverb", "conj_start", "mid_caps"]


def norm(s):
    """노트북이 Open Pool에서 train 중복을 뺄 때 쓰는 정규화와 같습니다."""
    return re.sub(r"\W+", " ", s.lower()).strip()


REFUSAL_RE = re.compile(
    r"^\W*(i can[’']?t|i cannot|i won[’']?t|i[’']m not (going|able|comfortable)|i[’']m going to (pass|decline)|"
    r"i[’']m sorry|sorry,|no, i|i (must|have to|need to) decline|i will not|i[’']d rather not|i don[’']t think i|"
    r"i[’']m unable|i am (unable|not able)|i (can|could) not|i[’']m declining|i[’']ll (pass|decline))", re.I)
MARKDOWN_RE = re.compile(r"\*\*|^#{1,6} |^\s*[-*•] |^\s*\d+[.)] |```|^\|", re.M)


def load_corpus():
    """short 뷰를 long 형식(셀 하나 = 응답 하나)으로 풀고 메타데이터와 출처를 붙입니다.

    quarantine_luna_* 파일은 읽지 않습니다. 반환 열: id, prompt, model, text, prompt_group_id, domain, source,
    word_count(공백 기준), 품질 플래그, norm.
    """
    wide = pd.read_parquet(CORPUS / "external_short_wide.parquet")
    long = wide.melt(id_vars=["id", "prompt"], var_name="model", value_name="text").dropna(subset=["text"])
    meta = pd.read_parquet(CORPUS / "external_cell_metadata.parquet")
    meta = meta[meta["view"] == "short"].drop(columns=["view"])
    long = long.merge(meta, on=["id", "model"], how="left", validate="one_to_one")
    assert long.prompt_group_id.notna().all()
    prov = pd.read_parquet(CORPUS / "external_cell_provenance.parquet",
                           columns=["cell_id_short", "source", "license_status"]).dropna(subset=["cell_id_short"])
    prov = prov.sort_values("source").groupby("cell_id_short").agg(
        source=("source", "first"), n_sources=("source", "nunique"), license_status=("license_status", "first"))
    long = long.merge(prov, left_on="cell_id", right_index=True, how="left")
    long["text"] = long["text"].astype(str)
    long["norm"] = long.text.map(norm)
    long["ws_words"] = long.text.str.split().str.len()
    return long.reset_index(drop=True)


def load_train():
    train = pd.read_csv(ROOT / "data" / "name-that-llm" / "train.csv")
    train["text"] = train["text"].astype(str)
    return train


def load_pool(columns=("text", "model", "match_weight")):
    """노트북 데이터 셀과 같은 필터(라벨 18개, train과 정규화 텍스트가 같은 지문 제외)를 거친 Open Pool."""
    pool = pd.read_parquet(ROOT / "data" / "open-pool" / "ntl_open_pool.parquet", columns=list(columns))
    pool = pool[pool.model.isin(LABELS)].reset_index(drop=True)
    return pool[~pool.text.map(norm).isin(set(load_train().text.map(norm)))].reset_index(drop=True)


def sample_pool(pool, n_per_label, seed=0):
    """노트북의 sample_pool과 같은 논리. n_per_label이 dict이면 라벨마다 다른 개수를 뽑습니다."""
    rng = np.random.default_rng(seed)
    picks = []
    for name, g in pool.groupby("model"):
        n = n_per_label[name] if isinstance(n_per_label, dict) else n_per_label
        w = g.match_weight.fillna(0).to_numpy(dtype=float)
        k = min(n, int((w > 0).sum()))
        picks.append(g.index.to_numpy()[rng.choice(len(g), k, replace=False, p=w / w.sum())])
    return pool.loc[np.concatenate(picks)].reset_index(drop=True)
