"""분석용 지문을 data/에 만든다. 사용법: python fetch_data.py <내려받은 parquet 폴더>

Kaggle에서 train.csv를 받을 수 없는 환경이라 Hugging Face 미러(HLDKNotFound/name-that-llm-dataset)를 썼다.
미러의 train_stage1/train_stage2/val_holdout/clean_pool parquet을 한 폴더에 내려받은 뒤 실행한다.
  https://huggingface.co/datasets/HLDKNotFound/name-that-llm-dataset/resolve/main/data/<split>-00000-of-00001.parquet

- source == "official_gold": 대회 train. id 기준 중복을 빼면 2,625개(원본 2,699개 중 74개 없음).
  oof_v4.npz의 (prompt_id, 모델)과 1:1로 맞춰 data/train_recovered.csv(oof 행 순서, 없는 행은 text 빈칸)로 저장한다.
- source == "open_pool": Open Pool. id 기준 중복을 빼면 49,050개(모델당 2,725개). data/pool_hf.parquet로 저장한다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
src = Path(sys.argv[1])
parts = [pd.read_parquet(src / f"{s}-00000-of-00001.parquet" if (src / f"{s}-00000-of-00001.parquet").exists()
                         else src / f"{s}.parquet") for s in ["train_stage1", "train_stage2", "val_holdout", "clean_pool"]]
g = pd.concat(parts)
gold = g[g.source == "official_gold"].drop_duplicates("id")[["id", "prompt_id", "text", "model"]]
pool = g[g.source == "open_pool"].drop_duplicates("id")[["id", "prompt_id", "text", "model"]]

d = np.load(ROOT / "analysis_v4" / "oof_v4.npz", allow_pickle=True)
labels = list(d["labels"])
key = pd.DataFrame({"prompt_id": d["prompt_id"], "model": [labels[i] for i in d["y"]]})
train = key.merge(gold, on=["prompt_id", "model"], how="left")
assert len(train) == len(key) and not train.duplicated(["prompt_id", "model"]).any()
print(f"train: {train.text.notna().sum()}/{len(train)}개 지문 찾음, pool: {len(pool)}개")

(ROOT / "data").mkdir(exist_ok=True)
train.to_csv(ROOT / "data" / "train_recovered.csv", index=False)
pool.reset_index(drop=True).to_parquet(ROOT / "data" / "pool_hf.parquet")
