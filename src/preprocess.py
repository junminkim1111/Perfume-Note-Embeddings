"""data/raw/perfumes.{csv,jsonl} -> data/perfumes_clean.csv, data/smells_like.csv (README 3장)"""
import json
import pandas as pd
from functions import parse_notes, format_notes, merge_notes

RAW = "data/raw"
LAYERS = ["top", "middle", "base"]


def fill_layers(top, middle, base, flat):
    """flat이면 세 레이어에 복사 (D1), 빈 레이어는 나머지 합집합으로 채움 (D2). 노트가 없으면 None."""
    is_flat = not (top or middle or base)
    layers = [flat] * 3 if is_flat else [top, middle, base]
    union = merge_notes(*layers)
    if not union:
        return None
    return is_flat, [layer or union for layer in layers]


def prepare_perfumes():
    raw = pd.read_csv(f"{RAW}/perfumes.csv")
    rows = []
    for r in raw.itertuples():
        filled = fill_layers(*(parse_notes(getattr(r, f"notes_{c}")) for c in LAYERS + ["flat"]))
        if filled is None:
            continue
        is_flat, layers = filled
        rows.append([r.id, r.name, r.brand, r.year, r.gender, r.vote_count, is_flat, *map(format_notes, layers)])
    cols = ["id", "name", "brand", "year", "gender", "vote_count", "is_flat"] + LAYERS
    df = pd.DataFrame(rows, columns=cols).astype({"year": "Int64"})
    df.to_csv("data/perfumes_clean.csv", index=False)
    return df


def prepare_similar(ids):
    """jsonl similar.* -> smells_like.csv (투표 정답), also_liked.csv (Fragrantica 추천 목록, 참고용)"""
    smells, liked = [], []
    with open(f"{RAW}/perfumes.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p["id"] not in ids:
                continue
            for t in p["similar"]["reminds_me_of"]:
                if t["id"] in ids and t["id"] != p["id"]:
                    smells.append([p["id"], t["id"], t["up_votes"], t["down_votes"]])
            targets = [t["id"] for t in p["similar"]["also_liked"] if t["id"] in ids and t["id"] != p["id"]]
            liked += [[p["id"], t, rank] for rank, t in enumerate(targets, 1)]
    smells = pd.DataFrame(smells, columns=["query_id", "target_id", "agree", "disagree"])
    liked = pd.DataFrame(liked, columns=["query_id", "target_id", "rank"])
    smells.to_csv("data/smells_like.csv", index=False)
    liked.to_csv("data/also_liked.csv", index=False)
    return smells, liked


if __name__ == "__main__":
    perfumes = prepare_perfumes()
    print(f"perfumes: {len(perfumes)} (flat {perfumes.is_flat.sum()})")
    pairs, liked = prepare_similar(set(perfumes.id))
    print(f"smells_like pairs: {len(pairs)}, queries: {pairs.query_id.nunique()}")
    print(f"also_liked pairs: {len(liked)}, queries: {liked.query_id.nunique()}")
