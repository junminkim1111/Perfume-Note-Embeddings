"""평가: "Smells like this" 투표를 정답으로 (README 5장)"""
import math
import os
import sys
import numpy as np
import pandas as pd
import torch
from functions import merge_notes

K = 100  # 순위 목록 길이. MRR은 MRR@100
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}


def get_split(ids, seed=0):
    """향수 id -> 'train'/'val'/'test'. 정답 기준(min_up)과 상관없이 고정."""
    ids = np.sort(np.asarray(ids))
    perm = np.random.default_rng(seed).permutation(len(ids))
    cut = np.cumsum([int(len(ids) * f) for f in SPLITS.values()])[:-1]
    labels = np.empty(len(ids), dtype=object)
    for name, part in zip(SPLITS, np.split(perm, cut)):
        labels[part] = name
    return dict(zip(ids, labels))


def load_truth(rec, min_up, split, seed=0):
    """{query_row: {target_row: gain}}, gain = agree − disagree. agree > disagree, agree ≥ min_up."""
    pairs = pd.read_csv("data/smells_like.csv")
    pairs = pairs[(pairs.agree > pairs.disagree) & (pairs.agree >= min_up)]
    split_of = get_split(rec.dataset["id"], seed)
    pairs = pairs[pairs.query_id.map(split_of) == split]
    truth = {}
    for q, t, gain in zip(pairs.query_id.map(rec.row_of), pairs.target_id.map(rec.row_of), pairs.agree - pairs.disagree):
        truth.setdefault(q, {})[t] = gain
    return truth


def model_lists(rec, query_rows, batch=512):
    """쿼리별 상위 K개 행 번호 (자기 제외)와 자기 자신 Hit@1 (자기 점수 == 최댓값, D15)"""
    lists, self_hit = [], []
    for i in range(0, len(query_rows), batch):
        rows = torch.tensor(query_rows[i:i + batch], device=rec.device)
        scores = rec.row_scores(rows)
        self_score = scores[torch.arange(len(rows)), rows]
        self_hit += (self_score >= scores.max(1).values - 1e-5).cpu().tolist()
        lists += rec.search(scores, K, exclude_rows=rows)[1].tolist()
    return lists, np.mean(self_hit)


def random_lists(n_db, query_rows, seed=0):
    rng = np.random.default_rng(seed)
    return [[r for r in rng.choice(n_db, K + 1, replace=False) if r != q][:K] for q in query_rows]


def also_liked_lists(rec, query_rows):
    liked = pd.read_csv("data/also_liked.csv").sort_values(["query_id", "rank"])
    by_query = liked.groupby(liked.query_id.map(rec.row_of)).target_id.apply(lambda s: [rec.row_of[t] for t in s])
    return [by_query.get(q, []) for q in query_rows]


def top_rows(scores, rows, seed=0):
    """(B, N) 점수에서 자기 자신을 빼고 상위 K개 행 번호. 공정한 동점 처리:
    점수를 소수 5자리로 반올림하고 (float32 잡음 제거), 같은 점수는 고정 시드 무작위 순서로 둔다.
    (torch.topk는 동점일 때 앞 행 = 작은 id = 인기 향수를 먼저 둬서 점수가 부풀려진다)"""
    n = scores.shape[1]
    key = np.round(scores.astype(np.float64) * 1e5).astype(np.int64) * n + np.random.default_rng(seed).permutation(n)
    key[np.arange(len(rows)), rows] = np.iinfo(np.int64).min + 1  # -min은 넘친다
    top = np.argpartition(-key, K, axis=1)[:, :K]
    return np.take_along_axis(top, np.argsort(-np.take_along_axis(key, top, 1), 1), 1).tolist()


def fair_model_lists(rec, query_rows, batch=512):
    """model_lists와 같은 점수, top_rows의 공정한 동점 처리"""
    lists = []
    for i in range(0, len(query_rows), batch):
        rows = np.array(query_rows[i:i + batch])
        lists += top_rows(rec.row_scores(torch.tensor(rows, device=rec.device)).cpu().numpy(), rows)
    return lists


def onehot_lists(rec, W, query_rows, delta=None, batch=256):
    """단어 유무 기준선 1: 노트 임베딩 대신 원-핫 (E = 단위행렬). IDF, 레이어, W, flat 규칙은 rec과 같다.
    점수 = Σ_ij W_ij cos(x_i, x_j), x_l = 레이어 l의 IDF 가중 노트 벡터.
    delta가 있으면 flat 향수가 낀 쌍은 노트 전체 집합 벡터의 cosine − delta"""
    unit_rows = lambda S: S.multiply(1 / np.sqrt(S.multiply(S).sum(1))).tocsr()
    X = [unit_rows(S) for S in rec.weight_matrices(rec.layers, 1.)]
    Xu = unit_rows(rec.weight_matrices([[merge_notes(*p)] for p in rec.layers], 1.)[0])
    flat = rec.flat.cpu().numpy()
    W = np.asarray(W) / np.sum(W)
    lists = []
    for i in range(0, len(query_rows), batch):
        rows = np.array(query_rows[i:i + batch])
        scores = sum((sum(W[a, b] * X[a][rows] for a in range(3)) @ X[b].T).toarray() for b in range(3))
        if delta is not None:
            union = (Xu[rows] @ Xu.T).toarray() - delta
            scores = np.where(flat[rows][:, None] | flat[None, :], union, scores)
        lists += top_rows(scores.astype(np.float32), rows)
    return lists


def paired_bootstrap(a, b, n=2000, seed=0):
    """쿼리별 지표 a, b (같은 쿼리 순서) -> 평균 차이 a − b와 95% 구간"""
    diff = np.asarray(a) - np.asarray(b)
    idx = np.random.default_rng(seed).integers(0, len(diff), (n, len(diff)))
    bs = diff[idx].mean(1)
    return float(diff.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def score_query(ranked, truth):
    """한 쿼리의 순위 목록과 정답 {row: gain} -> 지표"""
    rank = {r: i for i, r in enumerate(ranked, 1)}
    hits = sorted(rank[t] for t in truth if t in rank)
    dcg = sum(g / math.log2(rank[t] + 1) for t, g in truth.items() if rank.get(t, K + 1) <= 10)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(truth.values(), reverse=True)[:10]))
    out = {f"R@{k}": sum(h <= k for h in hits) / len(truth) for k in (5, 10, 20)}
    out["nDCG@10"] = dcg / idcg
    out["MRR"] = 1 / hits[0] if hits else 0.
    return out


def sanity(rec, query_rows, lists, n=10):
    """정상 동작 확인용: 상위 n개의 브랜드/성별 일치율, 노트 Jaccard"""
    notes = [set().union(*layers) for layers in rec.layers]
    brand, gender = rec.dataset["brand"].values, rec.dataset["gender"].values
    b, g, j = [], [], []
    for q, ranked in zip(query_rows, lists):
        for r in ranked[:n]:
            b.append(brand[q] == brand[r]); g.append(gender[q] == gender[r])
            j.append(len(notes[q] & notes[r]) / len(notes[q] | notes[r]))
    return {"brand": np.mean(b), "gender": np.mean(g), "jaccard": np.mean(j)}


def evaluate(rec, truth, lists, self_hit=float("nan")):
    """지표 표: 전체 / flat / pyramid (D14)"""
    query_rows = list(truth)
    per_query = pd.DataFrame([score_query(lst, truth[q]) for q, lst in zip(query_rows, lists)])
    per_query["subset"] = np.where(rec.dataset["is_flat"].values[query_rows], "flat", "pyramid")
    table = pd.concat([per_query.drop(columns="subset").mean().rename("all"),
                       *(g.drop(columns="subset").mean().rename(s) for s, g in per_query.groupby("subset"))], axis=1).T
    table["n"] = [len(per_query), *per_query.groupby("subset").size()]
    sanity_metrics = sanity(rec, query_rows, lists) | {"self@1": self_hit}
    return table, sanity_metrics


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)  # results/는 레포에 없다 (.gitignore)
    from sentence_transformers import SentenceTransformer
    from perfume_recommendation import Recommender, V1_LAYER_W

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
    rec = Recommender(pd.read_csv("data/perfumes_clean.csv"), model, device, alpha_min=.5, layer_weights=V1_LAYER_W, delta=None)  # 5단계 설정 그대로
    IDF_weight = rec.IDF_weight
    pd.set_option("display.precision", 4, "display.width", 200)

    if sys.argv[1:] == ["baselines"]:
        # README 결과 비교: 테스트 세트, up ≥ 5, 모든 방법을 같은 공정한 동점 처리(top_rows)로
        import pickle
        import time
        from functions import get_E, note_texts
        import json
        from perfume_recommendation import LAYER_W, FLAT_DELTA
        onehot_delta = json.load(open("results/flat_onehot.json"))["best delta"]  # optimize.py flat_onehot
        E = {i: get_E("minilm", i, note_texts(rec.vocab, i), device) for i in ("name", "name+desc")}
        rec = Recommender(rec.dataset, None, device, E=E["name+desc"])
        truth = load_truth(rec, 5, "test")
        query_rows = list(truth)
        print(f"test, up ≥ 5: {len(query_rows)} queries")

        def ours(input_type, W, delta):
            rec.E = E[input_type]
            rec.build_DB(1.)
            rec.set_layer_weights(W)
            rec.delta = delta
            return fair_model_lists(rec, query_rows)

        methods = {"random": lambda: random_lists(len(rec.dataset), query_rows),
                   "TF-IDF one-hot, final W, flat rule": lambda: onehot_lists(rec, LAYER_W, query_rows, delta=onehot_delta),
                   "MiniLM name, initial W (v1 setup on new data)": lambda: ours("name", V1_LAYER_W, None),
                   "MiniLM name+desc, initial W": lambda: ours("name+desc", V1_LAYER_W, None),
                   "MiniLM name+desc, final W (layered only)": lambda: ours("name+desc", LAYER_W, None),
                   "ours (MiniLM name+desc, final W, flat rule)": lambda: ours("name+desc", LAYER_W, FLAT_DELTA),
                   "Fragrantica also_liked (reference)": lambda: also_liked_lists(rec, query_rows)}
        rows, per_query, all_lists = [], {}, {}
        for name, make in methods.items():
            t0 = time.time()
            all_lists[name] = lists = make()
            table, s = evaluate(rec, truth, lists)
            per_query[name] = np.array([[score_query(l, truth[q])[m] for m in ("R@10", "nDCG@10")] for q, l in zip(query_rows, lists)])
            print(f"\n== {name} ({time.time() - t0:.0f}s) ==  " + "  ".join(f"{k} {v:.3f}" for k, v in s.items() if k != "self@1"))
            print(table.to_string())
            for subset, r in table.iterrows():
                rows.append({"method": name, "subset": subset} | r.to_dict() | s)
        pd.DataFrame(rows).to_csv("results/final_comparison_test.csv", index=False)
        with open("cache/final_comparison_lists.pkl", "wb") as f:
            pickle.dump({"query_rows": query_rows, "lists": all_lists}, f)
        ours_name = "ours (MiniLM name+desc, final W, flat rule)"
        for base in ("TF-IDF one-hot, final W, flat rule", "MiniLM name, initial W (v1 setup on new data)",
                     "MiniLM name+desc, final W (layered only)"):
            for j, m in enumerate(("R@10", "nDCG@10")):
                d = paired_bootstrap(per_query[ours_name][:, j], per_query[base][:, j])
                print(f"ours − {base}, test {m}: {d[0]:+.4f}  95% CI [{d[1]:+.4f}, {d[2]:+.4f}]")
        sys.exit()

    for min_up in (5, 10):
        truth = load_truth(rec, min_up, "val")
        query_rows = list(truth)
        print(f"\n######## val, up ≥ {min_up}: {len(query_rows)} queries ########")

        def run(name, lists, self_hit=float("nan")):
            table, s = evaluate(rec, truth, lists, self_hit)
            print(f"\n== {name} ==  " + "  ".join(f"{k} {v:.3f}" for k, v in s.items()))
            print(table.to_string())

        run("random", random_lists(len(rec.dataset), query_rows))
        for name, idf, alpha_min in [("IDF only (α_min=1)", IDF_weight, 1.),
                                     ("votes only (IDF=1)", {n: 1. for n in IDF_weight}, .5),
                                     ("IDF × votes (α_min=0.5)", IDF_weight, .5)]:
            rec.IDF_weight = idf
            rec.build_DB(alpha_min)
            run(name, *model_lists(rec, query_rows))
        rec.IDF_weight = IDF_weight
        run("Fragrantica also_liked (reference)", also_liked_lists(rec, query_rows))
