from functions import parse_notes
from prepare_data import fill_layers

assert parse_notes("Bergamot:49| Lemon :5|bergamot:60") == {"bergamot": 60, "lemon": 5}
assert parse_notes(float("nan")) == {}
assert parse_notes("Note: Odd:3") == {"note: odd": 3}  # 마지막 ':' 기준

a, b = {"a": 10}, {"b": 0}
assert fill_layers({}, {}, {}, a) == (True, [a, a, a])
assert fill_layers(a, {}, b, {}) == (False, [a, {"a": 10, "b": 0}, b])
assert fill_layers({}, {}, {}, {}) is None
print("ok")

import math
import pandas as pd
from functions import get_doc_IDF_weights

# flat 복사(첫 행)와 레이어 중복(둘째 행)이 향수당 1번만 세어지는지
toy = pd.DataFrame({"top": ["a:1", "a:5|b:0"], "middle": ["a:1", "a:0"], "base": ["a:1", "c:2"]})
idf = get_doc_IDF_weights(toy)
assert idf == {"a": 1.0, "b": math.log(2) + 1, "c": math.log(2) + 1}, idf
print("idf ok")

from functions import get_note_weights

idf = {"a": 1., "b": 3.}
idf_only = [{"a": .25, "b": .75}]
assert get_note_weights([{"a": 100, "b": 0}], idf, 1.) == idf_only      # α_min = 1 -> IDF만
assert get_note_weights([{"a": 0, "b": 0}], idf, .5) == idf_only        # 투표 전부 0 -> α = 1
w = get_note_weights([{"a": 100, "b": 0}], idf, .5)[0]  # a = 1·1, b = 3·0.5
assert math.isclose(w["a"], .4) and math.isclose(w["b"], .6), w
assert get_note_weights([{"a": 0, "b": 0}, {"a": 100}], idf, 0.)[0] == idf_only[0]  # 레이어 합 0 대비
print("weights ok")

import numpy as np
import torch
from torch import nn
from perfume_recommendation_v2 import Recommender


class FakeModel:
    def encode(self, notes, **kw):
        return np.random.default_rng(0).standard_normal((len(notes), 8)).astype(np.float32)


toy = pd.DataFrame({"id": [10, 20, 30, 40],
                    "top": ["a:100|b:20", "c:0", "a:0|d:0", "b:50"],
                    "middle": ["b:10", "c:0|d:0", "d:0", "e:100|a:3"],
                    "base": ["e:100", "c:0", "a:0|e:0", "c:40"]})
rec = Recommender(toy, FakeModel(), torch.device("cpu"), alpha_min=.5)

# 레이어 임베딩 = 정규화(Σ w·E), w는 get_note_weights 결과
E = torch.from_numpy(rec.E)
ref_db = torch.stack([torch.stack([nn.functional.normalize(sum(x * E[rec.note_idx[n]] for n, x in w.items()), dim=0)
                                   for w in get_note_weights(p, rec.IDF_weight, .5)]) for p in rec.layers])
assert torch.allclose(rec.D.view(4, 3, -1), ref_db, atol=1e-6)

# 행렬곱 점수 == v2 einsum 점수 (Σ_ij W_ij cos(q_i, d_j))
v2 = torch.einsum("id,bjd->bij", ref_db[0], ref_db) * rec.W
scores, _ = rec.search(rec.D[:1], k=4)
assert torch.allclose(scores[0], v2.sum(dim=(1, 2)).sort(descending=True).values, atol=1e-6)

# 자기 자신 제외, id로 반환
_, ids = rec.recommend_ids([10, 30], k=3)
assert 10 not in ids[0] and 30 not in ids[1] and set(ids[0].tolist()) == {20, 30, 40}
print("v2 ok")

from evaluate_v2 import score_query, get_split

m = score_query([7, 3, 9, 5, 1, 2], {3: 4, 1: 2, 8: 1})  # 정답 3은 2위, 1은 5위, 8은 목록에 없음
assert m["R@5"] == 2 / 3 and m["MRR"] == .5
idcg = 4 + 2 / math.log2(3) + 1 / 2
assert math.isclose(m["nDCG@10"], (4 / math.log2(3) + 2 / math.log2(6)) / idcg)
assert score_query([1, 2], {1: 5})["nDCG@10"] == 1.
s = get_split(range(1000))
assert sum(v == "train" for v in s.values()) == 700 and get_split(range(1000)) == s
print("eval ok")

from compare_embeddings import combine, cooc_embedding, unit

rng = np.random.default_rng(1)
Et, Ec = rng.standard_normal((5, 8)), rng.standard_normal((5, 3))
for beta in (0, .3, 1):
    E = combine(Et, Ec, beta)
    assert np.allclose(E @ E.T, (1 - beta) * unit(Et) @ unit(Et).T + beta * unit(Ec) @ unit(Ec).T, atol=1e-5)
# 대체 노트(a, b)는 같은 향수에 안 나와도 같은 이웃(c, d)과 쓰이면 가깝다
toy = [[{"a": 0, "c": 0}, {"d": 0}, {}], [{"b": 0, "c": 0}, {"d": 0}, {}], [{"e": 0, "f": 0}, {}, {}]]
Ec = unit(cooc_embedding(toy, ["a", "b", "c", "d", "e", "f"], dim=3))
assert Ec[0] @ Ec[1] > .9 and abs(Ec[0] @ Ec[4]) < .1
print("cooc ok")

from optimize_v2 import recall_at, params_to_W, W_to_params

truth = {0: {1: 3, 3: 1}, 2: {0: 5}}
rows = torch.tensor(list(truth))
lists = rec.search(rec.D[rows], 2, exclude_rows=rows)[1].tolist()
assert math.isclose(recall_at(rec, truth, k=2), np.mean([score_query(l, truth[q])["R@5"] for q, l in zip(truth, lists)]))
W0 = [[.8, .25, .1], [.25, 1., .35], [.1, .35, 1.2]]
assert np.allclose(params_to_W(W_to_params(W0)), W0)
print("optimize ok")

import warnings
from perfume_recommendation_v2 import FINAL_W

assert torch.allclose(Recommender(rec.dataset, FakeModel(), torch.device("cpu")).W, torch.tensor(FINAL_W) / sum(map(sum, FINAL_W)))
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    _, ids = rec.recommend_notes(top=["a", "zzz"], k=2)  # zzz는 DB에 없다
assert "zzz" in str(caught[0].message) and len(ids) == 2
try:
    rec.recommend_notes(top=["zzz"])
    assert False
except ValueError:
    pass
print("final ok")

import evaluate_v2
from evaluate_v2 import onehot_lists, jaccard_lists

evaluate_v2.K = 3  # 장난감 DB는 4개뿐
eye = Recommender(rec.dataset, None, torch.device("cpu"), E=np.eye(len(rec.vocab), dtype=np.float32))
assert onehot_lists(eye, FINAL_W, [0, 1, 2]) == evaluate_v2.fair_model_lists(eye, [0, 1, 2])
# 공정한 동점 처리: 같은 점수는 행 순서가 아니라 고정 무작위 순서
assert evaluate_v2.top_rows(np.array([[1., .5, .5, .5]]), [0]) == evaluate_v2.top_rows(np.array([[1., .5, .5, .5]]), [0])
# 노트 집합 {a,b,e} vs {b,c,e,a}: 교집합 3, 합집합 4 -> 가장 가깝다
assert jaccard_lists(eye, [0])[0][0] == 3
evaluate_v2.K = 100
print("baselines ok")
