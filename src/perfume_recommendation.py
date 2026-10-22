"""v2: 새 데이터(data/perfumes_clean.csv) + IDF 가중 평균 + 1152차원 행렬곱 검색 (handoff 5.3, 5.5)
최종 설정 (6단계): MiniLM name+desc 노트 임베딩, IDF만 (회원 투표 가중치 안 씀, α_min = 1), FINAL_W"""
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.sparse import csr_matrix
from functions import parse_notes, get_doc_IDF_weights, get_note_weights
from preprocess import LAYERS, fill_layers

INITIAL_W = [[0.8, 0.25, 0.1],  # v2의 W. 튜닝 시작점이자 비교 기준
             [0.25, 1., 0.35],
             [0.1, 0.35, 1.2]]
FINAL_W = [[0.3076, 0.0228, 0.0085],  # IDF만 설정으로 튜닝 (results/idf_only.json)
           [0.0228, 0.4044, 0.0209],
           [0.0085, 0.0209, 0.1836]]


class Recommender:
    def __init__(self, dataset, model, device, alpha_min=1., layer_weights=FINAL_W, IDF_weight=None, E=None):
        self.dataset = dataset.reset_index(drop=True)
        self.device = device
        self.row_of = {pid: i for i, pid in enumerate(self.dataset["id"])}
        self.layers = [[parse_notes(v) for v in row] for row in zip(*(self.dataset[c] for c in LAYERS))]
        self.IDF_weight = IDF_weight or get_doc_IDF_weights(self.dataset)
        self.vocab = sorted(self.IDF_weight)
        self.note_idx = {n: i for i, n in enumerate(self.vocab)}
        # (V, d), 노트당 1번. E를 주면 vocab(정렬된 노트) 순서여야 한다
        self.E = E if E is not None else model.encode(self.vocab, convert_to_numpy=True)
        self.set_layer_weights(layer_weights)
        self.build_DB(alpha_min)

    def set_layer_weights(self, layer_weights):
        """W만 바꿀 때는 D를 다시 만들 필요가 없다 (W는 쿼리 쪽에만 들어감)."""
        W = torch.tensor(layer_weights, dtype=torch.float32, device=self.device)
        self.W = W / W.sum()

    def weight_matrices(self, perfumes, alpha_min):
        """[[top, middle, base] dict] -> 레이어별 노트 가중치 희소 행렬 S_l (n × V), 행 합 1"""
        weights = [get_note_weights(p, self.IDF_weight, alpha_min) for p in perfumes]
        out = []
        for l in range(3):
            rows, cols, vals = [], [], []
            for r, w in enumerate(weights):
                for n, x in w[l].items():
                    rows.append(r); cols.append(self.note_idx[n]); vals.append(x)
            out.append(csr_matrix((vals, (rows, cols)), shape=(len(perfumes), len(self.vocab))))
        return out

    def embed(self, perfumes, alpha_min):
        """[[top, middle, base] dict] -> (n, 3·d) 레이어별 정규화된 가중 평균 임베딩"""
        blocks = [torch.from_numpy(S @ self.E).float() for S in self.weight_matrices(perfumes, alpha_min)]  # MPS 희소 연산 대신 CPU
        return torch.cat([nn.functional.normalize(b, dim=1) for b in blocks], dim=1).to(self.device)

    def build_DB(self, alpha_min):
        self.alpha_min = alpha_min
        self.D = self.embed(self.layers, alpha_min)  # (N, 3·d)

    def query_vectors(self, q):
        """q (B, 3·d) -> Q' (B, 3·d), Q'_j = Σ_i W_ij q̂_i. 점수 = Q' @ Dᵀ = Σ_ij W_ij cos(q_i, d_j)"""
        q = q.view(len(q), 3, -1)
        return torch.einsum("ij,bid->bjd", self.W, q).reshape(len(q), -1)

    def search(self, q, k, exclude_rows=None):
        scores = self.query_vectors(q) @ self.D.T  # (B, N)
        if exclude_rows is not None:
            scores[torch.arange(len(q)), exclude_rows] = -torch.inf
        top_scores, top_rows = torch.topk(scores, k)
        return top_scores.cpu(), top_rows.cpu()

    def recommend_ids(self, ids, k=5):
        """DB 향수 id 목록 -> (점수, 추천 id), 자기 자신 제외"""
        rows = torch.tensor([self.row_of[i] for i in ids], device=self.device)
        top_scores, top_rows = self.search(self.D[rows], k, exclude_rows=rows)
        return top_scores, torch.from_numpy(self.dataset["id"].values[top_rows.numpy()])

    def recommend_notes(self, top=(), middle=(), base=(), k=5):
        """직접 입력: 투표 가중치가 없으므로 IDF만 쓴다. 빈 레이어는 나머지 합집합 (D2).
        DB에 없는 노트는 임베딩이 없으므로 경고하고 뺀다."""
        layers = [{n.strip().lower(): 0 for n in names} for names in (top, middle, base)]
        unknown = sorted({n for layer in layers for n in layer if n not in self.note_idx})
        if unknown:
            warnings.warn(f"DB에 없는 노트라서 뺐다: {unknown}")
            layers = [{n: v for n, v in layer.items() if n in self.note_idx} for layer in layers]
        if not any(layers):
            raise ValueError("DB에 있는 노트가 하나도 없다")
        _, layers = fill_layers(*layers, {})
        top_scores, top_rows = self.search(self.embed([layers], self.alpha_min), k)
        return top_scores[0], torch.from_numpy(self.dataset["id"].values[top_rows[0].numpy()])

    def display(self, title, scores, ids):
        print("***" * 10, title, "***" * 10)
        for rank, (s, pid) in enumerate(zip(scores.tolist(), ids.tolist()), 1):
            r = self.dataset.iloc[self.row_of[pid]]
            print(f"NO.{rank} {r['name']} @ {r['brand']} (id {pid}) / Similarity Score: {s:.3f}")
        print()


if __name__ == "__main__":
    import random
    from functions import get_E, note_texts

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"USING DEVICE: {device}")
    dataset = pd.read_csv("data/perfumes_clean.csv")
    vocab = sorted(get_doc_IDF_weights(dataset))
    E = get_E("minilm", "name+desc", note_texts(vocab, "name+desc"), device)  # cache/에 없으면 MiniLM으로 인코딩
    recommender = Recommender(dataset, None, device, E=E)

    scores, ids = recommender.recommend_notes(top=["rose"], middle=["sunflower"], base=["pear"])
    recommender.display("Custom Query: rose / sunflower / pear", scores, ids)

    pid = random.choice(dataset["id"].tolist())
    r = recommender.dataset.iloc[recommender.row_of[pid]]
    scores, ids = recommender.recommend_ids([pid])
    recommender.display(f"{r['name']} @ {r['brand']} (id {pid})", scores[0], ids[0])
