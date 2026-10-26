"""노트 임베딩 향수 추천 (README 4장).
MiniLM name+desc 노트 임베딩, IDF 가중 평균 (회원 투표 가중치 안 씀, α_min = 1), 레이어 가중치 LAYER_W,
flat 향수가 낀 쌍은 노트 전체 집합으로 비교 (FLAT_DELTA)"""
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.sparse import csr_matrix
from functions import parse_notes, merge_notes, get_doc_IDF_weights, get_note_weights
from preprocess import LAYERS, fill_layers

V1_LAYER_W = [[0.8, 0.25, 0.1],  # v1에서 손으로 정한 W. 튜닝 시작점이자 비교 기준
             [0.25, 1., 0.35],
             [0.1, 0.35, 1.2]]
LAYER_W = [[0.3076, 0.0228, 0.0085],  # train에서 튜닝 (README 5.3)
           [0.0228, 0.4044, 0.0209],
           [0.0085, 0.0209, 0.1836]]
FLAT_DELTA = 0.125  # flat 쌍의 합집합 cosine에서 빼는 값, train에서 튜닝 (README 5.4)


class Recommender:
    def __init__(self, dataset, model, device, alpha_min=1., layer_weights=LAYER_W, IDF_weight=None, E=None, delta=FLAT_DELTA):
        """delta: None이면 모든 쌍을 레이어별로 비교한다.
        숫자면 flat 향수가 낀 쌍은 노트 전체 집합(합집합)끼리 비교하고, 점수에서 delta를 뺀다
        (합집합 cosine이 레이어별 점수보다 체계적으로 높아서 같은 척도로 맞추는 값)."""
        self.dataset = dataset.reset_index(drop=True)
        self.device = device
        self.delta = delta
        self.flat = torch.tensor(self.dataset["is_flat"].values.astype(bool), device=device)
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
        """[[layer dict, ...]] -> 레이어별 노트 가중치 희소 행렬 S_l (n × V), 행 합 1"""
        weights = [get_note_weights(p, self.IDF_weight, alpha_min) for p in perfumes]
        out = []
        for l in range(len(weights[0])):
            rows, cols, vals = [], [], []
            for r, w in enumerate(weights):
                for n, x in w[l].items():
                    rows.append(r); cols.append(self.note_idx[n]); vals.append(x)
            out.append(csr_matrix((vals, (rows, cols)), shape=(len(perfumes), len(self.vocab))))
        return out

    def embed(self, perfumes, alpha_min):
        """[[layer dict, ...]] -> (n, 레이어 수·d) 레이어별 정규화된 가중 평균 임베딩"""
        blocks = [torch.from_numpy(S @ self.E).float() for S in self.weight_matrices(perfumes, alpha_min)]  # MPS 희소 연산 대신 CPU
        return torch.cat([nn.functional.normalize(b, dim=1) for b in blocks], dim=1).to(self.device)

    def build_DB(self, alpha_min):
        self.alpha_min = alpha_min
        self.D = self.embed(self.layers, alpha_min)  # (N, 3·d)
        self.U = self.embed([[merge_notes(*p)] for p in self.layers], alpha_min)  # (N, d) 노트 전체 집합

    def query_vectors(self, q):
        """q (B, 3·d) -> Q' (B, 3·d), Q'_j = Σ_i W_ij q̂_i. 점수 = Q' @ Dᵀ = Σ_ij W_ij cos(q_i, d_j)"""
        q = q.view(len(q), 3, -1)
        return torch.einsum("ij,bid->bjd", self.W, q).reshape(len(q), -1)

    def scores(self, q, qu, q_flat):
        """쿼리 레이어 임베딩 q (B, 3·d) -> 점수 (B, N).
        delta가 있으면 쿼리나 후보가 flat인 쌍은 합집합 임베딩 qu (B, d)끼리의 cosine − delta"""
        scores = self.query_vectors(q) @ self.D.T
        if self.delta is not None:
            mixed = q_flat[:, None] | self.flat[None, :]
            scores = torch.where(mixed, qu @ self.U.T - self.delta, scores)
        return scores

    def self_scores(self, q, qu, q_flat):
        """쿼리와 쿼리 자신의 점수 (B,). 보여 줄 점수를 이것으로 나누면 노트가 같은 향수가 1.0 (순위는 그대로)"""
        s = (self.query_vectors(q) * q).sum(1)
        if self.delta is not None:
            s = torch.where(q_flat, (qu * qu).sum(1) - self.delta, s)
        return s

    def row_scores(self, rows):
        """DB 향수(행 번호)를 쿼리로 쓸 때의 점수 (B, N)"""
        return self.scores(self.D[rows], self.U[rows], self.flat[rows])

    def search_rows(self, rows, k):
        """DB 향수 쿼리 -> 자기 자신을 뺀 상위 k개 (점수, 행 번호)"""
        return self.search(self.row_scores(rows), k, exclude_rows=rows)

    def search(self, scores, k, exclude_rows=None):
        """점수 (B, N) -> 상위 k개 (점수, 행 번호)"""
        if exclude_rows is not None:
            scores[torch.arange(len(scores)), exclude_rows] = -torch.inf
        top_scores, top_rows = torch.topk(scores, k)
        return top_scores.cpu(), top_rows.cpu()

    def recommend_ids(self, ids, k=5):
        """DB 향수 id 목록 -> (점수, 추천 id), 자기 자신 제외. 점수는 쿼리 자기 점수로 나눈 값"""
        rows = torch.tensor([self.row_of[i] for i in ids], device=self.device)
        top_scores, top_rows = self.search_rows(rows, k)
        top_scores /= self.self_scores(self.D[rows], self.U[rows], self.flat[rows]).cpu()[:, None]
        return top_scores,torch.from_numpy(self.dataset["id"].values[top_rows.numpy()])

    def recommend_notes(self, top=(), middle=(), base=(), k=5):
        """직접 입력: 투표 가중치가 없으므로 IDF만 쓴다. 빈 레이어는 나머지 합집합 (D2).
        DB에 없는 노트는 임베딩이 없으므로 경고하고 뺀다. 점수는 쿼리 자기 점수로 나눈 값."""
        layers = [{n.strip().lower(): 0 for n in names} for names in (top, middle, base)]
        unknown = sorted({n for layer in layers for n in layer if n not in self.note_idx})
        if unknown:
            warnings.warn(f"DB에 없는 노트라서 뺐다: {unknown}")
            layers = [{n: v for n, v in layer.items() if n in self.note_idx} for layer in layers]
        if not any(layers):
            raise ValueError("DB에 있는 노트가 하나도 없다")
        q_flat = torch.tensor([sum(map(bool, layers)) == 1], device=self.device)  # 한 레이어만 입력하면 flat
        _, layers = fill_layers(*layers, {})
        qu = self.embed([[merge_notes(*layers)]], self.alpha_min)
        q = self.embed([layers], self.alpha_min)
        top_scores, top_rows = self.search(self.scores(q, qu, q_flat), k)
        top_scores /= self.self_scores(q, qu, q_flat).cpu()[:, None]
        return top_scores[0],torch.from_numpy(self.dataset["id"].values[top_rows[0].numpy()])

    def display(self, title, scores, ids, show_notes=False):
        """show_notes면 향수마다 노트도 붙인다 (flat 향수는 세 레이어가 같으므로 한 번만)"""
        print("***" * 10, title, "***" * 10)
        for rank, (s, pid) in enumerate(zip(scores.tolist(), ids.tolist()), 1):
            row = self.row_of[pid]
            r = self.dataset.iloc[row]
            line = f"NO.{rank} {r['name']} @ {r['brand']} (id {pid}) / Similarity Score: {s:.3f}"
            if show_notes:
                layers = self.layers[row]
                if self.flat[row]:
                    line += f" | notes: {', '.join(layers[0])}"
                else:
                    line += " | " + " / ".join(f"{l[0].upper()}: {', '.join(n)}" for l, n in zip(LAYERS, layers))
            print(line)
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
