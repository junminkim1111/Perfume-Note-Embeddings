"""임베딩 모델 × 입력 비교 (handoff 5.8). α_min = 0.5, 기본 W 고정, 검증 세트.
python src/compare_embeddings.py               # 가능한 입력 전부
python src/compare_embeddings.py name          # 입력 지정
python src/compare_embeddings.py cooc          # 5c: MiniLM name+desc + 동시 출현 벡터, β 비교"""
import gc
import os
import sys
import numpy as np
import pandas as pd
import torch
from perfume_recommendation_v2 import Recommender, INITIAL_W
from evaluate_v2 import load_truth, model_lists, evaluate
from scipy.sparse import csr_matrix
from functions import get_doc_IDF_weights

MODELS = {  # key: (모델, encode 인자)
    "minilm": ("all-MiniLM-L6-v2", {}),
    "bge": ("BAAI/bge-base-en-v1.5", {}),  # 대칭 작업이라 검색용 지시문은 붙이지 않는다
    "gemma": ("google/embeddinggemma-300m", {"prompt_name": "STS"}),
    "qwen": ("Qwen/Qwen3-Embedding-0.6B",
             {"prompt": "Instruct: Given the name of a perfume note, represent how it smells\nQuery: "}),
}
INPUTS = ["name", "desc", "name+desc"]
BASE = ("minilm", "name+desc")  # 5b에서 고른 텍스트 임베딩
BETAS = [0, 0.1, 0.2, 0.3, 0.5, 1]  # β = 0: 텍스트만, β = 1: 동시 출현만 (참고용)
PROBES = ["orange", "orange blossom", "amber", "ambergris", "ambroxan", "iso e super", "hedione", "cashmeran",
          "bergamot", "rose", "oud", "vanilla", "musk", "leather", "sea notes", "coffee", "tonka bean",
          "lavender", "patchouli", "black pepper"]


def note_texts(vocab, input_type):
    """노트별 임베딩 입력. UNKNOWN 노트는 이름으로 대신한다. 설명이 없는 노트가 있으면 None."""
    if input_type == "name":
        return vocab
    path = "data/note_descriptions.csv"
    if not os.path.exists(path):
        return None
    desc = pd.read_csv(path).set_index("note")
    if set(vocab) - set(desc.index):
        return None
    texts = []
    for n in vocab:
        d = desc.loc[n]
        if d["unknown"]:
            texts.append(n)
        else:
            texts.append(d["description"] if input_type == "desc" else f"{n}: {d['description']}")
    return texts


def get_E(key, input_type, texts, device):
    def encode():
        from sentence_transformers import SentenceTransformer
        name, kwargs = MODELS[key]
        model = SentenceTransformer(name, device=str(device))
        E = model.encode(texts, convert_to_numpy=True, batch_size=64, **kwargs)
        del model; gc.collect(); torch.mps.empty_cache() if device.type == "mps" else None
        return E
    return cached(f"cache/emb_{key}_{input_type}.pt", encode)


def cooc_embedding(layers, vocab, dim=128):
    """향수 = 노트 집합(세 레이어 합집합). 동시 출현 -> PPMI -> SVD 128차원 (handoff 5.8 C)"""
    idx = {n: i for i, n in enumerate(vocab)}
    rows, cols = [], []
    for r, p in enumerate(layers):
        for n in set().union(*p):
            rows.append(r); cols.append(idx[n])
    X = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(layers), len(vocab)))
    C = (X.T @ X).toarray()
    np.fill_diagonal(C, 0)
    p = C.sum(1) / C.sum()
    with np.errstate(divide="ignore"):
        pmi = np.log(C / C.sum() / np.outer(p, p))
    ppmi = np.where(C > 0, np.maximum(pmi, 0), 0)
    U, S, _ = np.linalg.svd(ppmi)
    return (U[:, :dim] * np.sqrt(S[:dim])).astype(np.float32)


def unit(E):
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)


def combine(E_text, E_cooc, beta):
    """내적 = (1−β)·cos_text + β·cos_cooc 이 되도록 정규화 후 가중해서 이어 붙인다"""
    return np.hstack([np.sqrt(1 - beta) * unit(E_text), np.sqrt(beta) * unit(E_cooc)]).astype(np.float32)


def cached(path, make):
    if os.path.exists(path):
        return torch.load(path).numpy()
    E = make()
    os.makedirs("cache", exist_ok=True)
    torch.save(torch.from_numpy(E), path)
    return E


def eval_row(rec, E, truths):
    """E로 D를 다시 만들고 검증 세트 지표 한 줄"""
    rec.E = E
    rec.build_DB(.5)
    row = {}
    for min_up, truth in truths.items():
        lists, self_hit = model_lists(rec, list(truth))
        table, sanity_metrics = evaluate(rec, truth, lists, self_hit)
        tag = f"up≥{min_up}"
        row |= {f"R@10 {tag}": table.loc["all", "R@10"], f"nDCG@10 {tag}": table.loc["all", "nDCG@10"],
                f"MRR {tag}": table.loc["all", "MRR"]}
        if min_up == 10:
            row |= {"R@10 flat": table.loc["flat", "R@10"], "R@10 pyramid": table.loc["pyramid", "R@10"],
                    "self@1": sanity_metrics["self@1"]}
    rec.D = None; gc.collect()
    print("  " + "  ".join(f"{k} {v:.4f}" for k, v in row.items()))
    return row


def nearest_notes(vocab, E, k=5):
    """사전 점검: 확인용 노트마다 가장 가까운 노트 k개"""
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    idx = {n: i for i, n in enumerate(vocab)}
    for p in PROBES:
        sims = En @ En[idx[p]]
        sims[idx[p]] = -np.inf
        print(f"  {p:15s} -> " + ", ".join(vocab[i] for i in np.argsort(-sims)[:k]))


def save_table(results, path):
    table = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    table.to_csv(path, index=False)
    print("\n" + table.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    dataset = pd.read_csv("data/perfumes_clean.csv")
    IDF_weight = get_doc_IDF_weights(dataset)
    vocab = sorted(IDF_weight)  # Recommender.vocab과 같은 순서
    rec, results = None, []

    def recommender(E):
        r = Recommender(dataset, None, device, alpha_min=.5, layer_weights=INITIAL_W, IDF_weight=IDF_weight, E=E)
        return r, {m: load_truth(r, m, "val") for m in (5, 10)}

    if sys.argv[1:] == ["cooc"]:
        key, input_type = BASE
        E_text = get_E(key, input_type, note_texts(vocab, input_type), device)
        rec, truths = recommender(E_text)
        E_cooc = cached("cache/emb_cooc_ppmi_svd128.pt", lambda: cooc_embedding(rec.layers, vocab))
        for beta in BETAS:
            E = cached(f"cache/emb_{key}_{input_type}+cooc_b{beta}.pt", lambda: combine(E_text, E_cooc, beta))
            print(f"\n=== {key} / {input_type} + cooc, β = {beta} (dim {E.shape[1]}) ===")
            nearest_notes(vocab, E)
            results.append({"beta": beta} | eval_row(rec, E, truths))
        save_table(results, "results/cooc_comparison.csv")
        sys.exit()

    for input_type in sys.argv[1:] or INPUTS:
        texts = note_texts(vocab, input_type)
        if texts is None:
            print(f"skip {input_type}: data/note_descriptions.csv가 없거나 빠진 노트가 있다")
            continue
        for key in MODELS:
            try:
                E = get_E(key, input_type, texts, device)
            except OSError as e:  # Hugging Face 접근 권한 없음 (gemma)
                print(f"skip {key}: {str(e).splitlines()[0]}")
                continue
            print(f"\n=== {key} / {input_type} (dim {E.shape[1]}) ===")
            nearest_notes(vocab, E)
            if rec is None:
                rec, truths = recommender(E)
            results.append({"model": key, "input": input_type} | eval_row(rec, E, truths))
    save_table(results, "results/embedding_comparison.csv")
