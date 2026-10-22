from collections import Counter
import gc
import math
import os



def get_IDF_weights(dataset, data_size = None, note_types = ['Top', 'Middle', 'Base']):
    all_notes = []

    if data_size is None:
        data_size = len(dataset)

    for note_type in note_types:
        notes = dataset[note_type].str.split(',').explode().str.strip().tolist()
        all_notes.extend(notes)

    note_counts = Counter(all_notes) # type: dict

    IDF_weight = {
        note : math.log(data_size / cnt) + 1
        for note, cnt in note_counts.items()
    }
    return IDF_weight

def parse_notes(s):
    """'Bergamot:49|Lemon:5' -> {'bergamot': 49, 'lemon': 5}. 빈 값이면 {}."""
    notes = {}
    if not isinstance(s, str):
        return notes
    for item in s.split('|'):
        name, _, weight = item.rpartition(':')
        name = name.strip().lower()
        if name:
            notes[name] = max(notes.get(name, 0), int(weight))
    return notes


def format_notes(notes):
    return '|'.join(f'{n}:{w}' for n, w in notes.items())


def merge_notes(*layers):
    """레이어 합집합. 같은 노트는 큰 가중치."""
    merged = {}
    for layer in layers:
        for n, w in layer.items():
            merged[n] = max(merged.get(n, 0), w)
    return merged


def get_doc_IDF_weights(dataset, layers=('top', 'middle', 'base')):
    """IDF = log(N / df) + 1, df = 노트를 (어느 레이어에든) 가진 향수 수 (handoff 5.2)"""
    df = Counter()
    for row in zip(*(dataset[c] for c in layers)):
        df.update(set().union(*map(parse_notes, row)))
    N = len(dataset)
    return {note: math.log(N / cnt) + 1 for note, cnt in df.items()}


def get_note_weights(layers, IDF_weight, alpha_min):
    """한 향수의 레이어들 [{note: 투표 가중치}] -> 레이어별 {note: w}, 레이어 안 합 1 (handoff 5.3)
    투표 가중치가 향수 전체에서 전부 0이면 α = 1 (IDF만)."""
    voted = any(v for layer in layers for v in layer.values())
    alpha = alpha_min if voted else 1.
    out = []
    for layer in layers:
        w = {n: IDF_weight[n] * (alpha + (1 - alpha) * v / 100) for n, v in layer.items()}
        if not sum(w.values()):  # α_min = 0이고 이 레이어만 투표가 전부 0이면 IDF로 대신
            w = {n: IDF_weight[n] for n in layer}
        total = sum(w.values())
        out.append({n: x / total for n, x in w.items()})
    return out


# 노트 embedding (handoff 5.8). key: (모델, encode 인자)
MODELS = {
    "minilm": ("all-MiniLM-L6-v2", {}),
    "bge": ("BAAI/bge-base-en-v1.5", {}),  # 대칭 작업이라 검색용 지시문은 붙이지 않는다
    "gemma": ("google/embeddinggemma-300m", {"prompt_name": "STS"}),
    "qwen": ("Qwen/Qwen3-Embedding-0.6B",
             {"prompt": "Instruct: Given the name of a perfume note, represent how it smells\nQuery: "}),
}


def note_texts(vocab, input_type):
    """노트별 임베딩 입력. UNKNOWN 노트는 이름으로 대신한다. 설명이 없는 노트가 있으면 None."""
    if input_type == "name":
        return vocab
    path = "data/note_descriptions.csv"
    if not os.path.exists(path):
        return None
    import pandas as pd
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


def cached(path, make):
    """path에 있으면 읽고, 없으면 make()로 만들어 저장 (numpy 배열)"""
    import torch
    if os.path.exists(path):
        return torch.load(path).numpy()
    E = make()
    os.makedirs("cache", exist_ok=True)
    torch.save(torch.from_numpy(E), path)
    return E


def get_E(key, input_type, texts, device):
    """노트 embedding E (V × d). cache/emb_{key}_{input_type}.pt에 저장해 두고 다시 쓴다"""
    import torch
    def encode():
        from sentence_transformers import SentenceTransformer
        name, kwargs = MODELS[key]
        model = SentenceTransformer(name, device=str(device))
        E = model.encode(texts, convert_to_numpy=True, batch_size=64, **kwargs)
        del model; gc.collect(); torch.mps.empty_cache() if device.type == "mps" else None
        return E
    return cached(f"cache/emb_{key}_{input_type}.pt", encode)
