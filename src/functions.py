from collections import Counter
import math



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
