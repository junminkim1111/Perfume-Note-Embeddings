<p align="right">
  <a href="./README.md">🇺🇸 English</a> |
  <a href="./README.ko.md">🇰🇷 한국어</a>
</p>

<h1 align="center">Perfume Recommendation with Note Embeddings</h1>

<p align="center">A model that measures how similar two perfumes are by turning their Top / Middle / Base notes into sentence embeddings. On a test split built from Fragrantica's "Smells like this" votes, the embedding model outperforms a model that only checks which note words are present</p>

<p align="center"><b><a href="https://junminkim1111.github.io/Perfume-Note-Embeddings/">Project page</a></b>: method, datasets, tuning, results, examples and limitations</p>


## Contents
1. [Installation & usage](#1-installation--usage)
2. [Project structure & reproducing the experiments](#2-project-structure--reproducing-the-experiments)
3. [License & acknowledgements](#3-license--acknowledgements)

---

## 1. Installation & usage

### Clone the repository

Package versions are pinned in `requirements.txt`. (Tested on an Apple-silicon Mac (MPS) with Python 3.13.)

```bash
git clone https://github.com/junminkim1111/perfume-note-embeddings.git
cd perfume-note-embeddings
pip install -r requirements.txt
```

### Prepare the data
Needs a Kaggle API token (kaggle.com → Settings → API).

```bash
pip install kaggle
kaggle datasets download ledecanteur/fragrantica-perfumes --unzip -p data/raw
python src/preprocess.py
```

`preprocess.py` writes `data/perfumes_clean.csv`, `data/smells_like.csv` and `data/also_liked.csv`.

### Run the demo
Shows recommendations for a custom note query and for a random perfume from the database.

```bash
python src/perfume_recommendation.py
```

The first run encodes the 2,589 notes and caches them in `cache/`.

### Use from Python

```python
import sys; sys.path.insert(0, "src")
import pandas as pd, torch
from perfume_recommendation import Recommender
from functions import get_E, note_texts, get_doc_IDF_weights

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
dataset = pd.read_csv("data/perfumes_clean.csv")
vocab = sorted(get_doc_IDF_weights(dataset))
E = get_E("minilm", "name+desc", note_texts(vocab, "name+desc"), device)   # note embeddings (cached)
rec = Recommender(dataset, None, device, E=E)                               # final settings are the defaults

# 1) your own notes (an empty layer is filled with the other layers' notes)
scores, ids = rec.recommend_notes(top=["bergamot", "lemon"], middle=["lavender"], base=["tonka bean", "musk"], k=5)
rec.display("custom query", scores, ids)

# 2) perfumes similar to one in the database (Fragrantica id; 9828 = Creed Aventus)
scores, ids = rec.recommend_ids([9828], k=5)
rec.display("Aventus", scores[0], ids[0])
```

Notes that do not occur in the database are dropped with a warning (there is no embedding for them).

---

## 2. Project structure & reproducing the experiments

```
src/
  perfume_recommendation.py      final recommender (Recommender class, demo)
  preprocess.py                  raw Kaggle dump → data/*.csv
  functions.py                   note parsing, per-perfume IDF, note weights, note embedding loader (get_E)
  optimize.py                    grid search + Nelder-Mead; `idf_only`, `flat`, `flat_w`, `flat_onehot` modes
  evaluate.py                    metrics, splits, baselines; `baselines` mode = final results
  v1(previous version)/          previous version: recommender, evaluation, W tuning, notebook (old dataset)
data/
  note_descriptions.csv          LLM note descriptions (committed)
  fra_cleaned.csv                old dataset used by v1
  raw/                           Kaggle download (not committed)
docs/                            project page (index.html, ko.html), architecture figure, figures
```

Re-running everything, in order:

```bash
python src/preprocess.py                 # seconds
python src/evaluate.py                   # baselines, ~1 min
python src/optimize.py                   # note weights with member votes + W, ~15 min
python src/optimize.py idf_only          # IDF only (final W), ~12 min
python src/optimize.py flat              # δ for flat perfumes, ~4 min
python src/optimize.py flat_w            # hybrid vs whole note set, W re-tune, ~15 min
python src/optimize.py flat_onehot       # δ for the word-presence baseline, ~5 min
python src/evaluate.py baselines         # final results, ~2 min
```

`cache/` holds note embeddings; delete `cache/emb_*desc*.pt` after editing `data/note_descriptions.csv`.

### Previous version
- **v1** (`src/v1(previous version)/`) used the 2024 [Fragrantica dataset](https://www.kaggle.com/datasets/olgagmiufana1/fragrantica-com-fragrance-dataset) (24,063 perfumes) with note names only, a hand-set W, and self-retrieval Hit@1 as the metric. This README describes **v2**.

---

## 3. License & acknowledgements

- Code: [MIT License](LICENSE).
- Data: [Fragrantica Perfumes: Ratings, Notes, Votes & More](https://www.kaggle.com/datasets/ledecanteur/fragrantica-perfumes), licensed **CC BY-NC-SA 4.0** (non-commercial, attribution, share-alike). The derived file `data/note_descriptions.csv` is shared under the same license. All perfume data originates from [Fragrantica](https://www.fragrantica.com).
- Embedding models: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m), [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5), [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
