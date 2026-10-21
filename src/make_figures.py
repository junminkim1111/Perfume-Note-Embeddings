"""README 그림: results/*.csv -> docs/figures/*.png
python src/make_figures.py"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "docs/figures"
BLUE, ORANGE, AQUA, GRAY, LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#8c8b86", "#c9c8c2"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white"})


def style(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def embedding_comparison():
    """5b: 모델 × 입력, 검증 R@10 (up ≥ 5). 점 그래프 (축이 0에서 시작하지 않으므로 막대 대신 점)"""
    df = pd.read_csv("results/embedding_comparison.csv")
    models = {"minilm": "all-MiniLM-L6-v2 (384-d)", "gemma": "EmbeddingGemma-300m (768-d)",
              "bge": "bge-base-en-v1.5 (768-d)", "qwen": "Qwen3-Embedding-0.6B (1024-d)"}
    inputs = {"name": ("note name", GRAY), "desc": ("description", ORANGE), "name+desc": ("name + description", BLUE)}
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for y, (m, label) in enumerate(models.items()):
        vals = [df[(df.model == m) & (df.input == i)]["R@10 up≥5"].item() for i in inputs]
        ax.plot([min(vals), max(vals)], [y, y], color=GRID, linewidth=2, zorder=1)
        for (i, (ilabel, c)), v in zip(inputs.items(), vals):
            ax.scatter(v, y, s=70, color=c, edgecolor="white", linewidth=1.5, zorder=2, label=ilabel if y == 0 else None)
        best = max(vals)
        ax.annotate(f"{best:.3f}", (best, y), xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(models)), models.values())
    ax.invert_yaxis()
    ax.set_xlabel("Recall@10 (validation, up ≥ 5)")
    ax.set_title("Note embedding: model × input text", loc="left", color=INK, fontsize=11, fontweight="bold")
    style(ax)
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.22), fontsize=9)
    fig.savefig(f"{OUT}/embedding_comparison.png")


def layer_weights():
    """W: 시작값 (v2) vs 튜닝 결과, 같은 색 척도"""
    from perfume_recommendation_v2 import INITIAL_W, FINAL_W
    Ws = {"Initial W (v2)": np.array(INITIAL_W) / np.sum(INITIAL_W), "Tuned W (final)": np.array(FINAL_W) / np.sum(FINAL_W)}
    vmax = max(W.max() for W in Ws.values())
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax, (title, W) in zip(axes, Ws.items()):
        ax.imshow(W, cmap="Blues", vmin=0, vmax=vmax)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{W[i, j]:.3f}", ha="center", va="center", fontsize=10,
                        color="white" if W[i, j] > vmax * 0.6 else INK)
        ax.set_xticks(range(3), ["Top", "Middle", "Base"])
        ax.set_yticks(range(3), ["Top", "Middle", "Base"])
        ax.set_xlabel("DB perfume layer j")
        ax.set_title(title, color=INK, fontsize=11, fontweight="bold")
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0].set_ylabel("query layer i")
    fig.savefig(f"{OUT}/layer_weights.png")


METHODS = {  # results/final_comparison_test.csv의 method -> 그림 라벨, 색
    "TF-IDF one-hot, final W": ("Word presence (TF-IDF one-hot)", GRAY),
    "MiniLM name, initial W (v2 on new data)": ("v2 setup (note names, initial W)", LIGHT),
    "MiniLM name+desc, initial W": ("+ note descriptions", LIGHT),
    "ours (MiniLM name+desc, final W)": ("Ours (+ tuned W)", BLUE),
    "Fragrantica also_liked (reference)": ("Fragrantica “also liked” (reference)", ORANGE),
}


def results_comparison():
    df = pd.read_csv("results/final_comparison_test.csv")
    df = df[df.subset == "all"].set_index("method")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    labels = [l for l, _ in METHODS.values()]
    for ax, metric in zip(axes, ["R@10", "nDCG@10"]):
        vals = [df.loc[m, metric] for m in METHODS]
        ax.barh(range(len(vals)), vals, color=[c for _, c in METHODS.values()], height=0.62)
        for y, v in enumerate(vals):
            ax.text(v + 0.002, y, f"{v:.3f}", va="center", fontsize=9, color=INK)
        ax.set_xlim(0, max(vals) * 1.2)
        ax.set_title(metric.replace("R@", "Recall@"), loc="left", color=INK, fontsize=11, fontweight="bold")
        style(ax)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    fig.suptitle("Test split, “Smells like this” votes (up ≥ 5)", x=0.02, ha="left", color=INK2, fontsize=9)
    fig.savefig(f"{OUT}/results_comparison.png")


def flat_vs_pyramid():
    df = pd.read_csv("results/final_comparison_test.csv").set_index(["method", "subset"])
    show = ["TF-IDF one-hot, final W", "ours (MiniLM name+desc, final W)", "Fragrantica also_liked (reference)"]
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    x = np.arange(2)
    w = 0.26
    for k, m in enumerate(show):
        label, c = METHODS[m]
        vals = [df.loc[(m, s), "R@10"] for s in ("pyramid", "flat")]
        ax.bar(x + (k - 1) * w, vals, w * 0.92, color=c, label=label)
        for xi, v in zip(x, vals):
            ax.text(xi + (k - 1) * w, v + 0.002, f"{v:.3f}", ha="center", fontsize=8, color=INK)
    n = [int(df.loc[(show[0], s), "n"]) for s in ("pyramid", "flat")]
    ax.set_xticks(x, [f"pyramid queries (n={n[0]:,})", f"flat queries (n={n[1]:,})"])
    ax.set_ylabel("Recall@10 (test)")
    style(ax, "y")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.savefig(f"{OUT}/flat_vs_pyramid.png")


if __name__ == "__main__":
    import sys
    os.makedirs(OUT, exist_ok=True)
    embedding_comparison()
    layer_weights()
    if os.path.exists("results/final_comparison_test.csv"):
        results_comparison()
        flat_vs_pyramid()
    print("saved to", OUT)
