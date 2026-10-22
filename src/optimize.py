"""하이퍼파라미터 튜닝 (README 5장). 목적함수: 학습 세트 Recall@10, 정답 up ≥ 5.
1) (α_min, γ) 격자 (기본 W)  2) W Nelder-Mead (학습 표본 10k)  3) 튜닝한 W로 격자 재확인
4) 검증 세트로 최종 선택  5) 테스트 세트 1회 측정
γ: IDF 지수. IDF^γ를 IDF_weight로 넘긴다 (γ = 0이면 IDF 끔).
python src/optimize.py idf_only   # IDF만 (α_min 1, γ 1)에 W를 따로 튜닝해서 튜닝된 v2(투표 사용)와 비교
python src/optimize.py flat       # flat 향수가 낀 쌍은 합집합으로 비교, delta를 train에서 튜닝하고 validation에서 비교
python src/optimize.py flat_w     # 위 혼합 방식에서 W를 다시 튜닝 + 합집합만 쓰는 방식과 부트스트랩 비교 (flat 먼저 실행)
python src/optimize.py flat_onehot  # 단어 유무 기준선(원-핫)에도 같은 flat 규칙을 주고 그 delta를 train에서 튜닝"""
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from perfume_recommendation import Recommender, V1_LAYER_W
from evaluate import load_truth, model_lists, fair_model_lists, evaluate, random_lists, also_liked_lists, score_query, paired_bootstrap, onehot_lists
from functions import get_E, note_texts
from functions import get_doc_IDF_weights

MIN_UP = 5
ALPHAS = [0, 0.25, 0.5, 0.75, 1]
GAMMAS = [0, 0.5, 1]
N_TUNE = 10000  # W 탐색용 학습 쿼리 표본 (평가 1회 ~16초 -> ~5초)


def recall_at(rec, truth, k=10, batch=512):
    """평균 Recall@k만 빠르게 (score_query의 R@k와 같다)"""
    q = list(truth)
    hits = []
    for i in range(0, len(q), batch):
        rows = torch.tensor(q[i:i + batch], device=rec.device)
        _, top = rec.search_rows(rows, k)
        hits += [len(truth[r].keys() & set(lst)) / len(truth[r]) for r, lst in zip(q[i:i + batch], top.tolist())]
    return float(np.mean(hits))


def params_to_W(params):
    """6개 파라미터 -> 3x3 대칭 행렬, softplus로 양수 (v2 optimize_weights.py와 같은 방식). 정규화는 Recommender가 한다."""
    p = np.log1p(np.exp(params))
    return [[p[0], p[1], p[2]], [p[1], p[3], p[4]], [p[2], p[4], p[5]]]


def W_to_params(W):
    W = np.asarray(W, dtype=float)
    v = np.array([W[0, 0], W[0, 1], W[0, 2], W[1, 1], W[1, 2], W[2, 2]])
    return np.log(np.expm1(v))


def set_weights(rec, IDF_weight, alpha_min, gamma):
    rec.IDF_weight = {n: v ** gamma for n, v in IDF_weight.items()}
    rec.build_DB(alpha_min)


def grid(rec, IDF_weight, truth, W, log):
    rec.set_layer_weights(W)
    for alpha_min in ALPHAS:
        for gamma in GAMMAS:
            set_weights(rec, IDF_weight, alpha_min, gamma)
            r = recall_at(rec, truth)
            log.append({"stage": "grid", "alpha_min": alpha_min, "gamma": gamma, "W": W, "train R@10": r})
            print(f"  α_min {alpha_min:<4} γ {gamma:<3} train R@10 {r:.4f}", flush=True)
    best = max((l for l in log if l["stage"] == "grid" and l["W"] == W), key=lambda l: l["train R@10"])
    return best["alpha_min"], best["gamma"]


def tune_W(rec, truth, W0, log):
    def objective(params):
        rec.set_layer_weights(params_to_W(params))
        r = recall_at(rec, truth)
        log.append({"stage": "nelder-mead", "W": np.round(params_to_W(params), 4).tolist(), "tune R@10": r})
        if len(log) % 10 == 0:
            print(f"  eval {len(log)}: tune R@10 {r:.4f}", flush=True)
        return -r
    res = minimize(objective, W_to_params(W0), method="Nelder-Mead",
                   options={"maxiter": 200, "xatol": 0.01, "fatol": 0.0005, "adaptive": True})
    W = params_to_W(res.x)
    W = (np.array(W) / np.sum(W)).round(4).tolist()
    return W


def delta_grid(rec, truth, deltas):
    """flat 쌍의 delta grid (train Recall@10) -> (기록, 최적 delta)"""
    log = []
    for delta in deltas:
        rec.delta = delta
        log.append({"delta": delta, "train R@10": recall_at(rec, truth)})
        print(f"  delta {delta:<5} train R@10 {log[-1]['train R@10']:.4f}", flush=True)
    return log, max(log, key=lambda l: l["train R@10"])["delta"]


def val_compare(rec, truth, configs):
    """configs: 이름 -> (delta, W, 모든 쌍을 합집합으로?). validation 표와 순위 목록 (공정한 동점 처리)"""
    flat_all, out = rec.flat.clone(), {}
    for name, (delta, W, union_only) in configs.items():
        rec.delta = delta
        rec.set_layer_weights(W)
        rec.flat = torch.ones_like(flat_all) if union_only else flat_all  # 모두 flat로 보면 모든 쌍이 합집합
        lists = fair_model_lists(rec, list(truth))
        rec.flat = flat_all
        table, _ = evaluate(rec, truth, lists)
        out[name] = (table, lists)
        print(f"\n== {name} ==\n" + table[["R@10", "nDCG@10", "MRR", "n"]].to_string(float_format="%.4f"), flush=True)
    return out


def save(obj, path):
    os.makedirs("results", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)  # results/는 레포에 없다 (.gitignore)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    dataset = pd.read_csv("data/perfumes_clean.csv")
    IDF_weight = get_doc_IDF_weights(dataset)
    vocab = sorted(IDF_weight)
    E = get_E("minilm", "name+desc", note_texts(vocab, "name+desc"), device)  # 5b/5c 최종 임베딩
    rec = Recommender(dataset, None, device, alpha_min=.5, IDF_weight=IDF_weight, E=E, delta=None)
    truth = {s: load_truth(rec, MIN_UP, s) for s in ("train", "val", "test")}
    rng = np.random.default_rng(0)
    tune_rows = rng.choice(list(truth["train"]), N_TUNE, replace=False)
    tune_truth = {r: truth["train"][r] for r in tune_rows}
    print({s: len(t) for s, t in truth.items()}, "tune", len(tune_truth))
    W0 = (np.array(V1_LAYER_W) / np.sum(V1_LAYER_W)).round(4).tolist()
    log = []

    if sys.argv[1:] == ["flat"]:
        # 최종 설정 그대로 (IDF만, LAYER_W). 바뀌는 것은 flat 향수가 낀 쌍의 점수뿐
        from perfume_recommendation import LAYER_W
        rec.IDF_weight = IDF_weight
        rec.build_DB(1.)
        rec.set_layer_weights(LAYER_W)
        print("\n[1] delta grid on train (Recall@10)", flush=True)
        grid_log, best_delta = delta_grid(rec, truth["train"], [0, .025, .05, .075, .1, .125, .15, .2, .25, .3])
        print(f"  best delta {best_delta}")

        print("\n[2] validation, fair tie-breaking", flush=True)
        configs = {"layered (current)": (None, LAYER_W, False), "union only": (0., LAYER_W, True),
                   "hybrid, delta 0": (0., LAYER_W, False), f"hybrid, delta {best_delta}": (best_delta, LAYER_W, False)}
        res = val_compare(rec, truth["val"], configs)
        out = {"delta grid": grid_log, "best delta": best_delta} | {n: t.round(4).to_dict(orient="index") for n, (t, _) in res.items()}
        save(out, "results/flat.json")
        sys.exit()

    if sys.argv[1:] == ["flat_onehot"]:
        from perfume_recommendation import LAYER_W
        rec.IDF_weight = IDF_weight
        rec.build_DB(1.)
        q = list(tune_truth)
        grid_log = []
        print("\n[1] delta grid for the one-hot baseline (train sample, Recall@10)", flush=True)
        for delta in [None, 0, .05, .1, .15, .2, .25, .3, .4, .5]:
            lists = onehot_lists(rec, LAYER_W, q, delta=delta)
            r = float(np.mean([score_query(l, tune_truth[x])["R@10"] for x, l in zip(q, lists)]))
            grid_log.append({"delta": delta, "train R@10": r})
            print(f"  delta {str(delta):<5} train R@10 {r:.4f}", flush=True)
        best = max((l for l in grid_log if l["delta"] is not None), key=lambda l: l["train R@10"])["delta"]
        print(f"  best delta {best}")
        save({"delta grid": grid_log, "best delta": best}, "results/flat_onehot.json")
        sys.exit()

    if sys.argv[1:] == ["flat_w"]:
        from perfume_recommendation import LAYER_W
        delta0 = json.load(open("results/flat.json"))["best delta"]
        rec.IDF_weight = IDF_weight
        rec.build_DB(1.)
        print(f"\n[1] Nelder-Mead on W, hybrid (delta {delta0})", flush=True)
        rec.delta = delta0
        W_hybrid = tune_W(rec, tune_truth, LAYER_W, log)
        print(f"  W {W_hybrid}  ({len(log)} evals)")

        print("\n[2] delta grid on train with the new W", flush=True)
        rec.set_layer_weights(W_hybrid)
        grid_log, delta1 = delta_grid(rec, truth["train"], [.075, .1, .125, .15, .175, .2])

        print("\n[3] validation, fair tie-breaking", flush=True)
        configs = {"union only": (0., LAYER_W, True), f"hybrid, LAYER_W, delta {delta0}": (delta0, LAYER_W, False),
                   f"hybrid, re-tuned W, delta {delta1}": (delta1, W_hybrid, False)}
        res = val_compare(rec, truth["val"], configs)
        out = {"W_hybrid": W_hybrid, "W evals": len(log), "delta grid": grid_log, "delta": delta1}
        out |= {n: t.round(4).to_dict(orient="index") for n, (t, _) in res.items()}
        q = list(truth["val"])
        per_query = {n: np.array([[score_query(l, truth["val"][x])[m] for m in ("R@10", "nDCG@10")] for x, l in zip(q, lists)])
                     for n, (_, lists) in res.items()}
        names = list(configs)
        print("\npaired bootstrap on validation (a − b, 95% CI)")
        for a, b in [(names[1], names[0]), (names[2], names[0]), (names[2], names[1])]:
            for j, m in enumerate(("R@10", "nDCG@10")):
                d = paired_bootstrap(per_query[a][:, j], per_query[b][:, j])
                out[f"{a} − {b} {m}"] = d
                print(f"  {a} − {b}  {m}: {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]")
        save(out, "results/flat_w.json")
        sys.exit()

    if sys.argv[1:] == ["idf_only"]:
        tuned = json.load(open("results/tuned_params.json"))
        print("\n[1] Nelder-Mead on W, IDF only (α_min 1, γ 1)", flush=True)
        set_weights(rec, IDF_weight, 1, 1)
        W_idf = tune_W(rec, tune_truth, W0, log)
        print(f"  W {W_idf}  ({len(log)} evals)")
        configs = {"IDF only, default W": (1, 1, W0), "IDF only, tuned W": (1, 1, W_idf),
                   "v2 tuned (votes)": (tuned["alpha_min"], tuned["gamma"], tuned["W"])}
        out, per_query, rows = {}, {}, []
        for name, (a, g, w) in configs.items():
            set_weights(rec, IDF_weight, a, g)
            rec.set_layer_weights(w)
            out[name] = {"alpha_min": a, "gamma": g, "W": w, "train R@10": recall_at(rec, truth["train"]),
                         "val R@10": recall_at(rec, truth["val"])}
            lists, self_hit = model_lists(rec, list(truth["test"]))
            table, sanity_metrics = evaluate(rec, truth["test"], lists, self_hit)
            per_query[name] = np.array([[score_query(l, truth["test"][q])[m] for m in ("R@10", "nDCG@10")]
                                        for q, l in zip(truth["test"], lists)])
            print(f"\n== {name} ==  train R@10 {out[name]['train R@10']:.4f}  val R@10 {out[name]['val R@10']:.4f}  "
                  + "  ".join(f"{k} {v:.3f}" for k, v in sanity_metrics.items()))
            print(table.to_string(float_format="%.4f"))
            for subset, r in table.iterrows():
                rows.append({"method": name, "subset": subset} | r.to_dict() | sanity_metrics)
        # 테스트 세트 쌍 부트스트랩: v2 tuned (votes) − IDF only (tuned W)
        diff = per_query["v2 tuned (votes)"] - per_query["IDF only, tuned W"]
        idx = rng.integers(0, len(diff), (2000, len(diff)))
        for j, m in enumerate(("R@10", "nDCG@10")):
            bs = diff[:, j][idx].mean(1)
            out[f"test {m} diff (v2 tuned (votes) − IDF only tuned W)"] = [float(diff[:, j].mean()),
                                                                   float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            print(f"v2 tuned (votes) − IDF only (tuned W) test {m}: {diff[:, j].mean():+.4f}  "
                  f"95% CI [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}]")
        save(out, "results/idf_only.json")
        pd.DataFrame(rows).to_csv("results/idf_only_test.csv", index=False)
        sys.exit()

    print("\n[1] grid, default W", flush=True)
    alpha_min, gamma = grid(rec, IDF_weight, truth["train"], W0, log)
    print(f"  best α_min {alpha_min}, γ {gamma}")

    print("\n[2] Nelder-Mead on W", flush=True)
    set_weights(rec, IDF_weight, alpha_min, gamma)
    nm_log = []
    W = tune_W(rec, tune_truth, W0, nm_log)
    log += nm_log
    print(f"  W {W}  ({len(nm_log)} evals)")

    print("\n[3] grid, tuned W", flush=True)
    alpha_min2, gamma2 = grid(rec, IDF_weight, truth["train"], W, log)
    print(f"  best α_min {alpha_min2}, γ {gamma2}")
    pd.DataFrame(log).to_csv("results/tuning_log.csv", index=False)

    print("\n[4] validation", flush=True)
    candidates = {"default": (.5, 1, W0), "grid": (alpha_min, gamma, W0), "grid + W": (alpha_min, gamma, W),
                  "W + regrid": (alpha_min2, gamma2, W)}
    val = {}
    for name, (a, g, w) in candidates.items():
        set_weights(rec, IDF_weight, a, g)
        rec.set_layer_weights(w)
        val[name] = {"alpha_min": a, "gamma": g, "W": w, "train R@10": recall_at(rec, truth["train"]),
                     "val R@10": recall_at(rec, truth["val"])}
        print(f"  {name:12s} {val[name]}", flush=True)
    best = max(val, key=lambda k: val[k]["val R@10"])
    params = {"embedding": "all-MiniLM-L6-v2 / name+desc", "min_up": MIN_UP} | val[best] | {"chosen": best}
    save(params, "results/tuned_params.json")
    print(f"  chosen: {best}")

    print("\n[5] test (once)", flush=True)
    t = truth["test"]
    q = list(t)
    rows = []

    def run(name, lists, self_hit=float("nan")):
        table, s = evaluate(rec, t, lists, self_hit)
        print(f"\n== {name} ==  " + "  ".join(f"{k} {v:.3f}" for k, v in s.items()))
        print(table.to_string(float_format="%.4f"))
        for subset, r in table.iterrows():
            rows.append({"method": name, "subset": subset} | r.to_dict() | s)

    run("random", random_lists(len(rec.dataset), q))
    configs = {"IDF only (MiniLM name, α_min=1, γ=1, default W)": ("name", 1, 1, W0),
               "v2 default (name+desc, α_min=0.5, γ=1, default W)": ("name+desc", .5, 1, W0),
               f"v2 tuned ({best})": ("name+desc", params["alpha_min"], params["gamma"], params["W"])}
    for name, (input_type, a, g, w) in configs.items():
        rec.E = get_E("minilm", input_type, note_texts(vocab, input_type), device)
        set_weights(rec, IDF_weight, a, g)
        rec.set_layer_weights(w)
        run(name, *model_lists(rec, q))
    run("Fragrantica also_liked (reference)", also_liked_lists(rec, q))
    pd.DataFrame(rows).to_csv("results/test_results.csv", index=False)
