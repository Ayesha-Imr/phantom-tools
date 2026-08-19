"""Idea 3 - H2 readout probes on Qwen3-4B (Phase 2).

Runs against the H1 cached states (h1_cache.npz + h1_queries.json +
h1_labels.json) produced by smoke.py. No model loading, no GPU needed - the
probes are sklearn logistic regressions on the cached per-layer hidden states.

Probes:
  - fab-vs-valid:  "will fabricate" vs "will call a valid tool", prompt-final
                   AND first-generated-token, layer-wise AUROC.
  - call-vs-nocall: any tool call vs decline, prompt-final, layer-wise AUROC.

Layer-wise AUROC curves use repeated stratified train/test splits (mean + std
across repeats) so the curves are stable enough for the paper.

Outputs: results/toolfabrication/probes.json (curves + best layers).
"""

import argparse
import json
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

OUT_DIR = os.environ.get("SMOKE_OUT", "results/toolfabrication")


def load_cache(out_dir):
    """Return (P, F, queries, labels) aligned by row index (len = n)."""
    z = np.load(os.path.join(out_dir, "h1_cache.npz"))
    P, F = z["P"], z["F"] if "F" in z else None
    with open(os.path.join(out_dir, "h1_queries.json")) as f:
        queries = json.load(f)
    with open(os.path.join(out_dir, "h1_labels.json")) as f:
        labels = json.load(f)
    assert len(P) == len(queries) == len(labels), "cache/queries/labels misaligned"
    return P, F, queries, labels


def probe_auc_curve(X, y, n_layers, n_repeats=5, test_size=0.3, seed=0):
    """Repeated stratified CV layer-wise AUROC. Returns (mean, std, best)."""
    idx = np.arange(len(y))
    sss = StratifiedShuffleSplit(n_splits=n_repeats, test_size=test_size,
                                 random_state=seed)
    aucs = np.zeros((n_repeats, n_layers))
    for r, (tr, te) in enumerate(sss.split(idx, y)):
        for L in range(n_layers):
            clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
            clf.fit(X[tr, L], y[tr])
            aucs[r, L] = roc_auc_score(y[te], clf.decision_function(X[te, L]))
    mean = aucs.mean(axis=0)
    std = aucs.std(axis=0)
    best = int(np.argmax(mean))
    return mean, std, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    P, F, queries, labels = load_cache(args.out)
    n_layers = P.shape[1]
    n = len(labels)
    labels_arr = np.array(labels)
    print(f"[{time.time()-t0:.0f}s] loaded cache: {n} queries, "
          f"P {P.shape}, layers {n_layers}", flush=True)

    # ---- contrast masks -----------------------------------------------------
    fab_idx = np.array([i for i, l in enumerate(labels) if l == "fabricated"])
    valid_idx = np.array([i for i, l in enumerate(labels)
                          if l in ("correct", "wrong_tool")])
    call_idx = np.array([i for i, l in enumerate(labels)
                         if l in ("fabricated", "correct", "wrong_tool")])
    print(f"fabricated={len(fab_idx)} valid={len(valid_idx)} "
          f"no_call={n - len(call_idx)}", flush=True)

    # ---- fab-vs-valid (prompt-final) ----------------------------------------
    summary = {"model": "Qwen/Qwen3-4B", "n": n, "n_layers": n_layers}
    if len(fab_idx) >= 4 and len(valid_idx) >= 4:
        both = np.concatenate([fab_idx, valid_idx])
        y = np.array([1] * len(fab_idx) + [0] * len(valid_idx))
        mean, std, best = probe_auc_curve(P[both], y, n_layers,
                                          args.repeats, args.test_size)
        summary["fab_vs_valid_prompt_final"] = {
            "auc_mean_by_layer": [float(a) for a in mean],
            "auc_std_by_layer": [float(a) for a in std],
            "best_probe_layer": int(best + 1),
            "best_model_layer": int(best),
            "best_auc": float(mean[best]),
            "n_fab": int(len(fab_idx)), "n_valid": int(len(valid_idx))}
        print(f"[{time.time()-t0:.0f}s] fab-vs-valid (prompt-final) best model "
              f"layer {best} AUROC={mean[best]:.4f}±{std[best]:.4f}", flush=True)
    else:
        summary["fab_vs_valid_prompt_final"] = None

    # ---- fab-vs-valid (first-token) -----------------------------------------
    if F is not None and summary.get("fab_vs_valid_prompt_final"):
        mean, std, best = probe_auc_curve(F[both], y, n_layers,
                                          args.repeats, args.test_size)
        summary["fab_vs_valid_first_token"] = {
            "auc_mean_by_layer": [float(a) for a in mean],
            "auc_std_by_layer": [float(a) for a in std],
            "best_probe_layer": int(best + 1),
            "best_model_layer": int(best),
            "best_auc": float(mean[best])}
        print(f"[{time.time()-t0:.0f}s] fab-vs-valid (first-token) best model "
              f"layer {best} AUROC={mean[best]:.4f}±{std[best]:.4f}", flush=True)
    else:
        summary["fab_vs_valid_first_token"] = None

    # ---- call-vs-nocall (prompt-final) --------------------------------------
    if len(call_idx) >= 4 and (n - len(call_idx)) >= 4:
        yc = (labels_arr == "fabricated") | (labels_arr == "correct") \
            | (labels_arr == "wrong_tool")
        yc = yc.astype(int)
        mean, std, best = probe_auc_curve(P, yc, n_layers,
                                          args.repeats, args.test_size)
        summary["call_vs_nocall"] = {
            "auc_mean_by_layer": [float(a) for a in mean],
            "auc_std_by_layer": [float(a) for a in std],
            "best_probe_layer": int(best + 1),
            "best_model_layer": int(best),
            "best_auc": float(mean[best]),
            "n_call": int(len(call_idx)), "n_nocall": int(n - len(call_idx))}
        print(f"[{time.time()-t0:.0f}s] call-vs-nocall best model layer {best} "
              f"AUROC={mean[best]:.4f}±{std[best]:.4f}", flush=True)
    else:
        summary["call_vs_nocall"] = None

    summary["wall_seconds"] = float(time.time() - t0)
    out_path = os.path.join(args.out, "probes.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("FINAL_SUMMARY " + json.dumps(summary), flush=True)
    print(f"results saved to {out_path}", flush=True)


if __name__ == "__main__":
    import sys
    import traceback
    try:
        main()
    except BrokenPipeError:
        pass
    except Exception as e:
        traceback.print_exc()
        print(f"FATAL: {e}", flush=True)
        sys.exit(1)
