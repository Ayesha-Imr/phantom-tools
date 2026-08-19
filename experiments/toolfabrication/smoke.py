"""Idea 3 - H1 phenomenon + H2 readout probes on Qwen3 (BFCL v3).

Fabrication = the model emits a tool NAME that is not in the provided registry
(distinct from misrouting to a valid-but-wrong tool, invalid args, or no-call).

Validates:
  H1: fabrication is a real, measurable phenomenon on BFCL v3 (fabrication rate
      per category, greedy vs sampled temp=1).
  H2: "will fabricate vs will call a valid tool" is linearly decodable from
      pre-generation hidden states (prompt-final AND first generated token),
      with a layer-wise AUROC curve.

Qwen3 specifics (vs the Qwen2.5 prototype): native <tools> chat template with
enable_thinking=False (forces a non-thinking response frame), <tool_call>
JSON output format. Greedy decode default; sampled temp=1 as fabrication
elicitation fallback (R1).

Design notes:
  - Auto-resume: if h1_cache.npz + h1_queries.json exist, only generate queries
    not already in the cache (so a small smoke run extends into a full run
    without regenerating). Labels are always recomputed from raw outputs joined
    against possible_answer ground truths (never trusted from a stale cache).
  - outputs.jsonl is appended, never clobbered on resume.

Model: Qwen/Qwen3-4B. Dataset: BFCL v3 from HF (public).
Outputs: results/toolfabrication/{results.json, outputs.jsonl, partial_results.json,
         h1_queries.json, h1_labels.json, h1_cache.npz}.
"""

import argparse
import json
import os
import random
import re
import time

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B"
OUT_DIR = os.environ.get("SMOKE_OUT", "results/toolfabrication")
BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_FILES = {
    "multiple": "BFCL_v3_multiple.json",
    "irrel": "BFCL_v3_irrelevance.json",
    "live_irrel": "BFCL_v3_live_irrelevance.json",
    "live_rel": "BFCL_v3_live_relevance.json",
    "simple": "BFCL_v3_simple.json",
}


def load_bfcl(filename):
    path = hf_hub_download(BFCL_REPO, filename, repo_type="dataset")
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_pa(filename):
    """Ground-truth tool names per item id from the possible_answer/ dir."""
    path = hf_hub_download(BFCL_REPO, f"possible_answer/{filename}", repo_type="dataset")
    pa = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            gts = []
            for entry in d.get("ground_truth", []):
                if isinstance(entry, dict):
                    gts.extend(entry.keys())
            pa[d["id"]] = gts
    return pa


def item_question(item):
    q = item.get("question") or item.get("conversations")
    if isinstance(q, list) and q:
        first = q[0]
        if isinstance(first, dict):
            return first.get("text") or first.get("content") or ""
        if isinstance(first, list) and first:
            d = first[0]
            if isinstance(d, dict):
                return d.get("text") or d.get("content") or ""
    return ""


def item_tools(item):
    return item.get("function") or []


def gt_names(item):
    for key in ("possible_answers", "possible_answer", "ground_truth"):
        pa = item.get(key)
        if pa:
            if isinstance(pa, list):
                return [p.get("name") for p in pa if isinstance(p, dict) and p.get("name")]
            if isinstance(pa, dict) and pa.get("name"):
                return [pa["name"]]
    return []


def build_prompt(tok, tools, question):
    """Render the Qwen3 native tool-calling prompt.

    Qwen3's chat template emits a <tools>...</tools> XML block when `tools` is
    passed, and with enable_thinking=False it adds the non-thinking response
    frame (" thinking\\n\\n response\\n\\n") to the generation prompt.
    """
    sys_msg = (
        "You are a helpful assistant with access to the following functions. "
        "Use the functions when they are relevant to the user's request. "
        "If no function is relevant, or required parameters are missing, say so "
        "instead of calling a function."
    )
    msgs = [{"role": "system", "content": sys_msg},
            {"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tools=tools or None, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


# Qwen3 emits tool calls as: <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
TOOLCALL_TAG_RE = re.compile(r"<tool_call>\s*\{.*?\"name\"\s*:\s*\"([^\"]+)\"", re.S)
# Some checkpoints / thinking outputs use the pipe form <|tool_call|>.
PIPE_TOOLCALL_RE = re.compile(r"<\|tool_call\|>\s*\{.*?\"name\"\s*:\s*\"([^\"]+)\"", re.S)
CALL_RE = re.compile(r"\[\s*([A-Za-z_]\w*)\s*\(")
JSON_NAME_RE = re.compile(r"\{[^{}]*\"name\"\s*:\s*\"([A-Za-z_][\w.]*)\"")
ANY_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\([^)]")
NONTOOL_WORDS = {"note", "see", "e.g", "i.e", "such", "like", "call", "use", "if",
                 "or", "and", "not", "no", "so", "then", "please", "here", "this",
                 "that", "example", "cases", "case", "check", "request", "answer",
                 "because", "since", "while", "before", "after", "for", "with"}


def parse_calls(text):
    names = [m.group(1) for m in TOOLCALL_TAG_RE.finditer(text)]
    if names:
        return names
    names = [m.group(1) for m in PIPE_TOOLCALL_RE.finditer(text)]
    if names:
        return names
    names = [m.group(1) for m in CALL_RE.finditer(text)]
    if names:
        return names
    names = [m.group(1) for m in JSON_NAME_RE.finditer(text)]
    if names:
        return names
    names = [m.group(1) for m in ANY_CALL_RE.finditer(text)]
    names = [n for n in names if n.lower() not in NONTOOL_WORDS]
    return names[:3]


def label_output(text, registry, gts):
    calls = parse_calls(text)
    if not calls:
        return "no_call", calls
    names = set(calls)
    if not names <= set(registry):
        return "fabricated", calls
    if gts and (names & set(gts)):
        return "correct", calls
    return "wrong_tool", calls


def load_model(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return tok, model


def gen(model, tok, prompt, max_new_tokens=220, do_sample=False, temperature=1.0):
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=0.9 if do_sample else 1.0,
            pad_token_id=tok.eos_token_id,
            output_hidden_states=True, return_dict_in_generate=True)
    text = tok.decode(out.sequences[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    hs0 = out.hidden_states[0]          # prompt forward, full sequence
    prompt_last = [h[0, -1, :].detach().float().cpu().numpy() for h in hs0]
    first_tok = None
    if len(out.hidden_states) > 1:
        hs1 = out.hidden_states[1]      # state after first generated token
        first_tok = [h[0, 0, :].detach().float().cpu().numpy() for h in hs1]
    return text, np.stack(prompt_last), np.stack(first_tok) if first_tok is not None else None


def probe_aucs(X, y, layers, test_size=0.3):
    """Per-layer logistic-regression AUROC. X: (N, L, D), y: binary labels."""
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, random_state=0, stratify=y)
    aucs = []
    for L in layers:
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(X[tr, L], y[tr])
        aucs.append(roc_auc_score(y[te], clf.decision_function(X[te, L])))
    return aucs


def save_partial(out_dir, p):
    try:
        with open(os.path.join(out_dir, "partial_results.json"), "w") as f:
            json.dump(p, f, indent=2)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--n-multiple", type=int, default=10000)
    ap.add_argument("--n-irrel", type=int, default=10000)
    ap.add_argument("--n-live-irrel", type=int, default=10000)
    ap.add_argument("--n-live-rel", type=int, default=10000)
    ap.add_argument("--n-simple", type=int, default=10000)
    ap.add_argument("--n-sampled", type=int, default=150)
    ap.add_argument("--max-tokens", type=int, default=220)
    ap.add_argument("--force", action="store_true",
                    help="regenerate everything even if a cache exists")
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    rng = random.Random(0)

    print("downloading BFCL v3 files...", flush=True)
    files = {key: load_bfcl(name) for key, name in BFCL_FILES.items()}
    counts = {"multiple": args.n_multiple, "irrel": args.n_irrel,
              "live_irrel": args.n_live_irrel, "live_rel": args.n_live_rel,
              "simple": args.n_simple}
    sets = {key: rng.sample(items, min(counts[key], len(items)))
            for key, items in files.items()}

    pa_maps = {}
    for key in ("multiple", "simple"):
        try:
            pa_maps[key] = load_pa(BFCL_FILES[key])
        except Exception as e:
            print(f"possible_answer load failed for {key}: {e}", flush=True)
            pa_maps[key] = {}

    def resolve_gts(item, category):
        return gt_names(item) or pa_maps.get(category, {}).get(item.get("id"), [])

    tok, model = load_model(args.model)
    n_layers = len(model.model.layers)
    print(f"[{time.time()-t0:.0f}s] model loaded: {args.model} ({n_layers} layers)",
          flush=True)

    # ---- build the full desired query set -----------------------------------
    queries = []
    for key, items in sets.items():
        for it in items:
            tools = item_tools(it)
            registry = [t["name"] for t in tools]
            gts = resolve_gts(it, key)
            prompt = build_prompt(tok, tools, item_question(it))
            queries.append({"prompt": prompt, "registry": registry, "gts": gts,
                            "category": key, "item_id": it.get("id"), "item": it})

    # ---- load existing cache / outputs for auto-resume ----------------------
    # The cache and outputs are keyed by item_id, which is stable across runs:
    # a smoke run's items are always a subset of a full run's items (full = all
    # items), so resume reuses states instead of regenerating or clobbering.
    cache_path = os.path.join(args.out, "h1_cache.npz")
    queries_path = os.path.join(args.out, "h1_queries.json")
    outputs_path = os.path.join(args.out, "outputs.jsonl")
    old_states, old_outputs = {}, []
    if os.path.exists(outputs_path):
        with open(outputs_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    o = json.loads(line)
                    old_outputs.append(o)
    out_by_item = {}
    for o in old_outputs:
        out_by_item[o["item_id"]] = o

    if not args.force and os.path.exists(cache_path) and os.path.exists(queries_path):
        with open(queries_path) as f:
            old_queries = json.load(f)
        z = np.load(cache_path)
        P_old, F_old = z["P"], z["F"] if "F" in z else None
        for i, q in enumerate(old_queries):
            old_states[q["item_id"]] = (P_old[i],
                                        F_old[i] if F_old is not None else None)
        print(f"[{time.time()-t0:.0f}s] cache: {len(old_queries)} existing queries, "
              f"P {P_old.shape}", flush=True)

    # ---- generate only the queries we do not have yet -----------------------
    P_rows, F_rows, outputs = [], [], []
    n_skip = 0
    for qi, q in enumerate(queries):
        old = old_states.get(q["item_id"])
        o = out_by_item.get(q["item_id"])
        if not args.force and old is not None and o is not None:
            P_rows.append(old[0])
            F_rows.append(old[1])
            o = dict(o)
            o["gts"] = q["gts"]
            lab, calls = label_output(o["raw"], q["registry"], q["gts"])
            o["label"], o["calls"] = lab, calls
            outputs.append(o)
            n_skip += 1
            continue
        text, ps, fs = gen(model, tok, q["prompt"], max_new_tokens=args.max_tokens)
        lab, calls = label_output(text, q["registry"], q["gts"])
        outputs.append({"qid": qi, "item_id": q["item_id"], "category": q["category"],
                        "registry": q["registry"], "gts": q["gts"], "raw": text,
                        "calls": calls, "label": lab})
        if qi < args.verbose:
            print(f"--- verbose [{qi}] cat={q['category']} registry={q['registry']} "
                  f"label={lab} calls={calls}\nRAW: {text[:400]}\n", flush=True)
        P_rows.append(ps)
        F_rows.append(fs if fs is not None else np.zeros_like(ps))
        if (qi + 1) % 25 == 0:
            print(f"[{time.time()-t0:.0f}s] generated {qi+1}/{len(queries)} "
                  f"(skipped {n_skip} cached)", flush=True)
    labels = [o["label"] for o in outputs]
    P = np.stack(P_rows)
    F = np.stack(F_rows) if F_rows and F_rows[0] is not None else None

    # persist merged cache + outputs keyed by item_id (extend, never clobber)
    merged_by_item = {}
    for o in old_outputs:
        merged_by_item[o["item_id"]] = o
    for o in outputs:
        merged_by_item[o["item_id"]] = o
    merged_outputs = list(merged_by_item.values())
    np.savez(cache_path, P=P, F=F)
    with open(queries_path, "w") as f:
        json.dump(queries, f)
    with open(os.path.join(args.out, "h1_labels.json"), "w") as f:
        json.dump(labels, f)
    with open(outputs_path, "w") as f:
        for o in merged_outputs:
            f.write(json.dumps(o) + "\n")
    print(f"[{time.time()-t0:.0f}s] cache saved: {cache_path} "
          f"({P.shape[0]} queries, {n_skip} reused)", flush=True)

    # ---- H1 rates ------------------------------------------------------------
    rates = {}
    for key in counts:
        idx = [i for i, q in enumerate(queries) if q["category"] == key]
        labs = [labels[i] for i in idx]
        rates[key] = {
            "n": len(idx),
            "fabricated": labs.count("fabricated") / max(1, len(idx)),
            "no_call": labs.count("no_call") / max(1, len(idx)),
            "valid": (labs.count("correct") + labs.count("wrong_tool")) / max(1, len(idx)),
        }
    total_fab = labels.count("fabricated") / max(1, len(labels))
    print(f"[{time.time()-t0:.0f}s] H1 greedy: total_fabricated={total_fab:.3f} "
          f"{json.dumps(rates)}", flush=True)

    # sampled subset (fabrication-prone categories, temp=1 elicitation)
    sampled_pool = [q for q in queries if q["category"] in ("irrel", "live_irrel", "live_rel")]
    sampled_pool = rng.sample(sampled_pool, min(args.n_sampled, len(sampled_pool)))
    s_labels = []
    for q in sampled_pool:
        text, _, _ = gen(model, tok, q["prompt"], max_new_tokens=args.max_tokens,
                         do_sample=True, temperature=1.0)
        lab, _ = label_output(text, q["registry"], q["gts"])
        s_labels.append(lab)
    sampled_fab = s_labels.count("fabricated") / max(1, len(s_labels))
    s_n = len(s_labels)
    print(f"[{time.time()-t0:.0f}s] H1 sampled(temp=1): fabricated={sampled_fab:.3f} "
          f"no_call={s_labels.count('no_call')/max(1,len(s_labels)):.3f} "
          f"(n={len(s_labels)})", flush=True)

    partial = {"h1_rates_greedy": rates,
               "h1_fabricated_total_greedy": float(total_fab),
               "h1_fabricated_sampled": float(sampled_fab)}
    save_partial(args.out, partial)

    # ---- H2 probes (fab-vs-valid + call-vs-nocall, prompt-final + first-tok) -
    t2 = time.time()
    probe_layers = list(range(1, n_layers + 1))  # probe idx i = model layer i-1
    y_fab = np.array([1 if l == "fabricated" else 0 for l in labels
                      if l in ("fabricated", "correct", "wrong_tool")])
    idx_fab = [i for i, l in enumerate(labels) if l in ("fabricated", "correct", "wrong_tool")]
    if y_fab.sum() >= 4 and (len(y_fab) - y_fab.sum()) >= 4:
        Xf = P[idx_fab]
        auc_fab_prompt = probe_aucs(Xf, y_fab, probe_layers)
        best_fab = probe_layers[int(np.argmax(auc_fab_prompt))]
        if F is not None:
            Xf1 = F[idx_fab]
            auc_fab_first = probe_aucs(Xf1, y_fab, probe_layers)
            best_fab_first = probe_layers[int(np.argmax(auc_fab_first))]
        else:
            auc_fab_first, best_fab_first = None, None
    else:
        auc_fab_prompt, best_fab, auc_fab_first, best_fab_first = None, None, None, None

    y_call = np.array([1 if l in ("fabricated", "correct", "wrong_tool") else 0
                       for l in labels])
    if y_call.sum() >= 4 and (len(y_call) - y_call.sum()) >= 4:
        auc_call = probe_aucs(P, y_call, probe_layers)
        best_call = probe_layers[int(np.argmax(auc_call))]
    else:
        auc_call, best_call = None, None

    def probe_summary(aucs, best_layer):
        if aucs is None:
            return None
        return {"auc_by_layer": [float(a) for a in aucs],
                "best_probe_layer": int(best_layer),
                "best_model_layer": int(best_layer) - 1,
                "best_auc": float(aucs[probe_layers.index(best_layer)])}

    print(f"[{time.time()-t2:.0f}s] H2 probes: "
          + (f"fab-vs-valid best L={best_fab} (model layer {best_fab-1}) "
             f"AUROC={auc_fab_prompt[probe_layers.index(best_fab)]:.3f} (prompt-final)"
             if auc_fab_prompt is not None else "fab-vs-valid skipped (class imbalance)")
          + " | "
          + (f"call-vs-nocall best L={best_call} (model layer {best_call-1}) "
             f"AUROC={auc_call[probe_layers.index(best_call)]:.3f}"
             if auc_call is not None else "call-vs-nocall skipped (class imbalance)"),
          flush=True)
    partial["h2_probe_fab_vs_valid"] = probe_summary(auc_fab_prompt, best_fab)
    partial["h2_probe_fab_first_token"] = probe_summary(auc_fab_first, best_fab_first)
    partial["h2_probe_call_vs_nocall"] = probe_summary(auc_call, best_call)
    save_partial(args.out, partial)

    summary = {
        "model": args.model,
        "n_queries": len(queries),
        "h1_rates_greedy": rates,
        "h1_fabricated_total_greedy": float(total_fab),
        "h1_fabricated_sampled": float(sampled_fab),
        "h1_sampled_n": s_n,
        "h2_probe_fab_vs_valid": probe_summary(auc_fab_prompt, best_fab),
        "h2_probe_fab_first_token": probe_summary(auc_fab_first, best_fab_first),
        "h2_probe_call_vs_nocall": probe_summary(auc_call, best_call),
        "wall_seconds": float(time.time() - t0),
    }
    out_path = os.path.join(args.out, "results.json")
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
