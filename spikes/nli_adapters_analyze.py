"""Analyze spikes/nli_adapters_dev.py output: can adapter probabilities map onto 3-way NLI labels?

Support score s per item, oriented so that higher = more entailment-like:
  factuality-detection  s = 1 - P("yes")   (P(not factually incorrect))
  answerability         s = P("answerable")
Rules, each scored with 3-class macro-F1 (unanswered items count as wrong, as in the Decision
Index) and compared with the dev-set baselines on the same items:
  fixed      entailment if s > 0.75, contradiction if s < 0.25, else neutral
  tuned      the (low, high) thresholds with the best macro-F1 on the items (in-sample, so
             optimistic) and the same search under 2-fold cross-validation split by request
  combined   answerability decides neutral (s_ans < a), then factuality splits the rest
             (s_fact >= f → entailment, else contradiction); in-sample and 2-fold CV

    uv run python spikes/nli_adapters_analyze.py [--dir runs/spikes/nli-adapters-dev]
"""

import argparse
import glob
import json
import random
import statistics
from pathlib import Path

LABELS = ("entailment", "neutral", "contradiction")
GRID = [i / 50 for i in range(51)]
BASELINES = {
    "default@3 (3B wide)": "runs/di-dev/oso-granite-3b-wide",
    "default@3 (Micro)": "runs/di-dev/oso-granite-micro-cf-v3",
    "Jev 1.13": "runs/di-dev/jev-1.13",
}


def macro_f1(pairs) -> float:
    f1s = []
    for c in LABELS:
        tp = sum(p == c and g == c for p, g in pairs)
        fp = sum(p == c and g != c for p, g in pairs)
        fn = sum(p != c and g == c for p, g in pairs)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f1s) / len(f1s)


def three_way(s, low, high):
    if s is None:
        return None
    return "entailment" if s > high else "contradiction" if s < low else "neutral"


def support(r):
    if r.get("p_yes") is None:
        return None
    return 1 - r["p_yes"] if r["adapter"] == "factuality-detection" else r["p_yes"]


def best_thresholds(items):
    best = (-1, None)
    for lo in GRID:
        for hi in GRID:
            if hi < lo:
                continue
            f = macro_f1([(three_way(s, lo, hi), g) for s, g in items])
            best = max(best, (f, (lo, hi)))
    return best


def best_combined(items):
    """items: (s_fact, s_ans, gold)."""
    best = (-1, None)
    for a in GRID:
        for f in GRID:
            preds = [
                None
                if sf is None or sa is None
                else "neutral"
                if sa < a
                else "entailment"
                if sf >= f
                else "contradiction"
                for sf, sa, _ in items
            ]
            best = max(best, (macro_f1(list(zip(preds, [g for *_, g in items]))), (a, f)))
    return best


def cross_validate(groups, fit, apply, seed=0):
    """2-fold CV over request groups (all hypotheses of one contract stay together)."""
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    folds = [keys[::2], keys[1::2]]
    pairs = []
    for i in (0, 1):
        train = [x for k in folds[1 - i] for x in groups[k]]
        test = [x for k in folds[i] for x in groups[k]]
        params = fit(train)[1]
        pairs += [(apply(x, params), x[-1]) for x in test]
    return macro_f1(pairs)


def baseline_score(run_dir, keys, gold):
    preds = {}
    for path in glob.glob(f"{run_dir}/shard-*/results.jsonl"):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r["status"] != "ok":
                    continue
                for k, a in r["response"]["answers"].items():
                    if "choice" in a:
                        preds[(r["run_id"], k)] = a["choice"]
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/spikes/nli-adapters-dev")
    args = ap.parse_args()
    rows = {}
    for name in ("factuality-detection", "answerability"):
        path = Path(args.dir) / f"{name}.jsonl"
        if path.exists():
            with path.open() as f:
                rows[name] = {(r["run_id"], r["key"]): r for r in map(json.loads, f)}

    # label text per baseline answer key (ANLI uses A/B/C)
    import gzip

    keymap = {}
    with gzip.open("evals/decision-index-dev/dev-rows.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["_evaluation"]["dataset"] in ("ANLI", "ContractNLI"):
                for k, q in r["questions"].items():
                    keymap[(r["_evaluation"]["run_id"], k)] = {
                        c: (d if r["_evaluation"]["dataset"] == "ANLI" else c)
                        .lower()
                        .replace("notmentioned", "neutral")
                        for c, d in q["criteria"].items()
                    }

    report = {}
    for bench in ("ANLI", "ContractNLI"):
        out = report[bench] = {}
        any_rows = next(iter(rows.values()))
        ids = [k for k, r in any_rows.items() if r["benchmark"] == bench]
        gold = {k: any_rows[k]["gold"] for k in ids}
        out["items"] = len(ids)
        out["gold_counts"] = {c: sum(g == c for g in gold.values()) for c in LABELS}
        for label, run_dir in BASELINES.items():
            preds = baseline_score(run_dir, ids, gold)
            out[f"baseline {label}"] = round(
                macro_f1(
                    [(keymap[k].get(preds.get(k)) if k in preds else None, gold[k]) for k in ids]
                ),
                3,
            )
        for name, rs in rows.items():
            items = [(support(rs[k]), gold[k]) for k in ids]
            per = out[name] = {
                "errors": sum(bool(rs[k]["error"]) for k in ids),
                "no_answer_position": sum(
                    rs[k].get("p_yes") is None and not rs[k]["error"] for k in ids
                ),
                "said_yes": sum(bool(rs[k].get("answer_yes")) for k in ids),
                "support_by_gold": {
                    c: {
                        "mean": round(statistics.mean(v), 3),
                        "q25": round(statistics.quantiles(v, n=4)[0], 3),
                        "median": round(statistics.median(v), 3),
                        "q75": round(statistics.quantiles(v, n=4)[2], 3),
                    }
                    for c in LABELS
                    if (v := [s for s, g in items if g == c and s is not None]) and len(v) > 1
                },
                "fixed_0.25_0.75": round(
                    macro_f1([(three_way(s, 0.25, 0.75), g) for s, g in items]), 3
                ),
            }
            f, (lo, hi) = best_thresholds(items)
            per["tuned_in_sample"] = {"macro_f1": round(f, 3), "low": lo, "high": hi}
            groups = {}
            for k in ids:
                groups.setdefault(k[0], []).append((support(rs[k]), gold[k]))
            per["tuned_2fold_cv"] = round(
                cross_validate(groups, best_thresholds, lambda x, p: three_way(x[0], *p)), 3
            )
        if len(rows) == 2:
            fr, ar = rows["factuality-detection"], rows["answerability"]
            items = [(support(fr[k]), support(ar[k]), gold[k]) for k in ids]
            f, (a, t) = best_combined(items)
            groups = {}
            for k, x in zip(ids, items):
                groups.setdefault(k[0], []).append(x)

            def apply(x, p):
                sf, sa, _ = x
                if sf is None or sa is None:
                    return None
                return "neutral" if sa < p[0] else "entailment" if sf >= p[1] else "contradiction"

            out["combined"] = {
                "in_sample": {
                    "macro_f1": round(f, 3),
                    "answerable_below": a,
                    "factual_at_least": t,
                },
                "2fold_cv": round(cross_validate(groups, best_combined, apply), 3),
            }
    print(json.dumps(report, indent=1))
    (Path(args.dir) / "analysis.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
