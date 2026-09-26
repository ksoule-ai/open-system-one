"""Markdown report for one eval run: overall, by question type, by category, per question."""

import statistics
from collections import defaultdict
from typing import Any


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def _pct(xs):
    xs = [x for x in xs if x is not None]
    return f"{100 * sum(xs) / len(xs):.0f}% ({sum(xs)}/{len(xs)})" if xs else "–"


def _f(x, digits=2):
    return "–" if x is None else f"{x:.{digits}f}"


def _quantile(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, round(q * (len(xs) - 1)))]


def _get(row, side, field):
    return (row.get(f"{side}_score") or {}).get(field)


def _group_table(rows, key) -> list[str]:
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    out = [
        "| group | n | ours correct | Jev correct | agree w/ Jev | mean abs Δp | ours P(exp) | Jev P(exp) | label mass |",
        "| --- | --: | --: | --: | --: | --: | --: | --: | --: |",
    ]
    for g in sorted(groups):
        rs = groups[g]
        out.append(
            f"| {g} | {len(rs)} | {_pct([_get(r, 'ours', 'correct') for r in rs])} "
            f"| {_pct([_get(r, 'jev', 'correct') for r in rs])} "
            f"| {_pct([(r.get('vs_jev') or {}).get('agree') for r in rs])} "
            f"| {_f(_mean([(r.get('vs_jev') or {}).get('dp') for r in rs]))} "
            f"| {_f(_mean([_get(r, 'ours', 'p_expected') for r in rs]))} "
            f"| {_f(_mean([_get(r, 'jev', 'p_expected') for r in rs]))} "
            f"| {_f(_mean([r.get('label_mass') for r in rs]), 3)} |"
        )
    return out


def _show(answer) -> str:
    if not answer:
        return "–"
    if answer["type"] == "noul":
        return f"{answer['noul']:.2f}"
    if answer["type"] == "choice":
        return f"{answer['choice']} ({answer['probabilities'][answer['choice']]:.2f})"
    return f"{answer['score']:.2f}"


def _expect(e) -> str:
    if e is None:
        return "–"
    if "band" in e:
        return f"[{e['band'][0]}, {e['band'][1]}]"
    if "answer_in" in e:
        return " / ".join(map(str, e["answer_in"]))
    return str(e["answer"]).lower() if isinstance(e["answer"], bool) else str(e["answer"])


def _mark(score) -> str:
    return "" if not score else ("✓" if score["correct"] else "✗")


def render(meta: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    ours_err = sum(1 for r in rows if r.get("ours_error"))
    jev_err = sum(1 for r in rows if r.get("jev_error"))
    masses = [r.get("label_mass") for r in rows if r.get("label_mass") is not None]
    conf = [
        r.get("jev_confidence_delta") for r in rows if r.get("jev_confidence_delta") is not None
    ]
    nouls = [r for r in rows if r["type"] == "noul"]
    server_ms = {r["case"]: r["ours_server_ms"] for r in rows}.values()
    wall_ms = {r["case"]: r["ours_wall_ms"] for r in rows}.values()
    jev_s = [s * 1000 for s in {r["case"]: r["jev_seconds"] for r in rows}.values() if s]

    ours_correct = _pct([_get(r, "ours", "correct") for r in rows])
    jev_correct = _pct([_get(r, "jev", "correct") for r in rows])
    ours_pexp = _f(_mean([_get(r, "ours", "p_expected") for r in rows]))
    jev_pexp = _f(_mean([_get(r, "jev", "p_expected") for r in rows]))
    ours_brier = _f(_mean([_get(r, "ours", "brier") for r in nouls]), 3)
    jev_brier = _f(_mean([_get(r, "jev", "brier") for r in nouls]), 3)
    vs = [r.get("vs_jev") or {} for r in rows]
    ours_latency = (
        f"{_f(_quantile(server_ms, 0.5), 0)} / {_f(_quantile(server_ms, 0.95), 0)} server, "
        f"{_f(_quantile(wall_ms, 0.5), 0)} / {_f(_quantile(wall_ms, 0.95), 0)} client"
    )
    jev_latency = (
        f"{_f(_quantile(jev_s, 0.5), 0)} / {_f(_quantile(jev_s, 0.95), 0)} (uncached calls)"
    )
    served = ", ".join(meta["jev_served_models"]) or "–"

    lines = [
        f"# Eval: {meta['target']} vs {meta['baseline']} — {meta['split']} split",
        "",
        (
            f"- Target: `{meta['target']}` ({meta['backend']}, `{meta['model_id']}`), strategy "
            f"`{meta['strategy']}`, prompt `{meta['prompt']}` (hash `{meta['prompt_hash']}`)"
        ),
        f"- Baseline: `{meta['baseline']}`, served as {served}; cache {meta['jev_cache']}",
        (
            f"- Cases: `{meta['cases']}`, split `{meta['split']}`: {meta['n_cases']} cases, "
            f"{len(rows)} questions. Run {meta['time']}."
        ),
        "",
        "## Overall",
        "",
        "| metric | ours | Jev |",
        "| --- | --: | --: |",
        f"| errors (questions) | {ours_err} | {jev_err} |",
        f"| correct vs expect | {ours_correct} | {jev_correct} |",
        f"| mean P(expected) | {ours_pexp} | {jev_pexp} |",
        f"| Brier (nouls with yes/no expect) | {ours_brier} | {jev_brier} |",
        f"| latency p50 / p95 per request (ms) | {ours_latency} | {jev_latency} |",
        "",
        (
            f"Agreement with Jev: argmax {_pct([v.get('agree') for v in vs])}, "
            f"mean |Δp| {_f(_mean([v.get('dp') for v in vs]), 3)}, "
            f"mean |Δscore| {_f(_mean([v.get('dscore') for v in vs]), 2)}."
        ),
        "",
        (
            f"Label mass (Σ P(label tokens) before normalization): mean {_f(_mean(masses), 4)}, "
            f"min {_f(min(masses) if masses else None, 4)}, "
            f"{sum(1 for m in masses if m < 0.9)} question(s) below 0.9; "
            f"{sum(1 for r in rows if r.get('missing_labels'))} with labels missing from the top-k."
        ),
        "",
        (
            "Confidence formula on Jev's own probabilities vs Jev's `confidence`: "
            f"mean |Δ| {_f(_mean(conf), 4)}, max {_f(max(conf) if conf else None, 4)} "
            f"over {len(conf)} answers."
        ),
        "",
        "## By question type",
        "",
        *_group_table(rows, lambda r: r["type"]),
        "",
        "## By category",
        "",
        *_group_table(rows, lambda r: r["category"]),
        "",
        "## Per question",
        "",
        "Sorted by |Δp| vs Jev, largest first. noul = P(yes); choice = argmax (its p); score = expected level.",
        "",
        "| case | q | type | expected | ours | | Jev | | abs Δp | mass | Jev doc |",
        "| --- | --- | --- | --- | --: | :-: | --: | :-: | --: | --: | --: |",
    ]
    ordered = sorted(rows, key=lambda r: -((r.get("vs_jev") or {}).get("dp") or 0))
    for r in ordered:
        ours = _show(r["ours"]) if r["ours"] else f"error: {(r['ours_error'] or '')[:40]}"
        lines.append(
            f"| {r['case']} | {r['question']} | {r['type']} | {_expect(r['expect'])} | {ours} "
            f"| {_mark(r.get('ours_score'))} | {_show(r['jev'])} | {_mark(r.get('jev_score'))} "
            f"| {_f((r.get('vs_jev') or {}).get('dp'))} | {_f(r.get('label_mass'), 3)} "
            f"| {_f(r['jev_reference']) if isinstance(r['jev_reference'], (int, float)) else '–'} |"
        )
    return "\n".join(lines) + "\n"
