"""Aggregate raw case results -> results.json + report.md, and diff vs baseline."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from jarvis_eval.config import BASELINE_FILE, REPO_ROOT, RESULTS_DIR, settings
from jarvis_eval.metrics import mean, pct

# metric -> (higher_is_better, regression threshold). Only these gate CI.
GATES = {
    "retrieval::recall@5": (True, 0.03),
    "rag_qa::answer_correctness": (True, 0.30),
    "agent_tasks::success": (True, 0.05),
}

_SUITE_METRICS = {
    "retrieval": ["recall@1", "recall@3", "recall@5", "recall@10",
                  "precision@5", "mrr@10", "ndcg@10", "hit", "top1_score"],
    "rag_qa": ["answer_correctness", "groundedness", "citation_recall",
               "retrieved_gold", "turns", "latency_s", "usd"],
    "agent_tasks": ["success", "tool_choice", "turns", "hitl_rounds", "latency_s", "usd"],
}


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip() or "?"
    except Exception:
        return "?"


def _per_case(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Average each metric across a case's repeats."""
    by_case: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_case.setdefault((r["suite"], r["case_id"]), []).append(r)
    out = {}
    for key, reps in by_case.items():
        keys = {k for rep in reps for k in rep["metrics"]}
        out[key] = {k: mean([rep["metrics"].get(k) for rep in reps]) for k in keys}
        out[key]["_repeats"] = len(reps)
        out[key]["_errors"] = sum(1 for rep in reps if rep["meta"].get("error"))
    return out


def aggregate(rows: list[dict]) -> dict:
    per_case = _per_case(rows)
    suites: dict[str, dict] = {}
    for suite, metric_names in _SUITE_METRICS.items():
        cases = {cid: m for (s, cid), m in per_case.items() if s == suite}
        if not cases:
            continue
        agg = {name: mean([m.get(name) for m in cases.values()]) for name in metric_names}
        if suite != "retrieval":
            lat = [m.get("latency_s") for m in cases.values()]
            agg["latency_p95"] = pct(lat, 95)
            agg["usd_total"] = round(sum(m.get("usd", 0) or 0 for m in cases.values()), 4)
        agg["n_cases"] = len(cases)
        agg["n_errors"] = sum(m["_errors"] for m in cases.values())
        suites[suite] = agg
    return {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runner_model": settings.RUNNER_MODEL,
            "judge_model": settings.JUDGE_MODEL,
            "git_sha": _git_sha(),
        },
        "suites": suites,
        "cases": {f"{s}/{c}": m for (s, c), m in per_case.items()},
    }


def write_run(rows: list[dict]) -> Path:
    agg = aggregate(rows)
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    d = RESULTS_DIR / ts
    d.mkdir(parents=True, exist_ok=True)
    (d / "raw.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    (d / "results.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False))
    (d / "report.md").write_text(render_md(agg, _load_baseline()))
    return d


def _load_baseline() -> dict | None:
    try:
        return json.loads(BASELINE_FILE.read_text())
    except (FileNotFoundError, ValueError):
        return None


def latest_run() -> Path | None:
    runs = sorted(p for p in RESULTS_DIR.glob("*/") if (p / "results.json").exists())
    return runs[-1] if runs else None


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _delta(cur, base, higher_better=True) -> str:
    if base is None or cur is None or not isinstance(cur, (int, float)) or not isinstance(base, (int, float)):
        return ""
    d = cur - base
    if abs(d) < 1e-9:
        return "±0"
    arrow = "▲" if d > 0 else "▼"
    good = (d > 0) == higher_better
    return f"{arrow}{abs(d):.3f} {'✅' if good else '⚠️'}"


def render_md(agg: dict, baseline: dict | None) -> str:
    m = agg["meta"]
    base_suites = (baseline or {}).get("suites", {})
    lines = [
        f"# Jarvis eval — {m['timestamp']}",
        "",
        f"- runner model: `{m['runner_model']}`  ·  judge: `{m['judge_model']}`  ·  eval sha: `{m['git_sha']}`",
    ]
    if baseline:
        lines.append(f"- baseline: `{baseline['meta']['timestamp']}` (runner `{baseline['meta']['runner_model']}`)")
    lines.append("")

    for suite, agg_metrics in agg["suites"].items():
        b = base_suites.get(suite, {})
        lines += [f"## {suite}", "",
                  f"_{agg_metrics['n_cases']} cases, {agg_metrics['n_errors']} errored_", "",
                  "| metric | value | vs baseline |", "|---|---|---|"]
        for name in _SUITE_METRICS[suite] + [k for k in ("latency_p95", "usd_total") if k in agg_metrics]:
            if name not in agg_metrics:
                continue
            hib = name not in ("turns", "hitl_rounds", "latency_s", "latency_p95", "usd", "usd_total")
            lines.append(f"| {name} | {_fmt(agg_metrics[name])} | {_delta(agg_metrics[name], b.get(name), hib)} |")
        lines.append("")

    # per-case detail
    lines += ["<details><summary>per-case</summary>", ""]
    for key, cm in agg["cases"].items():
        show = {k: round(v, 3) for k, v in cm.items() if not k.startswith("_") and isinstance(v, (int, float))}
        flag = " ⚠️errors" if cm.get("_errors") else ""
        lines.append(f"- `{key}`{flag}: {show}")
    lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


def regressions(agg: dict, baseline: dict | None) -> list[str]:
    if not baseline:
        return []
    out = []
    for gate, (higher_better, thr) in GATES.items():
        suite, metric = gate.split("::")
        cur = agg["suites"].get(suite, {}).get(metric)
        base = baseline["suites"].get(suite, {}).get(metric)
        if cur is None or base is None:
            continue
        drop = (base - cur) if higher_better else (cur - base)
        if drop > thr:
            out.append(f"{gate}: {base:.3f} → {cur:.3f} (drop {drop:.3f} > {thr})")
    return out


def promote_baseline(run_dir: Path) -> None:
    BASELINE_FILE.write_text((run_dir / "results.json").read_text())
