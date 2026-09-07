"""Turn a pile of per-case results into numbers and prose.

A benchmark run produces `rows` — one dict per (suite, case):
    {"suite": "beir_scifact", "case_id": "5", "repeat": 0,
     "metrics": {"ndcg@10": 1.0, ...}, "meta": {"error": None, ...}}

This module:
  * aggregate(rows)      -> mean each metric per suite  -> results.json shape
  * write_run(rows)      -> results/<ts>/{raw,results}.json + report.md
  * render_md(agg, base) -> the markdown table, with "vs baseline" deltas
  * regressions(...)     -> list of gated metrics that dropped too far (CI gate)
  * promote_baseline()   -> freeze a run into datasets/baseline.json
  * render_baseline_md() -> the standalone RESULTS.md scoreboard
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from jarvis_eval.config import BASELINE_FILE, REPO_ROOT, RESULTS_DIR, settings
from jarvis_eval.metrics import mean, pct
from jarvis_eval.benchmarks import BENCH_METRICS   # {suite: (metric_list, reference_string)}

# which metrics to surface per suite, and the published number to show beside them
_SUITE_METRICS = {s: metrics for s, (metrics, _ref) in BENCH_METRICS.items()}
_BENCH_REF = {s: ref for s, (_, ref) in BENCH_METRICS.items()}

# The only metrics that fail CI. metric -> (higher_is_better, allowed drop).
# A run scoring more than `thr` below baseline on one of these => exit 1.
GATES = {
    "beir_scifact::ndcg@10": (True, 0.03),
    "beir_nfcorpus::ndcg@10": (True, 0.03),
    "hotpotqa::support_recall@5": (True, 0.05),
    "hotpotqa::answer_f1": (True, 0.05),
}


def _git_sha() -> str:
    """Short HEAD of the jarvis-eval repo, stamped into each run for traceability."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip() or "?"
    except Exception:
        return "?"


def _per_case(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Collapse a case's repeats (usually 1) into one metric dict by averaging.
    Keyed by (suite, case_id). `_repeats` / `_errors` are bookkeeping."""
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
    """rows (per-case) -> {meta, suites: {suite: {metric: mean}}, cases: {...}}."""
    per_case = _per_case(rows)
    suites: dict[str, dict] = {}
    for suite, metric_names in _SUITE_METRICS.items():
        cases = {cid: m for (s, cid), m in per_case.items() if s == suite}
        if not cases:
            continue
        # average each headline metric across the suite's cases
        agg = {name: mean([m.get(name) for m in cases.values()]) for name in metric_names}
        # add latency p95 + total $ only if the suite actually ran the agent
        lat = [m.get("latency_s") for m in cases.values() if m.get("latency_s") is not None]
        if lat:
            agg["latency_p95"] = pct(lat, 95)
            agg["usd_total"] = round(sum(m.get("usd", 0) or 0 for m in cases.values()), 4)
        agg["n_cases"] = len(cases)
        agg["n_errors"] = sum(m["_errors"] for m in cases.values())
        suites[suite] = agg
    return {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runner_model": settings.RUNNER_MODEL,
            "git_sha": _git_sha(),
        },
        "suites": suites,
        "cases": {f"{s}/{c}": m for (s, c), m in per_case.items()},
    }


def write_run(rows: list[dict]) -> Path:
    """Persist one run to results/<UTC timestamp>/ and return the dir."""
    agg = aggregate(rows)
    d = RESULTS_DIR / time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    d.mkdir(parents=True, exist_ok=True)
    (d / "raw.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))       # every case, verbatim
    (d / "results.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False))    # the aggregate
    (d / "report.md").write_text(render_md(agg, _load_baseline()))                    # human view
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
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def _delta(cur, base, higher_better=True) -> str:
    """A '▲0.021 ✅' / '▼0.008 ⚠️' cell for the 'vs baseline' column."""
    if base is None or cur is None or not isinstance(cur, (int, float)) or not isinstance(base, (int, float)):
        return ""
    d = cur - base
    if abs(d) < 5e-4:                    # below the 3-decimal display precision
        return "±0"
    arrow = "▲" if d > 0 else "▼"
    good = (d > 0) == higher_better      # up is good for scores, bad for latency/$
    return f"{arrow}{abs(d):.3f} {'✅' if good else '⚠️'}"


def render_md(agg: dict, baseline: dict | None) -> str:
    """The per-run report: one table per suite, then a collapsible per-case
    dump. Suites that ran this time show live values + deltas vs baseline;
    suites only in the baseline are still shown (marked 'not run this time')
    so the report is always the full picture."""
    m = agg["meta"]
    base_suites = (baseline or {}).get("suites", {})
    run_suites = set(agg["suites"])
    lines = [
        f"# Jarvis eval — {m['timestamp']}",
        "",
        f"- agent model: `{m['runner_model']}`  ·  eval sha: `{m['git_sha']}`",
        f"- suites run: {', '.join(run_suites) or '(none)'}",
    ]
    if baseline:
        lines.append(f"- baseline: `{baseline['meta']['timestamp']}` (runner `{baseline['meta']['runner_model']}`)")
    lines.append("")

    # every known suite, in a stable order: what ran + anything only in baseline
    for suite in list(_SUITE_METRICS) + [s for s in base_suites if s not in _SUITE_METRICS]:
        live = suite in run_suites
        agg_metrics = agg["suites"].get(suite) or base_suites.get(suite)
        if not agg_metrics:
            continue
        b = base_suites.get(suite, {})
        lines += [f"## {suite}", ""]
        if suite in _BENCH_REF:
            lines.append(f"_reference: {_BENCH_REF[suite]}_")
        if live:
            lines.append(f"_{agg_metrics['n_cases']} cases, {agg_metrics['n_errors']} errored_")
        else:
            lines.append("_not run this time — showing baseline_")
        lines += ["", "| metric | value | vs baseline |", "|---|---|---|"]
        for name in _SUITE_METRICS.get(suite, list(agg_metrics)) + \
                [k for k in ("latency_p95", "usd_total") if k in agg_metrics]:
            if name not in agg_metrics:
                continue
            hib = name not in ("turns", "hitl_rounds", "latency_s", "latency_p95", "usd", "usd_total")
            delta = _delta(agg_metrics[name], b.get(name), hib) if live else ""
            lines.append(f"| {name} | {_fmt(agg_metrics[name])} | {delta} |")
        lines.append("")

    lines += ["<details><summary>per-case</summary>", ""]
    for key, cm in agg["cases"].items():
        show = {k: round(v, 3) for k, v in cm.items() if not k.startswith("_") and isinstance(v, (int, float))}
        flag = " ⚠️errors" if cm.get("_errors") else ""
        lines.append(f"- `{key}`{flag}: {show}")
    lines += ["", "</details>"]
    return "\n".join(lines) + "\n"


def regressions(agg: dict, baseline: dict | None) -> list[str]:
    """Gated metrics that fell more than their threshold below baseline.
    Empty list => the CI gate passes. No baseline => nothing to check."""
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


def render_baseline_md() -> str:
    """RESULTS.md — a plain scoreboard straight from datasets/baseline.json.
    Always shows every suite; needs no run. Regenerated by `jeval baseline`."""
    b = _load_baseline() or {"meta": {}, "suites": {}}
    m = b.get("meta", {})
    lines = [
        "# jarvis-eval — current scores",
        "",
        f"_Generated from `datasets/baseline.json` · agent model `{m.get('runner_model', '?')}` · "
        f"{m.get('timestamp', '?')}_",
        "",
        f"> {m['note']}" if m.get("note") else "",
        "",
    ]
    for suite, s in b.get("suites", {}).items():
        lines += [f"## {suite}", ""]
        if suite in _BENCH_REF:
            lines.append(f"_reference: {_BENCH_REF[suite]}_")
        lines += ["", "| metric | value |", "|---|---|"]
        for k, v in s.items():
            if k in ("n_cases", "n_errors"):
                continue
            lines.append(f"| {k} | {_fmt(v)} |")
        n = s.get("n_cases") or s.get("retrieval_n_cases")
        if n is not None:
            lines.append(f"| _cases_ | {n} ({s.get('n_errors', 0)} errored) |")
        lines.append("")
    return "\n".join(x for x in lines if x is not None) + "\n"


def promote_baseline(run_dir: Path) -> None:
    """Copy a run's aggregate scores into datasets/baseline.json, merging by
    suite (so promoting a beir-only run keeps the hotpotqa numbers, etc.).
    Per-case detail is dropped — the baseline is aggregates only."""
    new = json.loads((run_dir / "results.json").read_text())
    existing = _load_baseline() or {"meta": {}, "suites": {}}
    existing["meta"] = new["meta"]
    existing.pop("cases", None)
    existing.setdefault("suites", {}).update(new.get("suites", {}))
    BASELINE_FILE.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
