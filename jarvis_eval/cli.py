"""jeval — command line for the Jarvis benchmark harness.

    jeval setup                       create/enable the eval Keycloak users
    jeval seed <benchmark>            download + upload + embed a benchmark's corpus
    jeval run --suite <benchmark|all> [--model ID]
    jeval report [RUN_DIR] [--baseline] [--fail-on-regression]
    jeval baseline [RUN_DIR]          promote a run into datasets/baseline.json + RESULTS.md
    jeval baseline --show             print the current scoreboard (all suites)
    jeval embcompare <model> [--dims N] [--suite ...]
                                     score a candidate embedding model on BEIR
                                     offline (no cluster), vs the baseline
    jeval rerankcompare <reranker> [--suite ...] [--fetch N]
                                     dense-vs-reranked retrieval on beir_* /
                                     hotpotqa, offline (needs requirements-rerank.txt)

benchmarks: beir_scifact · beir_nfcorpus · hotpotqa · gaia

Structure: each subcommand is wired to a `_cmd_*` handler in main() via
argparse's `set_defaults(fn=...)`; parse_args() picks one and the last line
calls it. Handlers return a process exit code.
"""
import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

from jarvis_eval import report
from jarvis_eval.benchmarks import BENCHMARKS   # {name: (seed_fn, run_fn)}
from jarvis_eval.clients.auth import ensure_user
from jarvis_eval.config import settings

console = Console()
BENCH = list(BENCHMARKS)   # ["beir_scifact", "beir_nfcorpus", "hotpotqa", "gaia"]


def _bench_user(name: str) -> str:
    """The Keycloak username a benchmark runs as: `eval-beir-scifact` etc.
    (each gets its own so their corpora don't mix)."""
    return f"{settings.EVAL_USERNAME}-{name.replace('_', '-')}"


def _cmd_setup(_args) -> int:
    """`jeval setup` — create/enable every eval Keycloak user (idempotent)."""
    console.print(f"[green]{settings.EVAL_USERNAME}[/] — sub [bold]{ensure_user()}[/]")
    for b in BENCH:
        u = _bench_user(b)
        console.print(f"[green]{u}[/] — sub [bold]{ensure_user(u)}[/]")
    return 0


def _cmd_seed(args) -> int:
    """`jeval seed <benchmark>` — download its dataset from HF, upload the
    corpus into that benchmark's workspace, wait for embedding. One-off."""
    u = _bench_user(args.benchmark)
    ensure_user(u)                                  # make sure the user exists first
    console.rule(f"seed {args.benchmark}  (user {u})")
    console.print(BENCHMARKS[args.benchmark][0]())  # -> beir.seed(...) / hotpotqa.seed()
    return 0


def _cmd_run(args) -> int:
    """`jeval run --suite X[,Y] | all` — score one or more benchmarks, write
    a results/<ts>/ dir, print the report."""
    which = BENCH if args.suite == "all" else args.suite.split(",")
    bad = [s for s in which if s not in BENCH]
    if bad:
        console.print(f"[red]unknown suite(s): {bad}[/]  (choices: {', '.join(BENCH)}, all)")
        return 2
    if args.model:
        settings.RUNNER_MODEL = args.model          # override for this process only

    all_rows: list[dict] = []                       # one dict per (suite, case)
    for suite in which:
        ensure_user(_bench_user(suite))
        console.rule(f"{suite}")
        rows = BENCHMARKS[suite][1](None, repeats=1)   # -> beir.run(...) / hotpotqa.run()
        for r in rows:                                  # live progress line per case
            err = r["meta"].get("error")
            nums = {k: round(v, 3) for k, v in r["metrics"].items() if isinstance(v, (int, float))}
            console.print(f"  {'[red]ERR[/]' if err else '[green]ok[/]'} {r['case_id']}  "
                          f"{json.dumps(nums)}" + (f"  [red]{err}[/]" if err else ""))
        all_rows += rows

    run_dir = report.write_run(all_rows)            # aggregate + write raw/results/report
    console.print(f"\n[bold]wrote[/] {run_dir}/report.md")
    agg = json.loads((run_dir / "results.json").read_text())
    console.print(report.render_md(agg, report._load_baseline()))
    return 0


def _cmd_report(args) -> int:
    """`jeval report [dir]` — re-print a run's report; with --baseline show
    deltas; with --fail-on-regression exit 1 if a gated metric dropped too far."""
    run_dir = _resolve_run(args.run_dir)
    if run_dir is None:
        console.print("[red]no runs found[/] — `jeval run` first")
        return 1
    agg = json.loads((run_dir / "results.json").read_text())
    baseline = report._load_baseline() if args.baseline else None
    console.print(report.render_md(agg, baseline))
    if args.fail_on_regression:
        regs = report.regressions(agg, baseline)
        if regs:
            console.print("[red bold]REGRESSION[/]")
            for r in regs:
                console.print(f"  [red]{r}[/]")
            return 1
        console.print("[green]no regressions vs baseline[/]")
    return 0


def _cmd_baseline(args) -> int:
    """`jeval baseline` — freeze a run's scores as the new reference (updates
    datasets/baseline.json + RESULTS.md). `--show` just prints the current one."""
    if args.show:
        console.print(report.render_baseline_md())
        return 0
    run_dir = _resolve_run(args.run_dir)
    if run_dir is None:
        console.print("[red]no runs found[/]")
        return 1
    report.promote_baseline(run_dir)
    (report.REPO_ROOT / "RESULTS.md").write_text(report.render_baseline_md())
    console.print(f"[green]baseline + RESULTS.md updated[/] from {run_dir.name}")
    return 0


def _cmd_embcompare(args) -> int:
    """`jeval embcompare <model>` — offline BEIR score for a candidate
    embedding model, next to the deployed baseline."""
    from jarvis_eval import embcompare
    suites = args.suite.split(",") if args.suite else ["beir_scifact", "beir_nfcorpus"]
    embcompare.run(args.model, args.dims, suites)
    return 0


def _cmd_rerankcompare(args) -> int:
    """`jeval rerankcompare <reranker>` — offline dense-vs-reranked retrieval
    on beir_* / hotpotqa, next to the committed baseline."""
    from jarvis_eval import rerankcompare
    suites = args.suite.split(",") if args.suite else ["beir_scifact", "beir_nfcorpus", "hotpotqa"]
    rerankcompare.run(args.reranker, suites, args.fetch, args.backend, args.max_cost)
    return 0


def _cmd_hotpotsup(args) -> int:
    """`jeval hotpotsup <model>` — sentence-level Sup EM/F1 for HotpotQA
    (leaderboard metric) via an LLM supporting-fact selector."""
    from jarvis_eval import hotpotsup
    ctxs = (["distractor", "dense", "rerank"] if args.context == "all"
            else args.context.split(","))
    hotpotsup.run(args.model, ctxs, args.topk, args.rerank_model, args.max_cost)
    return 0


def _resolve_run(arg):
    """A run dir given on the CLI, or the newest one under results/."""
    if arg:
        p = Path(arg)
        return p if (p / "results.json").exists() else None
    return report.latest_run()


def main() -> None:
    p = argparse.ArgumentParser(prog="jeval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # each add_parser("<name>") declares a subcommand; set_defaults(fn=...)
    # attaches the handler that main() calls at the end.
    sub.add_parser("setup").set_defaults(fn=_cmd_setup)

    s = sub.add_parser("seed")
    s.add_argument("benchmark", choices=BENCH)
    s.set_defaults(fn=_cmd_seed)

    r = sub.add_parser("run")
    r.add_argument("--suite", required=True,
                   help=f"one, several comma-separated, or 'all' of: {', '.join(BENCH)}")
    r.add_argument("--model", default=None, help="override RUNNER_MODEL")
    r.set_defaults(fn=_cmd_run)

    rp = sub.add_parser("report")
    rp.add_argument("run_dir", nargs="?", default=None)
    rp.add_argument("--baseline", action="store_true")
    rp.add_argument("--fail-on-regression", action="store_true")
    rp.set_defaults(fn=_cmd_report)

    b = sub.add_parser("baseline")
    b.add_argument("run_dir", nargs="?", default=None)
    b.add_argument("--show", action="store_true", help="print the current scoreboard, don't promote")
    b.set_defaults(fn=_cmd_baseline)

    e = sub.add_parser("embcompare")
    e.add_argument("model", help="e.g. openai/text-embedding-3-large")
    e.add_argument("--dims", type=int, default=None, help="request this many dimensions")
    e.add_argument("--suite", default=None, help="comma list of BEIR suites (default: both)")
    e.set_defaults(fn=_cmd_embcompare)

    rr = sub.add_parser("rerankcompare")
    rr.add_argument("reranker", help="openrouter: a chat model id (e.g. google/gemini-2.5-flash); "
                                     "local: an HF cross-encoder (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2)")
    rr.add_argument("--backend", choices=["openrouter", "local"], default="openrouter",
                    help="openrouter = listwise LLM rerank (default); local = CPU cross-encoder")
    rr.add_argument("--suite", default=None,
                    help="comma list of beir_scifact/beir_nfcorpus/hotpotqa (default: all three)")
    rr.add_argument("--fetch", type=int, default=None,
                    help="dense candidates reranked per query (default: 20 openrouter / 100 local)")
    rr.add_argument("--max-cost", type=float, default=3.0, dest="max_cost",
                    help="openrouter: stop reranking once spend passes this (USD)")
    rr.set_defaults(fn=_cmd_rerankcompare)

    hs = sub.add_parser("hotpotsup")
    hs.add_argument("model", help="chat model that selects supporting sentences, e.g. google/gemini-2.5-flash")
    hs.add_argument("--context", default="all",
                    help="distractor | dense | rerank | all  (comma list ok; default all)")
    hs.add_argument("--topk", type=int, default=5, help="paragraphs shown for dense/rerank contexts")
    hs.add_argument("--rerank-model", default="google/gemini-2.5-flash", dest="rerank_model",
                    help="listwise reranker model for --context rerank")
    hs.add_argument("--max-cost", type=float, default=2.0, dest="max_cost", help="USD budget cap")
    hs.set_defaults(fn=_cmd_hotpotsup)

    args = p.parse_args()
    sys.exit(args.fn(args))     # args.fn is the _cmd_* picked by the subcommand


if __name__ == "__main__":
    main()
