"""jeval — command line for the Jarvis eval harness.

    jeval setup                     create/enable the `eval` Keycloak user
    jeval seed                      wipe + re-upload datasets/corpus, wait for indexing
    jeval run --suite retrieval|rag_qa|agent_tasks|smoke|all [--repeats N] [--model ID]
    jeval report [RUN_DIR] [--baseline] [--fail-on-regression]
    jeval baseline [RUN_DIR]        promote a run's results.json to datasets/baseline.json
"""
import argparse
import json
import sys

from rich.console import Console

from jarvis_eval import report
from jarvis_eval.config import settings
from jarvis_eval.dataset import SUITES, load, smoke

console = Console()


def _cmd_setup(args) -> int:
    from jarvis_eval.benchmarks import BENCHMARKS
    from jarvis_eval.clients.auth import ensure_user
    console.print(f"[green]{settings.EVAL_USERNAME}[/] — sub [bold]{ensure_user()}[/]")
    if getattr(args, "all_benchmark_users", False):
        for b in BENCHMARKS:
            u = f"{settings.EVAL_USERNAME}-{b.replace('_', '-')}"
            console.print(f"[green]{u}[/] — sub [bold]{ensure_user(u)}[/]")
    return 0


def _cmd_seed(args) -> int:
    from jarvis_eval.benchmarks import BENCHMARKS
    from jarvis_eval.clients import files

    if args.benchmark:
        from jarvis_eval.clients.auth import ensure_user
        u = f"{settings.EVAL_USERNAME}-{args.benchmark.replace('_', '-')}"
        ensure_user(u)
        console.rule(f"seed {args.benchmark}  (user {u})")
        console.print(BENCHMARKS[args.benchmark][0]())
        return 0

    removed = files.wipe()
    console.print(f"wiped {removed} top-level node(s)")
    stats = files.seed_corpus()
    console.print(f"uploaded {stats['files']} files into {stats['folders']} folders; waiting for indexing…")
    done = files.wait_for_indexing()
    by_status: dict[str, int] = {}
    for e in done:
        by_status[e["indexing_status"]] = by_status.get(e["indexing_status"], 0) + 1
    console.print(f"[green]indexed[/]: {by_status}")
    return 0 if by_status.get("failed", 0) == 0 else 1


def _cmd_run(args) -> int:
    from jarvis_eval.benchmarks import BENCHMARKS
    from jarvis_eval.runners import RUNNERS

    if args.suite in ("all", "smoke"):
        which = list(SUITES)
    else:
        which = [args.suite]
    pick = smoke if args.suite == "smoke" else load
    repeats = args.repeats or settings.REPEATS
    if args.model:
        settings.RUNNER_MODEL = args.model

    all_rows: list[dict] = []
    for suite in which:
        if suite in BENCHMARKS:
            from jarvis_eval.clients.auth import ensure_user
            ensure_user(f"{settings.EVAL_USERNAME}-{suite.replace('_', '-')}")
            console.rule(f"{suite} (benchmark)")
            rows = BENCHMARKS[suite][1](None, repeats=1)
        else:
            cases = pick(suite)
            console.rule(f"{suite} · {len(cases)} case(s) × {repeats}")
            rows = RUNNERS[suite](cases, repeats=repeats)
        for r in rows:
            err = r["meta"].get("error")
            tag = "[red]ERR[/]" if err else "[green]ok[/]"
            console.print(f"  {tag} {r['case_id']}#{r['repeat']}  "
                          f"{json.dumps({k: round(v, 3) for k, v in r['metrics'].items() if isinstance(v, (int, float))})}"
                          + (f"  [red]{err}[/]" if err else ""))
        all_rows += rows

    run_dir = report.write_run(all_rows)
    console.print(f"\n[bold]wrote[/] {run_dir}/report.md")
    agg = json.loads((run_dir / "results.json").read_text())
    console.print(report.render_md(agg, report._load_baseline()))
    return 0


def _cmd_report(args) -> int:
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
    run_dir = _resolve_run(args.run_dir)
    if run_dir is None:
        console.print("[red]no runs found[/]")
        return 1
    report.promote_baseline(run_dir)
    console.print(f"[green]baseline updated[/] from {run_dir.name}")
    return 0


def _resolve_run(arg):
    from pathlib import Path
    if arg:
        p = Path(arg)
        return p if (p / "results.json").exists() else None
    return report.latest_run()


def main() -> None:
    p = argparse.ArgumentParser(prog="jeval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    from jarvis_eval.benchmarks import BENCHMARKS
    bench = list(BENCHMARKS)

    st = sub.add_parser("setup")
    st.add_argument("--all-benchmark-users", action="store_true",
                    help="also create eval-<benchmark> users")
    st.set_defaults(fn=_cmd_setup)

    s = sub.add_parser("seed")
    s.add_argument("--benchmark", choices=bench, default=None,
                   help="seed a standard benchmark's corpus instead of the hand corpus")
    s.set_defaults(fn=_cmd_seed)

    r = sub.add_parser("run")
    r.add_argument("--suite", choices=[*SUITES, *bench, "smoke", "all"], default="smoke")
    r.add_argument("--repeats", type=int, default=0)
    r.add_argument("--model", default=None, help="override RUNNER_MODEL")
    r.set_defaults(fn=_cmd_run)

    rp = sub.add_parser("report")
    rp.add_argument("run_dir", nargs="?", default=None)
    rp.add_argument("--baseline", action="store_true")
    rp.add_argument("--fail-on-regression", action="store_true")
    rp.set_defaults(fn=_cmd_report)

    b = sub.add_parser("baseline")
    b.add_argument("run_dir", nargs="?", default=None)
    b.set_defaults(fn=_cmd_baseline)

    args = p.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
