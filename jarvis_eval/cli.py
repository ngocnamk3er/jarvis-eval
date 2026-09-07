"""jeval — command line for the Jarvis benchmark harness.

    jeval setup                       create/enable the eval Keycloak users
    jeval seed <benchmark>            download + upload + embed a benchmark's corpus
    jeval run --suite <benchmark|all> [--model ID]
    jeval report [RUN_DIR] [--baseline] [--fail-on-regression]
    jeval baseline [RUN_DIR]          promote a run's results.json to datasets/baseline.json

benchmarks: beir_scifact · beir_nfcorpus · hotpotqa · gaia
"""
import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

from jarvis_eval import report
from jarvis_eval.benchmarks import BENCHMARKS
from jarvis_eval.clients.auth import ensure_user
from jarvis_eval.config import settings

console = Console()
BENCH = list(BENCHMARKS)


def _bench_user(name: str) -> str:
    return f"{settings.EVAL_USERNAME}-{name.replace('_', '-')}"


def _cmd_setup(_args) -> int:
    console.print(f"[green]{settings.EVAL_USERNAME}[/] — sub [bold]{ensure_user()}[/]")
    for b in BENCH:
        u = _bench_user(b)
        console.print(f"[green]{u}[/] — sub [bold]{ensure_user(u)}[/]")
    return 0


def _cmd_seed(args) -> int:
    u = _bench_user(args.benchmark)
    ensure_user(u)
    console.rule(f"seed {args.benchmark}  (user {u})")
    console.print(BENCHMARKS[args.benchmark][0]())
    return 0


def _cmd_run(args) -> int:
    which = BENCH if args.suite == "all" else [args.suite]
    if args.model:
        settings.RUNNER_MODEL = args.model

    all_rows: list[dict] = []
    for suite in which:
        ensure_user(_bench_user(suite))
        console.rule(f"{suite}")
        rows = BENCHMARKS[suite][1](None, repeats=1)
        for r in rows:
            err = r["meta"].get("error")
            nums = {k: round(v, 3) for k, v in r["metrics"].items() if isinstance(v, (int, float))}
            console.print(f"  {'[red]ERR[/]' if err else '[green]ok[/]'} {r['case_id']}  "
                          f"{json.dumps(nums)}" + (f"  [red]{err}[/]" if err else ""))
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
    if arg:
        p = Path(arg)
        return p if (p / "results.json").exists() else None
    return report.latest_run()


def main() -> None:
    p = argparse.ArgumentParser(prog="jeval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("setup").set_defaults(fn=_cmd_setup)

    s = sub.add_parser("seed")
    s.add_argument("benchmark", choices=BENCH)
    s.set_defaults(fn=_cmd_seed)

    r = sub.add_parser("run")
    r.add_argument("--suite", choices=[*BENCH, "all"], required=True)
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
