"""Command-line entry point: ``ashdpf simulate`` / ``ashdpf experiments``."""

from __future__ import annotations

import argparse
import json
import sys



def _simulate(args) -> int:
    from .config import ASHDPFConfig
    from .experiments import make_world
    from .simulation import build_ashdpf

    env, hist, events = make_world(args.seed, args.history, args.events,
                                   n_pipelines=args.pipelines, n_groups=args.groups)
    fw = build_ashdpf(env, hist, ASHDPFConfig(threshold=args.threshold, seed=args.seed))
    fw.run(events)
    report = fw.report()
    if args.audit:
        fw.executor.export_audit(args.audit)
    if args.graph:
        fw.kg.save(args.graph)
    print(json.dumps({k: round(v, 4) for k, v in report.items()}, indent=2))
    return 0


def _experiments(args) -> int:
    from .experiments import run_all

    res = run_all(n_events=args.events, n_history=args.history, seed=args.seed,
                  workers=args.workers, out_dir=args.out, n_pipelines=args.pipelines,
                  n_groups=args.groups, quick=args.quick)
    print(f"wrote {args.out}/RESULTS.md and results.json "
          f"({res['meta']['runtime_seconds']:.0f}s)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ashdpf", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("simulate", "experiments"):
        s = sub.add_parser(name)
        s.add_argument("--events", type=int, default=2000 if name == "simulate" else 10000)
        s.add_argument("--history", type=int, default=500)
        s.add_argument("--seed", type=int, default=42)
        s.add_argument("--pipelines", type=int, default=500)
        s.add_argument("--groups", type=int, default=50)
        if name == "simulate":
            s.add_argument("--threshold", type=float, default=0.40)
            s.add_argument("--audit", help="write the audit log as JSONL")
            s.add_argument("--graph", help="write the knowledge graph as JSON")
            s.set_defaults(func=_simulate)
        else:
            s.add_argument("--out", default="results")
            s.add_argument("--workers", type=int, default=2)
            s.add_argument("--quick", action="store_true", help="skip threshold sweep")
            s.set_defaults(func=_experiments)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
