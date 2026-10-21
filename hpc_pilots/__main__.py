import argparse
import json
import os

from .arms import ARMS
from .common import config


def main(argv=None):
    """`argv` is accepted so the parser can be exercised without a subprocess."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    p = sub.add_parser("binding")
    p.add_argument("--pool", type=int, default=0)
    p.add_argument("--policy", choices=config()["binding"]["policies"], default="greedy")
    p.add_argument("--delay", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--control", choices=["permuted"],
                   help="Falsification arm: shuffle outcomes, then grade against the shuffle")
    p = sub.add_parser("binding-array")
    p.add_argument("--agent-only", action="store_true")
    p = sub.add_parser("preprocess-atlas")
    p.add_argument("--index", type=int, default=1)
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("structure")
    p.add_argument("--policy", choices=config()["structure"]["policies"], default="uniform")
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("structure-array")
    p.add_argument("--agent-only", action="store_true")
    sub.add_parser("calibration")
    sub.add_parser("summarize")
    # Domain layer: building-block registry, family evidence packets, pathogen landscapes
    # and the emergency decision episodes that consume the dossier state.
    sub.add_parser("chemistry")
    sub.add_parser("targets")
    # One parser pair per staged arm, generated from the registry. These were written out
    # by hand three times and the copies had already drifted -- `target` declared --control
    # without the default its neighbours carried -- which is the whole argument for not
    # writing them a fourth time.
    for name, spec in ARMS.items():
        if not spec.per_landscape:
            continue
        q = sub.add_parser(f"prepare-{name}", help=f"Stage one {name} landscape")
        q.add_argument("--landscape", required=True)
        q = sub.add_parser(name, help=f"Replay one acquisition policy against a {name} landscape")
        q.add_argument("--landscape", required=True)
        q.add_argument("--pool", type=int, default=0)
        q.add_argument("--policy", default="greedy")
        q.add_argument("--delay", type=int, default=0)
        q.add_argument("--smoke", action="store_true")
        q.add_argument("--control", choices=["permuted"], default=None,
                       help="Falsification arm: shuffle outcomes, then grade against the shuffle")
    p = sub.add_parser("emergency")
    p.add_argument("--family", default="coronaviridae")
    p.add_argument("--policy", default="rubric")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--control", choices=["zero_retrieval"],
                   help="Memorisation probe: forbid retrieval and push no evidence")
    p.add_argument("--repeat", type=int, default=1,
                   help="Independent episodes, each to its own artifact")
    # The decision-arm instruments. `twins` grades one decision per state against the floor
    # and ceiling the evaluator computes for it; `orbit` runs the same probe over a certified
    # nuisance group and splits the score into the part the structure supports and the part
    # the target's name supplied.
    p = sub.add_parser("twins", help="Grade one-step decisions over matched states")
    p.add_argument("--family", default="coronaviridae")
    p.add_argument("--policy", default="rubric")
    p.add_argument("--attempts", type=int, default=400,
                   help="Candidate states to sample before pairing")
    p.add_argument("--round", type=int, default=3, dest="now",
                   help="Decision round the probe is posed at")
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("orbit", help="Split a score into earned skill and supplied identity")
    p.add_argument("--family", default="coronaviridae")
    p.add_argument("--policy", default="rubric")
    p.add_argument("--count", type=int, default=4, help="Group elements per kind")
    p.add_argument("--attempts", type=int, default=400)
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("sweep", help="Run the acquisition grid in parallel on this machine")
    p.add_argument("--arms", default="binding,target,method,display")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-pools", type=int)
    p.add_argument("--no-controls", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("manifest", help="Rewrite the source digest record, or report its drift")
    p.add_argument("--check", action="store_true", help="Report drift and exit non-zero; write nothing")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from .data import prepare_binding, prepare_atlas_manifest
        prepare_binding()
        prepare_atlas_manifest()
    elif args.command in ("binding", "binding-array"):
        from .binding import run_binding
        if args.command == "binding-array":
            bc = config()["binding"]
            policies = ["agent"] if args.agent_only else [p for p in bc["policies"] if p != "agent"]
            tasks = [(pool, policy, delay) for pool in range(bc["evaluation_pools"])
                     for delay in bc["feedback_delays"] for policy in policies]
            run_binding(*tasks[int(os.environ["SLURM_ARRAY_TASK_ID"])])
        else:
            run_binding(args.pool, args.policy, args.delay, args.smoke, args.control)
    elif args.command == "preprocess-atlas":
        from .structure import preprocess
        preprocess(args.index, args.smoke)
    elif args.command in ("structure", "structure-array"):
        from .structure import run_structure
        if args.command == "structure-array":
            policies = ["agent"] if args.agent_only else [p for p in config()["structure"]["policies"] if p != "agent"]
            run_structure(policies[int(os.environ["SLURM_ARRAY_TASK_ID"])] )
        else:
            run_structure(args.policy, args.smoke)
    elif args.command == "calibration":
        from .calibration import run_calibration
        run_calibration()
    elif args.command == "chemistry":
        from .chemistry import build_registry
        build_registry()
    elif args.command == "targets":
        from .targets import build_packets
        build_packets()
    elif args.command.startswith("prepare-") and args.command[8:] in ARMS:
        from .data import prepare_display, prepare_method, prepare_target
        {"target": prepare_target, "method": prepare_method,
         "display": prepare_display}[args.command[8:]](args.landscape)
    elif args.command in ARMS:
        from .binding import run_arm
        run_arm(args.command, args.landscape, args.pool, args.policy, args.delay,
                args.smoke, args.control)
    elif args.command == "emergency":
        from .emergency import run_emergency
        run_emergency(args.family, args.policy, args.smoke, args.control, args.repeat)
    elif args.command == "twins":
        from .twins import run_twins
        run_twins(args.family, args.policy, attempts=args.attempts, smoke=args.smoke,
                  now=args.now)
    elif args.command == "orbit":
        from .invariance import run_orbit
        run_orbit(args.family, args.policy, count=args.count, attempts=args.attempts,
                  smoke=args.smoke)
    elif args.command == "sweep":
        from .sweep import run_sweep
        run_sweep([a for a in args.arms.split(",") if a], args.workers,
                  not args.no_controls, args.max_pools, args.dry_run)
    elif args.command == "manifest":
        from .common import source_manifest
        path, drift = source_manifest(check=args.check)
        print(json.dumps({"manifest": str(path), "wrote": not args.check, **drift}, indent=2))
        if args.check and any(drift.values()):
            raise SystemExit("source_manifest.json does not describe the working tree")
    else:
        from .summarize import summarize
        summarize()


if __name__ == "__main__":
    main()
