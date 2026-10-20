import argparse
import json
import os

from .common import config


def main():
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
    p = sub.add_parser("prepare-target")
    p.add_argument("--landscape", required=True)
    p = sub.add_parser("target")
    p.add_argument("--landscape", required=True)
    p.add_argument("--pool", type=int, default=0)
    p.add_argument("--policy", default="greedy")
    p.add_argument("--delay", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--control", choices=["permuted"],
                   help="Falsification arm: shuffle outcomes, then grade against the shuffle")
    p = sub.add_parser("prepare-method")
    p.add_argument("--landscape", required=True)
    p = sub.add_parser("method")
    p.add_argument("--landscape", required=True)
    p.add_argument("--pool", type=int, default=0)
    p.add_argument("--policy", default="greedy")
    p.add_argument("--delay", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--control", choices=["permuted"],
                   help="Falsification arm: shuffle outcomes, then grade against the shuffle")
    p = sub.add_parser("emergency")
    p.add_argument("--family", default="coronaviridae")
    p.add_argument("--policy", default="rubric")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--control", choices=["zero_retrieval"],
                   help="Memorisation probe: forbid retrieval and push no evidence")
    p.add_argument("--repeat", type=int, default=1,
                   help="Independent episodes, each to its own artifact")
    p = sub.add_parser("manifest", help="Rewrite the source digest record, or report its drift")
    p.add_argument("--check", action="store_true", help="Report drift and exit non-zero; write nothing")
    args = parser.parse_args()
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
    elif args.command == "prepare-target":
        from .data import prepare_target
        prepare_target(args.landscape)
    elif args.command == "target":
        from .binding import run_target
        run_target(args.landscape, args.pool, args.policy, args.delay, args.smoke, args.control)
    elif args.command == "prepare-method":
        from .data import prepare_method
        prepare_method(args.landscape)
    elif args.command == "method":
        from .binding import run_method
        run_method(args.landscape, args.pool, args.policy, args.delay, args.smoke, args.control)
    elif args.command == "emergency":
        from .emergency import run_emergency
        run_emergency(args.family, args.policy, args.smoke, args.control, args.repeat)
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
