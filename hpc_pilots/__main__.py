import argparse
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
            run_binding(args.pool, args.policy, args.delay, args.smoke)
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
    else:
        from .summarize import summarize
        summarize()


if __name__ == "__main__":
    main()
