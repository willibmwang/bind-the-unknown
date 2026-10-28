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
    p.add_argument("--limit", type=int,
                   help="Cap the probes scored, sampled rather than truncated. The cap is "
                        "what bounds inference spend on an agent policy")
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("orbit", help="Split a score into earned skill and supplied identity")
    p.add_argument("--family", default="coronaviridae")
    p.add_argument("--policy", default="rubric")
    p.add_argument("--count", type=int, default=4, help="Group elements per kind")
    p.add_argument("--attempts", type=int, default=400)
    p.add_argument("--limit", type=int,
                   help="Cap the probes scored per group element; bounds inference spend")
    p.add_argument("--kinds",
                   help="Comma-separated nuisance kinds to run, e.g. null,axis_order. "
                        "Defaults to all of them; naming a subset is how an expensive policy "
                        "buys one transformation instead of four")
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("improve", help="Search instrument variants behind the guard")
    p.add_argument("--family", default="coronaviridae")
    p.add_argument("--attempts", type=int, default=250)
    p.add_argument("--no-tests", action="store_true",
                   help="Skip the in-process invariant suite, which the guard otherwise runs")
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("intake",
                       help="Pathogen genome accession -> ranked candidate binding targets")
    p.add_argument("--accession", default="NC_045512.2",
                   help="NCBI nucleotide accession for the genome")
    p.add_argument("--exclude-organism", dest="exclude_organism",
                   help="Remove this organism from the reference before identifying, which "
                        "is how a genome whose proteins are already curated is scored as if "
                        "it were novel")
    p.add_argument("--top", type=int, default=12, help="Longest ORFs to identify")
    p.add_argument("--no-decompose", action="store_true",
                   help="Rank whole ORFs without expanding polyproteins into mature chains")
    p = sub.add_parser("structures",
                       help="Fetch per-target structural evidence with deposition dates")
    p.add_argument("--coverage", type=float, default=0.5,
                   help="Fraction of a declared range a deposited chain must cover to count")
    p = sub.add_parser("monitor", help="Audit which readouts can still tell two policies apart")
    p.add_argument("--family", action="append", dest="packets",
                   help="Restrict the audit to one packet; repeatable")
    p.add_argument("--attempts", type=int, default=300)
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("monomers", help="Hold monomers out of training and score the transfer")
    p.add_argument("--landscape", default="thrombin_cyclic")
    p.add_argument("--held", type=int, help="One holdout size; omit to sweep the configured set")
    p.add_argument("--no-controls", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("sweep", help="Run the acquisition grid in parallel on this machine")
    p.add_argument("--arms", default="binding,target,method,display")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-pools", type=int)
    p.add_argument("--no-controls", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("manifest", help="Rewrite the source digest record, or report its drift")
    p.add_argument("--check", action="store_true", help="Report drift and exit non-zero; write nothing")
    p = sub.add_parser("tools", help="List the tool registry, or snapshot and check it")
    p.add_argument("--snapshot", action="store_true",
                   help="Write configs/tools.v{version}.json; refuses to overwrite one")
    p.add_argument("--check", action="store_true",
                   help="Report drift against the committed snapshot and exit non-zero")
    p = sub.add_parser("selector",
                       help="Rank which tool to call next for a stated situation")
    p.add_argument("--known", nargs="*", default=["genome_accession"],
                   help="Facts the state holds; defaults to a bare genome accession, which is "
                        "the state a campaign actually starts in")
    p.add_argument("--unresolved", nargs="*", default=None,
                   help="Axes with no sufficient claim yet; defaults to all seven")
    p.add_argument("--proposed", action="store_true",
                   help="Include the declared-but-unbuilt lanes, to ask what building them "
                        "would buy. Nothing dispatches from them and the output says so")
    p = sub.add_parser("exercise",
                       help="Call each registered tool across its declared grid and record "
                            "what the call spent")
    p.add_argument("--tool", nargs="*", default=None, dest="tools_",
                   help="Which tools to exercise; defaults to every tool with a declared "
                        "grid. Cannot select which cells run")
    p.add_argument("--repeats-scale", type=int, default=1,
                   help="Multiply the per-cell repeats of the cells expected to succeed. "
                        "Scale 1 is sized to clear the cost floor exactly")
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("optimize",
                       help="Evaluate spec-edit proposals behind the guard; propose only")
    p.add_argument("--runs", default="runs/toolcalls",
                   help="Artifact tree a proposal must cite. Defaults to the tool-call corpus "
                        "`exercise` writes, because that is the only tree whose trace rows "
                        "name a tool: pointed at runs/emergency the report read 'no call to "
                        "this tool is on the record' with 23 instrumented rows on disk. Pass "
                        "runs/emergency for the episode view of the same fields")
    p.add_argument("--gap", action="store_true",
                   help="Report which proposable fields the record is silent about, and stop")
    p.add_argument("--smoke", action="store_true")
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
                  now=args.now, limit=args.limit)
    elif args.command == "orbit":
        from .invariance import run_orbit
        from .invariance import KINDS
        kinds = tuple(k for k in (args.kinds or "").split(",") if k) or KINDS
        unknown = set(kinds) - set(KINDS)
        if unknown:
            raise SystemExit(f"Unknown nuisance kinds {sorted(unknown)}; known kinds are "
                             f"{list(KINDS)}")
        run_orbit(args.family, args.policy, count=args.count, attempts=args.attempts,
                  smoke=args.smoke, limit=args.limit, kinds=kinds)
    elif args.command == "improve":
        from .improve import run_improve
        run_improve(args.family, attempts=args.attempts, run_tests=not args.no_tests,
                    smoke=args.smoke)
    elif args.command == "intake":
        from .intake import decompose, run_intake
        import json as _json
        payload = run_intake(args.accession, exclude_organism=args.exclude_organism,
                             top=args.top)
        rows = payload["targets"] if args.no_decompose else decompose(payload["targets"])
        # Two axes, not one. `score` is worth-binding and `conservation` is how-sure-we-are
        # what this is; `tier` says which identifier produced it. A row may hold a high score
        # on a remote or hydropathy-only assignment, and that is the row a reader has to be
        # able to see without opening `features`.
        print(_json.dumps([
            {"rank": i, "target": r.get("name"), "score": r.get("score"),
             "conservation": r.get("conservation"), "tier": r.get("tier"),
             "nearest_homolog": r.get("nearest_homolog"),
             "features": r.get("features")}
            for i, r in enumerate(rows[:args.top], 1) if r.get("identified")], indent=2))
    elif args.command == "structures":
        from .structures_evidence import build
        import json as _json
        payload = build(minimum=args.coverage)
        print(_json.dumps({t: {k: r[k] for k in ("accession", "structures",
                                                 "earliest_deposition", "best_resolution")}
                           for t, r in payload["targets"].items()}, indent=2))
    elif args.command == "monitor":
        from .monitor import run_monitor
        run_monitor(args.packets, attempts=args.attempts, smoke=args.smoke)
    elif args.command == "monomers":
        from .monomers import run_monomers, transfer
        if args.held:
            transfer(args.landscape, args.held)
        else:
            run_monomers(args.landscape, controls=not args.no_controls, smoke=args.smoke)
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
    elif args.command == "tools":
        from .tools import TOOL_SET_VERSION, TOOLS, snapshot, spec_set_digest
        from .common import ROOT, write_json
        path = ROOT / "hpc_pilots" / "configs" / f"tools.v{TOOL_SET_VERSION}.json"
        if args.snapshot:
            # Refuses to overwrite, because the snapshot is the thing that makes an already
            # recorded tool_set_sha256 verifiable. A changed spec goes in a new file.
            if path.exists():
                raise SystemExit(f"{path.name} exists and is byte-immutable; bump "
                                 f"TOOL_SET_VERSION and snapshot again")
            write_json(path, snapshot())
            print(json.dumps({"wrote": str(path), "tool_set_sha256": spec_set_digest()},
                             indent=2))
        elif args.check:
            committed = json.loads(path.read_text())["tool_set_sha256"] if path.exists() else None
            live = spec_set_digest()
            print(json.dumps({"snapshot": str(path), "committed": committed, "live": live,
                              "drift": committed != live}, indent=2))
            if committed != live:
                raise SystemExit(f"{path.name} does not describe the registry")
        else:
            print(json.dumps({name: {"summary": TOOLS[name].summary,
                                     "version": TOOLS[name].version,
                                     "return_kind": TOOLS[name].return_kind,
                                     "produces_evidence": TOOLS[name].produces_evidence,
                                     "parameters": [q.name for q in TOOLS[name].parameters]}
                              for name in sorted(TOOLS)}, indent=2))
    elif args.command == "selector":
        from .selector import AXES, PROPOSED, REQUIREMENTS, Situation, rank
        # Merged rather than substituted, so asking what an unbuilt lane would buy still ranks
        # it against the tools that exist. `includes_unbuilt_lanes` is in the output because a
        # reader who missed the flag would otherwise read a plan naming capabilities no action
        # can reach, which is the one way this command could mislead.
        requirements = {**REQUIREMENTS, **PROPOSED} if args.proposed else None
        situation = Situation(
            known=frozenset(args.known),
            unresolved=tuple(AXES if args.unresolved is None else args.unresolved))
        ordered = rank(situation, requirements=requirements)
        print(json.dumps(
            {"known": sorted(situation.known), "unresolved": list(situation.unresolved),
             "includes_unbuilt_lanes": bool(args.proposed),
             "first": next((p.tool for p in ordered if p.callable_now), None),
             "order": [{"tool": p.tool, "callable_now": p.callable_now,
                        "blocked_on": list(p.blocked_on), "unlocks": list(p.unlocks),
                        "gates_unresolved": list(p.gates_unresolved),
                        "why": p.why, "scope": p.scope} for p in ordered]}, indent=2))
    elif args.command == "exercise":
        from .exercise import run_exercise
        run_exercise(tools_=args.tools_, repeats_scale=args.repeats_scale, smoke=args.smoke)
    elif args.command == "optimize":
        from .optimizer import instrumentation_gap, run_optimize
        if args.gap:
            print(json.dumps(instrumentation_gap(args.runs), indent=2))
        else:
            run_optimize(args.runs, smoke=args.smoke)
    else:
        from .summarize import summarize
        summarize()


if __name__ == "__main__":
    main()
