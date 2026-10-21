"""Run the decision-arm grid off this laptop, keeping the provenance that makes it readable.

The decision arm became the expensive part of this project without anyone deciding that it
should. A single `run_twins` call samples several hundred candidate states, and grading one
state forks every dossier once per offered claim and recomputes the lead verdict on each fork;
an orbit multiplies that by the group elements, and the audits multiply it again by the
policies. None of it is numerically heavy and all of it is serial Python over small objects,
so it saturates a laptop's ten cores while the test suite is trying to use them.

It is also embarrassingly parallel over `(packet, policy, group element)` and -- the useful
part -- **needs no staged data at all**. The twin probes, the orbits and the readout audits run
entirely off the evidence packets in `targets.py`, which are source code. Only the acquisition
replay needs the 176MB under `data/processed`, and 166MB of that is one method landscape. So
the image for the decision arm is the twelve hundred kilobytes of this package, which builds in
seconds and is cheap to rebuild on every change.

**Why a remote cell is not simply a faster local cell.** Closing the provenance gap in the
run records turned up something that governs this: of 160 recorded GB1 replays re-run on
another machine, 154 reproduced exactly and six did not, all six in the two forest-backed
policies, with numpy, scipy and scikit-learn all matching the lock file. The divergence is
below the version pin -- floating-point reduction order differing across machines, flipping a
split or a tie. So a result computed elsewhere is a result from a different host, and the
honest handling is the one the project already chose: pin the same requirements so the machine
is the only thing that differs, record the host in every artifact, and never pool across hosts
without reading it. `common.provenance` already stamps host, platform, Python version, commit
and a dirty flag onto everything, so a remote artifact self-identifies; `verify` below is the
check that the lane itself is sound, run on a cell whose answer is known locally.

Nothing here writes inside the container. A container's filesystem goes away, so a remote cell
returns its result and the local driver writes it to the path it would have had, which also
means a remote sweep and a local one leave the same tree behind.

**This is a Modal script, not part of the scientific CLI.** It imports `modal`, which the
scientific environment deliberately does not have -- that environment is kept able to resolve
without CUDA pins and its smoke suite runs in under three seconds. Drive it with an
interpreter that has Modal installed:

    modal run hpc_pilots/offload.py::decision_grid --packets all --policies rubric,scripted
    modal run hpc_pilots/offload.py::verify

A GPU image is defined but deliberately unused by anything here, because nothing in this
repository is a GPU workload: the surrogate is a sixty-four-tree forest on at most 128 rows and
the policy is an HTTP call to a hosted model. It exists for the one thing that would be --
an in-context surrogate doing torch inference per acquisition round -- and `surrogates.py`
records what such a model would need so the dependency is named rather than discovered.
"""
import json
import pathlib

import modal

APP = "bind-the-unknown"
app = modal.App(APP)

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent

# The same pinned scientific environment, so the machine is the only difference between a
# remote cell and a local one. Installing anything else here would make a divergence
# impossible to attribute.
_base = (modal.Image.debian_slim(python_version="3.12")
         .pip_install_from_requirements(str(HERE / "requirements.lock.txt"))
         .add_local_dir(str(HERE), "/root/hpc_pilots", copy=True,
                        ignore=["__pycache__", "*.pyc"]))

# The decision arm reads no staged data. Kept separate from the image that does, so a change
# to a landscape does not invalidate the layer the decision grid runs on.
DECISION_IMAGE = _base

# The monomer transfer needs one small landscape and its descriptor matrix: 1.6MB against the
# 166MB the method landscapes would add.
MONOMER_IMAGE = _base.add_local_dir(
    str(ROOT / "data/processed/display"), "/root/data/processed/display", copy=True)

# Defined and unused. See the module docstring: there is no GPU workload here yet.
GPU_IMAGE = _base.pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cpu")


def _run(kind, arguments):
    """Dispatch one cell by name, inside the container.

    A name rather than a callable, because a callable would have to be pickled out of this
    process and the point of the registry pattern elsewhere in this package is that what
    varies between runs is data. The import is inside the function so the module can be read
    by a driver that has Modal but not scikit-learn.
    """
    if kind == "twins":
        from hpc_pilots.twins import run_twins
        return run_twins(**arguments)
    if kind == "orbit":
        from hpc_pilots.invariance import run_orbit
        return run_orbit(**arguments)
    if kind == "monitor":
        from hpc_pilots.monitor import run_monitor
        return run_monitor(**arguments)
    if kind == "monomers":
        from hpc_pilots.monomers import transfer
        return transfer(**arguments)
    raise ValueError(f"Unknown cell kind {kind!r}")


@app.function(image=DECISION_IMAGE, cpu=2.0, timeout=3600, max_containers=24)
def decision_cell(spec):
    kind, arguments = spec
    return {"kind": kind, "arguments": arguments, "result": _run(kind, arguments)}


@app.function(image=MONOMER_IMAGE, cpu=4.0, timeout=3600, max_containers=12)
def monomer_cell(spec):
    kind, arguments = spec
    return {"kind": kind, "arguments": arguments, "result": _run(kind, arguments)}


def _write(payload):
    """Write a returned result to the path the cell would have written locally.

    Imported here rather than at module scope because the driver's interpreter is the one with
    Modal, not necessarily the one with the scientific stack; this function only needs the
    path convention and an atomic write.
    """
    import sys

    sys.path.insert(0, str(ROOT))
    from hpc_pilots.common import write_json

    kind, result = payload["kind"], payload["result"]
    if kind == "twins":
        name = f"{result['family_packet']}_{result['policy']}"
        relative = f"runs/twins/{name}.json"
    elif kind == "orbit":
        relative = f"runs/orbit/{result['family_packet']}_{result['policy']}.json"
    elif kind == "monitor":
        relative = "runs/monitor/ledger.json"
    else:
        relative = (f"runs/monomers/{result['landscape']}_"
                    f"held{result['held_out_count']:02d}.json")
    write_json(ROOT / relative, result)
    return relative


@app.local_entrypoint()
def decision_grid(packets: str = "all", policies: str = "rubric,scripted",
                  attempts: int = 400, orbit: bool = False, elements: int = 2):
    """Fan the decision grid out over containers and write every result back locally.

    `orbit=True` runs the nuisance orbits instead of single probes, which is the same work
    multiplied by the group elements and is the case worth moving off a laptop.
    """
    names = (["coronaviridae", "orthomyxoviridae", "henipavirus"]
             if packets == "all" else [p for p in packets.split(",") if p])
    chosen = [p for p in policies.split(",") if p]
    cells = [("orbit", {"name": n, "policy": p, "count": elements, "attempts": attempts})
             if orbit else
             ("twins", {"name": n, "policy": p, "attempts": attempts})
             for n in names for p in chosen]
    print(json.dumps({"cells": len(cells), "orbit": orbit, "app": APP}))
    written = []
    for payload in decision_cell.map(cells):
        written.append(_write(payload))
        print(json.dumps({"wrote": written[-1]}))
    print(json.dumps({"written": len(written)}))


@app.local_entrypoint()
def monomer_grid(landscape: str = "thrombin_cyclic", counts: str = "2,4,6,9,12",
                 replicates: int = 5):
    """Fan the monomer-wise holdout out over containers, one cell per draw.

    The unit of analysis is the held-out *set*, so the draws are the thing being parallelised:
    each is an independent observation and they were costing five forest fits apiece in series.
    """
    import sys

    sys.path.insert(0, str(ROOT))
    from hpc_pilots.streams import stream_int

    sizes = [int(c) for c in counts.split(",") if c]
    cells = [("monomers", {"landscape": landscape, "held_count": size,
                           "seed": stream_int(20261021, "holdout-replicate", replicate),
                           "record": False})
             for size in sizes for replicate in range(replicates)]
    print(json.dumps({"cells": len(cells), "sizes": sizes, "replicates": replicates}))
    rows = []
    for payload in monomer_cell.map(cells):
        result = payload["result"]
        rows.append({"held_out_count": result["held_out_count"],
                     "held_out_monomers": result["held_out_monomers"],
                     "host": result["provenance"]["host"],
                     "by_cell": {f"{r['encoder']}/{r['model']}":
                                 {"held_out_spearman": r["held_out_spearman"],
                                  "tied_rate": r["paired"]["tied_rate"],
                                  "concordance": r["paired"]["concordance"]}
                                 for r in result["results"]}})
    print(json.dumps({"draws": len(rows)}, indent=2))
    return rows


@app.local_entrypoint()
def verify():
    """Check the lane before trusting it, on a cell whose answer is known locally.

    A deterministic policy on a fixed packet must return the same attainment remotely as it
    does here. If it does not, the difference is the host -- which is exactly what the
    re-run of the 160 GB1 replays found for the forest-backed policies -- and the result
    belongs in its own file rather than pooled with local ones. The check prints both hosts so
    an equal answer is still attributable.
    """
    import sys

    sys.path.insert(0, str(ROOT))
    from hpc_pilots.twins import run_twins

    arguments = {"name": "henipavirus", "policy": "rubric", "attempts": 200}
    here = run_twins(**arguments)
    there = decision_cell.remote(("twins", arguments))["result"]

    def summarise(result):
        return {family: result["summary"][family]["attainment"]
                for family in ("leader_best", "challenger_best")}

    print(json.dumps({
        "local": {"host": here["provenance"]["host"], "attainment": summarise(here),
                  "pairs": here["pairs"]},
        "remote": {"host": there["provenance"]["host"], "attainment": summarise(there),
                   "pairs": there["pairs"]},
        "identical": summarise(here) == summarise(there) and here["pairs"] == there["pairs"],
        "note": "An equal answer does not make the hosts interchangeable. Six of 160 recorded "
                "forest-backed replays failed to reproduce across machines below the version "
                "pin, so every artifact carries its host and none are pooled without it."},
        indent=2))
