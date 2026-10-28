"""Dispatch each registered tool under a declared grid and record what the call spent.

`optimizer.py` is a guard set with no corpus. Measured over all 45 recorded emergency
artifacts: not one `trace[]` row carries a `tool` key, so every one of the eight `PROPOSABLE`
spec fields reads as uninstrumented and the module's own correct output is to propose nothing
and say which fields are silent. The gap was never in the optimiser. It is that nothing in the
package had ever called a tool often enough to say what one costs, and the one live call site
recorded a hardcoded success string and nothing else. This module is the instrument that closes
that gap, and it is deliberately not part of the optimiser.

Four properties make the corpus safe to point the optimiser at, and each is the reason for a
specific choice here rather than a remark about one.

**The grid is declared in source, not chosen by the thing being measured.** If the optimiser
could pick the arguments its own evidence came from, cost deflation would be one line: call the
cheapest cell twenty times and declare the median. `GRID` is a module constant, this module is
not imported by `optimizer.py`, and the optimiser opens these artifacts read-only.

**Failures are induced through each tool's own declared contract, never faked.** Every invalid
cell names the `Failure` it is meant to reproduce, and the prediction is written before the run.
A cell producing a different status, or a symptom no declared failure predicts, is reported as a
finding rather than reconciled -- the discipline `improve.py` states with its `trips` field. A
hand-written failure string would be an answer planted in the corpus, which the optimiser would
then "discover"; every string in these artifacts is one a tool actually emitted.

**The corpus is written where the optimiser cannot write.** `optimizer.run_optimize` writes only
under `runs/tools/`, pinned by a test that intercepts `write_json` and checks every path. This
module writes under `runs/toolcalls/`, a disjoint tree. Had the corpus lived under `runs/tools/`
that test would still have passed while the optimiser could write into its own grader, so the
separation is load-bearing and not tidiness.

**A charged refusal is not a measurement of work.** `chief.call` leaves `seconds` and
`network_calls` as None for anything refused before dispatch, and this module records the None
rather than a zero. A zero would enter the cost record as a call that took no time, and
`cost_seconds` is exactly the field such a zero drags downwards.

What this module does not do: it does not decide which tool to call, it does not propose
anything, and it does not touch a spec. It calls, measures, and writes down what happened.
"""
import json
from dataclasses import dataclass, field

from . import chief
from .common import ROOT, provenance, write_json
from .tools import TOOLS, spec_set_digest, tool

# Where the corpus goes. Disjoint from `runs/tools/`, which is the optimiser's output tree.
# See the module docstring: nesting them would have left the propose-only test passing while
# the optimiser could write into the corpus it grades itself against.
CORPUS = "runs/toolcalls"

# A wall budget wide enough that the budget check never fires, because a refusal before
# dispatch records no wall time and a grid whose cells were being refused would measure
# nothing. Set above the slowest measured call by a wide margin rather than tuned to it.
WALL_BUDGET_SECONDS = 600


@dataclass(frozen=True)
class Cell:
    """One call to make, and what this grid's author predicts it will do.

    The prediction is recorded before the run and is never edited to match the outcome. A
    mispredicted cell is the useful output of a grid, in the same way a planted hack caught by
    the wrong guard is the useful output of a planted set.
    """

    arguments: dict
    expects: str
    """One of `chief.STATUSES`. What this cell is predicted to produce."""
    reproduces: str = ""
    """The `Failure.symptom` this cell is meant to reproduce, for an invalid cell. Empty for a
    cell expected to succeed. Not a free-text note: it is matched against the spec."""
    why: str = ""
    """Why this cell is in the grid at all."""
    repeats: int = 1
    """How many times to call it. Only a cell expected to succeed needs more than one, because
    a cost claim needs a sample and a failure string is a string however often it is seen."""


# One entry per registered tool. The invalid cells are drawn from each spec's own declared
# `failures`, so the grid tests the spec against the tool rather than against an opinion about
# the tool. `repeats` sits on the valid cells only, and the totals are chosen to clear
# `optimizer.MINIMUM_CALLS_FOR_A_COST_CLAIM` exactly rather than generously: twenty calls of
# `structures.evidence` is already ~18 minutes and ~720 outbound requests against a public
# host, and a floor is a floor rather than a target.
GRID = {
    "chemistry.library_spec": (
        Cell({}, "ok", repeats=5, why="the default library, which is what the one live call "
                                      "site in emergency.py passes"),
        Cell({"initiator": "ClAc-D-Tyr"}, "ok", repeats=5,
             why="the default initiator named explicitly, to check the argument path and the "
                 "omitted path cost the same"),
        Cell({"random_region": (4, 8)}, "ok", repeats=5,
             why="a tuple argument, which is the kind a strict JSON schema carries worst"),
        Cell({"elongator_classes": ("canonical", "N-methyl")}, "ok", repeats=5,
             why="a narrowed alphabet, the cheapest end of the combinatorial count"),
        Cell({"initiator": "ClAc-Nonsense"}, "failed",
             reproduces="'<name>' is not a registered chloroacetyl initiator",
             why="the initiator table is narrower than the set of strings the schema admits"),
        Cell({"random_region": (9, 2)}, "failed",
             reproduces="Invalid randomised-region bounds",
             why="1 <= low <= high is not expressible in a portable JSON schema"),
        Cell({"elongator_classes": ("not-a-monomer-class",)}, "failed",
             reproduces="Empty elongator alphabet",
             why="every requested class unknown leaves nothing to draw from"),
    ),
    "structures.evidence": (
        Cell({}, "ok", repeats=10,
             why="the default coverage threshold; the expensive cell, ~53 s and 36 outbound "
                 "requests each"),
        Cell({"minimum": 0.5}, "ok", repeats=10,
             why="the default named explicitly, to check the declared default is the one in "
                 "force"),
        Cell({"minimum": 5.0}, "rejected",
             why="above the declared maximum. A bound a strict JSON schema cannot carry is "
                 "charged rather than raised, so this cell is the one that writes a "
                 "`rejected` row -- a pointer the optimiser licenses and nothing has ever "
                 "written"),
    ),
    # Distinct arguments, one call each, and the reason is a result rather than a style note.
    # The first version of this grid repeated two accessions ten times and measured 5.2 us per
    # call against a first call of 4.08 s -- a spread of 777,000x -- because
    # `surface_for_accession` memoises in process on a mutable default argument. Repeating an
    # argument against a memoised function measures the dict lookup, not the tool. A cost grid
    # has to defeat whatever cache the tool holds or it is an instrument pointed at the cache.
    #
    # Six accessions cannot clear the twenty-call floor, so no cost claim is possible for this
    # tool from this grid, and that is the honest outcome rather than a gap to paper over: a
    # tool whose answer is memoised on `(accession, interval, replicates)` has exactly as many
    # distinct real calls as the registry has declared targets.
    "geometry.surface_markers": (
        Cell({"accession": "P0DTD1"}, "ok",
             why="SARS-CoV-2 ORF1ab polyprotein; five measurable chains, a 762 square-angstrom "
                 "patch, dated 2020-11-17"),
        Cell({"accession": "P0DTC2"}, "ok",
             why="SARS-CoV-2 spike; the largest patch measured anywhere in the registry"),
        Cell({"accession": "P03452"}, "ok",
             why="influenza haemagglutinin; the case where the spread across structures of one "
                 "domain reached a factor of six, which is the argument for the median"),
        Cell({"accession": "P15659"}, "ok", why="influenza PA-PB1 interface"),
        Cell({"accession": "Q9IH62"}, "ok", why="henipavirus G attachment glycoprotein"),
        Cell({"accession": "P03470"}, "empty",
             why="influenza neuraminidase. No deposited chain is long enough to carry the "
                 "target's surface, so the call succeeds and finds nothing -- a real zero and "
                 "the ordinary case for a novel pathogen, not a failure. This is also the "
                 "accession whose deep mutational scan a packet cites while assessing its "
                 "reagent tractability as strong, so the absence is worth having on the record"),
        Cell({"accession": "P0DTD1", "replicates": 1}, "ok",
             why="the same accession at a different replicate count. Before `replicates` was "
                 "added to the memo key this returned the five-structure answer, so a declared "
                 "parameter a policy can set was silently ignored"),
        Cell({"accession": "P0DTD1", "replicates": 0}, "rejected",
             why="below the declared minimum of 1. A bound a strict JSON schema cannot carry "
                 "is charged rather than raised, so this cell writes a `rejected` row"),
        Cell({"accession": "not-an-accession"}, "empty",
             why="a well-formed string that resolves to nothing. Charged, and distinguishable "
                 "from a target that exists and has no usable structure only by the reasons "
                 "the return carries"),
    ),
    "intake.rank_targets": (
        Cell({"accession": "NC_045512.2"}, "ok", repeats=2,
             why="one real accession. Kept to two calls: this pipeline calls a remote "
                 "aligner, and a cost claim it cannot support is better absent than drawn "
                 "from a sample of two"),
    ),
}


def _row(index, name, cell, envelope, cold):
    """One recorded call, in the shape `optimizer.fidelity` and `instrumentation_gap` read.

    `action` and `tool` are both written because the two pointers are different claims: the
    optimiser licenses `cost_actions` from `trace[].action` and a tool's failures from
    `trace[].result`, and a row carrying only one of them is readable for one field and silent
    for the other.

    `result` carries what the call produced and `tool_status` carries which kind of thing that
    is. Writing only the string is what the live call site does, and it is why a successful
    call and a failed one are indistinguishable in the existing corpus -- see the defect
    `optimizer.fidelity` had to be fixed for.
    """
    status = envelope["status"]
    return {
        "round": index,
        "action": "call_tool",
        "tool": envelope["tool"],
        "tool_version": envelope["tool_version"],
        "arguments": {key: list(value) if isinstance(value, tuple) else value
                      for key, value in cell.arguments.items()},
        "tool_status": status,
        # The string a reader would see. For anything other than a plain success this is the
        # tool's own note, never a sentence written here.
        "result": envelope["note"] or f"{status}: call returned with no note",
        # Measured, and None where nothing ran. Never coerced to zero.
        "tool_seconds": envelope["seconds"],
        "tool_network_calls": envelope["network_calls"],
        "rejected": ({"reason": envelope["note"]} if status == "rejected" else None),
        "citable": envelope["citable"],
        "source_kind": envelope["source_kind"],
        "rows": None if envelope["rows"] is None else len(envelope["rows"]),
        "expected_status": cell.expects,
        "reproduces": cell.reproduces or None,
        "status_as_predicted": status == cell.expects,
        # True for the first call to this tool in this process. It pays one-time setup the
        # later calls do not -- a module import, a table construction, a reference file read
        # -- and pooling it with them produces a median that is a steady-state figure being
        # consumed as the cost of *a* call. Measured, not supposed: the first
        # `chemistry.library_spec` call took 16.7 ms and the next twenty took ~5.4 us each, a
        # spread of about 3100x, and the median over all of them was about to be certified as
        # the tool's cost by the guard built to stop cost deflation.
        "cold": cold,
    }


def _mispredictions(name, rows, cells):
    """Cells whose outcome did not match the prediction written before the run.

    Reported rather than reconciled. A grid that silently agreed with whatever happened would
    measure nothing, and the two interesting cases are opposite: a cell expected to fail that
    succeeded means the spec declares a failure the tool does not have, and a cell expected to
    reproduce a declared symptom whose actual string no declared symptom matches means the
    spec's wording does not survive contact with the record.
    """
    spec = tool(name)
    declared = {failure.symptom for failure in spec.failures}
    out = []
    for row in rows:
        if not row["status_as_predicted"]:
            out.append({"arguments": row["arguments"], "expected": row["expected_status"],
                        "observed": row["tool_status"], "result": row["result"],
                        "finding": "the grid's prediction was wrong, or the spec is"})
        elif row["reproduces"]:
            quoted = row["result"]
            if not any(symptom in quoted or quoted in symptom for symptom in declared):
                out.append({
                    "arguments": row["arguments"], "expected": row["expected_status"],
                    "observed": row["tool_status"], "result": quoted,
                    "reproduces": row["reproduces"],
                    "finding": "the cell reproduced the declared failure, but the string the "
                               "run records matches no declared symptom by substring, so "
                               "`fidelity` will read a correctly declared failure as "
                               "unexplained"})
    return out


def exercise_tool(name, repeats_scale=1):
    """Call one tool across its declared grid and return the rows and what they cost.

    `repeats_scale` multiplies the per-cell repeats. It exists so a smoke run can take the same
    path at scale 0 rather than a different path, and so the expensive tool can be exercised
    separately from the cheap one. It cannot select *which* cells run: a scale that could drop
    a failing cell would let a caller assemble a corpus in which a spec looks accurate.
    """
    spec = tool(name)
    context = chief.context(None, now=0, wall_budget_seconds=WALL_BUDGET_SECONDS)
    rows, index = [], 0
    for cell in GRID[name]:
        for _ in range(max(1, cell.repeats * repeats_scale) if cell.repeats > 1 else 1):
            envelope = chief.call(spec, dict(cell.arguments), context)
            rows.append(_row(index, name, cell, envelope, cold=index == 0))
            index += 1
    timed = [row["tool_seconds"] for row in rows if row["tool_seconds"] is not None]
    # Both medians over the same population: calls that ran and succeeded. They were not, and
    # the mismatch was a silent deflation path into the one field a declared network cost is
    # licensed by -- a cell that fails before issuing a request contributed a zero to the
    # network median while being excluded from the seconds median.
    successful = [row for row in rows
                  if row["tool_status"] in ("ok", "empty") and row["tool_seconds"] is not None]
    warm = [row for row in successful if not row["cold"]]
    successes = [row["tool_seconds"] for row in successful]
    warm_seconds = [row["tool_seconds"] for row in warm]
    counted = [row["tool_network_calls"] for row in successful
               if row["tool_network_calls"] is not None]
    return {
        "tool": name,
        "tool_version": spec.version,
        "declared": {"cost_seconds": spec.cost_seconds,
                     "cost_network_calls": spec.cost_network_calls,
                     "cost_actions": spec.cost_actions,
                     "worst_case_seconds": spec.worst_case_seconds,
                     "reaches_network": spec.reaches_network},
        "calls": len(rows),
        "by_status": {status: sum(1 for row in rows if row["tool_status"] == status)
                      for status in chief.STATUSES
                      if any(row["tool_status"] == status for row in rows)},
        "measured": {
            "timed_calls": len(timed),
            "successful_timed_calls": len(successes),
            "seconds_median": (sorted(successes)[len(successes) // 2] if successes else None),
            "seconds_min": min(successes) if successes else None,
            "seconds_max": max(successes) if successes else None,
            "network_calls_median": (sorted(counted)[len(counted) // 2] if counted else None),
            # The cold call reported apart from the warm ones, because a single scalar
            # `cost_seconds` cannot represent both and the ratio says whether the distinction
            # matters for this tool. ~3100x for the in-process library definer, ~1.2x for the
            # network tool whose 53 s dwarfs any one-time setup.
            "cold_seconds": next((row["tool_seconds"] for row in successful if row["cold"]),
                                 None),
            "warm_calls": len(warm_seconds),
            "warm_seconds_median": (sorted(warm_seconds)[len(warm_seconds) // 2]
                                    if warm_seconds else None),
        },
        "mispredictions": _mispredictions(name, rows, GRID[name]),
        "trace": rows,
    }


def run_exercise(tools_=None, repeats_scale=1, smoke=False):
    """Exercise the declared grid and write one artifact per tool under `runs/toolcalls/`.

    One artifact per tool rather than one for the sweep, because `optimizer.fidelity` globs a
    directory and attributes rows by `trace[].tool`; a per-tool file means a long run of the
    expensive tool can be added to the corpus without rewriting the cheap one's record.
    """
    selected = sorted(tools_ or GRID)
    unknown = [name for name in selected if name not in GRID]
    if unknown:
        raise ValueError(f"no grid is declared for {unknown}; the registry holds "
                         f"{sorted(TOOLS)} and a grid is a source edit, not an argument")
    written = []
    for name in selected:
        result = exercise_tool(name, repeats_scale=0 if smoke else repeats_scale)
        document = {"experiment": "tool-call-exercise", "status": "complete", "smoke": smoke,
                    "tool_set_sha256": spec_set_digest(),
                    "grid_cells": len(GRID[name]),
                    "provenance": provenance(),
                    "interpretation":
                        "Measured spend per call, from a grid declared in source. `declared` "
                        "is what the spec advertises and `measured` is what the calls cost; "
                        "the two are different claims and a proposal to move the first must "
                        "cite the second. A refused call records None rather than zero.",
                    **result}
        output = ROOT / CORPUS / f"{'smoke_' if smoke else ''}{name.replace('.', '_')}.json"
        write_json(output, document)
        written.append(str(output.relative_to(ROOT)))
        print(json.dumps({k: document[k] for k in
                          ("tool", "calls", "by_status", "measured", "mispredictions")},
                         indent=2))
    return {"written": written, "tools": selected}
