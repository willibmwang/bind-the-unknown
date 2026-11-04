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

# The sequences the folding and homology grids submit, written out here rather than read off
# disk. A grid declared in source is the first property this module's docstring claims, and a
# cell whose argument came from a file is a cell whose argument something else can change
# between runs. Each is a real protein and each is identified by where it came from, so a
# reader can check the string rather than take it.
#
# SARS-CoV-2 main protease, nsp5, 306 residues, recovered from the CA trace of the committed
# fold in `runs/folded_structure/mpro_nsp5.pdb` and checked against that record's own
# `sequence_sha256` (50ec1901...91c92e). The same sequence the notebook folds, so the grid and
# the live panel submit one string.
MPRO_NSP5 = (
    "SGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDVVYCPRHVICTSEDMLNPNYEDLLIRKSNHNFLVQAGN"
    "VQLRVIGHSMQNCVLKLKVDTANPKTPKYKFVRIQPGQTFSVLACYNGSPSGVYQCAMRPNFTIKGSFLNGS"
    "CGSVGFNIDYDCVSFCYMHHMELPTGVHAGTDLEGNFYGPFVDRQTAQAAGTDTTITVNVLAWLYAAVINGD"
    "RWFLNRFTTTLNDFNLVAMKYNYEPLTQDHVDILGPLSAQTGIAVLDMCASLKELLQNGMNGRTILGSALLE"
    "DEFTPFDVVRQCSGVTFQ")
# SARS-CoV-2 envelope protein E, P0DTC4, 75 residues, and ORF7a, P0DTC7, 121 residues, both
# read from the reviewed UniProt reference this project already identifies against,
# `data/raw/reference/viral_reviewed.fasta`. Two further real proteins of the same pathogen at
# two further lengths, which is what the folding grid needs: the host caches by sequence, so a
# second length is the only way to see a cold call twice.
ENVELOPE_P0DTC4 = (
    "MYSFVSEETGTLIVNSVLLFLAFVVFLLVTLAILTALRLCAYCCNIVNVSLVKPSFYVYSRVKNLNSSRVPD"
    "LLV")
ORF7A_P0DTC7 = (
    "MKIILFLALITLATCELYHYQECVRGTTVLLKEPCSSGTYEGNSPFHPLADNKFALTCFSTQFAFACPDGVK"
    "HVYQLRARSVSPKLFIRQEEVQELYSPIFLIVAAIVFITLCFTLKRKTE")
# SARS-CoV-2 nucleoprotein, P0DTC9, 419 residues, from the same reference. Nineteen residues
# over the 400-residue ceiling `folded_structure.MAX_RESIDUES` measured, which is what makes it
# the cell for the declared length refusal: a real protein of the pathogen under study that
# this lane cannot fold, rather than a string lengthened until the bound tripped.
NUCLEOPROTEIN_P0DTC9 = (
    "MSDNGPQNQRNAPRITFGGPSDSTGSNQNGERSGARSKQRRPQGLPNNTASWFTALTQHGKEDLKFPRGQGV"
    "PINTNSSPDDQIGYYRRATRRIRGGDGKMKDLSPRWYFYYLGTGPEAGLPYGANKDGIIWVATEGALNTPKD"
    "HIGTRNPANNAAIVLQLPQGTTLPKGFYAEGSRGGSQASSRSSSRSRNSSRNSTPGSSRGTSPARMAGNGGD"
    "AALALLLLDRLNQLESKMSGKGQQQQGQTVTKKSAAEASKKPRQKRTATKAYNVTQAFGRRGPEQTQGNFGD"
    "QELIRQGTDYKHWPQIAQFAPSASAFFGMSRIGMEVTPSGTWLTYTGAIKLDDKDPNFKDQVILLNKHIDAY"
    "KTFPPTEPKKDKKKKADETQALPQRQKKQQTVTLLPAADLDDFSKQLQQSMSSADSTQA")


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
    # Gridded so the lookup a room is most likely to ask for by name is replayable without a
    # network. The predictions below were written before the first run, per this module's rule,
    # and the cell that turned out to be mispredicted is left as it was written.
    "structures.predicted": (
        Cell({"accession": "P0DTD1"}, "ok", repeats=3,
             why="SARS-CoV-2 ORF1ab, which carries nsp5, and the accession the notebook "
                 "dispatches live, so the recording and the live panel are the same call"),
        Cell({"accession": "P0DTC2"}, "ok", repeats=2,
             why="spike, the other target this project singles out, so the cost claim rests "
                 "on more than one protein and more than one model count"),
        Cell({"accession": "P0DTD1", "confident_plddt": 70.0}, "ok", repeats=2,
             why="the declared default named explicitly, to check the default in force is the "
                 "one the registry advertises"),
        Cell({"accession": "P0DTD1", "confident_plddt": 101.0}, "rejected",
             why="above the declared maximum of 100.0, the cell that writes a `rejected` row "
                 "for this tool"),
        Cell({"accession": "P00000"}, "failed",
             reproduces="urllib.error.HTTPError from the 3D-Beacons or AlphaFold host",
             why="a syntactically valid accession no host serves, which the spec declares "
                 "raises rather than returning an absent prediction"),
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
    # Four calls at the declared ten seconds each, which is the whole grid's budget against a
    # public keyless index with no published rate limit. `nsp12` is the term this module's own
    # endpoint survey measured, so the default cell's count can be read against a number
    # already on the record rather than only against itself.
    #
    # Neither declared failure has a cell, and the reason is a property of the failures rather
    # than a gap. "The date clause did not discriminate" is reported inside
    # `date_filter_verified` at HTTP 200 by design -- the control query exists precisely so it
    # never becomes a status -- so no argument can turn it into one. The HTTPError is a
    # condition of the host and the only way to provoke it from here is to abuse the endpoint.
    # A cell that claimed either would be a prediction about something it cannot cause.
    "literature.search": (
        Cell({"subject": "nsp12"}, "ok", repeats=3,
             why="the term this module's endpoint survey measured at 45481 hits unbounded, so "
                 "the default cell lands beside a recorded count and three calls give the "
                 "cost a sample"),
        Cell({"subject": "RNA-dependent RNA polymerase", "subject_kind": "family",
              "date_to": "2019-12-31"}, "ok",
             why="the family route under a publication-date ceiling, which is the only "
                 "configuration a historical replay may use. A second subject and a second "
                 "query shape, and the one cell that makes the tool run its control query"),
        Cell({"subject": "nsp5", "top": 101}, "rejected",
             why="above the declared maximum of 100. The function raises on the same bound, "
                 "so this cell also pins that the charged refusal comes first and nothing is "
                 "sent"),
        Cell({"subject": "Zzqxv-not-a-protein", "per_axis": False}, "empty",
             why="a well-formed query the index has never seen. A successful call that found "
                 "nothing, which for a novel pathogen is the ordinary answer and not a "
                 "failure; `per_axis` off so a true zero costs one request rather than eight"),
    ),
    # Three accessions, one call each, and the reason is the one the geometry grid gives: the
    # declared 21.2 s is already a median of eight calls whose spread ran 4.5 s to 24.9 s on
    # the *same* accession, so the variance here is the host's. Against variance that large a
    # second call on one accession buys less than a first call on another, and three distinct
    # proteins cost 65 s where repeats to the twenty-call floor would cost seven minutes and
    # 120 requests against a public endpoint.
    "families.entries": (
        Cell({"accession": "P0DTD1"}, "ok",
             why="SARS-CoV-2 ORF1ab, the accession every other lane in this grid set is keyed "
                 "on, and the long polyprotein whose entry count is what the 300 s worst case "
                 "is declared for"),
        Cell({"accession": "P0DTC2"}, "ok",
             why="spike, a single-domain glycoprotein rather than a polyprotein, so the cost "
                 "claim spans both shapes the registry's targets take"),
        Cell({"accession": "P03452"}, "ok",
             why="influenza haemagglutinin, a second organism, which is the axis the two "
                 "coronavirus cells cannot vary"),
        Cell({"accession": "P0DTD1", "max_families": 101}, "rejected",
             why="above the declared maximum of 100, the cell that writes a `rejected` row "
                 "for this tool"),
        Cell({"accession": "P00000"}, "failed",
             reproduces="urllib.error.HTTPError from the InterPro host",
             why="a syntactically valid accession InterPro serves nothing for. The spec draws "
                 "the line here explicitly -- a 404 from one member database is an absence, a "
                 "404 from the integrated endpoint raises -- so this cell is the one that "
                 "says which side of it an unknown accession falls on"),
    ),
    # The query is a sequence or a nucleotide accession, and both routes are exercised because
    # they are different amounts of work: the accession route resolves coding sequences first,
    # so its cost is not the sequence route's cost plus noise.
    "similarity.homologs": (
        Cell({"query": MPRO_NSP5}, "ok", repeats=2,
             why="the 306-residue query this module's endpoint comparison timed at two "
                 "seconds against SwissProt, which is the measurement the 9 s declaration was "
                 "drawn from"),
        Cell({"query": "NC_045512.2"}, "ok",
             why="the accession route, on the genome accession every arm of this project "
                 "starts from. A distinct argument and a distinct code path, so the cost "
                 "claim does not rest on one sequence handed straight to the submitter"),
        Cell({"query": MPRO_NSP5, "max_hits": 501}, "rejected",
             why="above the declared maximum of 500, the cell that writes a `rejected` row "
                 "for this tool"),
        Cell({"query": MPRO_NSP5, "max_wait_seconds": 1.0}, "failed",
             reproduces="the poll ceiling was crossed before the job finished",
             why="the declared minimum ceiling, so the call is charged rather than refused, "
                 "and one second is below the two this query has been measured to need. The "
                 "spec says a crossed ceiling raises rather than returning an empty hit list, "
                 "and this is the cell that holds it to that"),
    ),
    # Distinct sequences, one call each, for the reason the geometry grid learned the hard way
    # and this module's own docstring quotes: the ESM Atlas caches by sequence, so a repeated
    # call is about twenty times faster and a grid that repeated one sequence would be an
    # instrument pointed at the host's cache. Three lengths instead, 75, 121 and 306 residues,
    # against measured cold times of 1.8 s at 50 and 12.1 s at 300.
    #
    # Measured after the fact, and recorded here because it falsifies the reasoning above
    # rather than the predictions: the three folds took 0.631 s, 0.178 s and 0.162 s at one
    # request each, so none of them was cold. The host's cache is not per-machine, and three
    # reviewed proteins of the most-studied pathogen on the index are in it already. The `why`
    # on the second and third cells says they measure a cold fold, and that claim is wrong;
    # it is left as written, because the rule here is that nothing is edited to agree with the
    # outcome. What a cold fold of this lane costs is still only on the record in the module
    # docstring's own 1.8 s / 5.0 s / 12.1 s / 25.6 s series, and a sequence nobody has ever
    # submitted is what it would take to add to it.
    "structures.folded": (
        Cell({"sequence": MPRO_NSP5, "target_name": "Mpro (nsp5)"}, "ok",
             why="the sequence the committed fold in runs/folded_structure/ was taken from, so "
                 "the recording and the notebook's live panel submit the same string and the "
                 "two can be compared byte for byte"),
        Cell({"sequence": ENVELOPE_P0DTC4, "target_name": "E (P0DTC4)"}, "ok",
             why="75 residues, the short end. A sequence this host has not been asked for from "
                 "here before, so it measures a cold fold rather than a cache hit"),
        Cell({"sequence": ORF7A_P0DTC7, "target_name": "ORF7a (P0DTC7)"}, "ok",
             why="121 residues, a third length, which is what turns two readings into a claim "
                 "about how cost scales rather than two unconnected numbers"),
        Cell({"sequence": NUCLEOPROTEIN_P0DTC9, "target_name": "N (P0DTC9)"}, "empty",
             reproduces="the sequence exceeds what the endpoint accepts",
             why="419 residues against the measured 400-residue ceiling. Predicted `empty` "
                 "rather than `failed`: the spec says the refusal is a stated absence and not "
                 "a raise, and the module refuses locally, so this cell should also spend no "
                 "request at all"),
        Cell({"sequence": MPRO_NSP5, "confident_plddt": 101.0}, "rejected",
             why="above the declared maximum of 100.0, the same bound and the same expected "
                 "row as the predicted-structure grid's confidence cell"),
        Cell({"sequence": MPRO_NSP5, "neighbour_cutoff": 0.5}, "rejected",
             why="below the declared minimum of 1.0 angstrom. The function raises on a "
                 "non-positive cutoff of its own accord, so this cell pins that the charged "
                 "refusal arrives first and the sequence is never sent"),
    ),
    # The most expensive tool in the registry: 94 s and 16 requests per call across three
    # independent public hosts. Every valid cell is a single call, so there is no median here
    # and the artifact's `seconds_median` over one or two readings is the reading itself. That
    # is the honest trade. Twenty calls to clear the cost floor would be half an hour and
    # better than three hundred requests to ChEMBL, IEDB and MaveDB, and no cost claim is worth
    # that to three services that charge nothing.
    #
    # Neither declared failure has a cell. "A lane could not be reached" is recorded in
    # `lanes_unavailable` rather than raised, by design, so it is an outage and not an argument;
    # the HTTPError on the integrated query is the same kind of condition. The fourth cell is
    # the nearest thing an argument can reach, and it is written as a prediction rather than as
    # a reproduction of either.
    "bioactivity.prior_binders": (
        Cell({"accession": "P0DTD1"}, "ok",
             why="SARS-CoV-2 ORF1ab, the accession the notebook's refusal panel names and the "
                 "one every other grid here is keyed on"),
        Cell({"accession": "P0DTC2"}, "ok",
             why="spike, where the prior-binder question has a different answer in every lane, "
                 "so the cost claim rests on two proteins rather than one"),
        Cell({"accession": "P0DTD1", "max_molecules": 501}, "rejected",
             why="above the declared maximum of 500, the cell that writes a `rejected` row for "
                 "this tool and the only cheap cell in the grid"),
        Cell({"accession": "P00000"}, "ok",
             why="an accession no lane holds a record for. Predicted `ok` and not `empty`: "
                 "this tool declares no key that says whether it found anything, so three "
                 "lanes answering with nothing is a successful call reporting three zeros, and "
                 "distinguishing that from an unreached lane is what `lanes_unavailable` is "
                 "for. If the prediction is wrong the dispatcher reads absence from a payload "
                 "that does not carry it, which is worth finding out"),
    ),
}


# Registered tools with no grid here, and therefore with a declared cost no record can
# contradict. Stated rather than left to be noticed, because an unmeasured declaration is
# exactly what this module was built to end: `structures.evidence` declared zero seconds and
# spends sixty, and nobody could tell until something counted.
#
# Empty, as of the five onset grids above. Every registered tool now has a grid and a recording
# under `runs/toolcalls/`, so `set(GRID) | set(UNGRIDDED)` and `set(TOOLS)` agree with nothing
# in this tuple, and this is where a tool goes when a later one cannot be exercised here.
#
# Nothing in the test suite pins that partition today, which is worth saying plainly while the
# tuple is empty: an eleventh tool can be registered with no grid and no entry here, and the
# only thing that would notice is a reader comparing the registry against this file.
#
# What the grids above do *not* claim is a cost at the twenty-call floor. Six of the ten run
# one or three calls per cell, because clearing the floor for `bioactivity.prior_binders`
# alone would be half an hour and several hundred requests against three services that charge
# nothing for them. A median over one call is the reading, and the artifacts say so in
# `successful_timed_calls`; the optimiser reads the floor off that field and declines on its
# own, which is the behaviour wanted rather than a shortfall to apologise for.
#
# Deliberately not a silent gap. `python -m hpc_pilots exercise` runs only the grids declared
# above, and a reader comparing the registry against `runs/toolcalls/` would otherwise find a
# tool missing with nothing saying why.
UNGRIDDED = ()


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
