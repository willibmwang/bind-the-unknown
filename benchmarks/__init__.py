"""Two benchmarks for the co-scientist, with the baselines that make their numbers readable.

**Task 1 -- drugged-target retrieval.** Given one pathogen genome accession and nothing else,
rank the proteins it could encode. Score how high the ranking puts the proteins that an
approved or investigational antiviral actually binds. The labels are hand-curated and carry
their provenance; the metric is the rank-sum enrichment already written in
`intake.rank_enrichment`, which reports the best attainable p at each size so an underpowered
result is not read as a failure.

**Task 2 -- tool selection under preconditions.** From a state of known facts, choose the next
tool to call. Score against the shortest legal sequence that reaches a stated goal, computed by
breadth-first search over the precondition graph.

**Why Task 2's ground truth is not the selector's own rule.** The obvious way to score tool
selection here would be to compare a policy's ordering against `selector.rank`, and it would be
worthless: the system would be graded against its own preference and would score perfectly by
construction. So the two halves of `selector.REQUIREMENTS` are split by what they are claims
about.

`needs`, `needs_any` and `provides` are claims about the endpoints -- UniProt resolves an
accession, a folding endpoint takes a sequence -- and they are the *environment* here, as
checkable as any API contract. The ordering rule (`unlocks` before `gates` before `name`) is the
system's *policy*, and it is what gets graded. The grade comes from an external objective:
reach the goal in as few legal calls as possible, where "fewest" is a shortest path nobody in
this repository chose. The repo's selector is then one arm among several, and it is free to lose.
"""

from . import task1, task2  # noqa: F401

__all__ = ["task1", "task2"]
