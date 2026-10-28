"""Which tool to call next, given what is known and what is still unresolved.

The decision layer already has a move set and a caller. What it has never had is a reason to
prefer one tool over another, and `finder` -- which retrieves tools by matching a query against
summaries -- was measured as retrieving nothing at this scale, which is the expected result for
five tools whose summaries share a vocabulary. Ranking by text similarity was answering the
wrong question anyway.

**The question a selector can actually answer is reachability, not relevance.** A tool is not
a thing you may call because it sounds apposite; it is a thing you may call because its inputs
exist. From a pathogen sequence alone, a literature query has no target name to query for and a
structural lookup has no accession to look up. Those tools are not expensive and not irrelevant
-- they are *unreachable*, and no amount of summary matching discovers that, because the
obstruction is in the state and not in the text.

So this module does three things and declines to do a fourth.

* It declares, per tool, what the tool needs in the state and what its return puts there.
* It reports which tools are callable now, and names what each blocked one is blocked on.
* It orders the callable ones, preferring a call that unblocks others over a call that does not.

It does **not** rank by cost. `cost_seconds` is unmeasured on three of five specs and the
measured two were never measured here, so a cost-weighted order would be a preference computed
from numbers nobody has taken. When those numbers exist, the tie-break is the place to put them,
and `rank` returns a `score` tuple rather than a float so that an added term is visible in the
record instead of folded into one number.

**Why the requirement table lives here and not on `Tool`.** A spec in `tools.py` is a frozen,
digested, byte-immutable record: changing one bumps `TOOL_SET_VERSION` and obliges a new
`configs/tools.v{n}.json`, and every run that recorded the old digest keeps resolving. A
selector's opinion about what a tool needs is not that kind of fact -- it is this module's
model of the state, and it will change as the state grows. Keeping it here means a changed
opinion cannot silently reinterpret a recorded run, and it follows the device
`emergency.ACTION_TOOLS` already uses: a table beside the code that reads it, pinned in both
directions by a test so it cannot drift from the registry.
"""
from dataclasses import dataclass, field


# What a state can hold, enumerated rather than left to free strings, for the reason
# `chief.SOURCE_KINDS` is enumerated: a fact kind nobody enumerated is a precondition nobody
# decided about, and a typo in one would read as an unsatisfiable requirement for ever.
#
# These are kinds of identity and not values. `target_name` is "some target is named", not
# which one -- the selector reasons about whether a call is *possible*, and a module that
# needed the value would have to hold the state's contents and would become the state.
FACTS = (
    "genome_accession",
    # A nucleotide accession for the pathogen. The one fact a campaign can start from, and the
    # only one in this tuple that arrives from the task statement rather than from a tool.
    "target_name",
    # At least one candidate target is named. What a literature query needs in order to be a
    # query about something.
    "protein_accession",
    # A reviewed protein accession. What a structural lookup is keyed on; a sequence is not a
    # substitute, because the endpoints this repository reaches resolve accessions and not
    # sequences.
    #
    # That last clause was a statement about the endpoints reached, not about the world, and it
    # has been narrowed rather than deleted: it holds for PDBe coverage, for UniProt chain
    # locations and for AlphaFold DB, all of which resolve an accession. It does not hold for a
    # single-sequence folding endpoint, which is why `protein_sequence` is now a fact of its
    # own. The distinction matters because a genuinely unknown protein -- the project's premise
    # -- has no reviewed accession, so every accession-keyed lane returns a true and useless
    # zero for exactly the case the work is about.
    "protein_sequence",
    # One candidate's amino-acid sequence, which ORF calling yields directly from a genome and
    # which needs no identification at all. The only structural input a novel target can
    # supply. Distinct from `protein_accession` on purpose and in the same way
    # `homolog_accession` is: a sequence is what the pathogen has, an accession is what a
    # curated database has, and a pipeline that silently treats one as the other is the
    # conflation this tuple exists to prevent.
    "residue_ranges",
    # Declared or derived residue intervals for a named target. What makes structural coverage
    # a statement about a site rather than about a whole chain.
    "homolog_accession",
    # A similar sequence, with the identity and E-value that say how similar. Distinct from
    # `protein_accession` on purpose: identification by homology yields somebody else's
    # accession, and treating a 28%-identity hit as though it were the query's own identity is
    # the conflation `intake`'s own docstring warns about.
    "protein_family",
    # A family or domain assignment. The fact that connects a sequence to a literature that has
    # never heard of this pathogen: nothing is published about an unknown target, and a great
    # deal is published about its family, which is the whole premise of a prototype-pathogen
    # argument.
)

# Which dossier axis a tool's return can legitimately speak to. Deliberately narrow, and
# deliberately not "every axis the tool sounds related to": the axes are the Week 4 rubric and
# an overclaimed axis here would make an unresolved gate read as resolved, which is the one
# error that changes a recommendation rather than merely a ranking.
AXES = ("reagent_tractability", "essentiality", "conservation", "accessibility",
        "assay_availability", "public_labels", "precedent")


@dataclass(frozen=True)
class Requirement:
    """What one tool needs in the state, what it adds to it, and what it can speak to."""

    needs: tuple = field(default_factory=tuple)
    """Facts that must *all* be present before the call is possible. Every entry is in `FACTS`."""
    needs_any: tuple = field(default_factory=tuple)
    """Facts of which *at least one* must be present, or empty for no such requirement.

    Separate from `needs` because a literature query is answerable from either a target name or
    a family assignment, and folding that into a conjunction would make the tool unreachable
    until both existed -- which would hide the one route that works when the target itself is
    unknown. A disjunction and a conjunction are different claims about a precondition and
    collapsing them loses the weaker, more useful one.
    """
    provides: tuple = field(default_factory=tuple)
    """Facts the return puts in the state when the call succeeds. A tool whose return is not
    evidence can still provide a fact -- identification provides three -- and that is the whole
    reason reachability is a different question from citability."""
    gates: tuple = field(default_factory=tuple)
    """Dossier axes a successful return can bear on. Empty for a tool whose return is a design
    choice rather than a finding, which is the distinction `produces_evidence` draws in the
    spec and which this field must not contradict -- a test pins the two together."""
    scope: str = ""
    """A standing limit on when the tool is applicable at all, as prose, or empty.

    `structures.evidence` reads its targets from `configs/structures.json` and therefore cannot
    speak about a target that is not in it, whatever the state holds. That is not a missing
    precondition -- the call succeeds and returns a table -- so modelling it as one would be
    wrong, and leaving it out would let the selector propose a lookup for a target the lookup
    has never heard of. It is a scope note, surfaced in the proposal and never scored.
    """


# Pinned in both directions by a test: every registered tool has an entry here, and every entry
# names a registered tool. The first direction is the one that matters -- a tool added to the
# registry with no entry here would be silently unrankable, and an unrankable tool reads to a
# policy exactly like a tool that is never worth calling.
REQUIREMENTS = {
    "chemistry.library_spec": Requirement(
        needs=(),
        # Takes an initiator, length bounds and monomer classes, all with defaults, and reads
        # nothing out of the state. Callable from an empty state, which is correct and is also
        # why callability alone is a bad ranking: the one tool that needs nothing is the one
        # tool that learns nothing.
        provides=(),
        gates=(),
    ),
    "structures.evidence": Requirement(
        needs=(),
        provides=(),
        # Dated structural coverage bears on precedent -- a deposited structure is a positive
        # control and de-risks interpreting a failure. It does not bear on `accessibility`: a
        # structure count is a research-attention proxy, which the spec's own warning says must
        # not be pooled with a pocket score, and `accessibility` is the axis a pocket score
        # would speak to.
        gates=("precedent",),
        scope="Reads its targets from configs/structures.json and cannot speak about a target "
              "that is not declared there, however much the state knows about it.",
    ),
    "geometry.surface_markers": Requirement(
        needs=("protein_accession",),
        provides=(),
        # The only registered tool that bears on `accessibility`, which is the axis asking
        # whether an epitope is solvent-exposed in the relevant conformation. That is exactly
        # what a largest-contiguous-exposed-hydrophobic-patch measurement answers, and exactly
        # what `structures.evidence` declines to answer: its own warning says a structure count
        # is a research-attention proxy that must never be pooled with a pocket score.
        #
        # It does not gate `precedent`. A measured surface is not a prior campaign, and it does
        # not gate `reagent_tractability` either, however tempting: `exposed_fraction` speaks to
        # whether a domain presents itself, not to whether it can be expressed, folded and
        # immobilised through multiple selection rounds, which is the hard gate and which no
        # structural calculation answers.
        gates=("accessibility",),
        scope="Measures deposited conformations, so it is silent on a target with no usable "
              "deposited chain -- the ordinary case for a novel pathogen, and the case a "
              "sequence-based prediction would have to cover instead.",
    ),
    "intake.rank_targets": Requirement(
        needs=("genome_accession",),
        # The one tool in the registry that turns the fact a campaign starts with into the facts
        # every other tool is keyed on. Everything else is downstream of it by construction, not
        # by preference.
        #
        # `protein_sequence` is declared now because the ORF translation is carried into the
        # output rather than dropped from it. This is the only fact here that does not depend on
        # identification succeeding: the first three are what a reviewed database says about a
        # relative, and a genome with no recognisable homolog yields none of them -- the holdout
        # work measured exactly that, a cliff to zero identified ORFs at the species rung. The
        # sequence survives that cliff, because six-frame translation needs no reference at all.
        # So this is the fact that keeps a structural lane open for a target nothing recognises,
        # which is the case the whole project is named for.
        provides=("target_name", "protein_accession", "residue_ranges", "protein_sequence"),
        gates=(),
    ),
    "literature.search": Requirement(
        needs=(),
        # Either identity will do, and the family route is the one that works when the target
        # itself is unknown: nothing is published about a novel target and a great deal is
        # published about its family.
        # `genome_accession` belongs here and was missing, which understated the onset by a
        # whole lane. The query builder has an accession route, and it works on the one fact a
        # campaign starts with: `subject_kind="accession"` on NC_045512.2 sent
        # `(ACCESSION_ID:"NC_045512.2" OR ACCESSION_ID:"NC_045512")` and returned 4719 hits in
        # 2.5 seconds. Declaring the lane blocked at onset left the selector's first round with
        # one informative move where it has two.
        needs_any=("genome_accession", "target_name", "protein_family"),
        provides=(),
        # Gates nothing yet, deliberately. A hit count is attention and the dated claim lives
        # per hit, so substantiating an axis needs a row-set design this does not have. The
        # tool is the retrieval; admitting a retrieved paper into a dossier is the next step.
        # Worth naming: every row already carries `first_publication_date`, so this lane holds
        # the only genuine per-row date among the six and is the cheapest one to promote.
        gates=(),
        scope="Europe PMC indexes preprints and full text, so a count here is not comparable "
              "with a PubMed count -- measured at 1173 against 25 on one query.",
    ),
    "families.entries": Requirement(
        needs=(),
        # Either input works, and the sequence route is the one that matters here. The lane is
        # keyed on the MD5 of the sequence rather than on an accession, so a translated reading
        # frame that no reviewed database has catalogued can still be annotated -- the second
        # lane, beside folding, that reaches a protein nothing has named. Declaring this as
        # needing an accession would have shut it for exactly that case.
        needs_any=("protein_accession", "protein_sequence"),
        # `residue_ranges` too, and it is the more useful of the two. The lane derives the
        # boundaries this repository currently hand-authors in `configs/structures.json`: it
        # independently placed the coronavirus main protease at 3264-3575 of the polyprotein
        # and the nsp16-like domain at 6800-7095. A range is what turns a claim about a protein
        # into a claim about a site, which is what both the structural lookup and a binder need.
        provides=("protein_family", "residue_ranges"),
        # Gates nothing, and the author of this lane proposed four axes including
        # `reagent_tractability`. That one is refused on the lane's own description, which says
        # a family assignment "is never a measurement on this pathogen" and "predicts no
        # binding, no affinity and no druggability". Whether a soluble, correctly folded,
        # immobilisable reagent can be obtained is a wet-lab fact; inferring it from what
        # relatives of this protein are annotated with would put the hard gate on an
        # annotation. The other three are refused for the reason every retrieval lane here is:
        # the dates are per row -- two member databases are frozen at 2010 and 2020 releases --
        # and an axis claim needs a dated claim.
        gates=(),
        scope="An assignment is transferred from relatives and is never a measurement on this "
              "pathogen. Roughly half the matches on a long polyprotein come from a neural "
              "predictor that borrows member-database accessions, so curated hits and "
              "predictions are indistinguishable by accession alone and are never pooled; two "
              "member databases are also long frozen, which is why every row carries the "
              "release that served it.",
    ),
    "similarity.homologs": Requirement(
        needs=(),
        # A sequence is enough, and that is the point: this lane does not need an identity, so
        # it still answers for a target nothing recognises. `genome_accession` belongs here too
        # and was missing: the lane resolves coding sequences from a nucleotide accession
        # itself, which the spec's own `query` parameter says and which I had not carried into
        # the requirement. Verified on NC_045512.2 -- it resolved the CDS catalogue and
        # returned five homologs at identity 1.0 over six requests, from the one fact a
        # campaign starts with.
        needs_any=("genome_accession", "protein_sequence", "protein_accession"),
        provides=("homolog_accession", "protein_family"),
        gates=(),
        scope="EBI HMMER, keyless and about two seconds against SwissProt. The job model is "
              "pollable with a real ceiling, so the wall gate can refuse it -- unlike the "
              "BLAST dispatcher this project had declared, which requires an email address.",
    ),
    "structures.predicted": Requirement(
        needs=("protein_accession",),
        provides=(),
        # Gates nothing, and this is a correction. I registered it against `accessibility` on
        # the reasoning that a predicted model answers for a target nobody has solved. An
        # independent reviewer read the payload and the claim does not survive it: `coverage`
        # fetches a model inventory and confidence vectors, never downloads coordinates, and
        # computes nothing geometric. The recorded record for P03470 carries no key matching
        # exposure, solvent, access, surface or burial -- its one range block holds a single
        # number, a QMEANDisCo of 0.913, which is template agreement and not solvent exposure.
        # The module's own limitation says the confidence "does not say the site is
        # solvent-exposed, is a pocket, is bindable, or is accessible in vivo". The axis asks
        # whether the epitope is solvent-exposed in the relevant conformation, so gating it on
        # a model-quality score is the confidence-as-accessibility error with the label
        # changed.
        #
        # It can be earned back without a new endpoint: every model row carries a `model_url`,
        # so exposure is computable from those coordinates exactly as `geometry.py` computes it
        # from deposited ones. That is the work, and until it is done the axis stays open.
        #
        # It must not gate `precedent` either, for the separate reason: a prediction is not a
        # deposition, and AlphaFold modelled every reference proteome whether or not anyone
        # cared, so a predicted-model count carries no attention signal at all.
        gates=(),
        scope="Reports a predicted model, never an experimental one; models marked "
              "experimentally determined are partitioned out and counted separately.",
    ),
    "structures.folded": Requirement(
        needs=("protein_sequence",),
        provides=(),
        # Gates nothing, on the tool's own argument. The burial proxy is computed on one
        # monomer in vacuum, and for a viral glycoprotein the biological unit is usually a
        # trimer -- so a surface scored exposed here may be an oligomeric interface that is
        # never exposed in the assembled particle, which is `accessibility` answered wrongly.
        gates=(),
        scope="The only structural lane keyed on a sequence rather than an accession, so the "
              "only one that answers for a target with no reviewed entry.",
    ),
    "bioactivity.prior_binders": Requirement(
        needs=("protein_accession",),
        provides=(),
        gates=(),
        scope="Three independent public lanes. A lane that could not be reached is reported "
              "apart from a lane that returned nothing.",
    ),
}


@dataclass(frozen=True)
class Situation:
    """What is known, and what still needs resolving. The selector's whole input."""

    known: frozenset = frozenset()
    """Facts already in the state. Every entry must be in `FACTS`."""
    unresolved: tuple = ()
    """Axes with no sufficient claim yet. Every entry must be in `AXES`. Order is not
    significance: the selector counts how many a tool speaks to and does not weight them,
    because a weighting nobody elicited would be this module inventing the rubric."""

    def __post_init__(self):
        unknown = sorted(set(self.known) - set(FACTS))
        if unknown:
            raise ValueError(
                f"Situation names facts that are not in FACTS: {unknown}. A fact kind nobody "
                f"enumerated is a precondition nobody decided about.")
        stray = sorted(set(self.unresolved) - set(AXES))
        if stray:
            raise ValueError(f"Situation names axes that are not in AXES: {stray}")


@dataclass(frozen=True)
class Proposal:
    """One tool, whether it can be called now, and why it placed where it did."""

    tool: str
    callable_now: bool
    blocked_on: tuple = ()
    """Facts the tool needs that the state does not hold. Empty exactly when `callable_now`."""
    unlocks: tuple = ()
    """Tools that are blocked now and would stop being blocked if this call succeeded. Computed
    against the registry rather than declared, so it cannot go stale."""
    gates_unresolved: tuple = ()
    """Unresolved axes this tool's return can bear on."""
    scope: str = ""
    why: str = ""
    score: tuple = ()
    """The sort key, kept as a tuple so an added term stays visible in the record instead of
    being folded into a float nobody can decompose."""


def _requirement(name, requirements=None):
    table = REQUIREMENTS if requirements is None else requirements
    try:
        return table[name]
    except KeyError:
        raise ValueError(
            f"No requirement declared for tool {name!r}; declared tools are "
            f"{sorted(table)}. A tool with no entry is unrankable, which reads to a "
            f"policy exactly like a tool that is never worth calling.") from None


def blocked_on(name, situation, requirements=None):
    """Facts the tool needs and the situation does not hold, in declaration order.

    An unsatisfied disjunction is reported as one entry joining its alternatives with `|`,
    which is never a member of `FACTS` and so cannot be mistaken for a fact. Reported as one
    entry rather than several because listing both alternatives separately would read as two
    obstructions when either one alone would clear it.
    """
    requirement = _requirement(name, requirements)
    missing = [fact for fact in requirement.needs if fact not in situation.known]
    if requirement.needs_any and not any(f in situation.known for f in requirement.needs_any):
        missing.append("|".join(requirement.needs_any))
    return tuple(missing)


def unlocked_by(name, situation, names=None, requirements=None):
    """Tools blocked now that this call would unblock, assuming it provides what it declares.

    Assuming, and the assumption is stated in the proposal rather than hidden: a call can
    succeed and still provide nothing -- identification against a genome with no recognisable
    homolog returns rows that name no reviewed accession -- so this is what the call *can*
    unlock and not what it will. The alternative is a selector that promises outcomes it cannot
    inspect, which is the failure mode the `request_structure` absence string exists because of.
    """
    table = REQUIREMENTS if requirements is None else requirements
    gained = set(situation.known) | set(_requirement(name, table).provides)
    after = Situation(known=frozenset(gained), unresolved=situation.unresolved)
    return tuple(sorted(
        other for other in (names if names is not None else table)
        if other != name and blocked_on(other, situation, table)
        and not blocked_on(other, after, table)))


def rank(situation, names=None, requirements=None):
    """Every tool as a proposal, best first. Blocked tools are reported, never dropped.

    Reported because a policy that is shown only its callable moves cannot tell a tool that is
    useless from a tool that is one call away, and the second is the more useful thing to know
    under a budget. The ordering is: callable before blocked; among callable, more unblocking
    first, then more unresolved axes spoken to, then name. Name last and always, so the order is
    total and a run is reproducible -- a tie broken by dict order would make the recorded
    sequence depend on registration order, which is the defect `spec_set_digest` sorts to avoid.

    Unblocking is preferred over axis coverage deliberately, and it is the one real judgement
    here. A call that widens the action set changes what every later round may do, while a call
    that speaks to an axis changes one number; under a budget that binds, the option value is
    worth more than the increment. It is also the assumption most likely to be wrong, so it is
    a single comparison in a documented tuple rather than a weight buried in a sum, and
    `score[1]` is the term to change.
    """
    table = REQUIREMENTS if requirements is None else requirements
    names = sorted(table if names is None else names)
    proposals = []
    for name in names:
        requirement = _requirement(name, table)
        missing = blocked_on(name, situation, table)
        unlocks = unlocked_by(name, situation, names, table) if not missing else ()
        gated = tuple(axis for axis in requirement.gates if axis in situation.unresolved)
        if missing:
            why = (f"blocked: needs {', '.join(missing)}, which the state does not hold")
        elif unlocks:
            why = (f"callable now, and unblocks {', '.join(unlocks)}")
        elif gated:
            why = f"callable now, and bears on {', '.join(gated)}"
        else:
            why = "callable now, but unblocks nothing and bears on no unresolved axis"
        proposals.append(Proposal(
            tool=name, callable_now=not missing, blocked_on=missing, unlocks=unlocks,
            gates_unresolved=gated, scope=requirement.scope, why=why,
            # Negated counts so that a plain ascending sort puts the better proposal first and
            # no reverse flag has to agree with the comparison.
            score=(0 if not missing else 1, -len(unlocks), -len(gated), name)))
    return tuple(sorted(proposals, key=lambda p: p.score))


def first(situation, names=None, requirements=None):
    """The single best next call, or None when nothing is callable.

    None rather than the best blocked tool, because returning a tool whose inputs do not exist
    is how a selector launders an impossibility into an instruction.
    """
    for proposal in rank(situation, names, requirements):
        if proposal.callable_now:
            return proposal
    return None


# Tools that do not exist, kept strictly apart from the registry so the separation is
# structural rather than remembered.
#
# This table exists to answer one question the live one cannot: *which tool is worth building
# next*. Passed as `rank(..., requirements=...)`, it reports what the order would be if these
# existed, and the answer is the argument for building them -- with the current registry the
# dependency graph is flat, because nothing consumes what identification produces, so
# `intake.rank_targets` unlocks nothing and the first call is decided by axis coverage alone.
#
# A test pins that no name here is registered. That is the load-bearing property: a decider
# shown a spec for work nobody has commissioned would be reading an accurate description of a
# capability no action can reach, which `emergency.py` already identifies as a fabrication
# channel pointed at the policy. Nothing dispatches from this table and nothing may.
#
# The endpoints named in the scope notes are what each would reach. Two are already paid for:
# `intake` reaches EBI's BLAST REST service today behind its `remote` flag -- the sequence
# similarity lane is written, network-wired, provenance-recorded, and merely not separately
# callable -- and `structures.evidence` reaches PDBe and UniProt. InterPro's REST API is free
# and unauthenticated, keyed on the accession identification already yields, but the exact
# path below is from third-party summaries and is unverified here.
PROPOSED = {}
"""Empty, and empty for the reason the table existed: everything in it got built.

It held three lanes -- `similarity.homologs`, `families.entries` and `literature.search` --
and all three are now registered tools with dispatchers, so keeping them here would make
`--proposed` claim as unbuilt what the registry serves. A test pins that no name here is
registered, and it is the pin that forced this edit rather than letting the two tables drift.

Two of the three were not built as declared, which is the useful record. The homology lane
named EBI's NCBI-BLAST REST service; that endpoint answers HTTP 400 without an `email` field,
so it is not an unauthenticated tool and the lane shipped on EBI's HMMER phmmer API instead --
keyless, and about two seconds against SwissProt where the service this project blamed for a
fifteen-minute queue was the wrong one to blame. The literature lane named Europe PMC and kept
it, and the reason it was chosen held up: a publication-date ceiling is what makes a hindsight
audit falsifiable.

**What should be declared here next, stated as an argument rather than a table.** Five of the
seven dossier axes are gated by no registered tool: `reagent_tractability`, `essentiality`,
`conservation`, `assay_availability` and `public_labels`. The first is the hard gate, and it is
the one no database lookup plausibly answers -- whether a soluble, correctly folded,
immobilisable reagent can be obtained is a wet-lab fact, and a tool claiming it from public
records would be the worst overclaim available here. The other four are reachable in principle:
three of the built lanes already retrieve material bearing on them and none is declared as
evidence yet, because an axis claim needs a *dated* claim and their dates are per-row -- a
publication date, a bioactivity year, a database release. That row-set work is the next thing
to build, and it needs no new endpoint.

Entries go back in this table only with an endpoint verified live, which is how the three that
left it were written.
"""


def from_sequence(genome_accession=True, unresolved=AXES):
    """The situation a campaign actually starts in: an accession, and seven open axes.

    A convenience and a claim at once. The claim is that this is the real starting state -- the
    task statement supplies a nucleotide accession and nothing else, every protein-level fact
    is downstream of identification, and no axis is resolved before anything has been read.
    """
    known = frozenset({"genome_accession"} if genome_accession else set())
    return Situation(known=known, unresolved=tuple(unresolved))
