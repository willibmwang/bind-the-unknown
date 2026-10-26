"""What a tool is, in one place, so that adding one is a record and not a new call site.

Four pipelines here are already tools in everything but name. `intake.run_intake` takes an
accession and returns ranked targets. `structures_evidence.build` takes a coverage threshold
and returns dated structural evidence. `chemistry.library_spec` is *already* invoked as a
tool, from `emergency.py` under the `invoke_design_tool` action, by a call site that knows
its name, that it costs one action, and nothing else -- not what it accepts, not what it
returns, not how it fails. A second arm now has to decide which of them to spend an action
on, and that decision cannot be made from a function object.

A registry rather than a class hierarchy, for the reason `arms` and `surrogates` already
give: what varies between tools is a handful of facts and one entry point, which is a record
and not a subclass. One registry serves both arms, so a molecular pipeline and a literature
endpoint are described in the same vocabulary and compete for the same budget on the same
terms.

**This is not the action space, and the distinction is load-bearing.** `dossier.ACTIONS` has
six entries, of which three (`rerank_targets`, `recommend_experiment`, `stop`) are control
flow over the episode and one (`retrieve_evidence`) is an operation on the dossier. None of
those four is a callable and none belongs here. Only `request_structure` and
`invoke_design_tool` are tool-backed today. Conflating the two sets would mean either
inventing entry points for `stop` or dropping the three tools that no action yet names, and
the first is a lie while the second defeats the purpose. So `ACTIONS` stays where it is and
keeps its order -- `twins.py:409` reads `list(ACTIONS)` into an enum and `invariance`
measures an axis-order nuisance at 0.116, so the order is a recorded quantity -- and a test
pins that every tool-backed action names a registered tool.

Three further decisions are worth defending.

**Every entry point is a string, not a callable.** `entry` is `"hpc_pilots.intake:run_intake"`,
resolved by `resolve()` at call time, and `check` is the same shape. Three things follow.
This module imports nothing but the standard library and `common` -- itself pure stdlib --
so a caller that wants one tool's declared cost does not pay for biotite and pyrodigal to
get it, which is the property `arms` states as "nothing here reads a file or imports numpy".
The whole registry is JSON-serialisable, so a spec set has a content digest and a run can
name the tool set it ran against. And the specs can be diffed, which is what makes a
proposed edit a reviewable object rather than a patch.

**The spec is a claim about a function that nothing enforces.** A declared parameter list can
drift from the signature it describes and every call will still work, right up to the one
that does not. So the conformance check is a test over `inspect.signature` for every local
tool, and `tools --check` runs before a snapshot is cut. A spec that describes a function
nobody can call is worse than no spec, because the caller trusts it.

**An enum that varies per episode is bound late, by name.** `Parameter.enum` is for values
fixed in source; `enum_from` names a key the caller supplies at call time. A spec that
hard-coded target names would survive `invariance.relabel_packet`, which rewrites packet
content and not source constants -- so the identity orbit would read a renamed packet while
the model still saw the real names, and `attributable_to_identity` would stop meaning
anything. That number is pinned at exactly zero by two tests, and this is how it stays there.

What this is not: nothing here validates an argument, type-checks a return, or executes
anything. It is the description of the tools. The caller is a separate concern with a
separate artifact, and the optimiser proposing changes to these fields is a third.
"""
import json
from dataclasses import asdict, dataclass, field
from typing import Optional

from .common import content_digest

# Bumped by hand when a snapshot is cut. Deliberately not derived from the specs: a digest
# answers whether two spec sets differ, a version answers which file a reader should open.
TOOL_SET_VERSION = 1

# The declared types a parameter may take. Short on purpose, and not a coercion table: a
# string where an integer is declared is a contract violation and not an input to parse.
KINDS = ("str", "int", "float", "bool", "tuple", "dict")

# The JSON Schema type each kind emits. One source of truth, derived rather than declared
# twice, because two fields that can disagree is the failure `arms` was written to end.
JSON_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean",
              "tuple": "array", "dict": "object"}

# What the caller does with a return. `evidence` may become a dated dossier entry;
# `artifact` is recorded and charged but substantiates nothing; `none` returns no value.
RETURN_KINDS = ("evidence", "artifact", "none")

# Where a parameter's value is allowed to have come from, checked by the caller's policies.
# `any` is the default and means unconstrained -- it never means trusted.
TRUST = ("any", "user_only")

# The only fields an optimiser proposal may touch. Everything absent is either a claim about
# the world (`citation`, `produces_evidence`, `date_field`) or a statement about which code
# runs (`entry`, `check`, `endpoint`, `requires`, `returns`, `return_kind`), and no field any
# run records can support changing one. An edit outside this set is refused by name rather
# than evaluated on its merits.
PROPOSABLE = ("summary", "description", "parameters", "failures", "warning",
              "cost_seconds", "cost_network_calls", "cost_actions")


@dataclass(frozen=True)
class Parameter:
    """One argument a tool accepts, as the caller must supply it."""

    name: str
    """Must match the underlying signature exactly. Not a display label."""
    kind: str
    """One of `KINDS`: the type the value must already be, not one it is coerced to."""
    meaning: str
    """What the value selects. Not a usage hint, not an example, and not the rationale for a
    default -- a sentence saying which input usually works best is the answer leaking in."""
    required: bool = True
    """True when the signature has no default. Not a judgement about importance: `record=True`
    is unimportant and optional, `accession` is essential and required, and conflating the two
    is how a reader learns the spec was written from the docstring."""
    default: object = None
    """The value used when the argument is absent. Must equal the signature's default and must
    be JSON-serialisable. Not a value this record is free to choose."""
    enum: tuple = field(default_factory=tuple)
    """Admissible values that are fixed in source. Empty means unconstrained *by enumeration*;
    it never means every value is admissible, because `kind` still binds."""
    enum_from: Optional[str] = None
    """Key into the caller's `choices` mapping, for a vocabulary that varies per episode.
    Mutually exclusive with `enum`. See the module docstring: a target list written into
    source here would make the identity orbit unreadable."""
    minimum: Optional[float] = None
    """Inclusive lower bound, or None. Not expressible in a strict JSON schema -- OpenAI's
    strict mode drops it and both vLLM grammar backends ignore it -- so a violation is a
    well-typed wrong guess the caller charges and records, never an error it raises."""
    maximum: Optional[float] = None
    """Inclusive upper bound, with the same standing as `minimum`."""
    max_length: Optional[int] = None
    """Length ceiling for a string or sequence, with the same standing as `minimum`."""
    trust: str = "any"
    """One of `TRUST`. `user_only` means the value must trace to the task statement rather
    than to anything a tool returned. Read by the caller's policies, stripped before the
    schema is emitted so the model never sees the annotation."""

    def json_type(self):
        return JSON_TYPES[self.kind]

    def values(self, choices=None):
        """The admissible set, resolving `enum_from` against the caller's vocabulary.

        Returns `()` when the parameter is unconstrained by enumeration. A declared
        `enum_from` the caller did not supply raises rather than silently widening to
        unconstrained, which would turn a missing vocabulary into a permissive schema.
        """
        if self.enum:
            return tuple(self.enum)
        if self.enum_from is None:
            return ()
        if choices is None or self.enum_from not in choices:
            raise ValueError(
                f"Parameter {self.name!r} draws its values from {self.enum_from!r}, which the "
                f"caller did not supply; supplied keys are "
                f"{sorted(choices) if choices else []}")
        return tuple(choices[self.enum_from])


@dataclass(frozen=True)
class Failure:
    """One way a tool is known to fail: what the record shows, why, and what it raises."""

    symptom: str
    """What appears in an artifact when it happens, quoted where a run has quoted it."""
    cause: str
    """Why it happens. Not a guess -- a cause nothing recorded supports is an assertion from
    memory, and belongs in neither this tuple nor a proposal."""
    raises: str
    """`"ValueError"` for a contract violation, `"RuntimeError"` for the environment, or `""`
    when the tool returns a stated refusal instead of raising. The caller branches on this,
    so it is not a message and not a description."""
    observed_in: tuple = field(default_factory=tuple)
    """Paths under `runs/` where this failure is on the record. Empty is permitted only for a
    failure the tool's author declared at registration; an optimiser may never propose one,
    because an unobserved failure mode is the claim `targets.py` marks `verified=False`."""


@dataclass(frozen=True)
class Tool:
    """One tool: what it is called, how it is reached, what it costs, and how it fails."""

    name: str
    """Registry key, and the name a finder matches. Stable across versions: a renamed tool is
    a different tool, and every artifact naming the old one would stop resolving."""
    version: int
    """Monotone integer, bumped when any field changes. Not semver: the only question this
    project asks of a spec is whether it is the same bytes, and semver invites a judgement
    about whether a change is breaking that nothing here can check."""
    summary: str
    """One line, and the only field a finder indexes. Kept apart from `description` so that
    rewriting a description cannot move retrieval -- if retrieval read the paragraph, every
    description edit would silently change which tools are found, and the cheapest available
    hack would be writing the expected query into the prose."""
    description: str
    """What the tool does and what it does not, for whoever or whatever reads the registry.
    Not a parameter list; `parameters` is that."""
    entry: str
    """`"module.path:attribute"`, resolved lazily. Empty only for a remote-only tool."""
    endpoint: Optional[str] = None
    """Key into `configs/endpoints.json` for a remote tool; None means local. Not a URL: a URL
    in source cannot carry a token file, and a host move would read as a contract change."""
    parameters: tuple = field(default_factory=tuple)
    """`Parameter` records in signature order. Not necessarily every argument the signature
    accepts -- a pipeline's internal knobs stay unexposed -- but every one named here must
    exist in the signature, which is pinned by a test rather than trusted."""
    returns: tuple = field(default_factory=tuple)
    """`(key, kind, meaning)` triples for the top-level keys of the returned dict. Descriptive
    only: the caller dispatches on `return_kind`, because a dispatcher that had to read prose
    would be parsing documentation."""
    return_kind: str = "artifact"
    """One of `RETURN_KINDS`. This is what the caller branches on."""
    cost_seconds: float = 0.0
    """Median wall seconds per call, measured. 0.0 means unmeasured, which is an honest
    starting value: a guessed one would make the first optimiser proposal unfalsifiable. Not a
    timeout -- `worst_case_seconds` is the gate, and a declared cost the caller enforced would
    be unfalsifiable by construction."""
    worst_case_seconds: Optional[int] = None
    """The ceiling the caller refuses against before dispatch, or None for work with no
    bound it can enforce. A different question from `cost_seconds`: the median is for cost
    accounting, the worst case is the wall-budget gate, and one field would make the gate
    either toothless or permanently closed."""
    cost_network_calls: int = 0
    """Outbound requests one call makes, including retries. Not a rate limit -- a limit belongs
    in `failures`, where a caller can branch on hitting it."""
    cost_actions: int = 1
    """How much of an episode's action budget one call spends. One everywhere in v1, which is
    exactly what `emergency.py`'s per-round increment already charges, so no recorded budget
    arithmetic changes. Any other value is a new regime and must be visible in the artifact."""
    requires: tuple = field(default_factory=tuple)
    """Import names that must be available, checked by `available()` before dispatch so a
    missing backend names its cause instead of surfacing as an ImportError from inside the
    fifth round of an episode. Only what the *default* path needs: `intake` reaches `Bio` at
    line 1309 and `pyrodigal` at 1448, both off the default viral path, so declaring them
    would refuse a tool that works. Not a version pin -- `requirements.lock.txt` is that."""
    failures: tuple = field(default_factory=tuple)
    """`Failure` records. The one field a run's record can genuinely extend, because
    `trace[].result` and `agent.incidents[]` log failures as they happen. Never exhaustive,
    and never to be read as such: an absent failure mode means unobserved, not impossible."""
    check: Optional[str] = None
    """`"module.path:attribute"` for a cross-field validator taking the argument mapping and
    raising `ValueError`, or None. A string for the same reasons `entry` is one, and
    especially so that editing it moves the spec-set digest."""
    produces_evidence: bool = False
    """Whether a return may be observed into a `Dossier`. True obliges `date_field` and
    `citation`, because `dossier.evidence` requires `source` and `available_from`, and the
    date boundary is the leakage boundary this project is built on."""
    date_field: Optional[str] = None
    """Key in the return supplying `available_from`. Legal as None only when
    `produces_evidence` is False. Not the time of the call: a structure deposited in 2021 is
    evidence from 2021 however recently it was fetched."""
    citation: str = ""
    """What a claim derived from this tool must cite. Not a licence notice, and not optional
    for an evidence-producing tool -- an uncited claim cannot satisfy a hard gate, by
    `dossier.readiness`'s construction."""
    warning: str = ""
    """The caveat carried into every result that used this tool, so it travels with the number
    rather than with the paper. The precedent is `Arm.warning`."""

    def available(self):
        import importlib.util

        return [name for name in self.requires if importlib.util.find_spec(name) is None]

    def is_remote(self):
        return self.endpoint is not None

    def resolve(self, attribute="entry"):
        """Import an entry point and return it, unwrapped.

        Returns the function itself and never a wrapper. A retry, timeout or cache applied
        here would change what `runs/intake_validation.json` and `runs/surface_markers.json`
        already recorded, without changing any file that says so, which is the definition of a
        silently invalidated number. A caller that wraps writes its own artifact.
        """
        import importlib

        target = getattr(self, attribute)
        if not target:
            raise ValueError(f"Tool {self.name!r} declares no {attribute}")
        module_name, _, name = target.partition(":")
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            raise RuntimeError(f"Tool {self.name!r} names {target!r}, which will not import "
                               f"here: {error}") from error
        if not hasattr(module, name):
            raise ValueError(f"Tool {self.name!r} names {target!r}, and {module_name} has no "
                             f"attribute {name!r}")
        return getattr(module, name)

    def record(self):
        """The spec as plain JSON data, which is what gets digested and diffed."""
        return asdict(self)


# Three tools, chosen so the trio exercises every field rather than to be a useful library.
# `chemistry.library_spec` is already called as a tool from `emergency.py` under
# `invoke_design_tool`, so registering it tests the registry against a live call site at no
# risk. `structures.evidence` is the only one that produces dated evidence, which is the
# richest epistemic case. `intake.rank_targets` is the only one that reaches the network.
TOOLS = {
    "chemistry.library_spec": Tool(
        name="chemistry.library_spec", version=1,
        summary="Define a focused thioether-macrocycle library over a randomised region.",
        description="Returns a library definition: a fixed chloroacetyl initiator, a "
                    "randomised region of the given length bounds, a downstream cysteine "
                    "closure, and the elongator alphabet implied by the requested monomer "
                    "classes. It defines a library and does not synthesise, screen or score "
                    "one, and the theoretical diversity it reports is a count of sequences "
                    "the definition admits rather than a prediction that any of them bind.",
        entry="hpc_pilots.chemistry:library_spec",
        parameters=(
            Parameter("initiator", "str",
                      "Chloroacetyl initiator, which must be one the chemistry module "
                      "registers.", required=False, default="ClAc-D-Tyr"),
            Parameter("random_region", "tuple",
                      "Inclusive (low, high) length bounds for the randomised region.",
                      required=False, default=(4, 8)),
            Parameter("elongator_classes", "tuple",
                      "Monomer classes admitted into the elongator alphabet.",
                      required=False,
                      default=("canonical", "N-methyl", "D-alpha-amino")),
        ),
        returns=(("initiator", "str", "The initiator the library fixes."),
                 ("closure_residue", "str", "The residue that closes the macrocycle."),
                 ("random_region", "tuple", "The length bounds, as [low, high]."),
                 ("elongator_alphabet", "tuple", "Monomer codes the region draws from."),
                 ("theoretical_diversity", "int",
                  "Sequences the definition admits. Not a library that exists, and not a "
                  "count of anything measured.")),
        return_kind="artifact",
        cost_seconds=0.0, worst_case_seconds=None, cost_network_calls=0, cost_actions=1,
        requires=(),
        failures=(
            Failure("'<name>' is not a registered chloroacetyl initiator",
                    "The initiator is validated against the chemistry module's own table, "
                    "which is narrower than the set of strings the schema admits.",
                    "ValueError"),
            Failure("Invalid randomised-region bounds",
                    "The bounds must satisfy 1 <= low <= high, which a JSON schema cannot "
                    "express portably, so the function enforces it.",
                    "ValueError"),
            Failure("Empty elongator alphabet",
                    "Every requested monomer class was unknown, leaving nothing to draw "
                    "from.",
                    "ValueError"),
        ),
        produces_evidence=False, date_field=None,
        citation="A library definition is a design choice, not a cited claim, and cannot "
                 "substantiate a dossier axis on its own.",
        warning="Theoretical diversity is a combinatorial count over the definition. It is "
                "not a synthesised library, not a screened one, and no macrocycle has been "
                "measured anywhere in this repository."),
    "structures.evidence": Tool(
        name="structures.evidence", version=1,
        summary="Dated structural coverage per declared target, from PDBe and UniProt.",
        description="For each target in the structures config, resolves the UniProt entry, "
                    "the declared or derived residue ranges, and the PDBe entries covering "
                    "them, then reports the earliest deposition date and best resolution. "
                    "The structure count is a proxy for how much attention a target has "
                    "received; it is not a pocket score, not a druggability estimate, and "
                    "the two must never be pooled.",
        entry="hpc_pilots.structures_evidence:build",
        parameters=(
            Parameter("minimum", "float",
                      "Fraction of a declared range a structure must cover to count.",
                      required=False, default=0.5, minimum=0.0, maximum=1.0),
            Parameter("record", "bool",
                      "Whether the function writes its own artifact. A call made inside an "
                      "episode passes False, because the episode writes the record.",
                      required=False, default=True),
        ),
        returns=(("accession", "str", "The UniProt accession resolved for the target."),
                 ("structures", "int",
                  "Entries covering a declared range. An attention proxy, not a score."),
                 ("earliest_deposition", "str",
                  "ISO date of the earliest covering entry, which is what dates the claim."),
                 ("best_resolution", "float", "Best resolution among covering entries.")),
        return_kind="evidence",
        cost_seconds=0.0, worst_case_seconds=None, cost_network_calls=0, cost_actions=1,
        requires=(),
        failures=(
            Failure("urllib.error.HTTPError from the PDBe or UniProt host",
                    "Both are public endpoints with no declared rate limit here, and the "
                    "entry-summary call POSTs in batches of 400.",
                    "RuntimeError"),
        ),
        produces_evidence=True, date_field="earliest_deposition",
        citation="The UniProt accession and the PDBe entry identifiers covering the declared "
                 "range. A count with no entry identifiers is not a cited claim.",
        warning="A structure count is a research-attention proxy and a pocket score is not. "
                "They are different axes and must not be pooled. Coverage is computed per "
                "entry rather than per chain, because an obligate complex is two polymer "
                "entities and a per-chain test returns zero for every such structure ever "
                "deposited."),
    "intake.rank_targets": Tool(
        name="intake.rank_targets", version=1,
        summary="Pathogen genome accession to a ranked list of candidate binding targets.",
        description="Six-frame ORF calling, homology identification against a reviewed "
                    "UniProt reference, then a stated and unfitted ranking over "
                    "conservation, surface exposure, essentiality and druggability "
                    "features. Returns candidates with their features. It predicts no "
                    "binding, affinity or inhibition for any of them, and the worth-binding "
                    "score it reports is a separate axis from the confidence that the "
                    "identification itself is correct.",
        entry="hpc_pilots.intake:run_intake",
        parameters=(
            Parameter("accession", "str",
                      "NCBI nucleotide accession for the genome to read."),
            Parameter("exclude_organism", "str",
                      "Organism removed from the reference before identifying, which scores "
                      "a curated genome as if it were novel.", required=False, default=None),
            Parameter("top", "int", "How many of the longest ORFs to identify.",
                      required=False, default=12, minimum=1),
            Parameter("remote", "bool",
                      "Whether identification may fall back to the remote aligner. The "
                      "default local k-mer path is the one the recorded runs used.",
                      required=False, default=True),
            Parameter("record", "bool",
                      "Whether the function writes its own artifact. A call made inside an "
                      "episode passes False, because the episode writes the record.",
                      required=False, default=True),
        ),
        returns=(("targets", "tuple",
                  "Ranked candidate rows carrying score, conservation and tier."),
                 ("rubric", "dict", "The stated ranking weights, which are not fitted."),
                 ("provenance", "dict", "Host, commit and time, from common.provenance.")),
        return_kind="artifact",
        cost_seconds=0.0, worst_case_seconds=None, cost_network_calls=0, cost_actions=1,
        requires=(),
        failures=(
            Failure("A holdout that removes nothing raises rather than proceeding",
                    "`build_index(require_removal=True)` refuses an exclusion that matched "
                    "no reference entry, because a holdout that silently removed nothing "
                    "would report leakage as generalisation.",
                    "ValueError"),
            Failure("The remote aligner reports a non-finished status",
                    "The EBI service queues; a 15-minute RUNNING queue is on the record as "
                    "the reason the local k-mer path is the default.",
                    "RuntimeError"),
        ),
        produces_evidence=False, date_field=None,
        citation="Identification derives from the named reviewed UniProt release. A ranked "
                 "target row is not a cited claim and cannot satisfy a dossier axis alone.",
        warning="The worth-binding score and the identification confidence are different "
                "axes and must not be pooled. A high score resting on a hydropathy-only "
                "assignment is a row a reader has to be able to see."),
}


def tool(name):
    """Look up a registered tool, or raise naming what is registered.

    Never falls back to a default. An unknown tool name is a typo or a stale artifact, and
    both are worth stopping for; silently substituting one would make the substitution
    indistinguishable from the caller having asked for it.
    """
    if name not in TOOLS:
        raise ValueError(f"Unknown tool {name!r}; registered tools are {sorted(TOOLS)}")
    spec = TOOLS[name]
    missing = spec.available()
    if missing:
        raise RuntimeError(
            f"Tool {name!r} needs {missing}, which is not importable here. Declared "
            f"dependencies are checked before dispatch so that a missing backend names its "
            f"cause, rather than surfacing as an ImportError from inside an episode.")
    return spec


def register(spec):
    """Add a tool to the registry, validating the record rather than trusting it.

    Every check here is for a defect that would otherwise surface far from its cause: an
    unknown kind as a malformed schema the provider rejects, a missing date field as evidence
    entering a dossier at any clock, a non-serialisable default as a snapshot that will not
    cut. Validating at import means a bad record cannot reach a run at all.
    """
    if not isinstance(spec, Tool):
        raise TypeError("A tool must be a Tool record")
    if spec.name in TOOLS and TOOLS[spec.name] is not spec:
        raise ValueError(f"Tool {spec.name!r} is already registered")
    if not spec.entry and not spec.endpoint:
        raise ValueError(f"Tool {spec.name!r} declares neither an entry point nor an endpoint")
    if spec.return_kind not in RETURN_KINDS:
        raise ValueError(f"Tool {spec.name!r} declares return_kind {spec.return_kind!r}; "
                         f"known kinds are {list(RETURN_KINDS)}")
    if spec.produces_evidence and not (spec.date_field and spec.citation):
        raise ValueError(
            f"Tool {spec.name!r} produces evidence but declares date_field "
            f"{spec.date_field!r} and citation {spec.citation!r}. `dossier.evidence` requires "
            f"a source and an available_from, and an undated claim would enter a dossier at "
            f"any clock.")
    for parameter in spec.parameters:
        if parameter.kind not in KINDS:
            raise ValueError(f"Tool {spec.name!r} parameter {parameter.name!r} declares kind "
                             f"{parameter.kind!r}; known kinds are {list(KINDS)}")
        if parameter.trust not in TRUST:
            raise ValueError(f"Tool {spec.name!r} parameter {parameter.name!r} declares trust "
                             f"{parameter.trust!r}; known values are {list(TRUST)}")
        if parameter.enum and parameter.enum_from:
            raise ValueError(f"Tool {spec.name!r} parameter {parameter.name!r} declares both "
                             f"a fixed enum and an enum_from; one vocabulary, not two")
        try:
            json.dumps(parameter.default)
        except TypeError as error:
            raise ValueError(f"Tool {spec.name!r} parameter {parameter.name!r} has a default "
                             f"that will not serialise: {error}") from error
    TOOLS[spec.name] = spec
    return spec


def spec_set_digest():
    """A hash of what the registry says, independent of registration order.

    Stamped into a run artifact beside `config_sha256`, so a result names the tool set it ran
    against and a later spec edit cannot silently reinterpret it. `content_digest` rather than
    a digest over the snapshot file, because the snapshot carries a generation timestamp and
    re-cutting a byte-identical registry would otherwise hash differently -- which is the
    defect `content_digest` was written for.
    """
    return content_digest({name: TOOLS[name].record() for name in sorted(TOOLS)})


def snapshot():
    """The registry as a committed, byte-immutable record.

    A verifier and not a loader. Nothing in this package reads a snapshot at runtime; `tools
    --check` compares the live digest against it and exits non-zero on drift, the way
    `manifest --check` does. That is what makes an optimiser proposal unable to reach a run:
    the only thing it can write is JSON, and no code path loads JSON into the registry.
    """
    from datetime import datetime, timezone

    return {"version": TOOL_SET_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "tool_set_sha256": spec_set_digest(),
            "note": "Byte-immutable once committed, so the tool_set_sha256 embedded in "
                    "already-recorded runs still verifies. A changed spec goes in a new "
                    "tools.v{n+1}.json rather than editing this file, which is the device "
                    "configs/today.json already uses.",
            "tools": {name: TOOLS[name].record() for name in sorted(TOOLS)}}
