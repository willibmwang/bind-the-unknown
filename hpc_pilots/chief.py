"""Validate a tool call, execute it, and return an envelope that says what it is worth.

The caller is where a tool stops being a description and becomes a number in a record, so
the rules that matter are about what a return is allowed to claim rather than about dispatch.

**The line between raising and charging is drawn exactly where the emitted schema's
expressive power ends.** A violation of something the schema declared means constrained
decoding did not happen -- the provider ignored the grammar, or the enum was built wrongly --
and that is an instrument failure which must abort loudly so it gets fixed. A violation of
something only the spec declares is a well-typed wrong guess: the model was never constrained
against it, so a violation is a policy result and belongs in the record. Concretely, an
argument of the wrong type raises, and a `limit` outside its declared bounds is charged and
recorded, because a strict JSON schema cannot carry a numeric bound portably -- OpenAI's
strict mode drops `minimum` and both vLLM grammar backends ignore it.

That split is not invented here. `run_episode` already raises on an action outside the
allowed set and already charges an action that named a target and axis with no evidence
behind it. This module generalises the rule it was already following.

**Nothing is coerced.** A `"8"` where an integer was declared means the decoder did not
honour the schema, and repairing it here would hide the one failure this check exists to
surface. `True` is rejected for an integer parameter even though `isinstance(True, int)` is
True in Python: a model emitting `true` for a count is not off by a type, it is answering a
different question. The argument is the same one `RETRYABLE_FINISH` makes for leaving an
invalid action out of the retry set.

**A missing backend returns an explicit absence, charged.** `emergency.py` has done this for
one action since the structure arm was specified, and thirteen recorded artifacts carry its
string. Here it becomes a property of the dispatcher instead of something whoever writes the
next tool has to remember, which is the only way it stays true.

**A tool failure is not retried.** Retrying it is resampling, and resampling a wrong answer
until it is right is the objection this project raises against retrying an invalid action. A
tool may declare `retry_transport`, which retries only the 429/5xx/no-status classes
`Agent._post` already retries and counts them separately so the recorded wall time stays
attributable.

What this is not: it does not decide which tool to call -- `finder` does that -- and it does
not write anything. A tool returns; the episode loop writes. Five rules keep that true and
each is checked rather than documented, because `offload` already had to state the first one
and a container is a much more obvious boundary than a function call.
"""
import contextlib
import functools
import importlib.util
import socket
import threading
import time
from dataclasses import dataclass


# What happened to a call. `empty` is a successful call that found nothing, which is a
# different thing from a failure and must not be counted as one: the structures table
# legitimately returns undated rows for targets with no deposited structure.
STATUSES = ("ok", "empty", "unavailable", "rejected", "failed")

# Where a value came from, which is the only thing that decides whether it can be cited.
# Deliberately four and not a free string: a source kind nobody enumerated is a claim whose
# standing nobody has decided.
SOURCE_KINDS = ("external_record", "computed", "internal_assessment", "model_assertion")

# A claim from one of these may enter a dossier. `model_assertion` is absent on purpose, and
# that absence is the rule: anything a model said cannot substantiate an axis.
CITABLE_KINDS = ("external_record", "computed", "internal_assessment")

# And of those, only these carry `verified=True`. `internal_assessment` is excluded for the
# reason `targets.INTERNAL` already encodes: an in-house judgement is a real input and not a
# citation, and `dossier.assess` excludes unverified claims from substantiation.
VERIFIED_KINDS = ("external_record", "computed")

# Byte-identical to what `emergency.py` has emitted since the structure arm was specified,
# because `structure_requests_unserviced` counts it and thirteen recorded traces carry it.
# The template generalises the discipline without moving the one instance that exists.
NO_BACKEND = ("no {backend} backend configured; request recorded and charged, "
              "no {claim} produced")


class ToolCallRejected(ValueError):
    """Validated arguments do not satisfy what the emitted schema declared.

    A `ValueError` subclass deliberately, so it aborts an episode the way an action outside
    the allowed set does, and is not swallowed the way `ContentFiltered` is. `ContentFiltered`
    is missing data with a known provider cause and a defensible retry. This is a declared
    constraint that was not honoured, and recording a *rate* for it would turn a provider
    silently ignoring the schema into a property of the policy being measured.
    """


def _is_kind(value, kind):
    """Whether a value already is what the parameter declared, with no coercion offered.

    `bool` is excluded from `int` and `float` first, because Python says `isinstance(True,
    int)` and that is the wrong answer to this question.
    """
    if kind == "bool":
        return isinstance(value, bool)
    if isinstance(value, bool):
        return False
    if kind == "int":
        return isinstance(value, int)
    if kind == "float":
        return isinstance(value, (int, float))
    if kind == "str":
        return isinstance(value, str)
    if kind == "tuple":
        return isinstance(value, (list, tuple))
    if kind == "dict":
        return isinstance(value, dict)
    raise ValueError(f"Unknown kind {kind!r}")


def schema_conformance(spec, arguments, choices=None):
    """Re-check everything the emitted schema constrained. Raises; never repairs.

    Called after the model replies and before anything executes. Every check here corresponds
    to something the schema declared, so a failure means the grammar was not honoured and the
    run is not measuring what it claims to measure.

    Takes the spec record rather than its name, deliberately. The caller already holds the
    record it showed the model, and re-resolving a name here would make the global registry a
    hidden dependency of validation -- so a tool shown to the model but not registered, which
    is every tool in a test and every tool in a candidate spec set, could not be validated at
    all. Validate what you were handed.
    """
    declared = {p.name: p for p in spec.parameters}
    for parameter in spec.parameters:
        if parameter.name not in arguments or arguments[parameter.name] is None:
            if parameter.required:
                raise ToolCallRejected(
                    f"{spec.name} requires {parameter.name!r} and the reply left it out")
            continue
        value = arguments[parameter.name]
        if not _is_kind(value, parameter.kind):
            raise ToolCallRejected(
                f"{spec.name}.{parameter.name} must be {parameter.kind}, got "
                f"{type(value).__name__} {value!r}. The schema declared the type, so this is a "
                f"decoder that did not honour it and not a choice the policy made.")
        admissible = parameter.values(choices)
        if admissible and value not in admissible:
            raise ToolCallRejected(
                f"{spec.name}.{parameter.name} is outside the declared set; the schema "
                f"enumerated it, so the decoder did not honour the enum")
    for supplied in arguments:
        if supplied not in declared:
            raise ToolCallRejected(
                f"{spec.name} declares no parameter {supplied!r}; a flat schema carries a "
                f"property for every shown tool and it must arrive null here")
    return spec


def semantics(spec, arguments):
    """Limits and cross-field constraints the schema could not carry. Returns a reason or None.

    These are charged rather than raised because a strict JSON schema cannot express a numeric
    bound portably, so the model was never constrained against one and a violation is a wrong
    guess. This project does not resample a wrong guess away.
    """
    for parameter in spec.parameters:
        value = arguments.get(parameter.name)
        if value is None:
            continue
        # Statically declared enums are re-checked here even though `schema_conformance`
        # already raises on them, because `call` does not run that function and a caller may
        # not either. Without this a value the spec forbids reaches the tool and comes back as
        # the tool's own failure, which misattributes a caller's error to the tool and puts it
        # in the record `optimizer.fidelity` reads. Charged rather than raised, because `call`
        # returns an envelope and a refusal before dispatch is exactly what `rejected` means.
        #
        # Only `enum` is checked and never `enum_from`: a vocabulary-sourced set needs the
        # caller's `choices`, which this function is not given, and widening to unconstrained
        # when it is absent would turn a missing vocabulary into a permissive check. Those stay
        # with `schema_conformance`, which has the vocabulary.
        if parameter.enum and value not in parameter.enum:
            return (f"{parameter.name}={value!r} is not one of {tuple(parameter.enum)}")
        if parameter.minimum is not None and value < parameter.minimum:
            return (f"{parameter.name}={value!r} is below the declared minimum "
                    f"{parameter.minimum}")
        if parameter.maximum is not None and value > parameter.maximum:
            return (f"{parameter.name}={value!r} is above the declared maximum "
                    f"{parameter.maximum}")
        if parameter.max_length is not None and len(value) > parameter.max_length:
            return f"{parameter.name} is longer than the declared {parameter.max_length}"
    if spec.check:
        try:
            spec.resolve("check")(arguments)
        except ValueError as refused:
            return str(refused)[:200]
    return None


def envelope(spec, arguments, status, value=None, source=None, source_kind=None,
             available_from=None, rows=None, note="", detail=None, seconds=None,
             network_calls=None):
    """A tool return, carrying the provenance that decides whether it can become evidence.

    Three rules, enforced here rather than written down, because each one has a precedent in
    this repository for being true only as long as somebody remembered it.

    * **`citable` is False unless the call succeeded.** An `unavailable`, `failed`, `rejected`
      or `empty` return has nothing to cite, and the absence of this rule is exactly how a
      stub becomes a claim.
    * **`citable` requires a date.** `Dossier.observe` compares `available_from` against the
      decision clock, so an undated return would enter a dossier at any clock and silently
      open the one boundary the packet design exists to hold. `structures_evidence` already
      states the rule: a claim is only usable in an episode if it carries the date it became
      true.
    * **`verified` is a function of `source_kind` and nothing else.** `model_assertion` is
      never verified and never citable, which makes structural what `assess` and `readiness`
      already enforce downstream.

    A successful return with no declared `source_kind` raises rather than degrading to
    uncitable. An uncitable success is a legitimate thing for a tool to produce -- a library
    definition is one -- and it must *declare* itself, because the alternative is that every
    tool with a provenance bug contributes quietly to `unverified_claims` and the rubric reads
    the bug as policy behaviour.

    A table-returning tool carries one entry per row in `rows`, each separately dated, because
    one call on `structures.evidence` produces eleven claims of which some are genuinely
    undated -- a target with no deposited structure has no earliest deposition, and that is a
    real zero rather than a missing value.

    **`seconds` and `network_calls` are measured, never declared.** They are what this call
    spent; `cost_seconds` and `cost_network_calls` on the spec are what it was advertised to
    cost. Keeping them in different places is the whole point -- a record that reported the
    declaration back would make every cost proposal unfalsifiable, which is the objection this
    project already raises against a guessed `cost_seconds`.
    """
    if status not in STATUSES:
        raise ValueError(f"Unknown status {status!r}; known statuses are {list(STATUSES)}")
    if source_kind is not None and source_kind not in SOURCE_KINDS:
        raise ValueError(f"Unknown source_kind {source_kind!r}; known kinds are "
                         f"{list(SOURCE_KINDS)}")
    if status == "ok" and source_kind is None:
        raise ValueError(
            f"{spec.name} returned ok with no source_kind. An uncitable success must declare "
            f"itself rather than arrive as one, or a provenance bug becomes a policy result.")
    # Citability needs all three: a successful call, a source kind whose claims may be cited,
    # and a spec that declared its returns can become evidence at all. The third is not
    # redundant -- `chemistry.library_spec` computes its answer honestly and is still not
    # evidence, because a library definition is a design choice rather than a finding about
    # the world. Without this clause it came back citable, which is the bug this comment
    # exists because of.
    citable = (status == "ok" and source_kind in CITABLE_KINDS and spec.produces_evidence)
    if citable and rows is None and not (source and available_from):
        raise ValueError(
            f"{spec.name} claims citability with source={source!r} and "
            f"available_from={available_from!r}; a claim with no source or no date cannot "
            f"enter a dated dossier")
    return {"tool": spec.name, "tool_version": spec.version, "arguments": dict(arguments),
            "status": status, "value": value, "rows": rows,
            "citable": citable and (rows is None or any(r["citable"] for r in rows)),
            # Verification is a property of a claim, so a return that is not a claim is not
            # verified either. Reporting `verified=True` on a library definition would put a
            # truth value on a design choice, and `unverified_claims` counts the complement.
            "verified": citable and source_kind in VERIFIED_KINDS,
            "source": source, "source_kind": source_kind, "available_from": available_from,
            "note": note, "detail": detail, "seconds": seconds,
            # What the call actually spent, as opposed to what the spec declares it costs.
            # Both are left None by a dispatcher that does not measure them and filled by
            # `call`, which is the only place that can: a dispatcher sees neither the clock it
            # started on nor the requests its callee made.
            "network_calls": network_calls,
            "host": socket.gethostname(),
            "interpretation": "A tool return. Citability and verification are properties of "
                              "the source, not of the value, and a status other than ok "
                              "carries no claim at all."}


@dataclass(frozen=True)
class Allowed:
    """The call may proceed."""


@dataclass(frozen=True)
class Denied:
    """The call may not proceed, with a reason the record keeps."""

    reason: str
    redact: bool = False
    """Whether the reason was computed from something a tool returned. A reason that quotes
    untrusted bytes back into the deciding model's context is a channel, so a redacted one
    reaches the model and the full one reaches only the artifact."""


def _trusted_fields_policy(spec, arguments, context):
    """Every parameter declaring `user_only` must trace to the task rather than to a return.

    This is structural provenance, and it is the honest form of a capability check in a
    single-context loop. A value-level dependency graph would collapse here: with one context,
    everything the model has seen taints everything it emits, so a reader-intersection check
    would deny every call. Checking that an argument *is* a value from a trusted source is
    what the published policies actually lean on, and it catches the substitution class.
    """
    for parameter in spec.parameters:
        if parameter.trust != "user_only":
            continue
        value = arguments.get(parameter.name)
        if value is None:
            continue
        # An empty trusted set with a `user_only` parameter denies every call that supplies a
        # value, for ever, and `redact=True` means the policy sees only a refusal that reads
        # like its own mistake -- so a harness defect would present as policy behaviour while
        # the policy kept spending charged actions on it. No caller in this repository
        # populates `trusted`, so declaring `user_only` today is a kill switch wearing the
        # shape of a hardening. Found by an adversary, not by review. It fails loudly instead.
        if not context.get("trusted"):
            raise RuntimeError(
                f"{spec.name}.{parameter.name} declares trust='user_only' and the caller "
                f"supplied no trusted set, so every call naming it would be denied for ever. "
                f"Populate chief.context(trusted=...) before declaring a trusted field.")
        if value not in context["trusted"]:
            return Denied(f"{spec.name}.{parameter.name} does not trace to the task statement",
                          redact=True)
    return Allowed()


# Matched by exact name, longest first, then falling through to the generic field policy. A
# tool nothing matches is denied, which is the one default worth having.
POLICIES = ()

# Tools that change nothing outside their return and so need no policy at all. Being explicit
# about which those are is what makes default-deny affordable.
# `geometry.surface_markers` is here with one thing stated rather than waved past: it writes
# coordinate files into `data/raw/structures/` through `geometry.fetch_structure`. That is a
# download cache in the tree `common.download` already caches into, not an artifact a run
# names, and the rule this tuple encodes is about the latter -- a tool returns, the episode
# writes. Registering it without an entry here was the first thing default-deny caught, which
# is the behaviour working: the call came back `rejected` with "no policy covers" rather than
# succeeding because nobody had thought about it.
NO_SIDE_EFFECT = ("chemistry.library_spec", "intake.rank_targets", "structures.evidence",
                  "geometry.surface_markers", "literature.search", "families.entries",
                  "similarity.homologs", "structures.predicted", "structures.folded",
                  "bioactivity.prior_binders")


def check_policy(spec, arguments, context):
    """Three gates in order, then default-deny.

    The order is the design. A read-only tool short-circuits, because requiring a policy for
    something that changes nothing buys nothing and spends review attention that the
    state-changing tools need. Then the named policies. Then denial -- because a tool nobody
    wrote a policy for is a tool nobody decided about, and reading that as permission is how a
    policy set rots as the registry grows.
    """
    import fnmatch

    for pattern, policy in POLICIES:
        if fnmatch.fnmatch(spec.name, pattern):
            return policy(spec, arguments, context)
    verdict = _trusted_fields_policy(spec, arguments, context)
    if isinstance(verdict, Denied):
        return verdict
    if spec.name in NO_SIDE_EFFECT:
        return Allowed()
    return Denied(f"No policy covers {spec.name!r}, so it is denied rather than assumed safe")


def context(dossiers=None, now=0, wall_budget_seconds=600, trusted=()):
    """Everything a tool is allowed to know, built by the one function that can build it.

    Deliberately cannot reach `pool`, `probes`, `decision_relevant` or the twins answer key.
    That is structural rather than scored, and it catches a leak no readout could: handing a
    policy the graded moves does not change what a deterministic policy does, so no number
    moves and an objective would happily admit it. The check is the one `improve.py` already
    uses -- build this with the oracle patched to raise, and see that it still builds.
    """
    return {"now": now, "wall_budget_seconds": wall_budget_seconds,
            "targets": tuple(d.target for d in (dossiers or ())),
            "trusted": frozenset(trusted)}


# Outbound request accounting. Thread-local, and patched at the transport exactly once.
#
# `cost_network_calls` is a declared number with no instrument behind it. `optimizer` names
# `trace[].tool_network_calls` as the evidence a proposal to change that field would need, and
# nothing in this package has ever written that key -- so a proposal to change a network cost
# could never be contradicted by a run, which is the shape of an unfalsifiable proposal and the
# same objection this project raises against a guessed `cost_seconds`.
#
# Counted at the transport rather than at each call site, because a tool reaches the network
# through whatever it imports and a tally maintained by hand drifts the first time one of them
# changes. Thread-local because `sweep` runs cells concurrently and a shared counter would
# attribute one cell's requests to another.
#
# The honest limit, which `reaches_network` and not this number is what covers: this counts
# `urllib.request.urlopen`, which is what every networked tool in this repository uses. A tool
# reaching the network through another transport counts zero here, and a zero from this
# instrument therefore means "none seen" rather than "none made".
_REQUESTS = threading.local()


def _install_request_counter():
    """Wrap `urlopen` once, idempotently, so a dispatch window can count what it spent.

    Idempotent by a marker on the wrapper rather than by a module flag, so that a reload of
    this module cannot double-wrap a transport that is already counted -- two layers would
    count every request twice and the number would be quietly wrong rather than absent.
    """
    import urllib.request

    if getattr(urllib.request.urlopen, "_chief_counted", False):
        return
    original = urllib.request.urlopen

    @functools.wraps(original)
    def counted(*args, **kwargs):
        pending = getattr(_REQUESTS, "count", None)
        # Outside a window the count is None, and a request made there is deliberately not
        # recorded anywhere: this instrument reports what one dispatch spent, and a global
        # total would answer a question nobody asked of it.
        if pending is not None:
            _REQUESTS.count = pending + 1
        return original(*args, **kwargs)

    counted._chief_counted = True
    urllib.request.urlopen = counted


@contextlib.contextmanager
def counting_requests():
    """Count outbound requests made on this thread for the duration of the block.

    Yields a mapping whose `made` key is final once the block exits. A nested window adds its
    own total back into the enclosing one on exit, so an inner dispatch's requests are counted
    by both and not lost by the outer -- the alternative silently under-reports whichever call
    happened to nest.
    """
    _install_request_counter()
    previous = getattr(_REQUESTS, "count", None)
    _REQUESTS.count = 0
    tally = {"made": 0}
    try:
        yield tally
    finally:
        made = _REQUESTS.count
        tally["made"] = made
        _REQUESTS.count = None if previous is None else previous + made


def call(spec, arguments, context):
    """One validated tool call to one envelope. Never a fabricated value.

    Five steps, and the order is the design: dependencies before dispatch so a missing backend
    names itself rather than surfacing from inside round five; an absent backend to an explicit
    charged absence; the wall budget refused before the clock starts rather than discovered at
    round three; execution; and any exception to a `failed` envelope, caught at exactly one
    place so that no tool can smooth over its own failure.
    """
    missing = [name for name in spec.requires if importlib.util.find_spec(name) is None]
    if missing:
        return envelope(spec, arguments, "unavailable",
                        note=NO_BACKEND.format(backend=spec.name, claim=spec.return_kind),
                        detail={"missing_imports": missing})
    if spec.name not in DISPATCH:
        return envelope(spec, arguments, "unavailable",
                        note=NO_BACKEND.format(backend=spec.name, claim=spec.return_kind),
                        detail={"reason": "registered but not wired to a dispatcher"})
    verdict = check_policy(spec, arguments, context)
    if isinstance(verdict, Denied):
        return envelope(spec, arguments, "rejected", note=verdict.reason,
                        detail={"redact": verdict.redact})
    refused = semantics(spec, arguments)
    if refused:
        return envelope(spec, arguments, "rejected", note=f"rejected: {refused}",
                        detail={"charged": True})
    if (spec.worst_case_seconds is not None
            and spec.worst_case_seconds > context["wall_budget_seconds"]):
        return envelope(
            spec, arguments, "unavailable",
            note=f"{spec.name} declares a worst case of {spec.worst_case_seconds}s against a "
                 f"wall budget of {context['wall_budget_seconds']}s; refused before dispatch")
    started = time.perf_counter()
    try:
        with counting_requests() as tally:
            result = DISPATCH[spec.name](spec, arguments, context,
                                         lambda **kw: envelope(spec, arguments, **kw))
    except Exception as failure:
        # Caught here and nowhere else, and not retried: retrying a tool failure is resampling.
        return envelope(spec, arguments, "failed",
                        note=f"{type(failure).__name__}: {str(failure)[:200]}",
                        seconds=time.perf_counter() - started,
                        network_calls=tally["made"])
    # Filled by the caller, and only where the dispatcher left None. Before this, two of the
    # three dispatchers never passed `seconds` and the third passed an explicit None, so every
    # successful call in the record carried no wall time while every failed one did -- the
    # exact opposite of the asymmetry a cost record wants. A dispatcher that genuinely measures
    # something finer than this clock still wins, because only None is overwritten.
    if result["seconds"] is None:
        result["seconds"] = time.perf_counter() - started
    if result["network_calls"] is None:
        result["network_calls"] = tally["made"]
    return result


def _library_spec(spec, arguments, context, wrap):
    value = spec.resolve()(**arguments)
    return wrap(status="ok", value=value, source_kind="computed",
                source="hpc_pilots.chemistry, from the registered monomer tables",
                seconds=None)


def _structures_evidence(spec, arguments, context, wrap):
    payload = spec.resolve()(**{**arguments, "record": False})
    table = payload.get(spec.rows) or {}
    rows = tuple({"name": name,
                  "available_from": row.get(spec.date_field),
                  # A target with no deposited structure has no earliest deposition. That is a
                  # real zero and not a missing value, so the row is kept and marked uncitable
                  # rather than dropped -- dropping it would make the table read as though
                  # every target had been structurally characterised.
                  "citable": row.get(spec.date_field) is not None,
                  "value": row}
                 for name, row in sorted(table.items()))
    return wrap(status="ok" if rows else "empty", value={"coverage_threshold":
                                                         payload.get("coverage_threshold")},
                rows=rows, source_kind="external_record",
                source="PDBe SIFTS best_structures and PDBe entry summary, against the named "
                       "UniProt release",
                available_from=None)


def _surface_markers(spec, arguments, context, wrap):
    """Macrocycle surface markers for one accession.

    Three returns are distinguished where the underlying function conflates two of them. The
    function answers `None` when no deposited chain is long enough to be the target's surface,
    and a dict carrying `error` when chains were offered and none could be measured. Both are
    successful calls that produced no marker, so both are `empty` rather than `failed` -- the
    distinction `STATUSES` draws, and the reason a target with no usable structure must not be
    counted as a tool that broke. The reasons travel in `detail` so an empty answer can still
    be told apart from a broken machine.

    `source_kind` is `computed` and not `external_record`. The coordinates are an external
    record; the patch area is a calculation over them with a stated cut-off, and reporting it
    as a record would let a computed number inherit a deposition's standing.
    """
    value = spec.resolve()(**arguments)
    if value is None:
        return wrap(status="empty", value=None, source_kind="computed",
                    source="RCSB coordinates via PDBe best_structures, measured with freesasa",
                    note=f"no deposited chain for {arguments.get('accession')!r} is long "
                         f"enough to carry the target's surface")
    if value.get("error"):
        return wrap(status="empty", value=None, source_kind="computed",
                    source="RCSB coordinates via PDBe best_structures, measured with freesasa",
                    note=value["error"],
                    detail={"reasons": value.get("reasons"),
                            "structures_offered": value.get("structures_offered")})
    # One row, carrying the date and its own citability, rather than a flat dated return. The
    # envelope refuses a citable claim with no `available_from`, and a surface measured on
    # structures whose deposition dates could not be resolved is exactly that: a real
    # measurement that cannot be placed on the decision clock. Dropping it would hide a
    # computed surface; dating it at fetch time would launder a 2026 calculation into evidence
    # from whenever the caller likes. The row keeps it visible and uncitable, which is the
    # discipline `_structures_evidence` already applies to a target with no deposited
    # structure.
    return wrap(status="ok",
                value={key: value[key] for key in value if key != "deposition_dates"},
                rows=({"name": arguments.get("accession"),
                       "available_from": value.get("available_from"),
                       "citable": value.get("available_from") is not None,
                       "value": {"deposition_dates": value.get("deposition_dates"),
                                 "structures_used": value.get("structures_used")}},),
                source_kind="computed", available_from=value.get("available_from"),
                source="RCSB coordinates via PDBe best_structures, measured with freesasa",
                detail=({"dating_error": value["dating_error"]}
                        if value.get("dating_error") else None))


def _rank_targets(spec, arguments, context, wrap):
    payload = spec.resolve()(**{**arguments, "record": False})
    return wrap(status="ok", value=payload, source_kind="computed",
                source="Identification against the named reviewed UniProt release")


# Where each onset lane's return comes from, which is the only thing that decides whether it
# can be cited. Declared per tool rather than defaulted, because the lanes genuinely differ and
# one blanket value would be wrong for at least one of them: five retrieve what a public
# database already holds, while `structures.folded` asks a model to compute a structure that
# did not exist until the call was made. Reporting that as an `external_record` would give a
# fresh prediction the standing of a deposition, which is the conflation `structures.predicted`
# partitions its own return to avoid.
ONSET_SOURCE_KINDS = {
    "literature.search": "external_record",
    "families.entries": "external_record",
    "similarity.homologs": "external_record",
    "structures.predicted": "external_record",
    "structures.folded": "computed",
    "bioactivity.prior_binders": "external_record",
}

# Which payload key says whether the call found anything, per tool. A lane that found nothing
# returns `empty`, not `failed`: a successful call that found nothing is a finding, and
# `chief.STATUSES` separates them for the reason `structures.evidence` already pays for.
# Declared rather than sniffed, because guessing from whichever key happens to be present
# makes the meaning of `empty` depend on a dict's shape.
ONSET_FOUND_KEYS = {
    "literature.search": "hit_count",
    "families.entries": "families",
    "similarity.homologs": "homolog_count",
    "structures.predicted": "any_prediction_exists",
    "structures.folded": "folded",
    "bioactivity.prior_binders": None,
}


def _passthrough(spec, arguments, context, wrap):
    """Dispatch an onset lane whose payload is already the return, with `record` forced off.

    One adapter for six tools because they share a shape: each takes a `record` keyword and
    each returns a dict that is the payload. `record=False` is forced rather than defaulted --
    the episode writes the record, and a tool that wrote its own would put a second artifact on
    disk that no run names.

    The two per-tool facts an adapter cannot infer are declared above: where the return came
    from, and which key says whether it found anything.
    """
    name = spec.name
    payload = spec.resolve()(**{**arguments, "record": False})
    absence = payload.get("absence") if isinstance(payload, dict) else None
    key = ONSET_FOUND_KEYS.get(name)
    # The declared key decides, and `absence` is consulted only where no key is declared. The
    # first version had it the other way round and read `literature.search` as empty on a call
    # that found 4464 hits: that payload always carries an `absence` dict, because the field
    # reports *which per-axis queries* came back zero rather than whether the call found
    # nothing. A truthy dict is not a truthy claim, and treating one as the other turned a
    # successful retrieval into a recorded absence.
    found = bool(payload.get(key)) if key else not absence
    date = payload.get(spec.date_field) if (spec.date_field and isinstance(payload, dict)) \
        else None
    if not found:
        return wrap(status="empty", value=None,
                    source_kind=ONSET_SOURCE_KINDS[name],
                    source=f"{name} over its declared endpoints",
                    note=str(absence) if absence
                    else "the call succeeded and found nothing",
                    detail={"lanes_unavailable": payload.get("lanes_unavailable")}
                    if isinstance(payload, dict) and payload.get("lanes_unavailable")
                    else None)
    if spec.produces_evidence and date is None:
        # A dated tool that found something and cannot date it. The envelope refuses a citable
        # claim with no `available_from`, and rightly: `Dossier.observe` compares that field
        # against the decision clock, so an undated claim enters at any clock. Returned on an
        # uncitable row so the finding stays visible and unusable, which is the pattern
        # `_structures_evidence` applies to a target with no deposited structure.
        return wrap(status="ok", value=payload,
                    rows=({"name": str(arguments.get("accession") or name),
                           "available_from": None, "citable": False,
                           "value": {"reason": f"{name} found a result with no "
                                               f"{spec.date_field!r}, so it cannot be placed "
                                               f"on the decision clock"}},),
                    source_kind=ONSET_SOURCE_KINDS[name],
                    source=f"{name} over its declared endpoints")
    return wrap(status="ok", value=payload,
                source_kind=ONSET_SOURCE_KINDS[name],
                source=f"{name} over its declared endpoints",
                available_from=str(date) if date is not None else None)


# The only name-to-callable table. An adapter takes the spec, the validated arguments, the
# context and the envelope constructor, and returns an envelope. It never writes: every
# pipeline it wraps is called with `record=False`, because the episode writes the record and a
# tool that wrote its own would put a second artifact on disk that no run names.
DISPATCH = {
    "chemistry.library_spec": _library_spec,
    "structures.evidence": _structures_evidence,
    "intake.rank_targets": _rank_targets,
    "geometry.surface_markers": _surface_markers,
    "literature.search": _passthrough,
    "families.entries": _passthrough,
    "similarity.homologs": _passthrough,
    "structures.predicted": _passthrough,
    "structures.folded": _passthrough,
    "bioactivity.prior_binders": _passthrough,
}
