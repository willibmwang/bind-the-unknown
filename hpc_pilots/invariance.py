"""Certified nuisance transformations: how much of a score the instance's identity supplies.

The zero-retrieval control settled that the decision arm is contaminated. Forbidden from
retrieving anything and handed an empty dossier, the policy still named the reference-best
target in sixteen episodes of sixteen. That result establishes that a floor exists and says
nothing about where it is, because a control that fires tells you the answer was already
known, not how much of the reported score it accounts for. Reporting a number above a floor
you cannot locate is not a measurement.

This module locates it. The construction is a group of transformations that leave the
decision *structure* intact while destroying the instance's *identity*, and a score that
survives them is the part the structure supports.

Three kinds of transformation are applied, and the third is only interpretable beside the
other two.

* `null` re-runs the identical instance. Any spread here is the policy's own variance --
  sampling temperature, prompt position, provider nondeterminism -- and it is the floor
  against which the other two are read. Without it, a policy that is merely unstable looks
  exactly like a policy that was relying on the target's name.
* `presentation` permutes the order the dossiers are listed in. This is semantically
  nothing: the same evidence, the same names, the same answer. It separates sensitivity to
  the *arrangement* of the state from sensitivity to its *content*.
* `identity` replaces every declared identifying token -- target names, their aliases,
  pathogen, family, source strings and evidence identifiers -- with opaque codes. The
  decision problem is unchanged and the answer is the same target under its new code.

**What this measures and what it cannot.** Scrubbing is lexical. A claim reading
"polyprotein processing at eleven sites is obligatory for replication" names no target and
identifies one anyway to any reader who knows the biology, and no substitution table fixes
that: scientific evidence identifies its subject, which is what makes it evidence. So the
identity orbit bounds the *lexical* contribution of the name and bounds nothing about the
semantic one. `leaks` reports what survives the scrub against the declared lexicon, and
`residual_candidates` reports tokens that look like identifiers nobody declared, so the
limitation is visible in the artifact rather than resting on this docstring. Measuring the
semantic channel needs a transformation that changes the right answer; that is `twins`, and
the two instruments answer different questions.

**Why the group is certified rather than assumed.** A transformation is only a nuisance if
the oracle's answer moves with it and does not otherwise change. That is checkable here,
because the evaluator computes the oracle exactly: apply the relabelling, recompute the
reference ranking and the decision-relevant set, and require that they are the images of
the originals under the same relabelling. `certify` performs that check per element and
reports which part failed. An element that fails certification is not evidence about the
policy and must not be averaged into an orbit -- it is evidence about the harness.
"""
import hashlib
import json
import re
from dataclasses import dataclass

from .common import ROOT, provenance, write_json
from .dossier import AXES, Dossier, leaders, rank
from .targets import PACKETS, packet

KINDS = ("null", "presentation", "identity")

# Tokens that name an entity in claim text, source strings or identifiers, beyond the target
# and pathogen names the packet already declares. Data rather than a regular expression,
# because deciding whether a token identifies a target is a judgement about the biology and
# a judgement belongs in a reviewable list. An omission here is a leak, which is why `leaks`
# audits the result instead of trusting this table.
IDENTIFYING = {
    "coronaviridae": (
        "SARS-CoV-2", "SARS-CoV", "sarbecovirus", "sarbecoviruses", "betacoronavirus",
        "betacoronaviruses", "coronavirus", "coronaviruses", "Coronaviridae",
        "nsp1", "nsp2", "nsp3", "nsp5", "nsp10", "nsp13", "nsp14", "nsp16",
        "Mpro", "PLpro", "Spike", "RBD", "ACE2", "ISG15", "RaPID",
        "R1AB_SARS2_Flynn_2022", "2'-O-methyltransferase", "polyprotein",
    ),
    "orthomyxoviridae": (
        "influenza", "Influenza", "Orthomyxoviridae", "haemagglutinin", "hemagglutinin",
        "neuraminidase", "H1", "H3", "H1N1", "H3N2", "HA", "NA", "M2", "PB1", "PB2", "PA",
        "nucleoprotein", "sialic", "oseltamivir", "zanamivir", "stem", "stalk",
        # Public dataset identifiers quoted in claim text. `9INFA` names the organism
        # outright, so these leaked the family through the citation while every target name
        # was correctly replaced. Found by `residual_candidates` once its own false positive
        # was removed -- it had been reporting the bare `TGT` inside every opaque code, 9415
        # times, and a real leak would have arrived inside that noise.
        "A0A2Z5U3Z0_9INFA_Doud_2016", "A4D664_9INFA_Soh_2019", "C6KNH7_9INFA_Lee_2018",
        "H1-to-H3", "9INFA",
    ),
    "henipavirus": (
        "Nipah virus", "Nipah", "Henipavirus", "henipavirus", "NiV-B", "NiV-M",
        "Hendra", "Cedar", "Langya", "Ephrin-B2", "ephrin-B3", "Ephrin", "ephrin",
        "B2", "B3", "beta-propeller", "cathepsin L", "cathepsin",
    ),
}

# Declared identifiers must belong to a packet that exists. The table was first written with
# an entry for `filoviridae`, which is not one of the three packets, while `henipavirus` had
# none -- so that packet's identity orbit scrubbed five tokens and left `Nipah`, `Hendra`,
# `Langya`, `Cedar` and the ephrin receptors in the state it called anonymised. A table keyed
# by name does nothing at all when the name is wrong, and reports nothing while doing it.
_undeclared = set(IDENTIFYING) - set(PACKETS)
if _undeclared:
    raise RuntimeError(f"IDENTIFYING names packets that do not exist: {sorted(_undeclared)}; "
                       f"known packets are {sorted(PACKETS)}")

# A committed floor on each packet's lexicon, so shrinking the substitution table cannot
# shrink the audit that checks it. `leaks` is called with `lexicon(name)`, which is built from
# the same table `_codes` substitutes from, so emptying the table makes the scrub and its own
# auditor vanish together and `certify` passes with `surviving_tokens: {}` while eleven names
# survive into the state. Checked against this snapshot instead.
LEXICON_FLOOR = {"coronaviridae": 35, "orthomyxoviridae": 35, "henipavirus": 21}

# A token that looks like it names something: an internal capital, a digit welded to a
# letter, or a dotted abbreviation. Reported, never substituted -- a detector cannot tell a
# protein from a technique, and silently scrubbing "FRET" would destroy evidence the policy
# legitimately needs while silently keeping "ACE2" would leak. The judgement stays in
# `IDENTIFYING`; this only says where to look.
SUSPICIOUS = re.compile(r"\b(?:[A-Za-z]+[0-9]+[A-Za-z0-9'_-]*|[a-z]+[A-Z][A-Za-z]*|[A-Z]{2,})\b")

# Words the detector flags that are techniques, units or ordinary prose rather than entity
# names. Excluded from the residual report so that the report stays readable and a real leak
# is not lost among forty false positives.
BENIGN = frozenset({
    "FRET", "DMS", "WHO", "NIAID", "PDB", "SMILES", "RDKit", "MD", "NMR", "ATP", "NTP",
    "RNA", "DNA", "E", "coli", "ELISA", "SPR", "IC50", "KD", "EC50", "W4", "SynX",
    "ProteinGym", "UK", "US", "EU", "PCR", "LC", "MS", "HPLC", "CRISPR", "siRNA",
})


@dataclass(frozen=True)
class Relabelling:
    """One element of the nuisance group, serialisable so an orbit is reproducible.

    `names` carries the substitution table as pairs rather than a dict so the element is
    hashable and can be stamped into a result without being reordered on the way. `order`
    is the permutation applied to the dossier list, and is the identity permutation for
    every kind but `presentation`.
    """

    kind: str
    index: int
    names: tuple = ()
    order: tuple = ()

    def table(self):
        return dict(self.names)

    def code(self):
        """A short stable identifier for the element, for filenames and result keys."""
        payload = json.dumps([self.kind, self.index, sorted(self.names), list(self.order)],
                             sort_keys=True)
        return hashlib.blake2b(payload.encode(), digest_size=6).hexdigest()

    def record(self):
        return {"kind": self.kind, "index": self.index, "code": self.code(),
                "substitutions": len(self.names), "order": list(self.order)}


def _aliases(target):
    """Every lexical form of a target name a claim might use.

    Packet targets are written as `Mpro (nsp5)` and `nsp10/nsp16 interface`, so the whole
    string, each parenthesised alias and each slash-separated component are all names that
    appear in prose. Scrubbing only the whole string leaves the components behind, which is
    the leak this function exists to close.
    """
    out = {target}
    inner = re.findall(r"\(([^)]*)\)", target)
    out.update(inner)
    outer = re.sub(r"\s*\([^)]*\)", "", target).strip()
    out.add(outer)
    for piece in re.split(r"[/,]", outer):
        piece = piece.strip()
        # Single words only. Splitting "nsp10/nsp16 interface" yields "nsp16 interface",
        # whose second word is ordinary prose and must not become part of a code.
        if piece and " " not in piece:
            out.add(piece)
    return {o for o in out if len(o) > 1}


def lexicon(name, declared=True):
    """Every identifying token for a packet, longest first.

    Order matters: substitution is sequential, so `nsp10/nsp16 interface` has to be replaced
    before `nsp16`, or the longer name is left as a half-scrubbed hybrid of a code and a
    fragment. Sorting by descending length is what makes the sequence confluent.

    `declared=False` returns only the tokens recoverable from the packet itself -- the target
    names and their aliases, the pathogen, the family and the source strings -- and omits the
    hand-curated extras in `IDENTIFYING`. That subset is what a floor can be derived from for
    a packet nobody has curated a count for.
    """
    spec = PACKETS[name]
    tokens = {spec["family"], spec["pathogen"]}
    for target, (pathogen, _, axes) in spec["targets"].items():
        tokens |= _aliases(target)
        tokens.add(pathogen)
        tokens.update(entry[3] for entry in axes)
    if declared:
        tokens.update(IDENTIFYING.get(name, ()))
    return sorted((t for t in tokens if t), key=lambda t: (-len(t), t))


def lexicon_floor(name):
    """The committed floor for a packet, or one derived from the packet's own structure.

    This was `LEXICON_FLOOR.get(name, 0)`, which makes the check `len(declared) >= 0` -- true
    for every input. So the guard was vacuous for every packet the table does not name, which
    is the regression the floor exists to prevent, re-opened by default: the auditor would
    report a clean scrub it had not performed. The three curated packets have zero slack
    against their committed floors, so the omission was invisible as long as only they ran.

    A packet with no committed floor gets a floor derived from the tokens its own structure
    implies. That cannot catch a shrunken `IDENTIFYING` the way a committed count can, because
    the derived floor shrinks with it, so the two are not interchangeable and `certify`
    records which was in force rather than leaving a reader to assume the stronger one.
    """
    if name in LEXICON_FLOOR:
        return LEXICON_FLOOR[name], "committed"
    return len(lexicon(name, declared=False)), "derived from the packet's own tokens"


def _codes(name, rng):
    """Assign an opaque code to every declared token, under a permutation drawn from `rng`.

    Which code a target receives is itself a nuisance, so it is drawn rather than fixed.
    Codes are typed -- `TGT`, `ORG`, `SRC`, `TOK` -- because a policy handed seven
    indistinguishable strings cannot tell a target from a citation and the decision problem
    would have been changed rather than relabelled, which would fail certification for the
    wrong reason.
    """
    spec = PACKETS[name]
    targets = sorted(spec["targets"])
    organisms = sorted({spec["pathogen"], spec["family"]}
                       | {p for p, _, _ in spec["targets"].values()})
    sources = sorted({entry[3] for _, (_, _, axes) in spec["targets"].items()
                      for entry in axes})

    table = {}
    for pool, prefix in ((targets, "TGT"), (organisms, "ORG"), (sources, "SRC")):
        for slot, item in zip(rng.permutation(len(pool)), pool):
            for alias in (_aliases(item) if prefix == "TGT" else {item}):
                table.setdefault(alias, f"{prefix}-{int(slot) + 1:02d}")

    extra = [t for t in IDENTIFYING.get(name, ()) if t not in table]
    for slot, token in zip(rng.permutation(len(extra)), extra):
        table[token] = f"TOK-{int(slot) + 1:02d}"
    return table


def _substitute(text, table):
    """Replace declared tokens in free text, longest first and on word boundaries.

    Word boundaries matter in both directions. Without them `HA` rewrites the middle of
    `shared`; with them `2'-O-methyltransferase` would never match, because `'` is not a
    word character, so the pattern falls back to a literal match for tokens that cannot
    carry a boundary.
    """
    if not isinstance(text, str):
        return text
    for token in sorted(table, key=lambda t: (-len(t), t)):
        if re.fullmatch(r"[\w\s-]+", token):
            text = re.sub(rf"(?<![\w-]){re.escape(token)}(?![\w-])", table[token], text)
        else:
            text = text.replace(token, table[token])
    return text


def _scrub(obj, table):
    """Apply the substitution table to every string in a nested structure, keys included.

    Recursive rather than field-by-field because the packet specification is a nested dict
    whose *keys* are target names and whose leaves include claim prose that cross-references
    other targets -- "conserved but less so than Mpro across genera" is a claim filed under
    PLpro that names Mpro. A field-by-field scrub of the fields a policy reads left that
    specification untouched in the artifact, so the names were absent from the state and
    present in the file recording it, which is a leak for every later reader of the record
    and for anything trained on it.
    """
    if isinstance(obj, str):
        return _substitute(obj, table)
    if isinstance(obj, dict):
        return {_substitute(k, table): _scrub(v, table) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        scrubbed = [_scrub(v, table) for v in obj]
        return tuple(scrubbed) if isinstance(obj, tuple) else scrubbed
    return obj


def _relabel_entry(entry, table, target_code):
    """Relabel one evidence entry, keeping the conflict pointer consistent.

    Entry identifiers are built as `target:axis`, so the identifier carries the target name
    and leaks it wholesale. Rebuilding the identifier from the code rather than scrubbing
    the old string keeps `conflicts_with` resolvable: both sides of a contradiction are
    rebuilt by the same rule, so the pointer still finds its partner.
    """
    def rebuild(identifier):
        if identifier is None:
            return None
        head, _, tail = identifier.rpartition(":")
        if head.startswith("probe:"):
            kind = head.split(":")[1]
            return f"probe:{kind}:{target_code}:{tail}"
        return f"{target_code}:{tail}"

    return {**entry, "id": rebuild(entry["id"]),
            "claim": _substitute(entry["claim"], table),
            "source": _substitute(entry["source"], table),
            "conflicts_with": rebuild(entry["conflicts_with"])}


def relabel_packet(name, relabelling):
    """Build a packet under a group element, in the shape `targets.packet` returns.

    A drop-in for `packet`, so `run_episode` runs a relabelled instance through the
    identical code path rather than through a parallel one that could drift away from it.
    """
    spec, dossiers, entries, probes = packet(name)
    table = relabelling.table()
    if relabelling.kind == "identity" and not table:
        raise ValueError("An identity relabelling with an empty substitution table scrubs nothing")

    if relabelling.kind == "identity":
        codes = {d.target: table.get(d.target, d.target) for d in dossiers}
        spec = _scrub(spec, table)
        dossiers = [Dossier(codes[d.target], _substitute(d.pathogen, table),
                            _substitute(d.family, table), d.compartment) for d in dossiers]
        entries = [(codes[t], _relabel_entry(e, table, codes[t])) for t, e in entries]
        probes = [{**p, "target": codes[p["target"]],
                   "entry": _relabel_entry(p["entry"], table, codes[p["target"]])}
                  for p in probes]

    if relabelling.order:
        if sorted(relabelling.order) != list(range(len(dossiers))):
            raise ValueError("Order is not a permutation of the dossier list")
        dossiers = [dossiers[i] for i in relabelling.order]
    return spec, dossiers, entries, probes


def group(name, kind, seed, count):
    """`count` elements of one kind, derived from `seed` so an orbit is reproducible.

    The null kind returns `count` copies of the identity element rather than one. They are
    not redundant: each is a separate episode, and the spread across them is the policy
    variance every other orbit is read against.
    """
    if kind not in KINDS:
        raise ValueError(f"Unknown kind {kind!r}; known kinds are {list(KINDS)}")
    if name not in PACKETS:
        raise ValueError(f"Unknown family {name!r}")
    from .streams import stream

    size = len(PACKETS[name]["targets"])
    out = []
    for index in range(count):
        rng = stream(seed, "nuisance", kind, index)
        if kind == "null":
            out.append(Relabelling(kind, index))
        elif kind == "presentation":
            out.append(Relabelling(kind, index, order=tuple(int(i) for i in rng.permutation(size))))
        else:
            out.append(Relabelling(kind, index, names=tuple(sorted(_codes(name, rng).items()))))
    return out


def leaks(obj, tokens):
    """Declared tokens that survive into a structure the policy will read.

    Searched over the serialised form rather than over known fields, because a leak that
    mattered would be in whichever field nobody thought to check. Returns the tokens found,
    with a count, so a failure says what to add to the substitution table.
    """
    text = json.dumps(obj)
    found = {}
    for token in tokens:
        hits = len(re.findall(rf"(?<![\w-]){re.escape(token)}(?![\w-])", text)) \
            if re.fullmatch(r"[\w\s-]+", token) else text.count(token)
        if hits:
            found[token] = hits
    return dict(sorted(found.items(), key=lambda kv: (-kv[1], kv[0])))


def residual_candidates(obj):
    """Tokens in a scrubbed structure that look like identifiers nobody declared.

    The counterpart to `leaks`. That function proves the declared lexicon was applied; this
    one is the admission that the lexicon may be incomplete, and it is reported in the
    artifact so a reader can see what the scrub did not consider rather than inferring from
    a clean leak report that nothing was missed.
    """
    # The codes are removed from the text before scanning rather than filtered afterwards.
    # `SUSPICIOUS` matches runs of capitals on word boundaries and `-` is not a word
    # character, so it found the bare `TGT` inside every `TGT-01` and the filter -- which
    # tested the whole `TGT-01` form -- never fired. A clean identity run therefore reported
    # `{"TGT": 9415}`, and a real leak would have arrived in a report already full of noise,
    # which is the failure mode this function exists to prevent.
    text = re.sub(r"(?:TGT|ORG|SRC|TOK)-\d+", " ", json.dumps(obj))
    counts = {}
    for token in SUSPICIOUS.findall(text):
        if token in BENIGN:
            continue
        counts[token] = counts.get(token, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def certify(name, relabelling, deadline=8):
    """Check that the oracle moves with the relabelling and not otherwise.

    Three things must hold for an element to be a nuisance transformation. The reference
    ranking must be the image of the original ranking: same order, same substantiated
    counts, same gate verdicts, with every name replaced by its code. The decision-relevant
    set at the opening round must be the image of the original set. And an identity element
    must leave no declared token behind.

    Reported rather than asserted, and reported per check, because the three fail for
    different reasons and the distinction is the diagnostic. A ranking that fails while the
    relevant set passes means the ranking read something the relabelling moved -- which for
    `presentation` means the oracle is order-dependent, and that is a defect in the harness
    and not a property of any policy.
    """
    from .emergency import decision_relevant

    table = relabelling.table()
    base_spec, base_d, base_e, base_p = packet(name)
    spec, dossiers, entries, probes = relabel_packet(name, relabelling)

    def observed(ds, es, ps):
        by_target = {d.target: d for d in ds}
        for target, entry in es:
            by_target[target].observe(entry, deadline)
        for probe in ps:
            by_target[probe["target"]].observe(probe["entry"], deadline)
        return ds

    # Compared as a set of per-target verdicts and a lead verdict, not as a list of rows.
    # The row order among targets tied on the ranking key is the caller's list order and
    # means nothing, so requiring the rows to match positionally would fail a presentation
    # permutation for rearranging equals -- which is the one thing that transformation is
    # supposed to do. What has to be the image is the content.
    def verdicts(ds):
        return {(d.target, d.readiness()["substantiated_axes"],
                 d.readiness()["hard_gates_met"]) for d in ds}

    base_observed = observed(base_d, base_e, base_p)
    moved_observed = observed(dossiers, entries, probes)
    expected = {(table.get(t, t), s, g) for t, s, g in verdicts(base_observed)}
    base_lead = leaders(base_observed, deadline)
    moved_lead = leaders(moved_observed, deadline)
    ranking_ok = (expected == verdicts(moved_observed)
                  and {table.get(t, t) for t in base_lead["tied_for_lead"]}
                  == set(moved_lead["tied_for_lead"]))

    # Recomputed on unobserved dossiers at the opening round, which is the state the oracle
    # is actually evaluated in during an episode.
    _, fresh_base, fresh_base_e, _ = packet(name)
    _, fresh, fresh_e, _ = relabel_packet(name, relabelling)
    base_pool = {(t, e["axis"]): e for t, e in fresh_base_e}
    pool = {(t, e["axis"]): e for t, e in fresh_e}
    base_relevant = {(table.get(t, t), a)
                     for t, a in decision_relevant(fresh_base, base_pool, set(), 1)}
    moved_relevant = set(decision_relevant(fresh, pool, set(), 1))

    scrub = leaks([spec, [d.state(deadline) for d in dossiers], probes], lexicon(name)) \
        if relabelling.kind == "identity" else {}

    declared = lexicon(name)
    floor, floor_source = lexicon_floor(name)
    checks = {"reference_ranking_is_the_image": ranking_ok,
              "relevant_set_is_the_image": base_relevant == moved_relevant,
              "no_declared_token_survives": not scrub,
              "lexicon_is_not_smaller_than_its_floor": len(declared) >= floor}
    return {"element": relabelling.record(), "family_packet": name, "checks": checks,
            "certified": all(checks.values()),
            "declared_tokens": len(declared),
            "declared_tokens_floor": floor,
            # Which guard was in force. A derived floor shrinks with the table it audits, so
            # it cannot catch an emptied `IDENTIFYING` the way a committed count can, and a
            # reader must be able to tell the two apart.
            "declared_tokens_floor_source": floor_source,
            "surviving_tokens": scrub,
            "relevant_set_size": len(moved_relevant),
            "relevant_set_expected": len(base_relevant),
            "ranking_order_dependent": relabelling.kind == "presentation" and not ranking_ok,
            "interpretation": "A transformation is a nuisance only if the oracle is its image. "
                              "An uncertified element measures the harness, not the policy."}


def earned(nominal, orbits):
    """Split a score into the part the structure supports and the part identity supplied.

    `orbits` maps a kind to the scores observed over that kind's elements. The quantity to
    read is `attributable_to_identity`: the identity orbit's shortfall against the nominal
    run, with the null orbit's spread as the noise floor it has to clear.

    It was first defined as the shortfall against the *presentation* orbit, on the reasoning
    that a policy which is merely unstable would otherwise be indistinguishable from one
    reciting a name. That reasoning was wrong, and the guard in `improve` caught it: the
    fixed rubric, which reads no names at all, reported a 0.18 identity effect under a wider
    sampling band. The cause is that the two transformations are orthogonal rather than
    nested -- an identity relabelling renames with the order fixed, a presentation
    permutation reorders with the names fixed -- so their difference mixes two distinct
    sensitivities. The rubric walks the ranking and takes the first target, and `rank`
    resolves ties in the caller's list order, so permuting the list genuinely changes what it
    buys. Differencing the orbits charged that order-dependence to memorisation.

    The control for instability is the null orbit, which re-runs the identical instance and
    is what the noise floor is computed from. The presentation orbit measures something else
    worth reporting on its own: a policy with a large `attributable_to_presentation` is
    reading the arrangement of the state rather than its content, which is a defect in that
    policy and not evidence about the name.

    **The floor degenerates for a deterministic policy, and the hosted model is one.** Asked
    the identical question three times it returned the identical action each time, so its null
    orbit has a spread of exactly zero and `exceeds_variance_floor` is true for any non-zero
    effect, however small. The flag is therefore uninformative in precisely the case it was
    added for. Where the null spread is zero the quantity to read instead is the spread
    *across group elements* -- an identity orbit whose own deviation is larger than its
    distance from nominal has not shown an effect, it has shown that relabellings differ from
    each other.

    Returns `None` for any quantity whose orbit is absent rather than substituting nominal
    for a missing arm, so a partial run reports a gap instead of a number.
    """
    def summary(kind):
        rows = [r for r in orbits.get(kind, []) if r is not None]
        if not rows:
            return None
        mean = sum(rows) / len(rows)
        spread = (sum((r - mean) ** 2 for r in rows) / (len(rows) - 1)) ** 0.5 \
            if len(rows) > 1 else None
        return {"n": len(rows), "mean": round(mean, 4), "min": round(min(rows), 4),
                "max": round(max(rows), 4),
                "sd": None if spread is None else round(spread, 4)}

    null, presentation, identity = (summary(k) for k in KINDS)

    def shortfall(orbit):
        if nominal is None or orbit is None:
            return None
        return round(nominal - orbit["mean"], 4)

    gap = shortfall(identity)
    floor = None if null is None else null["sd"]
    return {"nominal": None if nominal is None else round(nominal, 4),
            "null_orbit": null, "presentation_orbit": presentation, "identity_orbit": identity,
            "earned_mean": None if identity is None else identity["mean"],
            "earned_floor": None if identity is None else identity["min"],
            "attributable_to_identity": gap,
            "attributable_to_presentation": shortfall(presentation),
            "policy_variance_floor": floor,
            "exceeds_variance_floor": None if gap is None or floor is None
            else bool(abs(gap) > floor),
            # Reported because the floor degenerates for a deterministic policy, and the
            # hosted model is one: its null orbit has a spread of exactly zero, so the flag
            # above is true for any non-zero effect. Where that happens, read the effect
            # against the identity orbit's own deviation instead -- an orbit that varies more
            # between its own elements than it differs from nominal has shown nothing.
            "exceeds_orbit_spread": None if gap is None or identity is None
            or identity["sd"] is None else bool(abs(gap) > identity["sd"]),
            "variance_floor_is_degenerate": None if floor is None else floor == 0.0,
            "interpretation": "attributable_to_identity is the nominal score less the identity "
                              "orbit's mean: what renaming the targets costs, with the order "
                              "held fixed. It is read against the null orbit's spread, which "
                              "is the same instance re-run and so is the policy's own "
                              "variance. attributable_to_presentation is a separate and "
                              "orthogonal sensitivity -- reordering with the names held fixed "
                              "-- and is a defect in a policy rather than evidence about the "
                              "name, so the two are reported side by side and never "
                              "differenced. Both bound the lexical channel only: evidence "
                              "identifies its subject, so a claim scrubbed of every name can "
                              "still name one, and that residue needs an intervention that "
                              "moves the answer."}


def axis_gate(dossiers, pool, retrieved, now):
    """Which axes an episode's decision currently turns on, as a set of axis labels.

    `decision_relevant` returns `(target, axis)` pairs; this projects onto the axis, because
    the question a policy is scored on in the twin construction is which *axis* to resolve.
    Kept here rather than in `emergency` so the projection is defined once and both the
    orbit machinery and the twin machinery read the same definition.
    """
    from .emergency import decision_relevant

    return {axis for _, axis in decision_relevant(dossiers, pool, retrieved, now)} & set(AXES)


# The scalar an orbit is taken over. The challenger stratum is the headline because it is the
# one a prior cannot answer, and the leader stratum travels with it as the positive control:
# an orbit element on which no policy scores there measured a broken state, not a hard one.
ORBIT_SCORES = ("challenger_best", "leader_best")


def run_orbit(name, policy, count=4, attempts=400, kinds=KINDS, smoke=False, limit=None):
    """Probe one policy over a whole nuisance orbit and split its score into earned and supplied.

    Each element is certified before it is run, and an element that fails certification stops
    the orbit rather than being skipped. Skipping would leave a mean taken over whichever
    elements happened to pass, which is a sample selected by the harness's own defects.

    The null orbit is run first and is not optional. Without it the identity orbit's shortfall
    cannot be read: a policy that answers differently every time it is asked the identical
    question would otherwise be reported as one relying on the target's name, and those two
    findings call for opposite responses.
    """
    from .twins import run_twins

    elements = {kind: group(name, kind, 20261021, 2 if smoke else count) for kind in kinds}
    for kind, pool in elements.items():
        for element in pool:
            certificate = certify(name, element)
            if not certificate["certified"]:
                raise ValueError(f"{kind} element {element.code()} is not certified: "
                                 f"{certificate['checks']}")

    runs, orbits = [], {score: {kind: [] for kind in kinds} for score in ORBIT_SCORES}
    # The artifact is named by the label the policy actually ran under, not by the string
    # asked for. A hosted model reports `agent-<provider>-<model>` and a local one reports
    # `agent`, and writing both to `<packet>_agent.json` would let a hosted orbit silently
    # overwrite a local-model orbit -- the exact pooling the replay arm already guards
    # against and which this runner had not inherited.
    label = policy
    for kind, pool in elements.items():
        for element in pool:
            # The null element is the identity transformation, so it is run with no
            # relabelling at all rather than with an empty substitution table, which would
            # take a different code path to reach the same instance.
            result = run_twins(name, policy, attempts=attempts, smoke=smoke, limit=limit,
                               relabelling=None if kind == "null" else element)
            label = result["policy"]
            for score in ORBIT_SCORES:
                orbits[score][kind].append(result["summary"][score]["attainment"])
            runs.append({"kind": kind, "element": element.record(),
                         "summary": result["summary"], "state_yield": result["state_yield"]})

    nominal = next((r["summary"] for r in runs if r["kind"] == "null"), None)
    split = {score: earned(None if nominal is None else nominal[score]["attainment"],
                           orbits[score]) for score in ORBIT_SCORES}
    out = {"experiment": "orbit", "family_packet": name, "policy": label,
           "policy_requested": policy,
           "elements_per_kind": 2 if smoke else count, "kinds": list(kinds),
           "probes_per_element": limit,
           "smoke": smoke, "status": "complete",
           "earned": split, "runs": runs,
           "provenance": provenance(),
           "interpretation": "The identity orbit bounds the lexical contribution of the "
                             "target's name and bounds nothing about the semantic one. "
                             "Evidence identifies its subject, so a claim scrubbed of every "
                             "declared name can still name one. What a prior cannot supply is "
                             "which target the decision currently turns on, which is what the "
                             "challenger stratum asks."}
    output = ROOT / "runs/orbit" / f"{'smoke_' if smoke else ''}{name}_{label}.json"
    write_json(output, out)
    print(json.dumps({"output": str(output), "earned": split}, indent=2))
    return out
