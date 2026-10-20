"""Target dossier decision state: the seven-axis schema the co-scientist reasons over.

This replaces the opaque-integer bandit state. The axes and their gating rationale are
the Week 4 target-selection rubric. Two properties matter:

* **Date-limited retrieval.** Every evidence entry carries `available_from`. A dossier
  refuses evidence dated after its decision clock. This is the leakage boundary that
  replaces opaque candidate identities.
* **Named targets reintroduce hindsight risk.** Because targets are named rather than
  integer-indexed, knowledge already in model weights can substitute for retrieval.
  The date boundary does *not* control that. It must be audited separately; see
  `hindsight_risk` in the state and the limitations in each evidence packet.

Dossier readiness is a measure of evidence coverage. It is not binding, inhibition or
antiviral activity, and it is not proof of therapeutic efficacy.
"""

# hard_gate axes end a campaign regardless of the other axes.
AXES = {
    "reagent_tractability": {
        "question": "Can a soluble, correctly folded, immobilisable target be obtained that "
                    "survives multiple selection rounds?",
        "gates": "Hard gate. An untractable reagent ends the campaign regardless of biology.",
        "hard_gate": True},
    "essentiality": {
        "question": "Does loss of function block replication?",
        "gates": "A binder to a dispensable protein cannot become a therapeutic.",
        "hard_gate": False},
    "conservation": {
        "question": "Is the epitope conserved across the family, and under what selective pressure?",
        "gates": "Gates every family-wide and prototype-pathogen claim.",
        "hard_gate": False},
    "accessibility": {
        "question": "Intracellular or extracellular; is the epitope solvent-exposed in the "
                    "relevant conformation?",
        "gates": "Sets whether a permeability story is required.",
        "hard_gate": False},
    "assay_availability": {
        "question": "Is there a functional orthogonal assay beyond enrichment?",
        "gates": "Without one, a single round can only substantiate enrichment.",
        "hard_gate": False},
    "public_labels": {
        "question": "Is there a public DMS or binding landscape for method development?",
        "gates": "Determines whether transfer tests are runnable now or blocked on SynX.",
        "hard_gate": False},
    "precedent": {
        "question": "Are there prior macrocycle or peptide campaigns against this target?",
        "gates": "Supplies positive controls and de-risks failure interpretation.",
        "hard_gate": False},
}

ASSESSMENTS = ("unknown", "weak", "moderate", "strong")

# The action space the project proposal actually specifies, replacing {greedy, explore, diverse}.
ACTIONS = {
    "retrieve_evidence": "Retrieve sourced evidence for one named target and axis.",
    "request_structure": "Request a structure or conformational calculation for one target.",
    "rerank_targets": "Revisit the target ranking using the evidence observed so far.",
    "invoke_design_tool": "Propose a focused macrocycle library against the leading target.",
    "recommend_experiment": "Recommend the single display round and state what it can substantiate.",
    "stop": "Stop and report the dossier with its open uncertainties.",
}


def evidence(id, axis, claim, strength, source, available_from, verified=True, conflicts_with=None):
    if axis not in AXES:
        raise ValueError(f"Unknown axis {axis!r}")
    if strength not in ASSESSMENTS:
        raise ValueError(f"Unknown strength {strength!r}")
    return {"id": id, "axis": axis, "claim": claim, "strength": strength, "source": source,
            "available_from": available_from, "verified": verified,
            "conflicts_with": conflicts_with}


class Dossier:
    """Evidence state for one candidate target. Claims are only as good as their citations."""

    def __init__(self, target, pathogen, family, compartment=None):
        self.target, self.pathogen, self.family = target, pathogen, family
        self.compartment = compartment
        self.observed = []
        self.rejected = []

    def observe(self, entry, now):
        """Accept an entry only if it was available at `now`; refusals are recorded, not silent."""
        if entry["available_from"] > now:
            self.rejected.append({"id": entry["id"], "available_from": entry["available_from"],
                                  "decision_clock": now, "reason": "not yet available"})
            return False
        self.observed.append(entry)
        return True

    def fork(self):
        """A copy sharing no mutable state, for counterfactual lookahead by the evaluator.

        The evaluator needs to ask what the ranking *would* be if one further claim were
        observed. It must not answer that by mutating the dossier the policy is reasoning
        over, and a shallow copy would share the observed list and do exactly that.
        """
        other = Dossier(self.target, self.pathogen, self.family, self.compartment)
        other.observed = list(self.observed)
        other.rejected = list(self.rejected)
        return other

    def axis_evidence(self, axis):
        return [e for e in self.observed if e["axis"] == axis]

    def assess(self, axis):
        """Strongest observed claim on the axis, with its citations. Unknown when uncited.

        Two readings are kept apart because they license different decisions.
        `assessment` is the strongest claim of any kind; `verified_assessment` ignores
        claims carrying no citation, so an assertion from memory cannot be the thing
        that satisfies a gate. `conflict` says another observed claim contradicts one
        of them, which `readiness` then refuses to treat as settled.
        """
        entries = self.axis_evidence(axis)
        if not entries:
            return {"assessment": "unknown", "verified_assessment": "unknown",
                    "citations": [], "unverified": [], "conflict": False}
        best = max(entries, key=lambda e: ASSESSMENTS.index(e["strength"]))
        return {"assessment": best["strength"],
                "verified_assessment": max((e["strength"] for e in entries if e["verified"]),
                                           key=ASSESSMENTS.index, default="unknown"),
                "citations": sorted({e["source"] for e in entries}),
                "unverified": [e["id"] for e in entries if not e["verified"]],
                "conflict": any(e["conflicts_with"] for e in entries)}

    def conflicts(self):
        """Contradictory entries coexist with timestamps; the newer one never overwrites the older."""
        out = []
        by_id = {e["id"]: e for e in self.observed}
        for e in self.observed:
            other = by_id.get(e["conflicts_with"])
            if other:
                pair = tuple(sorted((e["id"], other["id"])))
                entry = {"axis": e["axis"], "entries": list(pair),
                         "timestamps": {e["id"]: e["available_from"],
                                        other["id"]: other["available_from"]}}
                if entry not in out:
                    out.append(entry)
        return out

    def unsupported(self):
        """Axes asserted as better than unknown with no citation. Should always be empty."""
        return [axis for axis in AXES
                if self.assess(axis)["assessment"] != "unknown" and not self.assess(axis)["citations"]]

    def open_uncertainties(self):
        out = []
        for axis in AXES:
            a = self.assess(axis)
            if a["assessment"] == "unknown":
                out.append({"axis": axis, "issue": "no evidence observed"})
            elif a["unverified"]:
                out.append({"axis": axis, "issue": "evidence unverified", "entries": a["unverified"]})
            if a["conflict"]:
                out.append({"axis": axis, "issue": "contradictory evidence retained"})
        return out

    def readiness(self):
        """Evidence coverage, not efficacy. A failed hard gate blocks regardless of coverage.

        An axis is substantiated only when a *cited* claim reaches moderate or better and
        no observed claim contradicts it. Both exclusions are deliberate.

        Counting a contested axis as substantiated made retaining the contradiction
        decorative: the ranking read the axis as settled while the state recorded that it
        was not, so the one thing the design exists to preserve had no consequence for any
        decision taken downstream of it. Counting an uncited claim let an assertion from
        memory satisfy a hard gate, which is exactly what the `verified` flag was added to
        prevent. Both are reported, so a demotion can be read off the state.
        """
        assessed = {axis: self.assess(axis) for axis in AXES}
        substantiated = [a for a, v in assessed.items()
                         if v["verified_assessment"] in ("moderate", "strong") and not v["conflict"]]
        blocking = [a for a, spec in AXES.items() if spec["hard_gate"] and a not in substantiated]
        return {"substantiated_axes": len(substantiated), "total_axes": len(AXES),
                "substantiated": substantiated,
                "contested_axes": [a for a, v in assessed.items() if v["conflict"]],
                "uncited_axes": [a for a, v in assessed.items() if v["unverified"]],
                "hard_gates_met": not blocking, "blocking_axes": blocking,
                "unknown_axes": [a for a, v in assessed.items() if v["assessment"] == "unknown"],
                "interpretation": "Evidence coverage of the rubric, counting only cited and "
                                  "uncontested claims. Not binding, inhibition, antiviral "
                                  "activity or therapeutic efficacy."}

    def state(self, now):
        """The agent-visible decision state. Named targets, cited claims, explicit unknowns.

        This deliberately does *not* report the refusal count. A policy told that three
        claims exist which it is not allowed to see has been told that the axis is one
        somebody later thought worth writing about, which is information from after the
        decision clock arriving by a side channel the date boundary does not close. An
        axis with no evidence and an axis with only future evidence must be
        indistinguishable from here; the count stays available on the dossier for the
        episode record, which the policy never reads.
        """
        return {"target": self.target, "pathogen": self.pathogen, "family": self.family,
                "compartment": self.compartment, "decision_clock": now,
                "axes": {axis: {**AXES[axis], **self.assess(axis)} for axis in AXES},
                "readiness": self.readiness(), "conflicts": self.conflicts(),
                "open_uncertainties": self.open_uncertainties(),
                "hindsight_risk": "Targets are named, so pretrained knowledge may substitute for "
                                  "retrieval. The date boundary does not control this."}


def rank(dossiers, now):
    """Order targets by substantiated axes, with hard-gate failures demoted and the reason stated."""
    def key(d):
        r = d.readiness()
        return (r["hard_gates_met"], r["substantiated_axes"], -len(r["unknown_axes"]))
    ordered = sorted(dossiers, key=key, reverse=True)
    return [{"rank": i + 1, "target": d.target, "pathogen": d.pathogen,
             "substantiated_axes": d.readiness()["substantiated_axes"],
             "hard_gates_met": d.readiness()["hard_gates_met"],
             "blocking_axes": d.readiness()["blocking_axes"],
             "unknown_axes": d.readiness()["unknown_axes"],
             "contested_axes": d.readiness()["contested_axes"],
             "uncited_axes": d.readiness()["uncited_axes"],
             "rationale": _rationale(d)} for i, d in enumerate(ordered)]


def _rationale(d):
    """An axis with no retrieved evidence is reported as uninvestigated, never as a weak assessment.

    Conflating the two would let an unexplored target read as a rejected one, which is the
    failure mode that makes a ranking look decisive when it is merely incomplete.
    """
    strong = [a for a in AXES if d.assess(a)["assessment"] == "strong"]
    weak = [a for a in AXES if d.axis_evidence(a) and d.assess(a)["assessment"] == "weak"]
    unseen = [a for a in AXES if not d.axis_evidence(a)]
    contested = [a for a in AXES if d.assess(a)["conflict"]]
    uncited = [a for a in AXES if d.assess(a)["unverified"]]
    parts = []
    if strong:
        parts.append("strong on " + ", ".join(strong))
    if weak:
        parts.append("assessed weak on " + ", ".join(weak))
    if unseen:
        parts.append("no evidence retrieved for " + ", ".join(unseen))
    # Both of these demote an axis in `readiness`, so the rationale has to say so rather
    # than leaving the reader to wonder why a strong-looking axis did not count.
    if contested:
        parts.append("contested, not counted as substantiated: " + ", ".join(contested))
    if uncited:
        parts.append("claims awaiting a citation on " + ", ".join(uncited))
    return "; ".join(parts) or "no evidence observed"
