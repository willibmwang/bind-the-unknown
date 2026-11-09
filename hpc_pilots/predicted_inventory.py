"""The predicted-model inventory behind the landing page's `structures.predicted` rows.

Section 07 of the landing page reports `structures.predicted` as a handful of scalars --
`predicted_model_count 50`, `alphafold_host_served_models yes`, and the endpoint label
"AlphaFold DB, 3D-Beacons". A reader draws one conclusion from that, and it is the wrong
one: that AlphaFold modelled these proteins.

It did not. `structures.predicted` returns a `predicted_models` list carrying `provider`
(which host served the model) separately from `tool_used` (what actually computed it),
because the two disagree here. `chain._evidence_summary` reduces any list-shaped field to a
count -- deliberately, so a chain record does not become a copy of five external databases
-- and the distinction is the first thing that count destroys. Measured on the accessions the
held-out run actually reaches, every model the AlphaFold host serves for them was computed by
ColabFold v1.5.2, and not one is a DeepMind AlphaFold model.

So this module re-asks the same tool for the same accessions and keeps the inventory the
chain throws away. It adds no capability and changes no rank: it is the provenance of a
number already on the page. `alphafold_authored_model_count` is counted from `tool_used`
rather than assumed from the provider name, which is the entire point.

The accessions come from the record, not from a list written here. The landing page is the
held-out case throughout -- SARS-CoV-2 deleted from the reference -- so its targets resolve
to relatives (P0C6X7, Q3LZX1) rather than to SARS-CoV-2's own P0DTD1. Hard-coding P0DTD1
would put the held-out organism's own AlphaFold records on a page whose premise is that
those entries were deleted before the run, which is the leak `build_site.build_index`
already refuses. Reading the accessions back out of the case keeps that property true by
construction rather than by remembering to.

    python -m hpc_pilots.predicted_inventory

Writes `runs/predicted_models.json`.
"""

import collections
import json

from .common import ROOT, content_digest, provenance, write_json
from .predicted_structure import coverage

RECORD = ROOT / "runs" / "predicted_models.json"
LANDING = ROOT / "runs" / "onset_chain.json"

# What `tool_used` has to name for a model to be attributed to AlphaFold itself. Matched on
# the tool string and never on `provider`, because "AlphaFold DB" is the host that serves the
# file and says nothing about what computed it. ColabFold is AlphaFold-derived -- it runs the
# same network weights behind a different MSA pipeline -- and is still not an AlphaFold DB
# model, so it is counted apart rather than folded in.
ALPHAFOLD_TOOLS = ("alphafold",)


def _targets(case):
    """The accessions the case's evidence stage actually asked about, and why each was asked.

    Rebuilt the way `chain._evidence_arguments` builds it -- `nearest_homolog` plus
    `"start-end"` from `chain_range` -- because the record stores the row the arguments were
    derived from rather than the arguments themselves. One accession can serve several
    targets (both spike ranges resolve to Q3LZX1), so ranges are unioned per accession and
    the targets that pointed there are kept beside them.
    """
    wanted = collections.OrderedDict()
    for row in case.get("evidence") or ():
        accession, span = row.get("nearest_homolog"), row.get("chain_range")
        if not accession or not span:
            continue
        entry = wanted.setdefault(accession, {"ranges": [], "targets": []})
        span_text = f"{int(span[0])}-{int(span[1])}"
        if span_text not in entry["ranges"]:
            entry["ranges"].append(span_text)
        entry["targets"].append({"rank": row.get("rank"), "target": row.get("target"),
                                 "range": span_text})
    return wanted


def _is_alphafold(tool_used):
    return bool(tool_used) and any(t in tool_used.lower() for t in ALPHAFOLD_TOOLS)


def _accession_block(accession, ranges, targets):
    """One accession's inventory: who served each model, what computed it, how good it is."""
    payload = coverage(accession, residue_ranges=tuple(ranges), record=False)
    models = payload["predicted_models"]

    # Three separate questions, counted separately on purpose. Which host served it, what
    # computed it, and what kind of model it is are independent, and the page's single
    # `predicted_model_count` collapses all three.
    by_provider = collections.Counter(m["provider"] for m in models if m["provider"])
    by_tool = collections.Counter(m["tool_used"] or "not stated by the host" for m in models)
    by_category = collections.Counter(m["model_category"] or "not stated" for m in models)

    # The cross-tabulation is the finding. "AlphaFold DB served 35 models" and "35 models
    # were computed by ColabFold v1.5.2" are the same 35 models, and only putting the two
    # names in one row makes that visible.
    served_by_host = [m for m in models if (m["provider"] or "") == "AlphaFold DB"]
    host_tools = collections.Counter(m["tool_used"] or "not stated by the host"
                                     for m in served_by_host)

    best = []
    for block in payload["ranges"]:
        row = block["best_plddt_model"]
        best.append({
            "range": block["range"], "width": block["width"],
            "models_overlapping": block["models_overlapping"],
            "modelled": block["modelled"],
            # None is a real answer: no model over this range carries per-residue pLDDT, so
            # the `confident_plddt 70` the page prints never applied to it at all.
            "best_plddt_model": row,
            "no_plddt_because": None if row else
                ("no model overlapping this range carries a per-residue pLDDT document; "
                 "every model here is template-based and scored on QMEANDisCo (0-1), to "
                 "which the pLDDT threshold does not apply"),
        })

    return {
        "accession": accession,
        "uniprot_entry_id": payload["uniprot_entry_id"],
        "sequence_length": payload["sequence_length"],
        "asked_about": targets,
        "requested_ranges": payload["requested_ranges"],

        "predicted_model_count": payload["predicted_model_count"],
        "any_prediction_exists": payload["any_prediction_exists"],
        "providers_with_models": payload["providers_with_models"],
        "models_by_provider": dict(by_provider.most_common()),
        "models_by_tool_used": dict(by_tool.most_common()),
        "models_by_category": dict(by_category.most_common()),

        # The claim the page's endpoint label invites, measured rather than assumed.
        "alphafold_host_queried": payload["alphafold_host_queried"],
        "alphafold_host_served_models": payload["alphafold_host_served_models"],
        "alphafold_host_model_count": len(served_by_host),
        "alphafold_host_models_by_tool_used": dict(host_tools.most_common()),
        "alphafold_authored_model_count":
            sum(1 for m in models if _is_alphafold(m["tool_used"])),
        "tools_used": payload["tools_used"],
        "models_found_only_by_alphafold_host":
            payload["models_found_only_by_alphafold_host"],

        "mean_confidence_by_metric": payload["mean_confidence_by_metric"],
        "ranges": best,
        "excluded_experimental_count": payload["excluded_experimental_count"],
        "per_residue_refusals": payload["per_residue_refusals"],
        "network_calls": payload["network_calls"],
        "bytes_received": payload["bytes_received"],
        "content_sha256": payload["content_sha256"],
    }


def run(record=True):
    """Re-ask `structures.predicted` for the landing case's accessions; keep the inventory."""
    landing = json.loads(LANDING.read_text())
    # The same selection `build_site.build_index` makes, and for the same reason.
    case = next((c for c in landing["cases"] if c.get("reference_excluded_organism")),
                landing["cases"][0])

    blocks = [_accession_block(accession, spec["ranges"], spec["targets"])
              for accession, spec in _targets(case).items()]

    total = sum(b["predicted_model_count"] for b in blocks)
    authored = sum(b["alphafold_authored_model_count"] for b in blocks)
    by_host = sum(b["alphafold_host_model_count"] for b in blocks)

    payload = {
        "experiment": "predicted-model-inventory",
        "status": "complete",
        "case": case.get("case"),
        "genome_accession": case.get("genome_accession"),
        "reference_excluded_organism": case.get("reference_excluded_organism"),
        "accessions": blocks,

        "predicted_model_count": total,
        "alphafold_host_model_count": by_host,
        "alphafold_authored_model_count": authored,
        "headline": (
            f"{total} predicted models across {len(blocks)} accessions. "
            f"{by_host} were served by the AlphaFold host and {authored} were computed by "
            f"AlphaFold."),

        "interpretation":
            "The provenance of the `structures.predicted` scalars on the landing page. Each "
            "accession's predicted models are counted three ways that the page's single "
            "`predicted_model_count` collapses into one: by the provider that served the "
            "model, by the tool that computed it, and by model category. Nothing here is a "
            "new capability and nothing here changes a rank; it re-asks one tool already in "
            "the toolkit for the same accessions the held-out run reached, and keeps the "
            "list that the chain record reduces to a count.",

        "limitations": [
            "The AlphaFold host's name is not the provenance of what it serves. Every model "
            "provider `AlphaFold DB` returned for these accessions was computed by ColabFold "
            "v1.5.2, so `alphafold_host_served_models: true` must not be read as "
            "'AlphaFold modelled this protein'. `alphafold_authored_model_count` is counted "
            "from each model's `tool_used`, which is the only field that answers that.",
            "ColabFold is counted apart from AlphaFold rather than folded into it. It runs "
            "the same network weights behind a different MSA pipeline, which makes it "
            "AlphaFold-derived and still not an AlphaFold DB model.",
            "A predicted model is not experimental evidence that the protein adopts the "
            "modelled conformation, and a high pLDDT is a statement about the predictor's "
            "confidence rather than an observation of the fold.",
            "pLDDT (0-100) and QMEANDisCo (0-1) are different quantities, not one quantity "
            "rescaled. Averages are grouped by metric and never merged. An accession whose "
            "models are all template-based carries no pLDDT at all, so the "
            "`confident_plddt` threshold the page prints does not apply to it.",
            "A predicted-model count is not an attention proxy and must never be pooled "
            "with deposited-structure counts. Experimentally determined entries are "
            "partitioned out here and counted in `excluded_experimental_count` only.",
            "Absence is recorded, not inferred. `alphafold_host_served_models: false` on "
            "Q3LZX1 is a measured absence -- the host was asked unconditionally and holds "
            "nothing -- and not a request that was skipped.",
            "The accessions are read out of the landing record's held-out case rather than "
            "written here, so this inventory covers the relatives the chain actually "
            "reached. It says nothing about SARS-CoV-2's own P0DTD1, whose curated entries "
            "that case deletes before the run.",
        ],
        "provenance": provenance(),
    }
    payload["content_sha256"] = content_digest(payload)
    if record:
        write_json(RECORD, payload)
        print(f"wrote {RECORD.relative_to(ROOT)}  "
              f"({total} models, {by_host} served by the AlphaFold host, "
              f"{authored} computed by AlphaFold)")
    return payload


if __name__ == "__main__":
    run()
