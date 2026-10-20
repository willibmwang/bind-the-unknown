import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, config, content_digest, digest, download, write_json

PG_BASE = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/"
PG_META = "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv"
ATLAS_BASE = "https://www.dsimb.inserm.fr/ATLAS/"
ATLAS_META = ATLAS_BASE + "data/download/distributions/2024_11_18_ATLAS_info.tsv"
AA = "ACDEFGHIKLMNPQRSTVWY"


def prepare_binding():
    c = config()
    raw, out = ROOT / "data/raw", ROOT / "data/processed/binding"
    out.mkdir(parents=True, exist_ok=True)
    provenance = [download(PG_META, raw / "proteingym_metadata.csv"),
                  download(PG_BASE + "DMS_ProteinGym_substitutions.zip", raw / "proteingym_v1.3.zip")]
    meta = pd.read_csv(raw / "proteingym_metadata.csv")
    row = meta.loc[meta.DMS_id == c["binding"]["dataset"]].iloc[0]
    with zipfile.ZipFile(raw / "proteingym_v1.3.zip") as z:
        name = "DMS_ProteinGym_substitutions/" + row.DMS_filename
        df = pd.read_csv(z.open(name), usecols=["mutant", "DMS_score", "DMS_score_bin"])
    if df.mutant.duplicated().any() or not np.isfinite(df.DMS_score).all():
        raise ValueError("Duplicated mutant identifiers or non-finite outcomes")
    positions = sorted({int(m[1:-1]) for s in df.mutant for m in s.split(":")})
    wildtype = [row.target_seq[p - 1] for p in positions]
    digits = np.tile(np.array([AA.index(a) for a in wildtype], dtype=np.int16), (len(df), 1))
    pos_index = {p: i for i, p in enumerate(positions)}
    for i, s in enumerate(df.mutant):
        for m in s.split(":"):
            match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", m)
            if not match or match[1] != row.target_seq[int(match[2]) - 1]:
                raise ValueError("Invalid mutation or reference-sequence mismatch")
            digits[i, pos_index[int(match[2])]] = AA.index(match[3])
    df.to_csv(out / "outcomes.csv", index=False)
    np.save(out / "features.npy", digits)
    order = np.random.default_rng(c["seed"]).permutation(len(df))
    bc = c["binding"]
    required = bc["development_pool_size"] + bc["pool_size"] * bc["evaluation_pools"]
    if required > len(df):
        raise ValueError("Insufficient rows for disjoint development and evaluation pools")
    dev = order[:bc["development_pool_size"]]
    pools = order[bc["development_pool_size"]:required].reshape(bc["evaluation_pools"], bc["pool_size"])
    np.savez(out / "splits.npz", development=dev, evaluation=pools)
    manifest = {"dataset": row.DMS_id, "rows": len(df), "positions": positions,
                "feature_encoding": "canonical residue index at four variable positions",
                "endpoint": row.selection_assay, "selection_type": row.selection_type,
                "direction": "higher DMS_score is better; assay score, not physical binding energy",
                "sources": provenance, "config_sha256": digest(ROOT / "hpc_pilots/configs/today.json"),
                "outcomes_sha256": digest(out / "outcomes.csv"),
                "features_sha256": digest(out / "features.npy"),
                "splits_sha256": digest(out / "splits.npz"),
                "limitations": ["Single canonical-protein assay", "No non-canonical or antiviral claim",
                                "Candidate-pool repetitions are not independent biological assays"]}
    # Stamped here rather than through `write_landscape`, because the pilot writes its
    # artifacts earlier in this function and its manifest key order is already recorded in
    # results; the identity is what matters, not the route to it.
    manifest["content_sha256"] = content_digest(manifest)
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"binding_rows": len(df), "evaluation_pools": len(pools),
                      "content_sha256": manifest["content_sha256"],
                      "manifest": str(out / "manifest.json")}))


def pool_geometry(rows, pool_size, max_pools):
    """Largest set of genuinely disjoint pools that fits, shrinking the pool before overlapping.

    The GB1 landscape has 149,360 variants, so twenty disjoint 4,096-candidate pools plus a
    development pool fit easily. Mpro has 5,725 and the influenza scans about 10,700, so that
    geometry is impossible for them. Candidates are never silently reused across pools that
    are then described as disjoint: the pool size shrinks instead, and the shortfall is recorded.
    """
    size = pool_size
    while size >= 64:
        pools = rows // size - 1
        if pools >= 2:
            pools = min(pools, max_pools)
            return {"rows": rows, "pool_size": size, "evaluation_pools": pools,
                    "development_pool_size": size, "candidates_used": size * (pools + 1),
                    "candidates_unused": rows - size * (pools + 1), "disjoint": True,
                    "requested_pool_size": pool_size, "requested_pools": max_pools,
                    "reduced": size != pool_size or pools != max_pools}
        size //= 2
    raise ValueError(f"{rows} rows cannot support two disjoint pools of at least 64 candidates")


def prepare_target(landscape):
    """Stage one pathogen landscape as a (position, substitution) decision environment."""
    tc = json.loads((ROOT / "hpc_pilots/configs/targets.json").read_text())
    spec = tc["landscapes"][landscape]
    raw, out = ROOT / "data/raw", ROOT / "data/processed/targets" / landscape
    out.mkdir(parents=True, exist_ok=True)
    provenance = [download(PG_META, raw / "proteingym_metadata.csv"),
                  download(PG_BASE + "DMS_ProteinGym_substitutions.zip", raw / "proteingym_v1.3.zip")]
    meta = pd.read_csv(raw / "proteingym_metadata.csv")
    row = meta.loc[meta.DMS_id == spec["dms_id"]].iloc[0]
    with zipfile.ZipFile(raw / "proteingym_v1.3.zip") as z:
        df = pd.read_csv(z.open("DMS_ProteinGym_substitutions/" + row.DMS_filename),
                         usecols=["mutant", "DMS_score", "DMS_score_bin"])
    if df.mutant.duplicated().any() or not np.isfinite(df.DMS_score).all():
        raise ValueError("Duplicated mutant identifiers or non-finite outcomes")
    if df.mutant.str.contains(":").any():
        raise ValueError("Multi-substitution variants are not supported by this encoding")
    positions = sorted({int(m[1:-1]) for m in df.mutant})
    pos_index = {p: i for i, p in enumerate(positions)}
    # Each candidate differs from wildtype at exactly one site, so (position, residue) is a
    # complete and deliberately weak encoding: no sequence context, structure or MSA features.
    features = np.zeros((len(df), 2), dtype=np.int16)
    for i, s in enumerate(df.mutant):
        match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", s)
        if not match or match[1] != row.target_seq[int(match[2]) - 1]:
            raise ValueError(f"Invalid mutation or reference-sequence mismatch: {s}")
        features[i] = (pos_index[int(match[2])], AA.index(match[3]))
    geometry = pool_geometry(len(df), tc["pool_size"], tc["max_evaluation_pools"])
    dev, pools = _split_pools(len(df), geometry, tc["seed"])
    write_landscape(out, df, features, geometry, dev, pools, {
        "landscape": landscape, "dataset": row.DMS_id, "rows": len(df),
        "target": spec["target"], "pathogen": spec["pathogen"], "family": spec["family"],
        "variable_positions": len(positions), "sequence_length": int(row.seq_len),
        "feature_encoding": "(position index, substituted residue index) for a single substitution",
        "cardinalities": [len(positions), len(AA)],
        "endpoint": spec["endpoint"], "reported_selection_assay": str(row.selection_assay),
        "direction": "higher DMS_score is better within this assay only",
        "pool_geometry": geometry, "top_fraction": tc["top_fraction"],
        "top_fraction_note": tc["top_fraction_note"],
        "sources": provenance,
        "targets_config_sha256": digest(ROOT / "hpc_pilots/configs/targets.json"),
        "outcomes_sha256": None,
        "features_sha256": None,
        "splits_sha256": None,
        "limitations": [
            f"The measured endpoint is {spec['endpoint']}. It is not a macrocycle binding "
            "measurement and not an antiviral activity measurement.",
            "Canonical single substitutions only: no non-canonical residue, no cyclisation and "
            "no macrocycle appears in this landscape.",
            "Pools are sub-landscapes of one protein. Pool spread is descriptive variability, "
            "not biological replication across independent targets.",
            "Metrics must not be pooled across landscapes with different assay endpoints."]},
        summary={"landscape": landscape, "positions": len(positions)})


def _read_landscape(dms_id):
    """Fetch one ProteinGym assay table and its metadata row, with provenance."""
    raw = ROOT / "data/raw"
    provenance = [download(PG_META, raw / "proteingym_metadata.csv"),
                  download(PG_BASE + "DMS_ProteinGym_substitutions.zip", raw / "proteingym_v1.3.zip")]
    meta = pd.read_csv(raw / "proteingym_metadata.csv")
    row = meta.loc[meta.DMS_id == dms_id].iloc[0]
    with zipfile.ZipFile(raw / "proteingym_v1.3.zip") as z:
        df = pd.read_csv(z.open("DMS_ProteinGym_substitutions/" + row.DMS_filename),
                         usecols=["mutant", "DMS_score", "DMS_score_bin"])
    if df.mutant.duplicated().any() or not np.isfinite(df.DMS_score).all():
        raise ValueError(f"{dms_id}: duplicated mutant identifiers or non-finite outcomes")
    return row, df, provenance


def encode_variants(df, target_seq):
    """Integer-encode variants, choosing the narrower encoding when it is sufficient.

    Single-substitution landscapes are encoded as (position, residue), which is complete
    and keeps the surrogate small. Landscapes containing combined substitutions need the
    residue identity at every variable position instead, because a variant is no longer
    described by one change. Both stay deliberately weak: no sequence context, no
    structure, no evolutionary features.
    """
    split = [s.split(":") for s in df.mutant]
    positions = sorted({int(m[1:-1]) for parts in split for m in parts})
    index = {p: i for i, p in enumerate(positions)}
    for parts in split:
        for m in parts:
            match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", m)
            if not match or match[1] != target_seq[int(match[2]) - 1]:
                raise ValueError(f"Invalid mutation or reference-sequence mismatch: {m}")
    if max(len(parts) for parts in split) == 1:
        features = np.array([[index[int(parts[0][1:-1])], AA.index(parts[0][-1])]
                             for parts in split], dtype=np.int16)
        return (features, [len(positions), len(AA)], len(positions),
                "(position index, substituted residue index) for a single substitution")
    wildtype = np.array([AA.index(target_seq[p - 1]) for p in positions], dtype=np.int16)
    features = np.tile(wildtype, (len(split), 1))
    for i, parts in enumerate(split):
        for m in parts:
            features[i, index[int(m[1:-1])]] = AA.index(m[-1])
    return (features, [len(AA)] * len(positions), len(positions),
            "residue identity at each variable position, supporting combined substitutions")


def write_landscape(out, df, features, geometry, dev, pools, manifest, summary=None):
    """Write one staged landscape and its manifest.

    Every arm ends the same way -- three artifacts, three digests of those artifacts, a
    manifest and a summary line -- and each `prepare_*` used to spell that out again. The
    repetition had already produced a divergence: `prepare_target` inlined its own copy of
    the pool split instead of calling `_split_pools`, so the disjointness assertion lived in
    two places and could have drifted in one.

    The digests are taken after the files are on disk rather than from the arrays in
    memory, so a manifest can only ever describe bytes that were actually written.

    The caller supplies the manifest with the three digest keys already present and set to
    None. They are filled in place rather than appended, because a manifest's own hash is
    recorded in every result file that used it, and reordering its keys would change those
    bytes and silently break the tie between a recorded run and the data it ran on.
    """
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "outcomes.csv", index=False)
    np.save(out / "features.npy", features)
    np.savez(out / "splits.npz", development=dev, evaluation=pools)
    for key, name in (("outcomes_sha256", "outcomes.csv"), ("features_sha256", "features.npy"),
                      ("splits_sha256", "splits.npz")):
        if key not in manifest:
            raise KeyError(f"{key} must be present in the manifest so its position is the "
                           f"caller's choice, not this function's")
        manifest[key] = digest(out / name)
    # Stable identity for the staged data: the same inputs restage to the same value, which
    # the file's own hash cannot do while it carries observation timestamps.
    manifest["content_sha256"] = content_digest(manifest)
    write_json(out / "manifest.json", manifest)
    print(json.dumps({**(summary or {}), "rows": len(df), "geometry": geometry,
                      "manifest": str(out / "manifest.json")}))


def _split_pools(rows, geometry, seed):
    order = np.random.default_rng(seed).permutation(rows)
    dev = order[:geometry["development_pool_size"]]
    pools = order[geometry["development_pool_size"]:geometry["candidates_used"]].reshape(
        geometry["evaluation_pools"], geometry["pool_size"])
    if len(np.unique(np.r_[dev, pools.ravel()])) != geometry["candidates_used"]:
        raise ValueError("Pools are not disjoint")
    return dev, pools


def encode_library(sequences, separator=None):
    """Encode a library given as whole sequences rather than mutation strings.

    Display libraries are not scans around a wild type: every member is a draw from a
    randomised region, so there is no reference sequence to diff against and the
    `(position, substituted residue)` encoding used for deep mutational scans does not
    apply. Each position becomes a categorical over whatever symbols actually occur at it.

    `separator` splits a sequence into monomer tokens, for libraries built from named
    building blocks rather than single letters. With `separator=None` each character is a
    token, which recovers the canonical-residue case. Either way the alphabet is read off
    the data instead of being assumed, so non-canonical monomers need no special casing.
    """
    tokens = [list(s) if separator is None else s.split(separator) for s in sequences]
    widths = {len(t) for t in tokens}
    if len(widths) != 1:
        raise ValueError(f"Library members differ in length: {sorted(widths)}")
    vocabularies = [sorted({t[i] for t in tokens}) for i in range(widths.pop())]
    lookup = [{symbol: index for index, symbol in enumerate(v)} for v in vocabularies]
    features = np.array([[lookup[i][symbol] for i, symbol in enumerate(t)] for t in tokens],
                        dtype=np.int16)
    cardinalities = [len(v) for v in vocabularies]
    note = ("categorical symbol at each randomised position, vocabulary read from the "
            "library itself")
    return features, cardinalities, vocabularies, note


DESCRIPTORS = ("mw", "logp", "tpsa", "hbd", "hba", "rotb", "rings", "aromatic", "heavy", "fcsp3")


def monomer_descriptors(vocabularies, table):
    """Per-position chemical descriptors for a library built from named monomers.

    A token index says only *which* building block occupies a position; it carries no
    chemistry, so a model over token indices learns which monomers work and never why.
    This attaches the actual structures, so the same library can be modelled either way
    and the two compared. Returns `None` when any symbol has no structure, because a
    silently half-populated descriptor matrix is worse than none.

    The array is (positions, symbols, descriptors), aligned with the token vocabularies,
    so row `features[i, p]` indexes `descriptors[p, features[i, p]]`.
    """
    known = {r["monomer"]: r for r in table}
    missing = sorted({s for v in vocabularies for s in v if s not in known})
    if missing:
        return None, missing
    width = max(len(v) for v in vocabularies)
    out = np.full((len(vocabularies), width, len(DESCRIPTORS)), np.nan, dtype=np.float32)
    for p, vocabulary in enumerate(vocabularies):
        for s, symbol in enumerate(vocabulary):
            out[p, s] = [known[symbol][d] for d in DESCRIPTORS]
    return out, []


def prepare_display(landscape):
    """Stage one display-selection library: whole sequences with a measured endpoint.

    Unlike `prepare_method`, nothing here comes from ProteinGym. The input is a local
    two-column table of sequence and outcome, because the datasets of the molecule class
    this project actually designs -- macrocyclic, built from named monomers -- are not
    distributed in a mutation-string format and could not be expressed in one.
    """
    dc = json.loads((ROOT / "hpc_pilots/configs/display.json").read_text())
    spec = dc["landscapes"][landscape]
    source = ROOT / spec["path"]
    if not source.exists():
        raise FileNotFoundError(f"{source} is absent; see the acquisition note in display.json")
    out = ROOT / "data/processed/display" / landscape
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(source)
    df = raw[[spec["sequence_column"], spec["outcome_column"]]].copy()
    df.columns = ["sequence", "outcome"]
    df = df.dropna()
    if not np.isfinite(df.outcome).all():
        raise ValueError("Non-finite outcome values")
    # Duplicate members would break pool disjointness, which is enforced downstream by
    # index rather than by sequence, so they are collapsed here and the count recorded.
    duplicates = int(df.sequence.duplicated().sum())
    df = df.drop_duplicates(subset="sequence").reset_index(drop=True)

    features, cardinalities, vocabularies, encoding = encode_library(
        df.sequence.tolist(), spec.get("monomer_separator"))
    # A set too small for two disjoint pools is staged without them rather than being
    # forced into a geometry it cannot support. It is then usable for paired comparison
    # and nothing else, which the manifest states.
    if spec.get("pooled", True):
        geometry = pool_geometry(len(df), spec["pool_size"], dc["max_evaluation_pools"])
        dev, pools = _split_pools(len(df), geometry, dc["seed"])
    else:
        geometry = {"pooled": False, "rows": len(df),
                    "reason": "too few members for two disjoint pools; no replay is possible"}
        dev, pools = np.empty(0, dtype=int), np.empty((0, 0), dtype=int)

    # Chemistry, where the monomers have structures. Written beside the token features
    # rather than replacing them, so the replay path is unchanged and a token model and a
    # descriptor model can be compared on identical pools.
    chemistry, unmapped = None, []
    if spec.get("monomer_table"):
        table = pd.read_csv(ROOT / spec["monomer_table"]).to_dict("records")
        chemistry, unmapped = monomer_descriptors(vocabularies, table)
        if chemistry is not None:
            np.save(out / "descriptors.npy", chemistry)

    write_landscape(out, df, features, geometry, dev, pools, {
        "landscape": landscape, "rows": len(df), "duplicates_collapsed": duplicates,
        "descriptors": list(DESCRIPTORS) if chemistry is not None else None,
        "descriptors_sha256": digest(out / "descriptors.npy") if chemistry is not None else None,
        "monomers_without_structures": unmapped,
        "target": spec["target"], "modality": spec["modality"],
        "cyclisation": spec["cyclisation"], "noncanonical": spec["noncanonical"],
        "randomised_positions": len(cardinalities),
        "feature_encoding": encoding, "cardinalities": cardinalities,
        "vocabularies": vocabularies,
        "endpoint": spec["endpoint"], "direction": spec["direction"],
        "censoring": spec.get("censoring"),
        "pool_geometry": geometry, "top_fraction": spec["top_fraction"],
        "rationale": spec["why"], "source": spec["source"],
        "display_config_sha256": digest(ROOT / "hpc_pilots/configs/display.json"),
        "input_sha256": digest(source),
        "outcomes_sha256": None,
        "features_sha256": None,
        "splits_sha256": None,
        "limitations": dc["limitations"] + spec.get("limitations", [])},
        summary={"landscape": landscape, "randomised_positions": len(cardinalities),
                 "cardinalities": cardinalities, "duplicates_collapsed": duplicates})


def prepare_method(landscape):
    """Stage one method-development landscape. These are substrates, not target proposals."""
    mc = json.loads((ROOT / "hpc_pilots/configs/method.json").read_text())
    spec = mc["landscapes"][landscape]
    out = ROOT / "data/processed/method" / landscape
    out.mkdir(parents=True, exist_ok=True)
    row, df, provenance = _read_landscape(spec["dms_id"])
    features, cardinalities, n_positions, encoding = encode_variants(df, row.target_seq)
    combined = int((df.mutant.str.count(":") > 0).sum())
    geometry = pool_geometry(len(df), spec["pool_size"], mc["max_evaluation_pools"])
    dev, pools = _split_pools(len(df), geometry, mc["seed"])
    write_landscape(out, df, features, geometry, dev, pools, {
        "landscape": landscape, "dataset": row.DMS_id, "rows": len(df),
        "protein": spec["protein"], "interface": spec["interface"],
        "role": "method development substrate; not a pathogen target proposal",
        "variable_positions": n_positions, "sequence_length": int(row.seq_len),
        "combined_substitution_variants": combined,
        "feature_encoding": encoding, "cardinalities": cardinalities,
        "endpoint": spec["endpoint"], "reported_selection_assay": str(row.selection_assay),
        "reported_selection_type": str(row.selection_type),
        "direction": "higher DMS_score is better within this assay only",
        "pool_geometry": geometry, "top_fraction": spec["top_fraction"],
        "rationale": spec["why"], "sources": provenance,
        "method_config_sha256": digest(ROOT / "hpc_pilots/configs/method.json"),
        "outcomes_sha256": None,
        "features_sha256": None,
        "splits_sha256": None,
        "limitations": mc["limitations"] + [
            f"The measured endpoint is {spec['endpoint']}. It is not a macrocycle "
            "measurement and establishes nothing about non-canonical chemistry."]},
        summary={"landscape": landscape, "positions": n_positions,
                 "combined_substitution_variants": combined, "encoding": encoding})


def prepare_atlas_manifest():
    c, out = config(), ROOT / "data/processed/atlas"
    out.mkdir(parents=True, exist_ok=True)
    provenance = download(ATLAS_META, ROOT / "data/raw/atlas_metadata.tsv")
    df = pd.read_csv(ROOT / "data/raw/atlas_metadata.tsv", sep="\t")
    sc = c["structure"]
    mask = ((df.organism == "H. sapiens") & df.length.between(sc["minimum_length"], sc["maximum_length"]) &
            (df.non_redundant_protein.astype(str).str.lower() == "true"))
    eligible = df.loc[mask].sort_values("PDB").drop_duplicates("ECOD_F")
    n = sc["development_proteins"] + sc["evaluation_proteins"]
    selected = eligible.iloc[:n]
    if len(selected) != n:
        raise ValueError("Not enough eligible distinct-family entries")
    # Restrict selection metadata to information available before trajectory analysis.
    cols = ["PDB", "length", "protein_name", "organism", "ECOD_F", "sequence"]
    selected = selected[cols].copy()
    selected["split"] = ["development"] * sc["development_proteins"] + ["evaluation"] * sc["evaluation_proteins"]
    selected.to_csv(out / "selection.csv", index=False)
    manifest = {"metadata_rows": len(df), "selected_entries": len(selected),
                "selection_rule": "human, length 50-250, nonredundant, first PDB per ECOD family, sorted by PDB",
                "source": provenance, "selection_sha256": digest(out / "selection.csv"),
                "data_license": "CC BY-NC 4.0; noncommercial research pilot",
                "replicates": {"acquisition": [1, 2], "held_out_reference": 3},
                "limitations": ["MD reference is not experimental binding truth",
                                "Initial structures are equilibrated ATLAS starting structures",
                                "Commercial use requires separate terms from the dataset owner"]}
    manifest["content_sha256"] = content_digest(manifest)
    write_json(out / "manifest.json", manifest)
    print(selected[["PDB", "length", "split"]].to_string(index=False))


def fetch_atlas(index):
    selected = pd.read_csv(ROOT / "data/processed/atlas/selection.csv")
    pdb = str(selected.iloc[index].PDB)
    directory = ROOT / "data/raw/atlas" / pdb
    directory.mkdir(parents=True, exist_ok=True)
    provenance = download(ATLAS_BASE + "api/ATLAS/analysis/" + pdb, directory / "analysis.zip")
    with zipfile.ZipFile(directory / "analysis.zip") as z:
        for name in [pdb + ".pdb"] + [f"{pdb}_R{r}.xtc" for r in (1, 2, 3)]:
            # Extract only expected flat data files, never executable archive contents.
            with z.open(name) as source, (directory / name).open("wb") as target:
                import shutil
                shutil.copyfileobj(source, target)
    write_json(directory / "provenance.json", provenance)
    return pdb, directory
