import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, config, digest, download, write_json

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
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"binding_rows": len(df), "evaluation_pools": len(pools), "manifest": str(out / "manifest.json")}))


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
    order = np.random.default_rng(tc["seed"]).permutation(len(df))
    dev = order[:geometry["development_pool_size"]]
    pools = order[geometry["development_pool_size"]:geometry["candidates_used"]].reshape(
        geometry["evaluation_pools"], geometry["pool_size"])
    if len(np.unique(np.r_[dev, pools.ravel()])) != geometry["candidates_used"]:
        raise ValueError("Pools are not disjoint")
    df.to_csv(out / "outcomes.csv", index=False)
    np.save(out / "features.npy", features)
    np.savez(out / "splits.npz", development=dev, evaluation=pools)
    write_json(out / "manifest.json", {
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
        "outcomes_sha256": digest(out / "outcomes.csv"),
        "features_sha256": digest(out / "features.npy"),
        "splits_sha256": digest(out / "splits.npz"),
        "limitations": [
            f"The measured endpoint is {spec['endpoint']}. It is not a macrocycle binding "
            "measurement and not an antiviral activity measurement.",
            "Canonical single substitutions only: no non-canonical residue, no cyclisation and "
            "no macrocycle appears in this landscape.",
            "Pools are sub-landscapes of one protein. Pool spread is descriptive variability, "
            "not biological replication across independent targets.",
            "Metrics must not be pooled across landscapes with different assay endpoints."]})
    print(json.dumps({"landscape": landscape, "rows": len(df),
                      "positions": len(positions), "geometry": geometry,
                      "manifest": str(out / "manifest.json")}))


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
    write_json(out / "manifest.json", {"metadata_rows": len(df), "selected_entries": len(selected),
               "selection_rule": "human, length 50-250, nonredundant, first PDB per ECOD family, sorted by PDB",
               "source": provenance, "selection_sha256": digest(out / "selection.csv"),
               "data_license": "CC BY-NC 4.0; noncommercial research pilot",
               "replicates": {"acquisition": [1, 2], "held_out_reference": 3},
               "limitations": ["MD reference is not experimental binding truth",
                               "Initial structures are equilibrated ATLAS starting structures",
                               "Commercial use requires separate terms from the dataset owner"]})
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
