"""Download CycPeptMPDB and cut it into a landscape this project's stager can encode.

**Why a script and not a committed CSV.** `data/raw/` is gitignored and the one exception --
`thrombin_monomer_descriptors.csv` -- was force-added without a script that produces it, so
that landscape cannot be rebuilt from a clean checkout. This one can. It also keeps the
question of redistribution out of the repository: no licence or terms-of-use statement
appears on the CycPeptMPDB download page as fetched on 24 November 2026, the data is offered
for free download, and this repository is public. A script that fetches is safe where a
committed copy of someone else's unlicensed release is not.

**The four cuts, and why each is a cut and not a filter of convenience.**

*One assay.* The release carries PAMPA, Caco-2, MDCK and RRCK. Those are four instruments
reporting the same quantity, not four measurements of it, and pooling them is exactly the
scale-mixing defect `intake.withhold_remote_conservation` exists to repair on the other arm
of this project. PAMPA is taken because it is the largest at 7,298 entries. No figure from
this landscape may be compared with one computed on another assay.

*One monomer length.* `data.encode_library` represents a member as a tuple of per-position
tokens, so a set whose members differ in length has no position to encode and the stager
refuses it. Length 6 is the largest PAMPA set at 2,167 entries; 7 and 10 are comparable and
are not built here.

*Circle topology only.* A lariat carries a tail whose positions are not the ring's positions.
A fixed-position encoding would align a tail residue with a ring residue as though they were
the same place in the molecule, which is a claim about chemistry made by an array index.

*A "." separator, not the "-" the thrombin library uses.* Twenty-four of the 118 symbols
carry a hyphen inside their own name -- `(N->O)Val(3-OH)`, `1-Nal`, `3-pyridylethyl_Gly`.
Splitting on a hyphen fragments them into longer pseudo-sequences. The stager caught this as
a length mismatch rather than accepting it; "." is the separator HELM uses and no symbol in
the release contains one.

**What it does not do.** It does not deduplicate -- the stager collapses duplicate sequences
and records the count -- and it does not read the release's `Detection_Limit` columns, so a
censored value is carried here as if it were measured.

    python3 output/build_cycpept_landscape.py                 # length 6, PAMPA
    python3 output/build_cycpept_landscape.py --length 7
    python3 output/build_cycpept_landscape.py --keep-download  # leave the release on disk
"""
import argparse
import ast
import csv
import pathlib
import sys
import tempfile
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data/raw/cycpeptmpdb"
BASE = "http://cycpeptmpdb.com/static//download"
PEPTIDES = f"{BASE}/peptides/CycPeptMPDB_Peptide_All.csv"
MONOMERS = f"{BASE}/monomers/CycPeptMPDB_Monomer_All.csv"

# The ten the stager reads, in `data.DESCRIPTORS` order, against the release's own RDKit
# column names. Keeping the pairing here rather than renaming columns in place means a
# reader can check each one against the source file.
DESCRIPTORS = [("mw", "MolWt"), ("logp", "MolLogP"), ("tpsa", "TPSA"),
               ("hbd", "NumHDonors"), ("hba", "NumHAcceptors"),
               ("rotb", "NumRotatableBonds"), ("rings", "RingCount"),
               ("aromatic", "NumAromaticRings"), ("heavy", "HeavyAtomCount"),
               ("fcsp3", "FractionCSP3")]

# The site's own download table, so a silent change upstream is caught here rather than
# surfacing as a quietly different result. Checked, not assumed.
EXPECTED = {"PAMPA": 7298, "Caco2": 1332, "MDCK": 64, "RRCK": 186}
EXPECTED_LENGTH = {2: 4, 3: 69, 4: 55, 5: 88, 6: 2167, 7: 2071, 8: 120, 9: 423, 10: 1777,
                   11: 675, 12: 632, 13: 234, 14: 125, 15: 26}


def fetch(url, into):
    """Plain HTTP on purpose: the host's TLS rejects the SNI name and fails the handshake."""
    print(f"  fetching {url}")
    with urllib.request.urlopen(url, timeout=300) as response:
        into.write_bytes(response.read())
    return into


def rows_of(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as handle:
        return list(csv.DictReader(handle))


def build(length, assay, separator, download_dir):
    peptides = rows_of(fetch(PEPTIDES, download_dir / "peptides_all.csv"))
    monomers = {r["Symbol"]: r
                for r in rows_of(fetch(MONOMERS, download_dir / "monomers_all.csv"))}
    print(f"  release: {len(peptides)} peptides, {len(monomers)} monomers")

    for name, count in EXPECTED.items():
        seen = sum(1 for r in peptides if r[name].strip())
        if seen != count:
            print(f"  NOTE: {name} holds {seen} entries, the site's table said {count}. "
                  f"The release has changed since 24 Nov 2026; re-read the download page "
                  f"before trusting a comparison with an earlier run.", file=sys.stderr)

    kept, used = [], set()
    for row in peptides:
        if (row["Monomer_Length"] != str(length) or not row[assay].strip()
                or row["Molecule_Shape"] != "Circle"):
            continue
        sequence = ast.literal_eval(row["Sequence"])
        if len(sequence) != length:
            continue
        if any(separator in symbol for symbol in sequence):
            raise ValueError(f"separator {separator!r} appears inside a monomer symbol in "
                             f"{sequence}; splitting on it would fragment the sequence")
        used.update(sequence)
        kept.append({"peptide": separator.join(sequence), "target": float(row[assay]),
                     "source": row["Source"]})
    if not kept:
        raise ValueError(f"no Circle peptide of length {length} carries a {assay} value")

    missing = sorted(s for s in used if s not in monomers)
    if missing:
        raise ValueError(f"{len(missing)} monomers have no row in the release's monomer "
                         f"table and so would have no descriptors: {missing[:8]}")

    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"cycpept_len{length}_{assay.lower()}"
    peptide_file = OUT / f"{stem}.csv"
    with open(peptide_file, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["peptide", "target", "source"])
        writer.writeheader()
        writer.writerows(kept)

    monomer_file = OUT / f"{stem}_monomer_descriptors.csv"
    with open(monomer_file, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["monomer", "group", "smiles"]
                        + [name for name, _ in DESCRIPTORS] + ["is_absence"])
        for symbol in sorted(used):
            entry = monomers[symbol]
            blank = [c for _, c in DESCRIPTORS if not str(entry[c]).strip()]
            if blank:
                raise ValueError(f"monomer {symbol} has no value for {blank}; a zero here "
                                 f"would be read as a measurement")
            writer.writerow(
                [symbol, entry.get("Monomer_Type") or "",
                 entry.get("replaced_SMILES") or entry.get("CXSMILES") or ""]
                + [entry[column] for _, column in DESCRIPTORS] + ["False"])

    print(f"\nwrote {peptide_file.relative_to(ROOT)}")
    print(f"  {len(kept)} rows, {len({r['peptide'] for r in kept})} unique sequences, "
          f"{len({r['source'] for r in kept})} source publications")
    if EXPECTED_LENGTH.get(length) and length == 6:
        print(f"  (the site's table lists {EXPECTED_LENGTH[length]} entries at this length "
              f"across all assays and both topologies)")
    print(f"wrote {monomer_file.relative_to(ROOT)}")
    print(f"  {len(used)} monomers, every one with a complete descriptor row")
    print(f"\nnext: python3 -m hpc_pilots prepare-display --landscape {stem}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--length", type=int, default=6)
    parser.add_argument("--assay", default="PAMPA", choices=sorted(EXPECTED))
    parser.add_argument("--separator", default=".")
    parser.add_argument("--keep-download", action="store_true",
                        help="leave the full release on disk under data/raw/cycpeptmpdb")
    args = parser.parse_args()

    if args.keep_download:
        target = OUT / "release"
        target.mkdir(parents=True, exist_ok=True)
        build(args.length, args.assay, args.separator, target)
    else:
        with tempfile.TemporaryDirectory() as temporary:
            build(args.length, args.assay, args.separator, pathlib.Path(temporary))


if __name__ == "__main__":
    main()
