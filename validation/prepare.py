"""Standardize source archives and construct the prespecified balanced target panel."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import csv
import gzip
import hashlib
import io
import json
import os
from statistics import median

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

ROOT = Path(__file__).resolve().parent
TARGETS = {
    "ache": ("AChE", "CHEMBL220", "Other enzyme"),
    "adora2a": ("A2A", "CHEMBL251", "Class A GPCR"),
    "cnr2": ("CB2", "CHEMBL253", "Class A GPCR"),
    "egfr": ("EGFR", "CHEMBL203", "Kinase"),
    "faah": ("FAAH", "CHEMBL2243", "Other enzyme"),
    "herg": ("hERG", "CHEMBL240", "Ion channel"),
    "htr2a": ("5-HT2A", "CHEMBL224", "Class A GPCR"),
    "kdr": ("VEGFR2", "CHEMBL279", "Kinase"),
    "oprk1": ("KOR", "CHEMBL237", "Class A GPCR"),
    "oprm1": ("MOR", "CHEMBL233", "Class A GPCR"),
    "ptgs2": ("COX-2", "CHEMBL230", "Other enzyme"),
}


def prepare_target(target):
    RDLogger.DisableLog("rdApp.*")
    source = ROOT / "sources" / f"{target}_chembl.csv.gz"
    raw = gzip.decompress(source.read_bytes())
    reader = csv.DictReader(io.StringIO(raw.decode()))
    normalize = rdMolStandardize.Normalizer()
    choose = rdMolStandardize.LargestFragmentChooser()
    uncharge = rdMolStandardize.Uncharger()
    by_key = {}
    total = invalid = 0
    for row in reader:
        total += 1
        mol = Chem.MolFromSmiles(row["SMILES"])
        if mol is None:
            invalid += 1
            continue
        try:
            parent = uncharge.uncharge(choose.choose(normalize.normalize(mol)))
            Chem.SanitizeMol(parent)
            key = Chem.MolToInchiKey(parent)
            if not key:
                raise ValueError("Missing parent key")
        except (RuntimeError, ValueError):
            invalid += 1
            continue
        if key not in by_key:
            by_key[key] = (Chem.MolToSmiles(parent), [])
        by_key[key][1].append(float(row["pIC50"]))
    sample = sorted(by_key, key=lambda k: (hashlib.sha256(f"CommsChem-v1|{k}".encode()).digest(), k))[
        :1200
    ]
    rows = [(key, by_key[key][0], median(by_key[key][1]), len(by_key[key][1])) for key in sorted(sample)]
    destination = ROOT / "parents" / f"{target}.csv"
    with destination.open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["parent_key", "smiles", "pActivity", "source_rows_for_parent"])
        writer.writerows(rows)
    audit = dict(
        target=target,
        label=TARGETS[target][0],
        chembl=TARGETS[target][1],
        family=TARGETS[target][2],
        source_sha256=hashlib.sha256(raw).hexdigest(),
        parents_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        raw_rows=total,
        invalid_rows=invalid,
        parent_count=len(by_key),
        pooled_rows=total - invalid - len(by_key),
        modeled=len(sample),
    )
    print(target, "curated", len(by_key), "modeled", len(rows), flush=True)
    return audit


if __name__ == "__main__":
    (ROOT / "parents").mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=int(os.environ.get("PREP_WORKERS", "5"))) as pool:
        audits = list(pool.map(prepare_target, TARGETS))
    (ROOT / "input_manifest.json").write_text(json.dumps(audits, indent=2) + "\n")
