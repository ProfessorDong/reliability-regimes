"""Freeze complete, paginated ChEMBL activity responses as portable CSV files."""
import argparse
import csv
import gzip
import io
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
TARGETS = {"faah": "CHEMBL2243", "ptgs2": "CHEMBL230", "kdr": "CHEMBL279", "herg": "CHEMBL240"}
LABELS = {"faah": "FAAH", "ptgs2": "PTGS2", "kdr": "KDR", "herg": "hERG"}
FIELDS = ["activity_id", "canonical_smiles", "standard_type", "standard_value", "standard_units",
          "standard_relation", "document_year", "document_chembl_id", "assay_chembl_id",
          "target_organism", "target_chembl_id"]
API = "https://www.ebi.ac.uk/chembl/api/data"
URL = (API + "/activity.json?standard_relation=%3D&standard_units=nM&standard_type__in=IC50,Ki,Kd,EC50"
       + "&only=" + ",".join(FIELDS) + "&limit=1000&order_by=activity_id")


def freeze(target, directory, fetch=False):
    observed = []
    count = None
    offset = 0
    while count is None or offset < count:
        filename = directory / f"{target}_{offset}.json"
        if fetch:
            address = f"{URL}&target_chembl_id={TARGETS[target]}&offset={offset}"
            with urlopen(address, timeout=180) as response:
                filename.write_bytes(response.read())
        page = json.loads(filename.read_text())
        metadata = page["page_meta"]
        if count is None:
            count = metadata["total_count"]
        rows = page["activities"]
        if (metadata["total_count"] != count or metadata["offset"] != offset
                or len(rows) != min(1000, count - offset)):
            raise ValueError(f"Incomplete pagination: {target} page {offset}")
        if bool(metadata["next"]) != (offset + len(rows) < count):
            raise ValueError(f"Unexpected next page: {target} page {offset}")
        observed.extend(rows)
        offset += len(rows)
    ids = [r["activity_id"] for r in observed]
    if sorted(set(ids)) != ids or len(ids) != count:
        raise ValueError(f"Duplicate or nonmonotonic IDs for {target}")
    for row in observed:
        if (row["target_chembl_id"] != TARGETS[target] or row["standard_relation"] != "="
                or row["standard_units"] != "nM" or row["standard_type"] not in {"IC50", "Ki", "Kd", "EC50"}):
            raise ValueError(f"Unexpected record: {row['activity_id']}")
    output = HERE / "records" / f"{target}.csv.gz"
    output.parent.mkdir(exist_ok=True)
    with output.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(observed)
    print(f"{target}: {count} unique activity IDs in {(count+999)//1000} pages", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, help="Directory containing previously retrieved numbered JSON pages")
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    args = parser.parse_args()
    if args.pages:
        for target in args.targets:
            freeze(target, args.pages)
    else:
        with urlopen(API + "/status.json", timeout=60) as response:
            status = json.load(response)
        if status["chembl_db_version"] != "ChEMBL_37":
            raise RuntimeError("The live ChEMBL release has changed. Use the supplied frozen records.")
        with tempfile.TemporaryDirectory() as directory:
            for target in args.targets:
                freeze(target, Path(directory), fetch=True)
