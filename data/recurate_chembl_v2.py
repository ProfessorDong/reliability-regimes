"""Rigorous re-curation of the ChEMBL targets with full provenance and dates.

The original curated files contain only SMILES and pIC50, which prevents (a) a temporal
split, (b) assay-level quality control, and (c) the provenance reporting that Nature
Portfolio journals expect. This script re-downloads each target from the ChEMBL web
service and records, for every activity, the fields needed for all three.

Retained records must satisfy, at the activity level:
  * standard_relation '='            (censored values such as > or < are dropped)
  * standard_units 'nM'
  * standard_type in {IC50, Ki, Kd, EC50}
  * a parsable structure and a positive standard_value
Structures are standardized to a neutral parent (largest organic fragment, charges
neutralized) and grouped on the standard InChIKey, so that salts, counter-ions and
charge variants of the same compound collapse to one row. Activities are converted with
pAct = 9 - log10(value_nM) and aggregated by median.

Every filtering stage is counted and written to a provenance file so the curation flow
can be reported exactly.

    python data/recurate_chembl_v2.py
"""
from __future__ import annotations
import json, os, time, urllib.request, urllib.parse
from collections import defaultdict
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog('rdApp.*')
HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, 'chembl_v2')
API = 'https://www.ebi.ac.uk/chembl/api/data/activity.json'

# Verified against the ChEMBL target endpoint (human, SINGLE PROTEIN)
TARGETS = {
    'scd1': 'CHEMBL5555',    # Stearoyl-CoA desaturase
    'nk1r': 'CHEMBL249',     # Substance-P receptor (TACR1)
    'drd2': 'CHEMBL217',     # D(2) dopamine receptor
    'drd3': 'CHEMBL234',     # D(3) dopamine receptor
}
KEEP_TYPES = {'IC50', 'Ki', 'Kd', 'EC50'}
FIELDS = ['molecule_chembl_id', 'canonical_smiles', 'standard_type', 'standard_relation',
          'standard_value', 'standard_units', 'assay_chembl_id', 'assay_type',
          'document_chembl_id', 'document_year', 'target_organism']


def fetch_all(cid, page=1000):
    """Paginate every activity record for a target."""
    rows, offset = [], 0
    while True:
        u = (f"{API}?target_chembl_id={cid}&standard_relation=%3D&standard_units=nM"
             f"&limit={page}&offset={offset}")
        for attempt in range(4):
            try:
                r = json.load(urllib.request.urlopen(
                    urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'}), timeout=90))
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(3 * (attempt + 1))
        acts = r['activities']
        rows.extend([{k: a.get(k) for k in FIELDS} for a in acts])
        total = r['page_meta']['total_count']
        offset += page
        print(f"    fetched {min(offset, total)}/{total}", flush=True)
        if offset >= total or not acts:
            break
    return rows


_norm = rdMolStandardize.Normalizer()
_lfc = rdMolStandardize.LargestFragmentChooser()
_unc = rdMolStandardize.Uncharger()


def standard_parent(smi):
    """Neutral, largest-fragment, normalized parent -> (canonical SMILES, InChIKey)."""
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return None, None
    try:
        m = _unc.uncharge(_lfc.choose(_norm.normalize(m)))
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m), Chem.InchiToInchiKey(Chem.MolToInchi(m))
    except Exception:
        return None, None


def curate(name, cid):
    prov = {'target': name, 'chembl_id': cid}
    print(f"  == {name} ({cid})")
    raw = fetch_all(cid)
    prov['n_raw_records'] = len(raw)
    kept, drop = [], defaultdict(int)
    for a in raw:
        if a['standard_type'] not in KEEP_TYPES:
            drop['standard_type_not_affinity'] += 1; continue
        if a['target_organism'] != 'Homo sapiens':
            drop['non_human'] += 1; continue
        v = a['standard_value']
        try:
            v = float(v)
        except (TypeError, ValueError):
            drop['no_value'] += 1; continue
        if not (v > 0):
            drop['nonpositive_value'] += 1; continue
        smi, key = standard_parent(a['canonical_smiles'])
        if smi is None or key is None:
            drop['unparsable_structure'] += 1; continue
        a = dict(a); a['parent_smiles'] = smi; a['inchikey'] = key
        a['pAct'] = 9.0 - np.log10(v)
        kept.append(a)
    prov['n_after_filters'] = len(kept)
    prov['dropped'] = dict(drop)
    # aggregate by standardized parent InChIKey
    g = defaultdict(list)
    for a in kept:
        g[a['inchikey']].append(a)
    recs = []
    for key, items in g.items():
        vals = np.array([i['pAct'] for i in items], float)
        yrs = [i['document_year'] for i in items if i['document_year']]
        recs.append(dict(
            inchikey=key, SMILES=items[0]['parent_smiles'],
            pAct=float(np.median(vals)), n_records=len(items),
            sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            year_min=int(min(yrs)) if yrs else None,
            year_max=int(max(yrs)) if yrs else None,
            year_median=int(np.median(yrs)) if yrs else None,
            types=';'.join(sorted({i['standard_type'] for i in items})),
            assay_types=';'.join(sorted({str(i['assay_type']) for i in items})),
            n_documents=len({i['document_chembl_id'] for i in items}),
        ))
    prov['n_unique_parents'] = len(recs)
    prov['n_with_replicates'] = int(sum(1 for r in recs if r['n_records'] > 1))
    sds = [r['sd'] for r in recs if r['n_records'] > 1]
    prov['mean_within_compound_sd'] = float(np.mean(sds)) if sds else 0.0
    yy = [r['year_median'] for r in recs if r['year_median']]
    prov['year_range'] = [int(min(yy)), int(max(yy))] if yy else None
    prov['year_quartiles'] = [int(np.quantile(yy, q)) for q in (0.25, 0.5, 0.75)] if yy else None
    prov['type_mix'] = {t: sum(1 for r in recs if t in r['types']) for t in sorted(KEEP_TYPES)}
    os.makedirs(OUTD, exist_ok=True)
    import csv
    cols = ['inchikey', 'SMILES', 'pAct', 'n_records', 'sd', 'year_min', 'year_max',
            'year_median', 'types', 'assay_types', 'n_documents']
    with open(os.path.join(OUTD, f'{name}_chembl_v2.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in sorted(recs, key=lambda r: -r['pAct']):
            w.writerow({c: r[c] for c in cols})
    print(f"     raw {prov['n_raw_records']} -> filtered {prov['n_after_filters']} -> "
          f"unique parents {prov['n_unique_parents']}  years {prov['year_range']}", flush=True)
    return prov


if __name__ == '__main__':
    allprov = {}
    for name, cid in TARGETS.items():
        allprov[name] = curate(name, cid)
    os.makedirs(OUTD, exist_ok=True)
    with open(os.path.join(OUTD, 'curation_provenance.json'), 'w') as f:
        json.dump(allprov, f, indent=2)
    print('wrote', os.path.join(OUTD, 'curation_provenance.json'))
