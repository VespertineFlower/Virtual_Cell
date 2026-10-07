import json
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd


METADATA_PATH = "./data/WAYB_WAYC_metadata_train_val(1).csv"
OUTPUT_PATH = "chemical_smiles.csv"
PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/"
    "property/CanonicalSMILES,IsomericSMILES/JSON"
)

# Dataset labels occasionally use abbreviations or vendor-style names.
QUERY_ALIASES = {
    "(1R, 2S, 5R) - (-) - Menthol": "levomenthol",
    "1-10 Phenanthroline monohydrate": "1,10-phenanthroline monohydrate",
    "CHX": "cycloheximide",
    "FCCP": "carbonyl cyanide p-trifluoromethoxyphenylhydrazone",
    "LY 294002 hydrochloride": "LY294002 hydrochloride",
    "Oligomycin": "oligomycin A",
    "SDS": "sodium dodecyl sulfate",
    "U-73122": "U73122",
    "Tunicamycin": "tunicamycin V",
}

# Stable PubChem compound identifiers avoid occasional name-service ambiguity.
PUBCHEM_CIDS = {
    "Amphotericin B": 5280965,
    "Nocodazole": 4122,
}


def fetch_smiles(name: str) -> tuple[str, str, str]:
    if name in PUBCHEM_CIDS:
        url = PUBCHEM_URL.replace("name/{name}", f"cid/{PUBCHEM_CIDS[name]}")
    else:
        query = QUERY_ALIASES.get(name, name)
        url = PUBCHEM_URL.format(name=urllib.parse.quote(query, safe=""))
    request = urllib.request.Request(url, headers={"User-Agent": "baseline-smiles-fetch/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return "", "", f"error: {exc}"

    properties = payload["PropertyTable"]["Properties"][0]
    smiles = properties.get("ConnectivitySMILES") or properties.get("CanonicalSMILES") or ""
    isomeric = properties.get("SMILES") or properties.get("IsomericSMILES") or smiles
    return smiles, isomeric, "ok" if smiles else "missing"


def main() -> None:
    metadata = pd.read_csv(METADATA_PATH)
    chemicals = (
        metadata[["perturbation_no_concentration", "chemical_role"]]
        .drop_duplicates()
        .sort_values("perturbation_no_concentration")
    )

    rows = []
    roles = dict(chemicals.itertuples(index=False, name=None))
    cached = {}
    if Path(OUTPUT_PATH).exists():
        previous = pd.read_csv(OUTPUT_PATH)
        cached = {
            row["perturbation_no_concentration"]: row
            for _, row in previous.loc[previous["status"].eq("ok")].iterrows()
        }
    with ThreadPoolExecutor(max_workers=6) as executor:
        for name, row in cached.items():
            rows.append(row.to_dict())
            print(f"{name}: cached", flush=True)
        futures = {executor.submit(fetch_smiles, name): name for name in roles if name not in cached}
        for future in as_completed(futures):
            name = futures[future]
            canonical, isomeric, status = future.result()
            rows.append(
                {
                    "perturbation_no_concentration": name,
                    "chemical_role": roles[name],
                    "canonical_smiles": canonical,
                    "isomeric_smiles": isomeric,
                    "status": status,
                    "source": "PubChem PUG REST",
                }
            )
            print(f"{name}: {status}", flush=True)

    result = pd.DataFrame(rows).sort_values("perturbation_no_concentration")
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    unresolved = result.loc[result["canonical_smiles"].eq(""), "perturbation_no_concentration"].tolist()
    print(f"saved: {OUTPUT_PATH}")
    print(f"resolved: {len(result) - len(unresolved)}/{len(result)}")
    if unresolved:
        print("unresolved: " + ", ".join(unresolved))


if __name__ == "__main__":
    main()
