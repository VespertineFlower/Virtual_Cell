from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.parse
import warnings
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .baselines import protein_mean
from .constants import CONTROL_NAMES, EXCLUDED_TREATMENTS, MATCH_KEYS


PUBCHEM_PROPERTY_NAMES = (
    "Title,IUPACName,MolecularFormula,MolecularWeight,"
    "CanonicalSMILES,IsomericSMILES,ConnectivitySMILES,SMILES"
)

# Competition abbreviations and typographic variants that are unsafe to send
# to a name resolver without normalization.
PUBCHEM_QUERY_OVERRIDES: dict[str, str | None] = {
    "Quality Control": None,
    "CHX": "cycloheximide",
    "SDS": "sodium dodecyl sulfate",
    "MMS": "methyl methanesulfonate",
    "G418": "geneticin",
    "H2O2": "hydrogen peroxide",
    "NaCl": "sodium chloride",
    "DMSO": "dimethyl sulfoxide",
    "Water": "water",
    "EDTA": "edetic acid",
    "1-10 Phenanthroline monohydrate": "1,10-phenanthroline monohydrate",
    "(1R, 2S, 5R) - (-) - Menthol": "(-)-menthol",
    "LY 294002 hydrochloride": "LY294002 hydrochloride",
    "U-73122": "U73122",
}

AMBIGUOUS_REAGENT_NAMES = {
    "Oligomycin": (
        "Generic reagent name may denote oligomycin A, B, C, or a mixture; "
        "a product identifier is required before choosing a structure."
    ),
    "Tunicamycin": (
        "Generic reagent name denotes a family/mixture of tunicamycin homologs; "
        "a product identifier is required before choosing a structure."
    ),
}

OPEN_KNOWLEDGE_URLS = {
    "snp_distance": "http://1002genomes.u-strasbg.fr/files/1011DistanceMatrixBasedOnSNPs.tab.gz",
    "orf_distance": "http://1002genomes.u-strasbg.fr/files/1011DistanceMatrixBasedOnORFs.tab.gz",
    "peter_supplement": (
        "https://media.springernature.com/original/springer-static/esm/"
        "art%3A10.1038%2Fs41586-018-0030-5/MediaObjects/41586_2018_30_MOESM3_ESM.xls"
    ),
    "s288c_ncbi": (
        "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/GCF_000146045.2/"
        "download?include_annotation_type=GENOME_FASTA"
    ),
}


@dataclass
class FeatureStageConfig:
    morgan_bits: int = 2048
    morgan_radius: int = 2
    kmer_size: int = 6
    kmer_buckets: int = 4096
    pubchem_timeout_seconds: int = 30
    pubchem_pause_seconds: float = 0.20


def _download_if_missing(url: str, destination: Path, timeout_seconds: int = 120) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(
        url,
        stream=True,
        timeout=timeout_seconds,
        headers={"User-Agent": "GOAI-virtual-yeast-feature-stage/1.0"},
    ) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    stream.write(chunk)
    temporary.replace(destination)


def download_open_knowledge_resources(external_dir: Path) -> dict[str, Path]:
    """Download only the lightweight official resources required by this stage."""
    paths = {
        "snp_distance": external_dir / "1011DistanceMatrixBasedOnSNPs.tab.gz",
        "orf_distance": external_dir / "1011DistanceMatrixBasedOnORFs.tab.gz",
        "peter_supplement": external_dir / "Peter2018_supplementary_tables.xls",
        "s288c_zip": external_dir / "S288C_GCF_000146045.2.zip",
    }
    _download_if_missing(OPEN_KNOWLEDGE_URLS["snp_distance"], paths["snp_distance"])
    _download_if_missing(OPEN_KNOWLEDGE_URLS["orf_distance"], paths["orf_distance"])
    _download_if_missing(OPEN_KNOWLEDGE_URLS["peter_supplement"], paths["peter_supplement"])
    _download_if_missing(OPEN_KNOWLEDGE_URLS["s288c_ncbi"], paths["s288c_zip"])

    extraction_root = external_dir / "S288C_GCF_000146045.2"
    expected_fasta = (
        extraction_root
        / "ncbi_dataset"
        / "data"
        / "GCF_000146045.2"
        / "GCF_000146045.2_R64_genomic.fna"
    )
    if not expected_fasta.exists():
        extraction_root.mkdir(parents=True, exist_ok=True)
        resolved_root = extraction_root.resolve()
        with zipfile.ZipFile(paths["s288c_zip"]) as archive:
            for member in archive.infolist():
                target = (extraction_root / member.filename).resolve()
                if resolved_root not in target.parents and target != resolved_root:
                    raise ValueError(f"Unsafe path in NCBI archive: {member.filename}")
            archive.extractall(extraction_root)
    if not expected_fasta.exists():
        candidates = list(extraction_root.rglob("*.fna"))
        if len(candidates) != 1:
            raise FileNotFoundError("Could not identify a unique S288C genome FASTA in NCBI archive")
        expected_fasta = candidates[0]
    paths["s288c_fasta"] = expected_fasta
    return paths


def strict_feature_matched_controls(
    metadata: pd.DataFrame,
    y_log2: np.ndarray,
    target_rows: np.ndarray,
    control_pool: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Torch-free strict matcher used by the standalone feature runtime."""
    allowed = control_pool & metadata["perturbation_no_concentration"].isin(CONTROL_NAMES).to_numpy()
    groups: dict[tuple[object, ...], list[int]] = {}
    for position in np.flatnonzero(allowed):
        key = tuple(metadata.iloc[position][field] for field in MATCH_KEYS)
        groups.setdefault(key, []).append(int(position))
    rows: list[int] = []
    controls: list[np.ndarray] = []
    counts: list[int] = []
    values = np.asarray(y_log2)
    for position in np.flatnonzero(target_rows):
        key = tuple(metadata.iloc[position][field] for field in MATCH_KEYS)
        positions = groups.get(key)
        if not positions:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            control = np.nanmean(values[positions], axis=0).astype(np.float32)
        if np.isfinite(control).any():
            rows.append(int(position))
            controls.append(control)
            counts.append(len(positions))
    if not rows:
        return (
            np.empty(0, dtype=np.int64),
            np.empty((0, y_log2.shape[1]), dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
    return np.asarray(rows), np.stack(controls), np.asarray(counts)


def normalized_pubchem_query(name: str) -> str | None:
    if name in PUBCHEM_QUERY_OVERRIDES:
        return PUBCHEM_QUERY_OVERRIDES[name]
    return " ".join(str(name).strip().split())


def _extract_pubchem_record(payload: dict[str, Any]) -> dict[str, Any]:
    properties = payload["PropertyTable"]["Properties"]
    if not properties:
        raise ValueError("PubChem response contains no properties")
    item = properties[0]
    canonical = (
        item.get("ConnectivitySMILES")
        or item.get("CanonicalSMILES")
        or item.get("SMILES")
    )
    isomeric = item.get("SMILES") or item.get("IsomericSMILES") or canonical
    return {
        "pubchem_cid": item.get("CID"),
        "pubchem_title": item.get("Title"),
        "iupac_name": item.get("IUPACName"),
        "molecular_formula": item.get("MolecularFormula"),
        "pubchem_molecular_weight": item.get("MolecularWeight"),
        "canonical_smiles": canonical,
        "isomeric_smiles": isomeric,
    }


class PubChemResolver:
    def __init__(self, cache_path: Path, config: FeatureStageConfig):
        self.cache_path = cache_path
        self.config = config
        if cache_path.exists():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            self.cache: dict[str, Any] = {}

    def _save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def resolve(self, original_name: str, allow_network: bool) -> dict[str, Any]:
        query = normalized_pubchem_query(original_name)
        base: dict[str, Any] = {
            "chemical": original_name,
            "normalized_query": query,
            "mapping_source": "PubChem PUG REST",
        }
        if query is None:
            return {**base, "mapping_status": "not_a_chemical"}
        if original_name in AMBIGUOUS_REAGENT_NAMES:
            return {
                **base,
                "mapping_status": "ambiguous_mixture",
                "mapping_error": AMBIGUOUS_REAGENT_NAMES[original_name],
            }
        cached = self.cache.get(original_name)
        if cached is not None:
            return {**base, **cached, "mapping_status": cached.get("mapping_status", "resolved")}
        if not allow_network:
            return {**base, "mapping_status": "unresolved_network_disabled"}

        encoded = urllib.parse.quote(query, safe="")
        url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/"
            f"property/{PUBCHEM_PROPERTY_NAMES}/JSON"
        )
        try:
            response = requests.get(
                url,
                timeout=self.config.pubchem_timeout_seconds,
                headers={"User-Agent": "GOAI-virtual-yeast-feature-stage/1.0"},
            )
            response.raise_for_status()
            record = _extract_pubchem_record(response.json())
            record.update(
                {
                    "mapping_status": "resolved",
                    "source_url": url,
                    "retrieved_utc": pd.Timestamp.utcnow().isoformat(),
                }
            )
        except Exception as error:  # Every failed entity is retained for manual audit.
            record = {
                "mapping_status": "unresolved",
                "mapping_error": f"{type(error).__name__}: {error}",
                "source_url": url,
            }
        self.cache[original_name] = record
        self._save()
        time.sleep(self.config.pubchem_pause_seconds)
        return {**base, **record}


def resolve_chemical_table(
    chemical_names: list[str],
    output_dir: Path,
    config: FeatureStageConfig,
    allow_network: bool,
) -> pd.DataFrame:
    resolver = PubChemResolver(output_dir / "pubchem_cache.json", config)
    records = [resolver.resolve(name, allow_network) for name in chemical_names]
    columns = [
        "chemical",
        "normalized_query",
        "mapping_status",
        "pubchem_cid",
        "pubchem_title",
        "iupac_name",
        "molecular_formula",
        "pubchem_molecular_weight",
        "canonical_smiles",
        "isomeric_smiles",
        "mapping_source",
        "source_url",
        "retrieved_utc",
        "mapping_error",
    ]
    frame = pd.DataFrame(records)
    for column in columns:
        if column not in frame:
            frame[column] = pd.NA
    return frame[columns]


MOLECULAR_DESCRIPTOR_NAMES = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "HDonors",
    "HAcceptors",
    "RotatableBonds",
    "RingCount",
    "AromaticRings",
    "HeavyAtoms",
    "FractionCSP3",
    "FormalCharge",
    "HeteroAtoms",
)


def build_rdkit_features(
    mapping: pd.DataFrame,
    training_chemicals: set[str],
    config: FeatureStageConfig,
) -> tuple[np.ndarray, list[str], pd.DataFrame, dict[str, Any]]:
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import Descriptors, Lipinski, MACCSkeys, rdFingerprintGenerator
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError as error:
        raise RuntimeError(
            "RDKit is required for chemical features. Install with `python -m pip install rdkit`."
        ) from error

    morgan = rdFingerprintGenerator.GetMorganGenerator(
        radius=config.morgan_radius, fpSize=config.morgan_bits
    )
    raw_fingerprints: list[np.ndarray] = []
    raw_descriptors: list[np.ndarray] = []
    available: list[float] = []
    feature_smiles: list[str | None] = []
    feature_status: list[str] = []
    for _, row in mapping.iterrows():
        smiles = row.get("isomeric_smiles") or row.get("canonical_smiles")
        if pd.isna(smiles) or not str(smiles).strip():
            raw_fingerprints.append(
                np.zeros(config.morgan_bits + 167, dtype=np.float32)
            )
            raw_descriptors.append(np.zeros(len(MOLECULAR_DESCRIPTOR_NAMES), dtype=np.float32))
            available.append(0.0)
            feature_smiles.append(None)
            feature_status.append("missing_smiles")
            continue
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raw_fingerprints.append(
                np.zeros(config.morgan_bits + 167, dtype=np.float32)
            )
            raw_descriptors.append(np.zeros(len(MOLECULAR_DESCRIPTOR_NAMES), dtype=np.float32))
            available.append(0.0)
            feature_smiles.append(None)
            feature_status.append("rdkit_parse_failed")
            continue
        original_mol = mol
        try:
            parent = rdMolStandardize.FragmentParent(mol)
            if parent is not None and parent.GetNumAtoms() > 0:
                Chem.SanitizeMol(parent)
                mol = parent
        except Exception:
            mol = original_mol
        try:
            Chem.SanitizeMol(mol)
            Chem.GetSymmSSSR(mol)
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
            morgan_array = np.zeros(config.morgan_bits, dtype=np.int8)
            DataStructs.ConvertToNumpyArray(morgan.GetFingerprint(mol), morgan_array)
            maccs_array = np.zeros(167, dtype=np.int8)
            DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), maccs_array)
            fingerprint = np.concatenate([morgan_array, maccs_array]).astype(np.float32)
            descriptor = np.asarray(
                [
                    Descriptors.MolWt(mol),
                    Descriptors.MolLogP(mol),
                    Descriptors.TPSA(mol),
                    Lipinski.NumHDonors(mol),
                    Lipinski.NumHAcceptors(mol),
                    Lipinski.NumRotatableBonds(mol),
                    Lipinski.RingCount(mol),
                    Lipinski.NumAromaticRings(mol),
                    Lipinski.HeavyAtomCount(mol),
                    Descriptors.FractionCSP3(mol),
                    float(Chem.GetFormalCharge(mol)),
                    Lipinski.NumHeteroatoms(mol),
                ],
                dtype=np.float32,
            )
            if not np.isfinite(descriptor).all():
                raise ValueError("non-finite RDKit descriptor")
            raw_fingerprints.append(fingerprint)
            raw_descriptors.append(descriptor)
            available.append(1.0)
            feature_smiles.append(canonical)
            feature_status.append("ready")
        except Exception as error:
            raw_fingerprints.append(np.zeros(config.morgan_bits + 167, dtype=np.float32))
            raw_descriptors.append(np.zeros(len(MOLECULAR_DESCRIPTOR_NAMES), dtype=np.float32))
            available.append(0.0)
            feature_smiles.append(None)
            feature_status.append(f"rdkit_feature_failed:{type(error).__name__}")

    fingerprints = np.stack(raw_fingerprints)
    descriptors = np.stack(raw_descriptors)
    training_mask = mapping["chemical"].isin(training_chemicals).to_numpy() & (
        np.asarray(available) > 0
    )
    if not training_mask.any():
        raise ValueError("No resolved training chemicals are available to fit descriptor scaling")
    descriptor_mean = descriptors[training_mask].mean(axis=0)
    descriptor_std = descriptors[training_mask].std(axis=0)
    descriptor_std = np.where(descriptor_std > 1e-8, descriptor_std, 1.0)
    normalized = (descriptors - descriptor_mean) / descriptor_std
    normalized[np.asarray(available) == 0] = 0.0
    matrix = np.concatenate(
        [fingerprints, normalized.astype(np.float32), np.asarray(available, dtype=np.float32)[:, None]],
        axis=1,
    )
    feature_names = (
        [f"morgan_r{config.morgan_radius}_{i}" for i in range(config.morgan_bits)]
        + [f"maccs_{i}" for i in range(167)]
        + [f"descriptor_z_{name}" for name in MOLECULAR_DESCRIPTOR_NAMES]
        + ["molecule_feature_available"]
    )
    updated = mapping.copy()
    updated["feature_smiles"] = feature_smiles
    updated["feature_status"] = feature_status
    scaler = {
        "descriptor_names": list(MOLECULAR_DESCRIPTOR_NAMES),
        "mean": descriptor_mean.tolist(),
        "std": descriptor_std.tolist(),
        "fit_chemicals": sorted(training_chemicals),
        "fit_resolved_count": int(training_mask.sum()),
        "fingerprint_standardization": "RDKit FragmentParent before fingerprinting; original PubChem match retained",
    }
    return matrix.astype(np.float32), feature_names, updated, scaler


@dataclass
class ContextFeatureEncoder:
    biological_vocabularies: dict[str, list[str]]
    observation_vocabularies: dict[str, list[str]]
    temperature_mean: float
    temperature_std: float
    max_train_time: float
    train_time_centers: list[float]

    BIO_CATEGORICAL = ("Medium", "pert_time_unit")
    OBSERVATION_CATEGORICAL = (
        "data_source",
        "instrument",
        "Yeast_cell_plate",
        "protein_well",
    )

    @classmethod
    def fit(cls, train_metadata: pd.DataFrame) -> "ContextFeatureEncoder":
        bio = {
            field: sorted(train_metadata[field].astype(str).unique().tolist())
            for field in cls.BIO_CATEGORICAL
        }
        observation = {
            field: sorted(train_metadata[field].astype(str).unique().tolist())
            for field in cls.OBSERVATION_CATEGORICAL
        }
        temperature = train_metadata["Temperature"].astype(float).to_numpy()
        times = train_metadata["pert_time"].astype(float).to_numpy()
        return cls(
            biological_vocabularies=bio,
            observation_vocabularies=observation,
            temperature_mean=float(temperature.mean()),
            temperature_std=float(max(temperature.std(), 1e-8)),
            max_train_time=float(times.max()),
            train_time_centers=sorted(np.unique(times).astype(float).tolist()),
        )

    @staticmethod
    def _one_hot(
        metadata: pd.DataFrame, vocabularies: dict[str, list[str]]
    ) -> tuple[list[np.ndarray], list[str]]:
        blocks: list[np.ndarray] = []
        names: list[str] = []
        for field, vocabulary in vocabularies.items():
            lookup = {value: i for i, value in enumerate(vocabulary)}
            values = metadata[field].astype(str).to_numpy()
            block = np.zeros((len(metadata), len(vocabulary)), dtype=np.float32)
            indices = np.fromiter((lookup.get(value, -1) for value in values), dtype=int)
            known = indices >= 0
            block[np.arange(len(metadata))[known], indices[known]] = 1.0
            blocks.append(block)
            names.extend([f"{field}={value}" for value in vocabulary])
        return blocks, names

    def transform_biological(self, metadata: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        blocks, names = self._one_hot(metadata, self.biological_vocabularies)
        temperature = metadata["Temperature"].astype(float).to_numpy(dtype=np.float32)
        times = metadata["pert_time"].astype(float).to_numpy(dtype=np.float32)
        continuous = [
            ((temperature - self.temperature_mean) / self.temperature_std)[:, None],
            (times / self.max_train_time)[:, None],
            (np.log1p(times) / np.log1p(self.max_train_time))[:, None],
            (times > self.max_train_time).astype(np.float32)[:, None],
        ]
        names.extend(["temperature_z", "time_scaled", "time_log1p_scaled", "time_exceeds_train_max"])
        centers = np.asarray(self.train_time_centers, dtype=np.float32)
        if len(centers) > 1:
            gaps = np.diff(centers)
            width = float(max(np.median(gaps), 1.0))
        else:
            width = 1.0
        rbf = np.exp(-0.5 * ((times[:, None] - centers[None, :]) / width) ** 2).astype(np.float32)
        continuous.append(rbf)
        names.extend([f"time_rbf_{center:g}" for center in centers])
        return np.concatenate(blocks + continuous, axis=1).astype(np.float32), names

    def transform_observation(self, metadata: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        blocks, names = self._one_hot(metadata, self.observation_vocabularies)
        return np.concatenate(blocks, axis=1).astype(np.float32), names

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_fasta_sequence(path: Path) -> str:
    sequence: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if not line.startswith(">"):
                sequence.append(line.strip().upper())
    return "".join(sequence)


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def hashed_kmer_features(sequence: str, k: int, buckets: int) -> np.ndarray:
    if k <= 0 or buckets <= 0:
        raise ValueError("k and buckets must be positive")
    counts = np.zeros(buckets, dtype=np.float64)
    base = {"A": 0, "C": 1, "G": 2, "T": 3}
    forward = 0
    reverse = 0
    valid_length = 0
    bit_mask = (1 << (2 * k)) - 1
    reverse_high_shift = 2 * (k - 1)
    for character in sequence:
        value = base.get(character)
        if value is None:
            forward = reverse = valid_length = 0
            continue
        forward = ((forward << 2) | value) & bit_mask
        reverse = (reverse >> 2) | ((3 - value) << reverse_high_shift)
        valid_length += 1
        if valid_length >= k:
            counts[min(forward, reverse) % buckets] += 1
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts.astype(np.float32)


def prepare_strain_mapping(strain_names: list[str], path: Path) -> pd.DataFrame:
    required = [
        "strain",
        "genome_path",
        "source_name",
        "source_url",
        "database_version",
        "download_date",
        "is_proxy",
        "notes",
    ]
    if path.exists():
        frame = pd.read_csv(path, keep_default_na=False)
        for column in required:
            if column not in frame:
                frame[column] = ""
        existing = set(frame["strain"].astype(str))
    else:
        frame = pd.DataFrame(columns=required)
        existing = set()
    additions = []
    for strain in strain_names:
        if strain in existing:
            continue
        additions.append(
            {
                "strain": strain,
                "genome_path": "",
                "source_name": "SGD S288C reference (suggested proxy)" if strain == "DHY210" else "1011 Yeast Genomes Project (mapping required)",
                "source_url": "",
                "database_version": "",
                "download_date": "",
                "is_proxy": strain == "DHY210",
                "notes": "Do not use until the exact genome mapping is verified.",
            }
        )
    if additions:
        frame = pd.concat([frame, pd.DataFrame(additions)], ignore_index=True)
    frame = frame[frame["strain"].astype(str).isin(strain_names)].copy()
    frame["strain"] = frame["strain"].astype(str)
    frame = frame.sort_values("strain").reset_index(drop=True)
    return frame[required]


def build_strain_features(
    mapping: pd.DataFrame, config: FeatureStageConfig
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    vectors: list[np.ndarray] = []
    statuses: list[str] = []
    hashes: list[str | None] = []
    for _, row in mapping.iterrows():
        raw_path = str(row.get("genome_path", "")).strip()
        path = Path(raw_path) if raw_path else None
        if path is None or not path.is_file():
            vectors.append(np.zeros(config.kmer_buckets, dtype=np.float32))
            statuses.append("genome_missing")
            hashes.append(None)
            continue
        sequence = _read_fasta_sequence(path)
        vectors.append(hashed_kmer_features(sequence, config.kmer_size, config.kmer_buckets))
        statuses.append("ready")
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    available = np.asarray([status == "ready" for status in statuses], dtype=np.float32)[:, None]
    matrix = np.concatenate([np.stack(vectors), available], axis=1)
    names = [f"canonical_kmer{k}" for k in range(config.kmer_buckets)] + ["genome_feature_available"]
    updated = mapping.copy()
    updated["feature_status"] = statuses
    updated["genome_sha256"] = hashes
    return matrix.astype(np.float32), names, updated


def build_population_distance_features(
    strain_names: list[str],
    snp_distance_path: Path,
    orf_distance_path: Path,
    components_per_source: int = 32,
) -> tuple[np.ndarray, list[str], list[str], dict[str, Any]]:
    """Embed official 1,011-isolate distance profiles without assay labels."""
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        raise RuntimeError("scikit-learn is required for population distance PCA") from error

    blocks: list[np.ndarray] = []
    feature_names: list[str] = []
    source_reports: dict[str, Any] = {}
    available_by_source: list[np.ndarray] = []
    for source_name, path in (("snp", snp_distance_path), ("orf", orf_distance_path)):
        frame = pd.read_csv(path, sep="\t", index_col=0, compression="gzip")
        if frame.shape[0] != frame.shape[1] or frame.index.astype(str).tolist() != frame.columns.astype(str).tolist():
            raise ValueError(f"{source_name} distance matrix is not square/aligned")
        values = frame.to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"{source_name} distance matrix contains non-finite values")
        scaled = StandardScaler().fit_transform(values)
        n_components = min(components_per_source, min(scaled.shape) - 1)
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=20260805)
        embedding = pca.fit_transform(scaled)
        # Whiten component scales so a few leading population axes do not
        # dominate k-mer, metadata, or molecular feature branches downstream.
        component_scale = np.sqrt(np.maximum(pca.explained_variance_, 1e-8))
        embedding = (embedding / component_scale.reshape(1, -1)).astype(np.float32)
        lookup = {str(name): i for i, name in enumerate(frame.index.astype(str))}
        selected = np.zeros((len(strain_names), n_components), dtype=np.float32)
        available = np.zeros(len(strain_names), dtype=np.float32)
        for row, strain in enumerate(strain_names):
            position = lookup.get(strain)
            if position is not None:
                selected[row] = embedding[position]
                available[row] = 1.0
        blocks.extend([selected, available[:, None]])
        feature_names.extend(
            [f"population_{source_name}_pca_{i}" for i in range(n_components)]
            + [f"population_{source_name}_available"]
        )
        available_by_source.append(available)
        source_reports[source_name] = {
            "path": str(path.resolve()),
            "source_url": f"http://1002genomes.u-strasbg.fr/files/{path.name}",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "matrix_shape": list(frame.shape),
            "pca_components": n_components,
            "explained_variance_fraction": float(pca.explained_variance_ratio_.sum()),
            "component_scaling": "PCA scores divided by sqrt(explained_variance) using external population only",
            "matched_competition_strains": [
                strain for strain in strain_names if strain in lookup
            ],
            "unmatched_competition_strains": [
                strain for strain in strain_names if strain not in lookup
            ],
            "fit_scope": "all 1,011 external isolates; no competition proteome labels",
        }
    status = []
    combined_available = np.maximum.reduce(available_by_source)
    for flag in combined_available:
        status.append("ready" if flag > 0 else "not_in_1011_project")
    return np.concatenate(blocks, axis=1), feature_names, status, source_reports


def build_population_metadata_features(
    strain_names: list[str], supplementary_xls: Path
) -> tuple[np.ndarray, list[str], pd.DataFrame, dict[str, Any]]:
    """Encode stable isolate metadata from Peter et al. 2018 Supplementary Table S1."""
    frame = pd.read_excel(supplementary_xls, sheet_name="Table S1", header=3)
    frame = frame.rename(columns=lambda value: str(value).strip())
    key = "Standardized name"
    if key not in frame:
        raise ValueError("Supplementary Table S1 lacks the Standardized name column")
    frame = frame[frame[key].notna()].copy()
    frame[key] = frame[key].astype(str)
    frame = frame.drop_duplicates(key, keep="first").set_index(key)

    numeric_columns = [
        "Ploidy",
        "Plasmid CN",
        "Total number of SNPs",
        "Proportion of clean heterozygous SNPs (whole dataset)",
        "Number of singletons",
        "Mean coverage",
    ]
    categorical_columns = [
        "Ecological origins",
        "HO deletion",
        "Plasmid class",
        "Zygosity",
        "Clades",
    ]
    blocks: list[np.ndarray] = []
    names: list[str] = []
    selected_audit = pd.DataFrame({"strain": strain_names})
    available = np.asarray([strain in frame.index for strain in strain_names], dtype=np.float32)

    for column in numeric_columns:
        population = pd.to_numeric(frame[column], errors="coerce")
        mean = float(population.mean())
        std = float(population.std())
        if not np.isfinite(std) or std <= 1e-8:
            std = 1.0
        values = []
        audit_values = []
        for strain in strain_names:
            value = population.get(strain, np.nan)
            audit_values.append(value)
            values.append((float(value) - mean) / std if np.isfinite(value) else 0.0)
        blocks.append(np.asarray(values, dtype=np.float32)[:, None])
        names.append(f"peter2018_numeric_z_{column}")
        selected_audit[column] = audit_values

    for column in categorical_columns:
        population = frame[column].fillna("__MISSING__").astype(str)
        vocabulary = sorted(population.unique().tolist())
        lookup = {value: i for i, value in enumerate(vocabulary)}
        block = np.zeros((len(strain_names), len(vocabulary)), dtype=np.float32)
        audit_values = []
        for row, strain in enumerate(strain_names):
            if strain in population.index:
                value = population.loc[strain]
                block[row, lookup[value]] = 1.0
                audit_values.append(value)
            else:
                audit_values.append(None)
        blocks.append(block)
        names.extend([f"peter2018_{column}={value}" for value in vocabulary])
        selected_audit[column] = audit_values

    blocks.append(available[:, None])
    names.append("peter2018_metadata_available")
    selected_audit["metadata_available"] = available.astype(int)
    report = {
        "path": str(supplementary_xls.resolve()),
        "source_url": (
            "https://media.springernature.com/original/springer-static/esm/"
            "art%3A10.1038%2Fs41586-018-0030-5/MediaObjects/41586_2018_30_MOESM3_ESM.xls"
        ),
        "sha256": hashlib.sha256(supplementary_xls.read_bytes()).hexdigest(),
        "source_rows": int(len(frame)),
        "matched_competition_strains": [strain for strain in strain_names if strain in frame.index],
        "unmatched_competition_strains": [strain for strain in strain_names if strain not in frame.index],
        "fit_scope": "Peter et al. 2018 external isolate metadata only; no competition proteome labels",
    }
    return np.concatenate(blocks, axis=1), names, selected_audit, report


def build_statistical_priors(
    metadata: pd.DataFrame,
    y_log2: np.ndarray,
    observation_mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    train_rows = metadata["split_final"].eq("train").to_numpy()
    train_controls = train_rows & metadata["perturbation_no_concentration"].isin(CONTROL_NAMES).to_numpy()
    train_treatments = train_rows & ~metadata["perturbation_no_concentration"].isin(EXCLUDED_TREATMENTS).to_numpy()
    mean = protein_mean(y_log2, train_rows)
    values = np.asarray(y_log2)
    control_values = values[train_controls]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        global_control_mean = np.nanmean(control_values, axis=0)
        global_control_std = np.nanstd(control_values, axis=0)
    global_control_mean = np.where(np.isfinite(global_control_mean), global_control_mean, mean).astype(np.float32)
    global_control_std = np.where(np.isfinite(global_control_std), global_control_std, 0.0).astype(np.float32)

    pair_rows, pair_controls, pair_counts = strict_feature_matched_controls(
        metadata, y_log2, train_treatments, train_rows
    )
    treated = values[pair_rows]
    valid_delta = np.isfinite(treated) & np.isfinite(pair_controls)
    delta = np.where(valid_delta, treated - pair_controls, np.nan).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        global_delta = np.nanmean(delta, axis=0)
    global_delta = np.where(np.isfinite(global_delta), global_delta, 0.0).astype(np.float32)

    chemical_names = sorted(metadata.iloc[pair_rows]["perturbation_no_concentration"].astype(str).unique())
    chemical_mean: list[np.ndarray] = []
    chemical_observations: list[np.ndarray] = []
    chemical_samples: list[int] = []
    pair_chemicals = metadata.iloc[pair_rows]["perturbation_no_concentration"].astype(str).to_numpy()
    for name in chemical_names:
        selected = pair_chemicals == name
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            group_mean = np.nanmean(delta[selected], axis=0)
        group_mean = np.where(np.isfinite(group_mean), group_mean, global_delta)
        chemical_mean.append(group_mean.astype(np.float32))
        chemical_observations.append(np.isfinite(delta[selected]).sum(axis=0).astype(np.int32))
        chemical_samples.append(int(selected.sum()))

    strain_names = sorted(metadata.loc[train_controls, "Strains"].astype(str).unique())
    strain_mean: list[np.ndarray] = []
    strain_std: list[np.ndarray] = []
    strain_observations: list[np.ndarray] = []
    strain_samples: list[int] = []
    control_strains = metadata.loc[train_controls, "Strains"].astype(str).to_numpy()
    for name in strain_names:
        selected = control_strains == name
        group = control_values[selected]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            group_mean = np.nanmean(group, axis=0)
            group_std = np.nanstd(group, axis=0)
        strain_mean.append(np.where(np.isfinite(group_mean), group_mean, global_control_mean).astype(np.float32))
        strain_std.append(np.where(np.isfinite(group_std), group_std, global_control_std).astype(np.float32))
        strain_observations.append(np.isfinite(group).sum(axis=0).astype(np.int32))
        strain_samples.append(int(selected.sum()))

    arrays = {
        "global_protein_mean": mean,
        "global_control_mean": global_control_mean,
        "global_control_std": global_control_std,
        "global_delta_mean": global_delta,
        "chemical_names": np.asarray(chemical_names, dtype=str),
        "chemical_delta_mean": np.stack(chemical_mean),
        "chemical_observed_counts": np.stack(chemical_observations),
        "chemical_sample_counts": np.asarray(chemical_samples, dtype=np.int32),
        "strain_names": np.asarray(strain_names, dtype=str),
        "strain_control_mean": np.stack(strain_mean),
        "strain_control_std": np.stack(strain_std),
        "strain_observed_counts": np.stack(strain_observations),
        "strain_sample_counts": np.asarray(strain_samples, dtype=np.int32),
    }
    info = {
        "fit_scope": "split_final=train only",
        "n_train_controls": int(train_controls.sum()),
        "n_train_treatments": int(train_treatments.sum()),
        "n_strict_pairs": int(len(pair_rows)),
        "strict_pair_coverage": float(len(pair_rows) / max(int(train_treatments.sum()), 1)),
        "mean_controls_per_pair": float(pair_counts.mean()),
        "training_usage_warning": (
            "Entity target priors must be cross-fitted/leave-fold-out for training samples. "
            "The full-train tables are safe for validation/test lookup only."
        ),
    }
    return arrays, info


def save_npz_with_names(
    path: Path,
    sample_ids: np.ndarray,
    biological: np.ndarray,
    biological_names: list[str],
    observation: np.ndarray,
    observation_names: list[str],
    chemical_indices: np.ndarray,
    strain_indices: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        sample_ids=np.asarray(sample_ids, dtype=str),
        biological_context=biological.astype(np.float32),
        biological_feature_names=np.asarray(biological_names, dtype=str),
        observation_context=observation.astype(np.float32),
        observation_feature_names=np.asarray(observation_names, dtype=str),
        chemical_indices=chemical_indices.astype(np.int32),
        strain_indices=strain_indices.astype(np.int32),
    )


def entity_indices(values: pd.Series, entities: list[str]) -> np.ndarray:
    lookup = {value: i for i, value in enumerate(entities)}
    result = np.fromiter((lookup.get(str(value), -1) for value in values), dtype=np.int32)
    if (result < 0).any():
        missing = sorted(set(values.astype(str)[result < 0]))
        raise ValueError(f"Entity contract missing values: {missing}")
    return result
