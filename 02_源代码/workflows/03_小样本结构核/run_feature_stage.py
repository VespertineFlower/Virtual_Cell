from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vc_stage01.advanced_features import (  # noqa: E402
    ContextFeatureEncoder,
    FeatureStageConfig,
    build_rdkit_features,
    build_population_distance_features,
    build_population_metadata_features,
    build_statistical_priors,
    build_strain_features,
    download_open_knowledge_resources,
    entity_indices,
    prepare_strain_mapping,
    resolve_chemical_table,
    save_npz_with_names,
)


def discover_test_metadata(stage0_dir: Path, explicit_data_dir: Path | None) -> Path:
    if explicit_data_dir is not None:
        return explicit_data_dir / "WAYB_WAYC_metadata_test(1).csv"
    audit = json.loads((stage0_dir / "stage0_audit.json").read_text(encoding="utf-8"))
    return Path(audit["sources"]["metadata_test"]["path"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build auditable transferable entity, context, and statistical features."
    )
    parser.add_argument("--stage0-dir", type=Path, default=PROJECT_ROOT / "outputs" / "stage0")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "features")
    parser.add_argument(
        "--resolve-pubchem",
        action="store_true",
        help="Resolve uncached chemical names through the official PubChem PUG REST API.",
    )
    parser.add_argument(
        "--strain-mapping",
        type=Path,
        default=None,
        help="Optional editable CSV containing verified genome FASTA paths and provenance.",
    )
    parser.add_argument("--kmer-size", type=int, default=6)
    parser.add_argument("--kmer-buckets", type=int, default=4096)
    parser.add_argument(
        "--download-open-knowledge",
        action="store_true",
        help="Download cached official 1011-genomes, Nature supplement, and NCBI S288C resources.",
    )
    parser.add_argument(
        "--snp-distance-matrix",
        type=Path,
        default=None,
        help="Official 1011DistanceMatrixBasedOnSNPs.tab.gz; auto-detected under output/external.",
    )
    parser.add_argument(
        "--orf-distance-matrix",
        type=Path,
        default=None,
        help="Official 1011DistanceMatrixBasedOnORFs.tab.gz; auto-detected under output/external.",
    )
    parser.add_argument(
        "--peter-supplementary-xls",
        type=Path,
        default=None,
        help="Peter et al. 2018 Nature supplementary tables; auto-detected under output/external.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = None
    if args.download_open_knowledge:
        downloaded = download_open_knowledge_resources(args.output_dir / "external")

    metadata_train_val = pd.read_csv(args.stage0_dir / "metadata_aligned.csv")
    metadata_test = pd.read_csv(discover_test_metadata(args.stage0_dir, args.data_dir))
    if metadata_train_val["sample_ID"].duplicated().any() or metadata_test["sample_ID"].duplicated().any():
        raise ValueError("sample_ID must be unique within each metadata table")
    if set(metadata_train_val["sample_ID"]) & set(metadata_test["sample_ID"]):
        raise ValueError("train/validation and test sample_ID sets overlap")

    config = FeatureStageConfig(kmer_size=args.kmer_size, kmer_buckets=args.kmer_buckets)
    combined = pd.concat(
        [metadata_train_val.assign(feature_partition="train_val"), metadata_test.assign(feature_partition="test")],
        ignore_index=True,
    )
    train_rows = metadata_train_val["split_final"].eq("train")
    training_chemicals = set(
        metadata_train_val.loc[train_rows, "perturbation_no_concentration"].astype(str)
    )

    chemical_names = sorted(combined["perturbation_no_concentration"].astype(str).unique())
    chemical_mapping = resolve_chemical_table(
        chemical_names, args.output_dir, config, args.resolve_pubchem
    )
    chemical_matrix, chemical_feature_names, chemical_mapping, chemical_scaler = build_rdkit_features(
        chemical_mapping, training_chemicals, config
    )
    chemical_mapping.to_csv(args.output_dir / "chemical_mapping.csv", index=False)
    np.savez_compressed(
        args.output_dir / "chemical_features.npz",
        entity_names=np.asarray(chemical_mapping["chemical"].astype(str), dtype=str),
        features=chemical_matrix,
        feature_names=np.asarray(chemical_feature_names, dtype=str),
    )
    (args.output_dir / "chemical_descriptor_scaler.json").write_text(
        json.dumps(chemical_scaler, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    strain_names = sorted(combined["Strains"].astype(str).unique())
    mapping_path = args.strain_mapping or (args.output_dir / "strain_mapping.csv")
    strain_mapping = prepare_strain_mapping(strain_names, mapping_path)
    if downloaded is not None:
        dhy = strain_mapping["strain"].eq("DHY210")
        strain_mapping.loc[dhy, "genome_path"] = str(downloaded["s288c_fasta"])
        strain_mapping.loc[dhy, "source_name"] = "NCBI RefSeq S288C reference (DHY210 proxy)"
        strain_mapping.loc[dhy, "source_url"] = (
            "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/"
            "GCF_000146045.2/download?include_annotation_type=GENOME_FASTA"
        )
        strain_mapping.loc[dhy, "database_version"] = "GCF_000146045.2"
        strain_mapping.loc[dhy, "download_date"] = str(pd.Timestamp.utcnow().date())
        strain_mapping.loc[dhy, "is_proxy"] = True
        strain_mapping.loc[dhy, "notes"] = (
            "S288C is used as an explicit proxy for DHY210; not assumed identical."
        )
    strain_matrix, strain_feature_names, strain_mapping = build_strain_features(
        strain_mapping, config
    )
    snp_path = args.snp_distance_matrix or (
        args.output_dir / "external" / "1011DistanceMatrixBasedOnSNPs.tab.gz"
    )
    orf_path = args.orf_distance_matrix or (
        args.output_dir / "external" / "1011DistanceMatrixBasedOnORFs.tab.gz"
    )
    population_report = None
    if snp_path.exists() and orf_path.exists():
        population_matrix, population_names, population_status, population_report = (
            build_population_distance_features(strain_names, snp_path, orf_path)
        )
        strain_matrix = np.concatenate([strain_matrix, population_matrix], axis=1)
        strain_feature_names.extend(population_names)
        strain_mapping["population_feature_status"] = population_status
        genome_ready = strain_mapping["feature_status"].eq("ready")
        population_ready = strain_mapping["population_feature_status"].eq("ready")
        strain_mapping["feature_status"] = np.where(
            genome_ready | population_ready,
            "ready",
            "missing_all_content_features",
        )
    else:
        strain_mapping["population_feature_status"] = "distance_matrices_missing"
    peter_path = args.peter_supplementary_xls or (
        args.output_dir / "external" / "Peter2018_supplementary_tables.xls"
    )
    peter_report = None
    if peter_path.exists():
        peter_matrix, peter_names, peter_audit, peter_report = build_population_metadata_features(
            strain_names, peter_path
        )
        strain_matrix = np.concatenate([strain_matrix, peter_matrix], axis=1)
        strain_feature_names.extend(peter_names)
        peter_audit.to_csv(args.output_dir / "strain_external_metadata.csv", index=False)
    strain_mapping.to_csv(args.output_dir / "strain_mapping.csv", index=False)
    np.savez_compressed(
        args.output_dir / "strain_features.npz",
        entity_names=np.asarray(strain_mapping["strain"].astype(str), dtype=str),
        features=strain_matrix,
        feature_names=np.asarray(strain_feature_names, dtype=str),
    )

    context_encoder = ContextFeatureEncoder.fit(metadata_train_val.loc[train_rows])
    train_bio, bio_names = context_encoder.transform_biological(metadata_train_val)
    train_observation, observation_names = context_encoder.transform_observation(metadata_train_val)
    test_bio, test_bio_names = context_encoder.transform_biological(metadata_test)
    test_observation, test_observation_names = context_encoder.transform_observation(metadata_test)
    if bio_names != test_bio_names or observation_names != test_observation_names:
        raise ValueError("train/test context feature contracts differ")
    chemical_entities = chemical_mapping["chemical"].astype(str).tolist()
    strain_entities = strain_mapping["strain"].astype(str).tolist()
    save_npz_with_names(
        args.output_dir / "train_val_context_features.npz",
        metadata_train_val["sample_ID"].astype(str).to_numpy(),
        train_bio,
        bio_names,
        train_observation,
        observation_names,
        entity_indices(metadata_train_val["perturbation_no_concentration"], chemical_entities),
        entity_indices(metadata_train_val["Strains"], strain_entities),
    )
    save_npz_with_names(
        args.output_dir / "test_context_features.npz",
        metadata_test["sample_ID"].astype(str).to_numpy(),
        test_bio,
        bio_names,
        test_observation,
        observation_names,
        entity_indices(metadata_test["perturbation_no_concentration"], chemical_entities),
        entity_indices(metadata_test["Strains"], strain_entities),
    )
    (args.output_dir / "context_encoder.json").write_text(
        json.dumps(context_encoder.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    y_log2 = np.load(args.stage0_dir / "y_log2.npy", mmap_mode="r")
    mask = np.load(args.stage0_dir / "mask.npy", mmap_mode="r")
    aligned_metadata = metadata_train_val.set_index("sample_ID")
    priors, prior_info = build_statistical_priors(aligned_metadata, y_log2, mask)
    np.savez_compressed(args.output_dir / "statistical_priors.npz", **priors)

    chemical_status = chemical_mapping["feature_status"].value_counts(dropna=False).to_dict()
    strain_status = strain_mapping["feature_status"].value_counts(dropna=False).to_dict()
    manifest = {
        "stage": "feature_engineering",
        "purpose": "standalone auditable features between baseline modeling and ES-IFM",
        "fit_scope": "all encoders/scalers/statistical priors fit on split_final=train only",
        "n_train_val_samples": len(metadata_train_val),
        "n_test_samples": len(metadata_test),
        "n_chemical_entities": len(chemical_mapping),
        "n_strain_entities": len(strain_mapping),
        "chemical_feature_dim": int(chemical_matrix.shape[1]),
        "strain_feature_dim": int(strain_matrix.shape[1]),
        "biological_context_dim": int(train_bio.shape[1]),
        "observation_context_dim": int(train_observation.shape[1]),
        "chemical_feature_status": {str(key): int(value) for key, value in chemical_status.items()},
        "strain_feature_status": {str(key): int(value) for key, value in strain_status.items()},
        "pubchem_network_enabled": bool(args.resolve_pubchem),
        "statistical_priors": prior_info,
        "population_distance_features": population_report,
        "population_metadata_features": peter_report,
        "contracts": {
            "chemical_entity_order": "chemical_features.npz/entity_names",
            "strain_entity_order": "strain_features.npz/entity_names",
            "sample_to_entity_indices": "*_context_features.npz",
            "missing_external_entity_policy": "zero content vector plus explicit availability flag",
            "technical_feature_policy": "observation_context is stored separately from biological_context",
        },
    }
    (args.output_dir / "feature_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
