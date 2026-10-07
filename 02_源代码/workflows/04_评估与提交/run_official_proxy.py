from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vc_stage01.constants import VALIDATION_SPLITS  # noqa: E402
from vc_stage01.official_proxy import fit_proxy_reference, score_official_proxy  # noqa: E402


DEFAULT_MODELS = {
    "pooled_mse": PROJECT_ROOT / "outputs/loss_ablation/pooled_mse/validation_predictions.npy",
    "huber": PROJECT_ROOT / "outputs/loss_ablation/masked_huber/validation_predictions.npy",
    "huber_fc001": PROJECT_ROOT / "outputs/loss_ablation_huber_fc001/masked_huber/validation_predictions.npy",
    "huber_fc005": PROJECT_ROOT / "outputs/loss_ablation_huber_fc005_fixed/masked_huber/validation_predictions.npy",
    "strain_content_huber": PROJECT_ROOT / "outputs/content_ablation_huber/strain_content/validation_predictions.npy",
    "multihead_response": PROJECT_ROOT / "outputs/multihead_ablation_deep_pooled/response/validation_predictions.npy",
}

DEFAULT_ROUTES = {
    "route_huber_chem_fc005_else": {
        "base": "huber_fc005", "overrides": {"val_chem_only": "huber"}},
    "route_fc001_chem_fc005_else": {
        "base": "huber_fc005", "overrides": {"val_chem_only": "huber_fc001"}},
    "route_fc001_fc005_time": {
        "base": "huber_fc001", "overrides": {"val_time": "huber_fc005"}},
    "route_huber_strain_content_time": {
        "base": "huber", "overrides": {"val_time": "strain_content_huber"}},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Local proxy for the six official scoring modules.")
    parser.add_argument("--stage0-dir", type=Path, default=PROJECT_ROOT / "outputs/stage0")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/official_proxy")
    parser.add_argument("--model", action="append", default=[], metavar="NAME=PATH",
                        help="Prediction array for validation rows; repeat for multiple models.")
    parser.add_argument("--no-default-routes", action="store_true")
    args = parser.parse_args()

    models = dict(DEFAULT_MODELS)
    if args.model:
        models = {}
        for item in args.model:
            if "=" not in item:
                raise ValueError("--model must use NAME=PATH")
            name, path = item.split("=", 1)
            models[name] = Path(path)
    missing = [f"{name}={path}" for name, path in models.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing prediction files: {missing}")

    metadata = pd.read_csv(args.stage0_dir / "metadata_aligned.csv").set_index("sample_ID")
    y = np.load(args.stage0_dir / "y_log2.npy", mmap_mode="r")
    train = metadata["split_final"].eq("train").to_numpy()
    validation = metadata["split_final"].isin(VALIDATION_SPLITS).to_numpy()
    validation_positions = np.flatnonzero(validation)
    reference = fit_proxy_reference(metadata, y, train)
    print({"reference": reference.summary(), "validation_rows": len(validation_positions)}, flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded_predictions = {}
    for name, path in models.items():
        prediction = np.load(path)
        if prediction.shape != (len(validation_positions), y.shape[1]):
            raise ValueError(f"{name} prediction shape {prediction.shape} does not match "
                             f"{(len(validation_positions), y.shape[1])}")
        loaded_predictions[name] = np.asarray(prediction, dtype=np.float32)
    if not args.model and not args.no_default_routes:
        validation_metadata = metadata.iloc[validation_positions]
        for route_name, route in DEFAULT_ROUTES.items():
            routed = loaded_predictions[route["base"]].copy()
            for split, source in route["overrides"].items():
                selected = validation_metadata["split_final"].eq(split).to_numpy()
                routed[selected] = loaded_predictions[source][selected]
            loaded_predictions[route_name] = routed

    split_frames = []
    summaries = []
    for name, prediction in loaded_predictions.items():
        print(f"Scoring {name}", flush=True)
        full = np.full(y.shape, np.nan, dtype=np.float32)
        full[validation_positions] = prediction
        split_records, summary = score_official_proxy(metadata, y, full, train, reference)
        split_frame = pd.DataFrame(split_records)
        split_frame.insert(0, "model", name)
        split_frames.append(split_frame)
        summaries.append({"model": name, **summary})
        del full

    split_metrics = pd.concat(split_frames, ignore_index=True)
    summary_frame = pd.DataFrame(summaries).sort_values("proxy_score", ascending=False)
    split_metrics.to_csv(args.output_dir / "split_module_metrics.csv", index=False)
    summary_frame.to_csv(args.output_dir / "model_ranking.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "reference": reference.summary(),
        "models": summaries,
        "limitations": [
            "Evaluation uses held-out true controls as a proxy for controls held by the official server.",
            "Context/drug residual references are fitted only from split_final=train treatment-control pairs.",
            "Official server control pairing and module aggregation details are not public.",
            "PCC values are mapped to [0,1] and R2 values are clipped to [0,1].",
            "DEP is the mean of direction accuracy, mapped high-effect PCC, Recall@K, F1 and AUPRC.",
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_frame[["model", "proxy_score", "module1_fc", "module2_fidelity",
                         "module3_context_residual", "module4_drug_residual",
                         "module5_both_time", "module6_dep"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
