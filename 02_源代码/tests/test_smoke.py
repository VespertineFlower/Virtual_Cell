from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vc_stage01.baselines import matched_control_predictions, protein_mean  # noqa: E402
from vc_stage01.features import ConditionFeatureEncoder  # noqa: E402
from vc_stage01.metrics import masked_regression_metrics  # noqa: E402
from vc_stage01.stage2 import feature_slices, strict_matched_controls  # noqa: E402
from vc_stage01.content_mlp import selected_base_columns, selected_strain_columns  # noqa: E402
from vc_stage01.control_delta import context_fold_ids, fit_pca_embedding  # noqa: E402
from vc_stage01.control_delta import MaskedProteomeAutoencoder  # noqa: E402
from vc_stage01.anchor_bridge import GatedResidualBridge, predict_bridge  # noqa: E402
from vc_stage01.anchor_splits import make_anchor_folds  # noqa: E402
from vc_stage01.losses import (MaskedLossConfig, fit_masked_loss_state, masked_loss,
                               masked_objective_numpy, masked_rowwise_pearson_loss)  # noqa: E402
from vc_stage01.multihead_mlp import (MultiHeadConditionMLP, ProteinGroupingConfig,
                                      make_protein_groups)  # noqa: E402
from vc_stage01.official_proxy import (_average_precision, aggregate_proxy_score,
                                       _dep_metrics)  # noqa: E402
from vc_stage01.batch_calibration import (apply_batch_calibration,
                                          fit_batch_calibrator)  # noqa: E402
from vc_stage01.physics_residual import (PhysicsResidualCorrector,
                                         make_time_triples)  # noqa: E402
from vc_stage01.latent_diffusion_bridge import (ConditionalLatentBridge,
                                                sample_latent_bridge)  # noqa: E402
from vc_stage01.advanced_features import (  # noqa: E402
    ContextFeatureEncoder,
    _extract_pubchem_record,
    hashed_kmer_features,
)


class PipelineSmokeTests(unittest.TestCase):
    def test_zero_initialized_latent_bridge_is_exact_zero_residual(self) -> None:
        model = ConditionalLatentBridge(3, 4, 8, 0.0)
        anchor = torch.randn(2, 4)
        features = torch.randn(2, 3)
        residual = sample_latent_bridge(model, anchor, features, steps=4)
        np.testing.assert_allclose(residual.detach().numpy(), 0.0, atol=1e-7)

    def test_physics_corrector_starts_at_zero_and_time_triples_are_ordered(self) -> None:
        model = PhysicsResidualCorrector(3, 5, 8, 0.0)
        np.testing.assert_allclose(model(torch.randn(2, 3)).detach().numpy(), 0.0)
        metadata = pd.DataFrame({
            "data_source": ["D"] * 4, "instrument": ["I"] * 4,
            "Strains": ["S"] * 4, "perturbation_no_concentration": ["Drug"] * 4,
            "Medium": ["M"] * 4, "Temperature": [30] * 4,
            "pert_time_unit": ["min"] * 4, "pert_time": [15, 30, 60, 120],
        })
        triples, weights = make_time_triples(metadata, np.ones(4, dtype=bool))
        np.testing.assert_array_equal(triples, [[0, 1, 2], [1, 2, 3]])
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)

    def test_batch_calibration_learns_control_batch_offset(self) -> None:
        metadata = pd.DataFrame({
            "data_source": ["D"] * 4, "instrument": ["I"] * 4,
            "Yeast_cell_plate": ["P"] * 4,
            "perturbation_no_concentration": ["Water", "DMSO", "Drug", "Drug"],
        })
        truth = np.array([[2.], [2.], [4.], [5.]], dtype=np.float32)
        prediction = truth - 1.0
        train = np.ones(4, dtype=bool)
        calibrator, _ = fit_batch_calibrator(
            metadata, truth, prediction, train,
            modes=("controls",), shrinkages=(0.0,), scales=(1.0,))
        calibrated, correction = apply_batch_calibration(calibrator, metadata, prediction)
        np.testing.assert_allclose(correction, 1.0)
        np.testing.assert_allclose(calibrated, truth)

    def test_proxy_average_precision_is_exact_for_perfect_ranking(self) -> None:
        self.assertAlmostEqual(_average_precision(
            np.array([0.9, 0.8, 0.1]), np.array([True, True, False])), 1.0)

    def test_proxy_dep_is_perfect_for_perfect_delta(self) -> None:
        delta = np.array([[2.0, -3.0, 0.2, 0.1]], dtype=np.float32)
        metrics = _dep_metrics(delta, delta, np.ones_like(delta, dtype=bool))
        self.assertAlmostEqual(metrics["module6_score"], 1.0, places=6)

    def test_proxy_aggregation_uses_all_six_weights(self) -> None:
        records = []
        for split in ("val_chem_only", "val_strain_only", "val_both", "val_time"):
            records.append({"split": split, "module1_score": 1.0,
                            "module2_score": 1.0, "module3_score": 1.0,
                            "module4_score": 1.0, "module5_score": 1.0,
                            "module6_score": 1.0})
        summary = aggregate_proxy_score(records)
        self.assertAlmostEqual(summary["proxy_score"], 1.0)
        self.assertAlmostEqual(summary["available_weight"], 1.0)

    def test_multihead_output_shape_and_zero_residual_initialization(self) -> None:
        assignments = np.array([0, 1, 0, 1, 1])
        model = MultiHeadConditionMLP(3, 5, 8, 4, 0.0, assignments, shared_depth=2)
        output = model(torch.randn(2, 3))
        self.assertEqual(tuple(output.shape), (2, 5))
        np.testing.assert_allclose(output.detach().numpy(), 0.0, atol=1e-7)

    def test_protein_grouping_is_train_only_and_complete(self) -> None:
        y = np.array([[1., 2., 3., 4.], [2., 3., 4., 5.],
                      [100., -100., 100., -100.]], dtype=np.float32)
        mask = np.ones_like(y, dtype=bool)
        train = np.array([True, True, False])
        for method in ("random", "missingness", "response"):
            config = ProteinGroupingConfig(method=method, n_heads=2,
                                           signature_components=1, seed=7)
            assignments, audit = make_protein_groups(y, mask, train, config)
            changed = y.copy()
            changed[~train] *= -999
            repeated, _ = make_protein_groups(changed, mask, train, config)
            np.testing.assert_array_equal(assignments, repeated)
            self.assertEqual(sum(audit["head_sizes"]), y.shape[1])

    def test_masked_rowwise_pearson_loss(self) -> None:
        target = torch.tensor([[1., 2., 3.], [4., 2., 9.]])
        prediction = torch.tensor([[2., 4., 6.], [8., 4., -3.]])
        mask = torch.tensor([[1., 1., 1.], [1., 1., 0.]])
        self.assertAlmostEqual(float(masked_rowwise_pearson_loss(prediction, target, mask)), 0.0, places=6)

    def test_masked_loss_variants_match_numpy_objective(self) -> None:
        truth = np.array([[1., 2.], [3., np.nan], [5., 8.]], dtype=np.float32)
        mask = np.isfinite(truth)
        filled = np.where(mask, truth, 0.).astype(np.float32)
        prediction = np.array([[2., 0.], [4., 9.], [3., 9.]], dtype=np.float32)
        rows = np.ones(3, dtype=bool)
        for config in (MaskedLossConfig(), MaskedLossConfig("protein_balanced_mse"),
                       MaskedLossConfig("variance_standardized_mse"),
                       MaskedLossConfig("masked_huber")):
            state = fit_masked_loss_state(filled, mask, rows, config)
            actual = masked_loss(torch.from_numpy(prediction), torch.from_numpy(filled),
                                 torch.from_numpy(mask.astype(np.float32)),
                                 torch.from_numpy(state.protein_weights), config)
            expected = masked_objective_numpy(prediction, filled, mask, state)
            self.assertAlmostEqual(float(actual), expected, places=6)

    def test_ood_anchor_splits_have_no_entity_or_time_leakage(self) -> None:
        frame = pd.DataFrame({
            "Strains": ["A", "A", "B", "B", "C", "C"],
            "perturbation_no_concentration": ["x", "y", "x", "z", "y", "z"],
            "pert_time": [15, 30, 15, 30, 15, 30],
        })
        train = np.ones(len(frame), dtype=bool)
        strain_folds, strain_rows, _ = make_anchor_folds(frame, train, "leave_one_strain_out")
        self.assertTrue(strain_rows.all())
        for fold in strain_folds:
            self.assertTrue(set(frame.loc[fold.train_rows, "Strains"]).isdisjoint(
                set(frame.loc[fold.validation_rows, "Strains"])))

        compound_folds, compound_rows, _ = make_anchor_folds(
            frame, train, "leave_compound_group_out", n_folds=2)
        self.assertTrue(compound_rows.all())
        for fold in compound_folds:
            self.assertTrue(set(frame.loc[fold.train_rows, "perturbation_no_concentration"]).isdisjoint(
                set(frame.loc[fold.validation_rows, "perturbation_no_concentration"])))

        time_folds, time_rows, audit = make_anchor_folds(frame, train, "late_time_holdout")
        self.assertEqual(audit["cutoff"], 30.0)
        self.assertTrue((frame.loc[time_rows, "pert_time"] == 30).all())
        self.assertLess(frame.loc[time_folds[0].train_rows, "pert_time"].max(),
                        frame.loc[time_folds[0].validation_rows, "pert_time"].min())

    def test_zero_initialized_bridge_is_exact_anchor(self) -> None:
        bridge = GatedResidualBridge(2, 1, 2, 2, 4, 2, 0.0)
        decoder = MaskedProteomeAutoencoder(3, 4, 2)
        anchor = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        latent = np.array([[0.2, -0.1]], dtype=np.float32)
        prediction = predict_bridge(
            bridge, decoder, anchor, latent,
            np.zeros((1, 1), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 2), dtype=np.float32),
            torch.device("cpu"), 1,
        )
        np.testing.assert_allclose(prediction, anchor, atol=1e-7)

    def test_control_folds_keep_exact_context_together(self) -> None:
        frame = pd.DataFrame({
            "data_source": ["D", "D"], "instrument": ["I", "I"],
            "Yeast_cell_plate": ["P", "P"], "Strains": ["S", "S"],
            "Medium": ["M", "M"], "Temperature": [30, 30],
            "pert_time": [15, 15], "pert_time_unit": ["min", "min"],
        })
        folds = context_fold_ids(frame, 5)
        self.assertEqual(int(folds[0]), int(folds[1]))

    def test_entity_pca_is_finite_and_label_free_shape(self) -> None:
        x = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [1.0, 1.0, 0.0]])
        embedding, contract = fit_pca_embedding(x, 2)
        self.assertEqual(embedding.shape, (3, 2))
        self.assertTrue(np.isfinite(embedding).all())
        self.assertEqual(contract["components"].shape, (2, 3))

    def test_content_ablation_feature_selection(self) -> None:
        encoder_info = {
            "categorical_fields": ["Strains", "perturbation_no_concentration", "Medium"],
            "vocabularies": {
                "Strains": ["a", "b"],
                "perturbation_no_concentration": ["x", "y", "z"],
                "Medium": ["m"],
            },
            "output_dim": 8,
        }
        columns = selected_base_columns(
            8, feature_slices(encoder_info), keep_strain_id=False, keep_chemical_id=True
        )
        self.assertEqual(columns.tolist(), [2, 3, 4, 5, 6, 7])
        names = np.array([
            "canonical_kmer_0", "population_snp_pca_0",
            "peter2018_region=X", "genome_feature_available",
        ])
        self.assertEqual(selected_strain_columns(names).tolist(), [1, 2])

    def test_masked_metrics_ignore_nan(self) -> None:
        truth = np.array([[1.0, np.nan], [3.0, 5.0]])
        pred = np.array([[1.0, 100.0], [2.0, 5.0]])
        metrics = masked_regression_metrics(truth, pred)
        self.assertEqual(metrics["valid_values"], 3)
        self.assertAlmostEqual(metrics["log2_rmse"], np.sqrt(1.0 / 3.0))

    def test_unseen_category_maps_to_zero_block(self) -> None:
        train = pd.DataFrame(
            {
                "Strains": ["A"],
                "perturbation_no_concentration": ["drug"],
                "Medium": ["M"],
                "Temperature": [30],
                "data_source": ["D"],
                "instrument": ["I"],
                "Yeast_cell_plate": ["P"],
                "pert_time": [15],
            }
        )
        encoder = ConditionFeatureEncoder.fit(train)
        unseen = train.copy()
        unseen["Strains"] = "UNSEEN"
        encoded = encoder.transform(unseen)
        self.assertEqual(float(encoded[0, 0]), 0.0)

    def test_protein_mean_and_matched_control(self) -> None:
        metadata = pd.DataFrame(
            {
                "sample_ID": ["c", "t"],
                "data_source": ["D", "D"],
                "instrument": ["I", "I"],
                "Yeast_cell_plate": ["P", "P"],
                "Strains": ["S", "S"],
                "Medium": ["M", "M"],
                "Temperature": [30, 30],
                "pert_time": [15, 15],
                "pert_time_unit": ["min", "min"],
                "perturbation_no_concentration": ["Water", "Drug"],
                "split_final": ["train", "val_chem_only"],
            }
        ).set_index("sample_ID")
        y = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        mean = protein_mean(y, np.array([True, False]))
        np.testing.assert_allclose(mean, [1.0, 2.0])
        rows, pred, counts = matched_control_predictions(
            metadata, y, np.array([False, True])
        )
        np.testing.assert_array_equal(rows, [1])
        np.testing.assert_allclose(pred, [[1.0, 2.0]])
        np.testing.assert_array_equal(counts, [1])

    def test_strict_matching_does_not_read_validation_control(self) -> None:
        metadata = pd.DataFrame(
            {
                "sample_ID": ["c", "t"],
                "data_source": ["D", "D"],
                "instrument": ["I", "I"],
                "Yeast_cell_plate": ["P", "P"],
                "Strains": ["S", "S"],
                "Medium": ["M", "M"],
                "Temperature": [30, 30],
                "pert_time": [15, 15],
                "pert_time_unit": ["min", "min"],
                "perturbation_no_concentration": ["Water", "Drug"],
                "split_final": ["val_chem_only", "train"],
            }
        ).set_index("sample_ID")
        y = np.array([[1.0], [2.0]], dtype=np.float32)
        rows, _, _ = strict_matched_controls(
            metadata,
            y,
            np.array([False, True]),
            np.array([False, True]),
        )
        self.assertEqual(len(rows), 0)

    def test_pubchem_payload_parser_accepts_connectivity_smiles(self) -> None:
        payload = {
            "PropertyTable": {
                "Properties": [
                    {
                        "CID": 702,
                        "Title": "Ethanol",
                        "ConnectivitySMILES": "CCO",
                        "SMILES": "CCO",
                    }
                ]
            }
        }
        record = _extract_pubchem_record(payload)
        self.assertEqual(record["pubchem_cid"], 702)
        self.assertEqual(record["canonical_smiles"], "CCO")

    def test_canonical_kmers_are_reverse_complement_invariant(self) -> None:
        forward = hashed_kmer_features("AACGTTAC", k=4, buckets=64)
        reverse = hashed_kmer_features("GTAACGTT", k=4, buckets=64)
        np.testing.assert_allclose(forward, reverse)
        self.assertAlmostEqual(float(forward.sum()), 1.0)

    def test_context_encoder_separates_biology_and_observation(self) -> None:
        frame = pd.DataFrame(
            {
                "Medium": ["M1", "M2"],
                "pert_time_unit": ["h", "h"],
                "Temperature": [25, 30],
                "pert_time": [2, 24],
                "data_source": ["D1", "D2"],
                "instrument": ["I1", "I2"],
                "Yeast_cell_plate": ["P1", "P2"],
                "protein_well": ["W1", "W2"],
            }
        )
        encoder = ContextFeatureEncoder.fit(frame)
        biological, biological_names = encoder.transform_biological(frame)
        observation, observation_names = encoder.transform_observation(frame)
        self.assertEqual(biological.shape[0], 2)
        self.assertEqual(observation.shape[0], 2)
        self.assertFalse(any("instrument" in name for name in biological_names))
        self.assertTrue(any("instrument" in name for name in observation_names))


if __name__ == "__main__":
    unittest.main()
