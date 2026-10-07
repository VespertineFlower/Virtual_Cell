from __future__ import annotations

METADATA_TRAIN_VAL = "WAYB_WAYC_metadata_train_val(1).csv"
METADATA_TEST = "WAYB_WAYC_metadata_test(1).csv"
PROTEOME_TRAIN_VAL = "WAYB_WAYC_proteome_raw_train_val.csv"
PROTEOME_TEST = "WAYB_WAYC_proteome_raw_test.csv"

VALIDATION_SPLITS = (
    "val_chem_only",
    "val_strain_only",
    "val_both",
    "val_time",
)

CONTROL_NAMES = frozenset({"Water", "DMSO"})
EXCLUDED_TREATMENTS = CONTROL_NAMES | frozenset({"Quality Control"})

# Exact-match fields stated in section 4.2.2 of the supplied tutorial PDF.
MATCH_KEYS = (
    "data_source",
    "instrument",
    "Yeast_cell_plate",
    "Strains",
    "Medium",
    "Temperature",
    "pert_time",
    "pert_time_unit",
)

# A deliberately simple condition MLP. Technical variables are concatenated
# rather than separated; the user's later method can improve on this baseline.
MLP_CATEGORICAL_FIELDS = (
    "Strains",
    "perturbation_no_concentration",
    "Medium",
    "Temperature",
    "data_source",
    "instrument",
    "Yeast_cell_plate",
)

