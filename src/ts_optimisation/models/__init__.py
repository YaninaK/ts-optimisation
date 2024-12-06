MODEL_CONFIG = {
    "input_stat_feature_names": [
        "Ni_Feed_mom",
        "Cu_Feed_mom",
        "Density_Feed_mom",
        "Rec_Conc_mom",
    ],
    "input_sequence_length": 4,
    "n_features": 12,
    "stat_units_max": 32,
    "stat_units_min": 16,
    "lstm_n_units_max": 32,
    "lstm_n_units_min": 16,
}


OUTPUT_FILE_NAME = "output.txt"
