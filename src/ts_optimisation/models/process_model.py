import logging
from typing import Optional

import pandas as pd
import tensorflow as tf

logger = logging.getLogger(__name__)

__all__ = ["define_process_model"]

from . import MODEL_CONFIG


def get_model(df_stat: pd.DataFrame, config: Optional[dict] = None) -> tf.keras.Model:
    """
    Генерирует модель, работающую с последовательностями (LSTM) и
    статическими признаками для фиксации закономерностей процесса флотации.

    """
    if config is None:
        config = MODEL_CONFIG

    norm_Ni_Feed_mom = tf.keras.layers.Normalization(axis=None, name="norm_Ni_Feed_mom")
    norm_Ni_Feed_mom.adapt(
        df_stat["Ni_Feed_mom"].infer_objects(copy=False).fillna(0).values
    )

    norm_Cu_Feed_mom = tf.keras.layers.Normalization(axis=None, name="norm_Cu_Feed_mom")
    norm_Cu_Feed_mom.adapt(
        df_stat["Cu_Feed_mom"].infer_objects(copy=False).fillna(0).values
    )

    norm_Density_Feed_mom = tf.keras.layers.Normalization(
        axis=None, name="norm_Density_Feed_mom"
    )
    norm_Density_Feed_mom.adapt(
        df_stat["Density_Feed_mom"].infer_objects(copy=False).fillna(0).values
    )

    norm_Rec_Conc_mom = tf.keras.layers.Normalization(
        axis=None, name="norm_Rec_Conc_mom"
    )
    norm_Rec_Conc_mom.adapt(
        df_stat["Rec_Conc_mom"].infer_objects(copy=False).fillna(0).values
    )
    norm_n_periods = tf.keras.layers.Normalization(axis=None, name="norm_n_periods")
    norm_n_periods.adapt(
        df_stat["n_periods"].infer_objects(copy=False).fillna(0).values
    )

    inputs = {
        "Ni_Feed_mom": tf.keras.Input(shape=(1,), dtype=int, name=f"Ni_Feed_mom"),
        "Cu_Feed_mom": tf.keras.Input(shape=(1,), dtype=float, name=f"Cu_Feed_mom"),
        "Density_Feed_mom": tf.keras.Input(
            shape=(1,), dtype=int, name=f"Density_Feed_mom"
        ),
        "Rec_Conc_mom": tf.keras.Input(shape=(1,), dtype=float, name=f"Rec_Conc_mom"),
        "n_periods": tf.keras.Input(shape=(1,), dtype=int, name=f"n_periods"),
        "LSTM input": tf.keras.Input(
            shape=(config["input_sequence_length"], config["n_features"]),
            name="LSTM input",
        ),
    }
    layers = []
    layers.append(norm_Ni_Feed_mom(inputs["Ni_Feed_mom"]))
    layers.append(norm_Cu_Feed_mom(inputs["Cu_Feed_mom"]))
    layers.append(norm_Density_Feed_mom(inputs["Density_Feed_mom"]))
    layers.append(norm_Rec_Conc_mom(inputs["Rec_Conc_mom"]))
    layers.append(norm_n_periods(inputs["n_periods"]))

    stat_features = tf.keras.layers.Concatenate(axis=-1, name="stat_features")(layers)

    X_stat = tf.keras.layers.Dense(
        config["stat_units_max"], activation="relu", name="dense_1"
    )(stat_features)
    X_stat = tf.keras.layers.Dense(
        config["stat_units_min"], activation="relu", name="dense_2"
    )(X_stat)

    encoder_output1, state_h1, state_c1 = tf.keras.layers.LSTM(
        config["lstm_n_units_max"],
        return_sequences=True,
        return_state=True,
        name="encoder_output1",
    )(inputs["LSTM input"])
    encoder_states1 = [state_h1, state_c1]

    encoder_output2 = tf.keras.layers.LSTM(
        config["lstm_n_units_min"], return_state=False, name="encoder_output2"
    )(encoder_output1)

    concatenated = tf.keras.layers.Concatenate(axis=-1, name="all_features")(
        [encoder_output2, X_stat]
    )
    X_all = tf.keras.layers.Dense(
        config["stat_units_min"], activation="relu", name="dense_3"
    )(concatenated)

    outputs = tf.keras.layers.Dense(
        config["n_features"], activation="linear", name="dense_4"
    )(X_all)

    model = tf.keras.Model(inputs, outputs)

    return model
