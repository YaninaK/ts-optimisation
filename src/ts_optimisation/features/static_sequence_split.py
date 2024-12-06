import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src", "ts_optimisation"))

import logging
from typing import Optional, Tuple

import pandas as pd
from models import MODEL_CONFIG as CONFIG

logger = logging.getLogger(__name__)

__all__ = ["generate_static_and_sequence_datasets"]


def generate_static_and_sequence_datasets(
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Генерирует 2 датасета:
        df_stat - датасет со статичными данными
        df_seq - датасет с временными рядами.
    """
    if config is None:
        config = CONFIG

    df_stat = (
        df[config["input_stat_feature_names"]]
        .reset_index()
        .groupby("fm_id", as_index=False)
        .agg(
            n_periods=("time", "count"),
            Ni_Feed_mom=("Ni_Feed_mom", "first"),
            Cu_Feed_mom=("Cu_Feed_mom", "first"),
            Density_Feed_mom=("Density_Feed_mom", "first"),
            Rec_Conc_mom=("Rec_Conc_mom", "first"),
        )
    )
    df_seq = df[
        [col for col in df.columns if col not in config["input_stat_feature_names"]]
    ]

    return df_stat, df_seq
