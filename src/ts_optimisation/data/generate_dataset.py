import logging
from typing import Optional

import pandas as pd

from . import DATA_CONFIG

logger = logging.getLogger(__name__)

__all__ = ["generate_dataset"]


class DatasetGenerator:
    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config = DATA_CONFIG

        self.n = DATA_CONFIG["n_records"]
        self.data_freq = DATA_CONFIG["data_freq"]
        self.cols = DATA_CONFIG["columns"]
        self.extra_cols = [str]

    def fit_transform(self, text: str) -> pd.DataFrame:
        df = self._get_df_from_text(text)
        df = self._generate_dataset(df)
        self._check_sequence_continuity(df)
        self.extra_cols = list(set(df.columns) - set(self.cols))

        return df[self.cols + self.extra_cols]

    def _get_df_from_text(self, text: str) -> pd.DataFrame:
        """
        Generates DataFrame from text input
        """
        input_s = [
            (s[:16], s[17:].split()[0], s[17:].split()[1:])
            for s in text.split("\n")
            if s
        ]
        df = pd.DataFrame(input_s, columns=["time", "fm_num", "data"])
        df["time"] = pd.to_datetime(df["time"])

        return df

    def _generate_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates dataset from the given DataFrame
        """
        data = pd.DataFrame(
            index=pd.MultiIndex.from_tuples(
                sorted(list(set(zip(df["time"], df["fm_num"])))),
                names=["time", "fm_id"],
            )
        )
        for i in range(len(df)):
            row = df.loc[i, :]
            ind = (row["time"], row["fm_num"])
            s = row["data"]
            if len(s) == self.n:
                data.loc[ind, f"{s[0]}_{s[1]}_low"] = float(s[2])
                data.loc[ind, f"{s[0]}_{s[1]}_high"] = float(s[3])
            else:
                data.loc[ind, f"{s[0]}_{s[1]}_mom"] = float(s[2])

        return data

    def _check_sequence_continuity(self, df: pd.DataFrame) -> None:
        """
        Checks floating machine data sequence continuity
        """
        fm_id = list(set(df.index.get_level_values("fm_id")))
        for i in fm_id:
            n_periods = df.xs(f"{i}", level="fm_id").shape[0]
            t0 = df.xs(f"{i}", level="fm_id").index[0]
            if set(df.xs(f"{i}", level="fm_id").index) != set(
                pd.date_range(t0, freq=self.data_freq, periods=n_periods)
            ):
                print(f"Data sequence of floating machine {i} is broken")
                logger.warning(
                    f"Data sequence of floating machine {i} is broken",
                    UserWarning,
                )
