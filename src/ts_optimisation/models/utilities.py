import logging
from typing import Optional

import pandas as pd

from . import OUTPUT_FILE_NAME

logger = logging.getLogger(__name__)

__all__ = ["write_results_to_output_file"]


def get_final_result(result: pd.DataFrame, filename: Optional[str] = None) -> None:
    """
    Writes final result to output file.
    """
    if filename is None:
        filename = OUTPUT_FILE_NAME

    df = result[
        ["Ni_Conc_low", "Ni_Conc_high", "Ni_Tail_low", "Ni_Tail_high"]
    ].reset_index()
    with open(filename, "w", encoding="utf-8") as f:
        for i in range(len(df)):
            t = str(df.iloc[i, 0])[:-3]
            fm_num = df.iloc[i, 1]
            prod_1 = df.columns[2].split("_")
            prod_2 = df.columns[4].split("_")
            f.write(
                f"{t} {fm_num} {prod_1[0]} {prod_1[1]} {df.iloc[i, 2]} {df.iloc[i, 3]}\n"
            )
            f.write(
                f"{t} {fm_num} {prod_2[0]} {prod_2[1]} {df.iloc[i, 4]} {df.iloc[i, 5]}\n"
            )
