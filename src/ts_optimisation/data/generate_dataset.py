import pandas as pd


class DatasetGenerator:
    def __init__(self, n: int = 4):
        self.n = n
        self.cols = [
            "Ni_Feed_mom",
            "Cu_Feed_mom",
            "Density_Feed_mom",
            "Ni_Conc_mom",
            "Ni_Tail_mom",
            "Ni_Conc_low",
            "Ni_Conc_high",
            "Ni_Tail_low",
            "Ni_Tail_high",
            "Rec_Conc_mom",
        ]
        self.extra_cols = [str]

    def fit_transform(self, text: str) -> pd.DataFrame:
        df = self._get_df_from_text(text)
        df = self._generate_dataset(df)
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
                sorted(list(set(zip(df["time"], df["fm_num"]))))
            )
        )
        for i in range(len(df)):
            row = df.loc[i, :]
            ind = (row["time"], row["fm_num"])
            s = row["data"]
            if len(s) == self.n:
                data.loc[ind, f"{s[0]}_{s[1]}_low"] = s[2]
                data.loc[ind, f"{s[0]}_{s[1]}_high"] = s[3]
            else:
                data.loc[ind, f"{s[0]}_{s[1]}_mom"] = s[2]

        return data
