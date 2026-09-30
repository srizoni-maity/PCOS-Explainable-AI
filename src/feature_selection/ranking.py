"""
=========================================================
Consensus Ranking using Borda Count

Author:
Project:

=========================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class BordaRankAggregator:

    """
    Aggregates multiple feature rankings
    using Borda Count.
    """

    def __init__(self):

        self.score_columns = []

    # --------------------------------------------------------

    def rank_features(self, df, score_column):

        """
        Higher score = better feature

        Rank 1 = Best
        """

        ranks = (

            df[score_column]

            .rank(

                ascending=False,

                method="min"

            )

            .astype(int)

        )

        return ranks

    # --------------------------------------------------------

    def borda_points(self, ranks):

        """
        Rank 1 gets maximum points.

        Example

        32 Features

        Rank1 -> 32

        Rank2 -> 31

        Rank32 -> 1
        """

        n = len(ranks)

        return n - ranks + 1

    # --------------------------------------------------------

    def aggregate(self, dataframe):

        """
        dataframe

        Feature

        MI_Score

        RF_Score

        ...

        """

        df = dataframe.copy()

        self.score_columns = [

            c for c in df.columns

            if c.endswith("_Score")

        ]

        borda_columns = []

        for column in self.score_columns:

            rank_column = column.replace(

                "_Score",

                "_Rank"

            )

            point_column = column.replace(

                "_Score",

                "_Points"

            )

            df[rank_column] = self.rank_features(

                df,

                column

            )

            df[point_column] = self.borda_points(

                df[rank_column]

            )

            borda_columns.append(

                point_column

            )

        df["BordaScore"] = df[

            borda_columns

        ].sum(axis=1)

        df = df.sort_values(

            "BordaScore",

            ascending=False

        ).reset_index(drop=True)

        df["ConsensusRank"] = np.arange(

            1,

            len(df) + 1

        )

        return df