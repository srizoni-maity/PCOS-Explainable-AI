"""
=========================================================
Subset Selector

Ranks subsets.

=========================================================
"""

class SubsetSelector:

    def select(

        self,

        df

    ):

        ranking = df.sort_values(

            by=[

                "MeanROC",

                "MeanMCC",

                "MeanPR",

                "MeanF1",

                "MeanBrier",

                "NumFeatures"

            ],

            ascending=[

                False,
                False,
                False,
                False,
                True,
                True

            ]

        ).reset_index(drop=True)

        return ranking, ranking.iloc[0]