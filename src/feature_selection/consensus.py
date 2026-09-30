"""
=========================================================
Consensus Feature Selection

Runs every selector

Combines them using
Borda Count

=========================================================
"""

from pathlib import Path

import pandas as pd

from .mutual_information import MutualInformationSelector
from .anova import AnovaSelector
from .random_forest import RandomForestSelector
from .lasso import LassoSelector
from .xgboost_selector import XGBoostSelector

from .ranking import BordaRankAggregator


class ConsensusFeatureSelector:

    def __init__(self):

        self.selectors = {

            "MI": MutualInformationSelector(),

            "ANOVA": AnovaSelector(),

            "RF": RandomForestSelector(),

            "LASSO": LassoSelector(),

            "XGB": XGBoostSelector()

        }

        self.ranker = BordaRankAggregator()

    # --------------------------------------------------

    def run(self, X, y):

        ranking = pd.DataFrame()

        ranking["Feature"] = X.columns

        for name, selector in self.selectors.items():

            scores = selector.fit(X, y)

            ranking[f"{name}_Score"] = scores

        ranking = self.ranker.aggregate(ranking)

        return ranking

    # --------------------------------------------------

    def save(self, ranking, output_directory):

        output_directory = Path(output_directory)

        output_directory.mkdir(

            parents=True,

            exist_ok=True

        )

        ranking.to_csv(

            output_directory / "ranking.csv",

            index=False

        )

        subsets = {

            "top5.csv":5,

            "top8.csv":8,

            "top10.csv":10,

            "top12.csv":12,

            "all.csv":len(ranking)

        }

        for filename,k in subsets.items():

            ranking.head(k).to_csv(

                output_directory/filename,

                index=False

            )