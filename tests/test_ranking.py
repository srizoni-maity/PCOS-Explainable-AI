import numpy as np
import pandas as pd

from src.feature_selection.ranking import BordaRankAggregator

features = [

    "A",

    "B",

    "C",

    "D",

    "E"

]

df = pd.DataFrame({

    "Feature": features,

    "MI_Score": np.random.rand(5),

    "ANOVA_Score": np.random.rand(5),

    "RF_Score": np.random.rand(5),

    "LASSO_Score": np.random.rand(5),

    "XGB_Score": np.random.rand(5)

})

ranking = BordaRankAggregator().aggregate(df)

print(ranking)