from .logistic_model import LogisticModel
from .random_forest_model import RandomForestModel
from .xgboost_model import XGBoostModel
from .catboost_model import CatBoostModel


def get_models():

    return [

        LogisticModel(),

        RandomForestModel(),

        XGBoostModel(),

        CatBoostModel()

    ]

def get_catboost():
    return CatBoostModel()