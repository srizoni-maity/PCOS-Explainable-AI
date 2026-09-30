"""
Consensus Feature Selection Framework
"""
from .base import BaseSelector
from .mutual_information import MutualInformationSelector
from .anova import AnovaSelector
from .random_forest import RandomForestSelector
from .lasso import LassoSelector
from .xgboost_selector import XGBoostSelector