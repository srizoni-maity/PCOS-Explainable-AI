"""
=========================================================
Generic Optuna Search
=========================================================
"""

import optuna

from .objective import Objective


class OptunaSearch:
    """
    Generic Optuna hyperparameter optimization.
    """

    def __init__(
        self,
        n_trials=30,
        scoring="ROC_AUC",
        random_state=42,
    ):
        self.n_trials = n_trials
        self.scoring = scoring
        self.random_state = random_state

    def optimize(
        self,
        model_definition,
        X,
        y,
        scoring=None,
    ):
        """
        Perform Optuna hyperparameter optimization.

        Parameters
        ----------
        model_definition : ModelDefinition
            Wrapper containing estimator() and search_space().
        X : pandas.DataFrame
            Training features.
        y : pandas.Series
            Training labels.
        scoring : str, optional
            Metric to optimize. If None, uses the class default.

        Returns
        -------
        dict
            Optimization results.
        """

        scoring = scoring or self.scoring

        objective = Objective(
            model_definition=model_definition,
            X=X,
            y=y,
            scoring=scoring,
        )

        sampler = optuna.samplers.TPESampler(
            seed=self.random_state
        )

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=model_definition.name,
        )

        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
        )

        return {
            "study": study,
            "best_score": study.best_value,
            "best_params": study.best_params,
            "best_trial": study.best_trial.number,
            "num_trials": len(study.trials),
            "scoring": scoring,
        }