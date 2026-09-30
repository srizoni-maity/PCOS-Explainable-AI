"""
=========================================================
Bootstrap Confidence Interval Evaluation

IEEE Publication Version

=========================================================
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.utils import resample

from src.training.metrics import calculate_metrics


class BootstrapEvaluator:

    """
    Bootstrap confidence interval estimation.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap iterations.

    confidence : float
        Confidence level.
    """

    def __init__(

        self,

        n_bootstrap=2000,

        confidence=0.95,

        random_state=42

    ):

        self.n_bootstrap = n_bootstrap

        self.confidence = confidence

        self.random_state = random_state

    # ----------------------------------------------------------

    def bootstrap(

        self,

        model,

        X,

        y

    ):

        rng = np.random.RandomState(

            self.random_state

        )

        results = []

        n = len(y)

        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        for i in range(self.n_bootstrap):

            idx = rng.choice(

                n,

                size=n,

                replace=True

            )

            X_sample = X.iloc[idx]

            y_sample = y.iloc[idx]

            y_prob = model.predict_proba(

                X_sample

            )[:,1]

            y_pred = model.predict(

                X_sample

            )

            metrics = calculate_metrics(

                y_sample,

                y_pred,

                y_prob

            )

            metrics["Iteration"] = i + 1

            results.append(metrics)

        return pd.DataFrame(results)

    # ----------------------------------------------------------

    def confidence_interval(

        self,

        samples

    ):

        alpha = 1.0 - self.confidence

        lower = alpha / 2

        upper = 1 - lower

        rows = []

        metric_columns = [

            c for c in samples.columns

            if c != "Iteration"

        ]

        for metric in metric_columns:

            values = samples[metric].values

            rows.append({

                "Metric": metric,

                "Mean": np.mean(values),

                "Std": np.std(values, ddof=1),

                "Median": np.median(values),

                "Lower95":

                    np.quantile(

                        values,

                        lower

                    ),

                "Upper95":

                    np.quantile(

                        values,

                        upper

                    ),

                "Minimum":

                    np.min(values),

                "Maximum":

                    np.max(values)

            })

        return pd.DataFrame(rows)

    # ----------------------------------------------------------

    def print_summary(

        self,

        ci_df

    ):

        display = ci_df.copy()

        display = display[

            [

                "Metric",

                "Mean",

                "Lower95",

                "Upper95"

            ]

        ]

        return display

    # ----------------------------------------------------------

    def plot_distributions(

        self,

        samples,

        output_folder

    ):

        metrics = [

            c for c in samples.columns

            if c != "Iteration"

        ]

        for metric in metrics:

            plt.figure(

                figsize=(6,4)

            )

            plt.hist(

                samples[metric],

                bins=30

            )

            plt.title(

                metric

            )

            plt.xlabel(

                metric

            )

            plt.ylabel(

                "Frequency"

            )

            plt.tight_layout()

            plt.savefig(

                output_folder /

                f"{metric}_distribution.png",

                dpi=600

            )

            plt.close()

    # ----------------------------------------------------------

    def plot_boxplots(

        self,

        samples,

        output

    ):

        metrics = [

            c for c in samples.columns

            if c != "Iteration"

        ]

        plt.figure(

            figsize=(11,6)

        )

        plt.boxplot(

            [

                samples[m]

                for m in metrics

            ],

            labels=metrics,

            vert=True

        )

        plt.xticks(

            rotation=45,

            ha="right"

        )

        plt.tight_layout()

        plt.savefig(

            output,

            dpi=600

        )

        plt.close()