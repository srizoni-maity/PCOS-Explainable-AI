from sklearn.base import clone


class Trainer:

    def train(

        self,

        model,

        params,

        X_train,

        y_train

    ):

        estimator = clone(model)

        estimator.set_params(

            **params

        )

        estimator.fit(

            X_train,

            y_train

        )

        return estimator