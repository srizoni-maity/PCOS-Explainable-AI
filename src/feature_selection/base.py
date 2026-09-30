from abc import ABC, abstractmethod

import numpy as np


class BaseSelector(ABC):

    """
    Base class for every selector.
    """

    def normalize(self, scores):

        scores = np.asarray(scores, dtype=float)

        minimum = np.min(scores)
        maximum = np.max(scores)

        if minimum == maximum:
            return np.ones_like(scores)

        return (scores - minimum) / (maximum - minimum)

    @abstractmethod
    def fit(self, X, y):
        pass