from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract base class for all models.
    """
    @property
    @abstractmethod
    def name(self):
        """Model name"""
        pass

    @abstractmethod
    def estimator(self):
        """Return initialized estimator"""
        pass

    @abstractmethod
    def search_space(self, trial):
        """
        Return Optuna search space.
        """
        pass