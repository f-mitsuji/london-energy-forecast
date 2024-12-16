from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseModel(ABC):
    """Base class for all machine learning models."""

    def __init__(self):
        self.model = None
        self.model_params: dict[str, Any] = {}
        self.is_fitted = False

    @abstractmethod
    def train(self, X, y) -> None:
        """Train the model with given data.

        Args:
            X: Training features
            y: Target values
        """

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Make predictions on new data.

        Args:
            X: Features to predict on

        Returns:
            Predictions
        """

    def get_params(self) -> dict[str, Any]:
        """Get model parameters.

        Returns:
            Dictionary of model parameters
        """
        return self.model_params

    def set_params(self, **params: Any) -> None:
        """Set model parameters.

        Args:
            **params: Parameters to set
        """
        self.model_params.update(params)
