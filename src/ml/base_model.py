from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    def __init__(self):
        self.model = None
        self.model_params = {}
        self.is_fitted = False

    @abstractmethod
    def train(self, X, y) -> None:
        pass

    @abstractmethod
    def predict(self, X, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def predict_72h(self, X, *args, **kwargs) -> Any:
        pass

    def get_params(self) -> dict[str, Any]:
        return self.model_params

    def set_params(self, **params: Any) -> None:
        self.model_params.update(params)
