"""Abstract base model. """
from abc import ABCMeta, abstractmethod

from tensorflow import keras

from base_data_loader import DataLoader


class BaseModel(metaclass=ABCMeta):
    """Abstract Model class that is inherited to all models. """

    def __init__(self, config: dict, data_loader: DataLoader) -> None:
        self.config = config
        self.model = self.build()
        self.data_loader: DataLoader = data_loader

    @abstractmethod
    def build(self) -> keras.Model:
        pass

    @abstractmethod
    def train(self) -> dict:
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def predict(self):
        pass
