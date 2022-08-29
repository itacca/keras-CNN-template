from abc import ABCMeta, abstractmethod
from typing import Optional

import tensorflow as tf


class DataLoader(metaclass=ABCMeta):
    """Abstract Model class that is inherited to all models. """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.train_dataset: Optional[tf.data.Dataset] = None
        self.validation_dataset: Optional[tf.data.Dataset] = None
        self.test_dataset: Optional[tf.data.Dataset] = None

    @abstractmethod
    def get_train_data(self) -> tf.data.Dataset:
        pass

    @abstractmethod
    def get_validation_data(self) -> tf.data.Dataset:
        pass

    @abstractmethod
    def get_test_data(self) -> tf.data.Dataset:
        pass

