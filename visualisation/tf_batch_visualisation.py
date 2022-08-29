import math
from abc import ABCMeta, abstractmethod
from typing import Dict

import numpy as np
import tensorflow as tf
from loguru import logger
from matplotlib import pyplot as plt
from tensorflow.keras import layers


class DataVisualisation(metaclass=ABCMeta):
    """Models visualisation for the 'tf.data' Datasets. """

    def __init__(self, dataset: tf.data.Dataset) -> None:
        self.dataset: tf.data.Dataset = dataset

    def visualise_dataset(
            self,
            data_augmentation: layers.Layer,
            batch_ordinary_number: int = 1,
            number_of_samples: int = 20,
            original_data: bool = True,
            augmented_data: bool = True
    ) -> None:
        """Visualise original and the augmented dataset.

        The 'original_data' controls whether the original dataset would be
        visualised.
        The 'augmented_data' controls whether the augmented dataset would be
        visualised.
        Visualisation is important in order to find out what actually model
        needs to learn:
            1. It validates that our dataset is correctly loaded - and that
                we loaded what we intended!
            2. It shows us the input to the model after augmentations are
                applied - it could happen that the augmentations are too
                intensive so the model is not able to learn target traits,
                and it could lead to over-fitting rather than learning
                important features.
        """
        # Take the specified batch from the dataset
        # Shape (Tuple in List): [ ((128, 28, 28, 1), (128, 10)) ]
        batch_x_y_list = list(self.dataset.take(batch_ordinary_number))
        # Take the first (and only) element from the list
        batch_x_y = batch_x_y_list[0]
        # Take the first element from the Tuple, shape: (128, 28, 28, 1)
        batch_x = batch_x_y[0]
        if original_data:
            # Visualise original train data
            logger.info("Visualisation of original dataset.")
            self._visualise_data_sample(
                batch_x,
                number_of_samples=number_of_samples,
                window_title="Original dataset"
            )
        if augmented_data:
            # Visualise augmented train data
            # Note: when augmenting a single image, expand its dimension
            # by using: np.expand_dims(image, axis=0)
            logger.info("Visualisation of augmented dataset.")
            augmented_batch_x = data_augmentation(batch_x, training=True)
            self._visualise_data_sample(
                augmented_batch_x,
                number_of_samples=number_of_samples,
                window_title="Augmented dataset"
            )

    @staticmethod
    def _visualise_data_sample(
            data: np.array,
            number_of_samples: int,
            window_title: str = "Dataset visualisation"
    ) -> None:
        """Visualise specified number of images from the provided dataset."""
        grid_size = math.ceil(math.sqrt(number_of_samples))

        # Create a figure where the subplots will be added
        fig = plt.figure()
        fig.canvas.manager.set_window_title(window_title)
        plt.suptitle(f"Subsample of {number_of_samples} images:")

        for i in range(number_of_samples):
            # Define subplot
            plt.subplot(grid_size, grid_size, i + 1)
            # Plot raw pixel data
            plt.imshow(
                data[i],
                # Show as grayscale if necessary
                # cmap=plt.get_cmap('gray')
            )
        # Show the figure
        plt.show()
