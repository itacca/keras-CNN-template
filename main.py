"""MNIST Classification using Keras CNN

This script allows training and evaluating the simple CNN architecture,
designed to solve MNIST classification task.

All steps in the pipeline (data loading / visualisation / augmentation,
model build, model training and evaluation) are implemented as separate
modules, following the OOP concept.

By running this script, the full pipeline would be run:
1. Dataset loading, batching and prefetching using 'tf.data' Dataset
2. Dataset visualisation: inspection of both original samples from the
    dataset, and the images after the augmentation layer is applied.
3. Model build-up: create the custom architecture specified in the
    configuration file. All parameters (number of ConvLayers,
    existence of Batch Normalization and Pooling layers, etc.) could
    be specified via configuration file.
4. Model training: the whole process is supported by logging tool -
    MLflow, so we are able to track performance across individual
    experiments (where each experiment is denoted with one set of
    the hyper-parameters).
5. Model evaluation on the test dataset: simple accuracy metric.
"""
import os
from typing import List

import mlflow
import tensorflow as tf

from base_data_loader import DataLoader
from json_handling import load_json

from mnist_data_loader import MNISTDataLoader
from mnist_model import MNISTModel
from plotting import plot_learning_curves, PlotType
from tf_batch_visualisation import DataVisualisation

# Enable auto-logging to MLflow to capture TensorBoard metrics.
mlflow.tensorflow.autolog()

# Note: this line solves a problem with CuDNN
# The origin of this issue could be missmatch between versions
# of CuDNN and TF. It is only reflected when using CNN layers.
# Log from TF:
# ' Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH
#   environment variable is set. Original config value was 0. '
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

CONFIG_PATH: str = "config_files/config.json"


def check_gpu_visibility() -> bool:
    """Checks whether the GPU is visible from TensorFlow."""

    # Run the following command before using GPU (if needed):
    # sudo ldconfig /usr/lib/cuda/lib64

    recognized_gpu_devices: List = tf.config.list_physical_devices("GPU")
    print("List of recognized GPU devices:")
    print(recognized_gpu_devices)

    if not recognized_gpu_devices:
        return False

    # Sanity check - GPU is found, now test with simple matrix multiplication
    with tf.device('/gpu:0'):
        a = tf.constant(
            [1.0, 2.0, 3.0],
            shape=[1, 3],
            name='a'
        )
        b = tf.constant(
            [1.0, 2.0, 3.0],
            shape=[3, 1],
            name='b'
        )
        c = tf.matmul(a, b)
    print("Matrix multiplication result:")
    tf.print(c)
    return True


def visualise_dataset_sample(
        data_loader: DataLoader,
        augmentation_config,
        batch_ordinary_number: int = 1,
        number_of_samples: int = 20,
        original_data: bool = True,
        augmented_data: bool = True
) -> None:
    """Visualises provided dataset.

    It is possible to visualise both original and augmented images,
    on the specified batch and provided number of samples.
    """
    # Prefetched dataset, shape (Tuple): ((None, 28, 28, 1), (None, 10))
    train_dataset: tf.data.Dataset = data_loader.get_train_data()
    data_augmentation = MNISTModel.get_augmentations(augmentation_config)

    visualiser = DataVisualisation(train_dataset)
    visualiser.visualise_dataset(
        data_augmentation=data_augmentation,
        batch_ordinary_number=batch_ordinary_number,
        number_of_samples=number_of_samples,
        original_data=original_data,
        augmented_data=augmented_data
    )


def main() -> None:
    # Load configuration file
    config = load_json(CONFIG_PATH)
    mnist_data_loader = MNISTDataLoader(config)

    # Visualise training dataset (original & augmented images)
    augmentation_config = config["model"]["augmentation"]
    visualise_dataset_sample(
        data_loader=mnist_data_loader,
        augmentation_config=augmentation_config
    )

    mnist_model = MNISTModel(config, mnist_data_loader)
    mnist_model.build()

    history = mnist_model.train()
    mnist_model.evaluate()

    plot_learning_curves(history, PlotType.ACCURACY)


if __name__ == '__main__':
    main()
