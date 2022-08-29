import math
import os
from enum import Enum
from typing import Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from json_handling import load_json, save_json
from loguru import logger
from tensorflow import keras
from tensorflow.keras import layers

# Enable auto-logging to MLflow to capture TensorBoard metrics.
mlflow.tensorflow.autolog()

# Note: this line solves a problem with CuDNN
# The origin of this issue could be missmatch between versions
# of CuDNN and TF. It is only reflected when using CNN layers.
# Log from TF:
# ' Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH
#   environment variable is set. Original config value was 0. '
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class PlotType(Enum):
    LOSS = "LOSS"
    ACCURACY = "ACCURACY"


def prepare_mnist_data() -> Tuple[Tuple, Tuple]:
    """Prepare data for training and testing. """
    # this dataset has 10 distinct categories
    num_classes = 10

    # load and split MNIST dataset to train and test
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # scale images to [0, 1]
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # dimension correction - (28, 28) -> (28, 28, 1)
    # we want to provide one extra dimension for our one channel
    # (e.g. RGB images would have 3)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    logger.info(f"x_train shape: {x_train.shape}")
    logger.info(f"x_train samples: {x_train.shape[0]}")
    logger.info(f"x_test samples: {x_test.shape[0]}")

    # convert class vectors to binary class matrices - one-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)


def visualise_data_sample(
        data: np.array,
        number_of_samples: int,
        window_title: str = "Dataset visualisation"
) -> None:
    """Visualise specified number of images from the provided dataset. """
    grid_size = math.ceil(math.sqrt(number_of_samples))

    # create a figure where the subplots will be added
    fig = plt.figure()
    fig.canvas.manager.set_window_title(window_title)
    plt.suptitle(f"Subsample of {number_of_samples} images:")

    for i in range(number_of_samples):
        # define subplot
        plt.subplot(grid_size, grid_size, i + 1)
        # plot raw pixel data
        plt.imshow(data[i], cmap=plt.get_cmap('gray'))
    # show the figure
    plt.show()


def build_model(num_classes: int = 10) -> keras.Model:
    """Build simple CNN model. """
    input_shape = (28, 28, 1)
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(
        32, kernel_size=(3, 3), activation="relu", padding="same", strides=2
    )(inputs)
    # x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(
        64, kernel_size=(3, 3), activation="relu", padding="same", strides=2
    )(x)
    # x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.7)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


def train_model(
        data: Tuple[np.array, np.array],
        batch_size: int = 128,
        epochs: int = 30,
        history_saving: bool = True
) -> keras.Model:
    """Train the architecture on provided data. """
    x_train = data[0]
    y_train = data[1]

    model = build_model()
    model.summary()

    # sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.8)
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    # 'patience' defines how many epochs can pass without improvement
    # of desired metric (accuracy or loss) until the training is stopped
    # 'min_delta' specifies minimal amount of units (1 unit for mean
    # squared error, or 1 % for accuracy) that is considered improvement.
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1
    )
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    callbacks = [early_stopping_callback, model_checkpoint_callback]

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=callbacks
    )
    if history_saving:
        save_json(history.history, "history.json")
    return model


def evaluate_model(
        model: keras.Model,
        test_data: Tuple[np.array, np.array]
) -> None:
    """Evaluate trained model on test data with corresponding labels. """
    x_test, y_test = test_data
    score = model.evaluate(x_test, y_test, verbose=1)
    print(f"Test loss: {score[0]}")
    print(f"Test accuracy: {score[1]}")


def plot_learning_curves(
        training_history_path: str,
        plot_type: PlotType,
        window_title: str = "Learning curves"
):
    """Plot the learning curve for either loss or accuracy. """
    history = load_json(training_history_path)
    metric = str(plot_type.value).lower()

    window_title += f": {metric.title()}"
    fig = plt.figure()
    fig.canvas.manager.set_window_title(window_title)
    plt.xlabel("Epoch")
    plt.ylabel(metric.title())
    plt.plot(history[metric], label="train")
    plt.plot(history["val_" + metric], label="val")
    plt.legend()
    plt.show()


def main() -> None:
    (x_train, y_train), (x_test, y_test) = prepare_mnist_data()
    # visualise_data_sample(x_train, 55)

    model = train_model((x_train, y_train))

    # model = keras.models.load_model("best_model.h5")

    evaluate_model(model, (x_test, y_test))

    plot_learning_curves("history.json", PlotType.ACCURACY)


if __name__ == '__main__':
    main()
