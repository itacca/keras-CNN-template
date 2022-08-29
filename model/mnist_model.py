import os
from typing import Dict

from loguru import logger

from base_data_loader import DataLoader
from base_model import BaseModel
from tensorflow import keras
from tensorflow.keras import layers, Sequential


class MNISTModel(BaseModel):
    """MNIST dataset classification with modular building blocks."""

    def __init__(self, config: dict, data_loader: DataLoader) -> None:
        super().__init__(config, data_loader)

    def _apply_preprocessing(self, inputs: layers.Layer) -> layers.Layer:
        """Preprocess input data during the forward pass.

        The preprocessing includes rescaling and augmentation. """
        # Rescale inputs - scale images to [0, 1]
        x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
        x = self._apply_augmentation(x)
        return x

    def _apply_augmentation(self, inputs: layers.Layer) -> layers.Layer:
        """Apply basic augmentation layers to the provided input layer.

        The included augmentation layers perform random flipping, random
        rotation, random zoom and random translation. It should work well
        for the majority of downstream tasks, but for specific use cases,
        e.g. the ones whose inputs are not translation invariant, these
        augmentation layers should be skipped or changed."""
        augmentation_config = self.config["model"]["augmentation"]
        if not augmentation_config["active"]:
            logger.info("Skipping augmentations")
            return inputs
        data_augmentation = self.get_augmentations(augmentation_config)

        augmented_input = data_augmentation(inputs)
        logger.info("Augmentations applied.")

        return augmented_input

    @staticmethod
    def get_augmentations(augmentation_config: Dict) -> layers.Layer:
        """Return the augmentations stored in a Sequential layer.

        The method could be used while training to perform augmentation
        on the training dataset, or to visualize how the augmentation
        functions would change the original training dataset.
        """
        factors = augmentation_config["random_translation"]["factors"]
        height, width = factors[0], factors[1]
        data_augmentation = Sequential([
            layers.experimental.preprocessing.RandomFlip(
                augmentation_config["random_flip"]["mode"]
            ),
            layers.experimental.preprocessing.RandomRotation(
                augmentation_config["random_rotation"]["factor"]
            ),
            layers.experimental.preprocessing.RandomZoom(
                augmentation_config["random_zoom"]["height_factor"],
                augmentation_config["random_zoom"]["width_factor"]
            ),
            layers.experimental.preprocessing.RandomTranslation(
                height_factor=height, width_factor=width,
                fill_mode='reflect', interpolation='bilinear',
            )
        ])
        return data_augmentation

    def build(self) -> None:
        """Build simple CNN model for this peculiar problem. """
        input_shape = tuple(self.config["model"]["input"])
        inputs = keras.Input(shape=input_shape)

        x = self._apply_preprocessing(inputs)

        conv_layers_stack = self.config["model"]["stack"]
        for conv_layer in conv_layers_stack:
            x = layers.Conv2D(
                conv_layer["filters"],
                kernel_size=(3, 3),
                activation="relu",
                padding=conv_layer["padding"],
                strides=conv_layer["strides"],
                name=conv_layer["name"]
            )(x)
            # Note: there is an open discussion in Deep Learning
            # community whether Batch normalization layer should
            # be placed before or after activation function. Because
            # of the following explanation, this proposed architecture
            # has Batch normalization layer applied before activation
            # function.

            # Explanation: Take a look at the answer: (link to the
            # Stackoverflow comment: https://stackoverflow.com/
            # questions/34716454/where-do-i-call-the-batchnormalization-
            # function-in-keras#comment78826470_37979391 ).
            if conv_layer["bn_next"]:
                x = layers.BatchNormalization()(x)
            if conv_layer["relu_next"]:
                x = layers.ReLU()(x)

            if conv_layer["strides"] == 1:
                # We don't want to apply both pooling and
                # stride step of size 2 since it would reduce
                # the spatial size of the image significantly.
                x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        if self.config["model"]["closing_layer"] == "max_pooling":
            x = layers.GlobalMaxPooling2D()(x)
        elif self.config["model"]["closing_layer"] == "flatten":
            x = layers.Flatten()(x)
            x = layers.Dropout(0.7)(x)
        else:
            raise NotImplementedError(
                f"The provided choice {self.config['model']['closing_layer']}"
                f" is not supported / implemented."
            )
        outputs = layers.Dense(
            self.config["model"]["output"],
            activation="softmax"
        )(x)

        self.model = keras.Model(inputs, outputs)

    def train(self) -> dict:
        """Train the model on the provided data. """
        self.model.summary()

        # sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.8)
        self.model.compile(
            # We don't need 'sparse' categorical CE function since our
            # labels are one-hot encoded.
            loss=self.config["train"]["loss"],
            optimizer=self.config["train"]["optimizer"],
            metrics=self.config["train"]["metrics"]
        )

        callbacks = self.initialize_callbacks()

        train_dataset = self.data_loader.get_train_data()
        validation_dataset = self.data_loader.get_validation_data()
        history = self.model.fit(
            train_dataset,
            batch_size=self.config["train"]["batch_size"],
            epochs=self.config["train"]["epochs"],
            validation_data=validation_dataset,
            callbacks=callbacks
        )
        return history.history

    def initialize_callbacks(self):
        """Creates and retrieves callback functions for the upcoming training.

        Instantiated callback functions:
            1. Early stopping
            2. Model checkpointing
            3. Tensor board
        """

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
            os.path.join(
                self.config["train"]["checkpoint_dir"],
                "checkpoint-{epoch:02d}-loss{val_loss:.3f}.h5"
            ),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        tensor_board = keras.callbacks.TensorBoard(log_dir="./logs")
        callbacks = [
            early_stopping_callback,
            model_checkpoint_callback,
            tensor_board
        ]
        return callbacks

    def evaluate(self):
        """Evaluate trained model on test data with corresponding labels."""

        test_dataset = self.data_loader.get_test_data()
        score = self.model.evaluate(test_dataset, verbose=1)
        print(f"Test loss: {score[0]}")
        print(f"Test accuracy: {score[1]}")

    def predict(self):
        pass
