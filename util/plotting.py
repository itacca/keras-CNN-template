from enum import Enum
from typing import Dict, Optional

from matplotlib import pyplot as plt

from json_handling import load_json, save_json


class PlotType(Enum):
    LOSS = "LOSS"
    ACCURACY = "ACCURACY"


def plot_learning_curves(
        history: Optional[Dict],
        plot_type: PlotType,
        window_title: str = "Learning curves",
        save_path: str = "history.json",
        training_history_path: str = "history.json"
):
    """Plot the learning curve for either loss or accuracy."""
    if history:
        save_json(history, save_path)
    else:
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
