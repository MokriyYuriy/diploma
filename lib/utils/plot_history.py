from collections import OrderedDict

from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SETTINGS = dict(
    smoothed=True,
    xlabel="iterations"
)


def plot_history(history):
    clear_output()
    rows = (len(history) + 1) // 2
    plt.figure(figsize=(14, 4 * rows))
    for i, (key, value) in enumerate(history.items()):
        values, settings = value
        plt.subplot(rows, 2, i + 1)
        plt.title(key)
        plt.xlabel(settings.get("xlabel", DEFAULT_SETTINGS.get("xlabel")))
        plt.plot(values, label="values")
        if settings.get("smoothed", DEFAULT_SETTINGS.get("smoothed")):
            plt.plot([np.mean(values[i:i+30]) for i in range(len(values) - 30)], label="smoothed values")
            plt.legend()
    plt.show()

def update_history(history, new_values):
    for key, value in new_values.items():
        history[key][0].append(value)

def build_history(metrics_names):
    return OrderedDict([(metric_name, ([], settings)) for metric_name, settings in metrics_names])