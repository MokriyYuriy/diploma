from collections import OrderedDict

from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np


def plot_history(history):
    clear_output()
    rows = (len(history) + 1) // 2
    plt.figure(figsize=(10, 4 * rows))
    for i, (key, values) in enumerate(history.items()):
        plt.subplot(rows, 2, i + 1)
        plt.title(key)
        plt.plot(values, label="values")
        plt.plot([np.mean(values[i:i+30]) for i in range(len(values) - 30)], label="smoothed values")
        plt.legend()
    plt.show()

def update_history(history, new_values):
    for key, value in new_values.items():
        history[key].append(value)

def build_history(metrics_names):
    return OrderedDict([(metric_name, []) for metric_name in metrics_names])