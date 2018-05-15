from collections import OrderedDict

from IPython.display import clear_output
import matplotlib.pyplot as plt


def plot_history(history):
    clear_output()
    rows = (len(history) + 1) // 2
    plt.figure(figsize=(4 * rows, 8))
    for i, (key, values) in enumerate(history.items()):
        plt.subplot(rows, 2, i + 1)
        plt.title(key)
        plt.plot([x[0] for x in values], label="values")
        plt.plot([x[1] for x in values], label="smoothed values")
        plt.legend()
    plt.show()

def update_history(history, new_values):
    for key, value in new_values.items():
        history[key].append(value)

def build_history(metrics_names):
    return OrderedDict([(metric_name, []) for metric_name in metrics_names])