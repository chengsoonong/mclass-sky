""" Visualisations

"""

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pandas import DataFrame
from matplotlib.ticker import FuncFormatter


def plot_class_distribution(targets, title):
    """ Input is a target column of the DataFrame """
    
    target_df = DataFrame(targets)
    fig, ax = plt.subplots(figsize=(9, 5))
    counts = target_df.apply(pd.value_counts)
    counts.plot(ax=ax, kind="bar", fontsize=12, legend=False)
    ax.set_title(title)
    ax.set_xticklabels(labels=counts.index, rotation=0)
    
    format_thousands = lambda x, pos: format(int(x), ',')
    ax.get_yaxis().set_major_formatter(FuncFormatter(format_thousands))
    
    
def plot_balanced_accuracy_expected(results, title, classifier_names):
    """ Results are dict"""
    
    results = DataFrame(results, index=["Expected Balanced Accuracy"])
    results = results.reindex(columns=classifier_names)
    
    format_as_percent_plot = lambda x, pos: "{:.0f}%".format(x * 100)
    fig, ax = plt.subplots(figsize=(9, 5))
    results.plot(ax=ax, kind="bar", title=title, fontsize=12)
    ax.legend(bbox_to_anchor = (1.5, 0.6))
    ax.set_xticklabels([], rotation=0)
    ax.get_yaxis().set_major_formatter(FuncFormatter(format_as_percent_plot))
    
    
    
def plot_balanced_accuracy_violin(balanced_accuracy_samples, names):
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.violinplot(DataFrame(balanced_accuracy_samples, columns=names), ax=ax,
                   names=[1, 2, 3, 4, 5, 6, 7])
    ax.set_title("Posterior Balanced Accuracy")
    
    format_as_percent_plot = lambda x, pos: "{:.0f}%".format(x * 100)
    ax.get_yaxis().set_major_formatter(FuncFormatter(format_as_percent_plot))
    
    handles = []
    colours = sns.color_palette("husl", 7)
    
    for c, n in zip(colours, names):
        handles.append(mpatches.Patch(color=c, label=n))
    
    ax.legend(handles=handles, bbox_to_anchor = (1.4, 0.6))