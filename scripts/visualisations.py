""" Visualisations """

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pandas import DataFrame
from matplotlib.ticker import FuncFormatter


def plot_class_distribution(targets, title):
    """ Plot the distribution of the classes.
        
        Parameters
        ----------
        target : array
            The target column of the dataset.
            
        title : str
            Title of the plot.
    """
    
    target_df = DataFrame(targets)
    fig, ax = plt.subplots(figsize=(9, 5))
    counts = target_df.apply(pd.value_counts)
    counts.plot(ax=ax, kind="bar", fontsize=12, legend=False)
    ax.set_title(title)
    ax.set_xticklabels(labels=counts.index, rotation=0)
    
    format_thousands = lambda x, pos: format(int(x), ',')
    ax.get_yaxis().set_major_formatter(FuncFormatter(format_thousands))
    
    plt.show()
    
    
def plot_scores(scores, title, x_label, classifier_names):
    """ Make a barplot of the scores of some performance measure.
        
        Parameters
        ----------
        scores : dict
            Where the keys are the classifier names and the values are the scores.
        
        title : str
            Title of the plot.
            
        x_label : str
            Label for the x-axis
            
        classifier_names : array
            List of the names of the classifiers, the order of which will be used
            to order the bars.
    """
    
    scores = DataFrame(scores, index=[x_label])
    scores = scores.reindex(columns=classifier_names)
    
    format_as_percent_plot = lambda x, pos: "{:.0f}%".format(x * 100)
    fig, ax = plt.subplots(figsize=(9, 5))
    scores.plot(ax=ax, kind="bar", title=title, fontsize=12)
    ax.legend(bbox_to_anchor = (1.5, 0.6))
    ax.set_xticklabels([], rotation=0)
    ax.get_yaxis().set_major_formatter(FuncFormatter(format_as_percent_plot))
    
    plt.show()
    
    
def plot_balanced_accuracy_violin(balanced_accuracy_samples, classifier_names):
    """ Make a violin plot of the balanced posterior accuracy.
        
        Parameters
        ----------
        balanced_accuracy_samples : dict
            Where the keys are the classifier names and the each value is an array
            of sample points from which an empirical pdf can be approxmiated.
            
        classifier_names : array
            List of classifier names, the order of which will be used
            to order the bars.
    """

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.violinplot(DataFrame(balanced_accuracy_samples, columns=classifier_names),
                   inner=None, ax=ax, names=[1, 2, 3, 4, 5, 6, 7])
    ax.set_title("Posterior Balanced Accuracy")
    
    format_as_percent_plot = lambda x, pos: "{:.0f}%".format(x * 100)
    ax.get_yaxis().set_major_formatter(FuncFormatter(format_as_percent_plot))
    
    handles = []
    colours = sns.color_palette("husl", 7)
    
    for c, n in zip(colours, classifier_names):
        handles.append(mpatches.Patch(color=c, label=n))
    
    ax.legend(handles=handles, bbox_to_anchor = (1.4, 0.6))
    
    plt.show()
    
    
def plot_learning_curve(sample_sizes, scores, title):
    """ Plot the learning curve.
        
        Parameters
        ----------
        sample_sizes : array
            The sample sizes in which the classifier is run.
            
        scores : array
            The corresponding score for each sample size.
            
        title : str
            The title of the plot.
    """
    
    fig, ax = plt.subplots(figsize=(9, 5))
    plt.plot(sample_sizes, scores)
    ax.set_title(title)
    
    format_as_percent_plot = lambda x, pos: "{:.1f}%".format(x * 100)
    ax.get_yaxis().set_major_formatter(FuncFormatter(format_as_percent_plot))
    
    format_thousands = lambda x, pos: format(int(x), ',')
    ax.get_xaxis().set_major_formatter(FuncFormatter(format_thousands))
    
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Mean Balanced Accuracy')
    
    plt.show()

