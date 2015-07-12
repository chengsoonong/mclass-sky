""" Selected plots commonly used in astronomy and active learning. """

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pandas import DataFrame
from matplotlib.ticker import FuncFormatter

# These are the "Tableau 20" colors as RGB.  
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]  
  
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.  
for i in range(len(tableau20)):  
    r, g, b = tableau20[i]  
    tableau20[i] = (r / 255., g / 255., b / 255.)  


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
    ax.set_xscale('log')
    
    plt.show()
    
    return (fig, ax)



def plot_hex_map(ra, dec, origin=0, title='Distribution of Objects', projection='mollweide', milky_way=True,
                C=None, reduce_C_function=np.mean, vmin=0, vmax=1500, mincnt=1, cmap=plt.cm.bone_r):
    """ Plot density of objects on a map. """
    
    # shift ra values to range [-180, +180]
    ra = np.remainder(ra + 360 - origin, 360)
    ra[ra > 180] -= 360
    
    # reverse scale so that East is to the left
    ra = -ra
    
    # set tick labels to correct values
    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels + 360 + origin, 360)
    
    # plot data on map
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111, projection=projection, axisbg='white')
    hex_quasar = ax.hexbin(np.radians(ra), np.radians(dec), cmap=cmap, mincnt=mincnt,
                           zorder=-1, vmin=vmin, vmax=vmax, C=C, reduce_C_function=reduce_C_function)
    fig.colorbar(hex_quasar)
    ax.set_xticklabels(tick_labels)
    ax.set_title(title)
    ax.set_xlabel('ra')
    ax.set_ylabel('dec')
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    ax.grid(True)
    
    # draw the Milky Way
    if milky_way:
        lons = np.arange(0, 360)
        ra_dec = np.zeros((360, 2))
        for lon in lons:
            gal_coords = ephem.Galactic(np.radians(lon), 0)
            equatorial_coords = ephem.Equatorial(gal_coords)
            ra_dec[lon] = np.degrees(equatorial_coords.get())
        milky_ra = ra_dec[:, 0]
        milky_dec = ra_dec[:, 1]
        milky_ra = np.remainder(milky_ra + 360 - origin, 360)
        milky_ra[milky_ra > 180] -= 360
        milky_ra = -milky_ra
        
        # sort so the line does not loop back
        sort_index = np.argsort(milky_ra)
        milky_ra_sorted = milky_ra[sort_index]
        milky_dec_sorted = milky_dec[sort_index]
        
        ax.plot(np.radians(milky_ra_sorted), np.radians(milky_dec_sorted))
    plt.show()