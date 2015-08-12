""" Selected plots commonly used in astronomy and active learning. """

import mclearn
import pandas as pd
import numpy as np
import ephem
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pandas import DataFrame
from matplotlib.ticker import FuncFormatter

# These are the "Tableau 20" colors as RGB.  
tableau10 = [(214, 39, 40), (31, 119, 180), (44, 160, 44),
             (255, 127, 14), (148, 103, 189), (140, 86, 75),
             (127, 127, 127), (23, 190, 207), (188, 189, 34), (227, 119, 194)]
  
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.  
for i in range(len(tableau10)):  
    r, g, b = tableau10[i]  
    tableau10[i] = (r / 255., g / 255., b / 255.)  


def plot_class_distribution(target, ax=None):
    """ Plot the distribution of the classes.
        
        Parameters
        ----------
        target : array
            The target column of the dataset.
            
        ax : Matplotlib Axes object
            A matplotlib Axes instance.

        Returns
        -------
        ax : Matplotlib Axes object
            The matplotlib Axes instance where the figure is drawn.
    """

    if not ax:
        ax = plt.gca()
    
    counts = DataFrame(target).apply(pd.value_counts)
    counts.plot(ax=ax, kind="bar", fontsize=12, legend=False)
    ax.set_xticklabels(labels=counts.index, rotation=0)
    
    format_thousands = lambda x, pos: format(int(x), ',')
    ax.get_yaxis().set_major_formatter(FuncFormatter(format_thousands))
    ax.xaxis.grid(False)
    
    return ax
    
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
    
    
def plot_balanced_accuracy_violin(balanced_accuracy_samples, ax=None):
    """ Make a violin plot of the balanced posterior accuracy.
        
        Parameters
        ----------
        balanced_accuracy_samples : dict
            Where the keys are the classifier names and the each value is an array
            of sample points from which an empirical pdf can be approxmiated.

        ax : Matplotlib Axes object
            A matplotlib Axes instance.

        Returns
        -------
        ax : Matplotlib Axes object
            The matplotlib Axes instance where the figure is drawn.
    """

    if not ax:
        ax = plt.gca()

    sns.violinplot(data=balanced_accuracy_samples, ax=ax, inner='box')
    
    format_as_percent_plot = lambda x, pos: "{:.2f}%".format(x * 100)
    ax.get_yaxis().set_major_formatter(FuncFormatter(format_as_percent_plot))
    
    return ax
    
    
def plot_learning_curve(sample_sizes, learning_curves, curve_labels, xscale='log', ax=None):
    """ Plot the learning curve.
        
        Parameters
        ----------
        sample_sizes : array
            The sample sizes in which the classifier is run.
            
        learning_curves : array
            List of learning_curves to be plotted
            
        curve_labels : array
            The labels of the learning curves.

        xscale : str
            The scale of the x-axis. Default is 'log'.

        ax : Matplotlib Axes object
            A matplotlib Axes instance.

        Returns
        -------
        ax : Matplotlib Axes object
            The matplotlib Axes instance where the figure is drawn.
    """
    
    if not ax:
        ax = plt.gca()

    for curve, label in zip(learning_curves, curve_labels):
        ax.plot(sample_sizes[:len(curve)], curve, label=label)

    format_as_percent_plot = lambda x, pos: "{:.1f}%".format(x * 100)
    ax.get_yaxis().set_major_formatter(FuncFormatter(format_as_percent_plot))
    ax.legend(loc='lower right', frameon=True)
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Balanced Accuracy Rate')
    ax.set_xscale(xscale)
    ax.grid(False)
    
    return ax


def plot_average_learning_curve(sample_sizes, learning_curves, curve_labels, no_trials=10, ax=None):
    """ Plot the average learning curve from many trials.

        Parameters
        ----------
        sample_sizes : array
            The sample sizes in which the classifier is run.
            
        learning_curves : array
            List of learning_curves to be plotted
            
        curve_labels : array
            The labels of the learning curves.

        no_trials : int
            The number of trials that were run for each learning curve.

        ax : Matplotlib Axes object
            A matplotlib Axes instance.

        Returns
        -------
        ax : Matplotlib Axes object
            The matplotlib Axes instance where the figure is drawn.

    """

    mean_curves = []
    for learning_curve in learning_curves:
        learning_curve = np.array(learning_curve)
        mean_curve = np.zeros(len(sample_sizes))
        for i in range(len(sample_sizes)):
            mean_curve[i] = np.mean(learning_curve[:, i])
        mean_curves.append(mean_curve)

    if not ax:
        ax = plt.gca()

    for mean_curve, curve_label in zip(mean_curves, curve_labels):
        ax.plot(sample_sizes, mean_curve, label=curve_label)

    ax.set_xlabel('Number of Training Examples')
    ax.set_ylabel('Balanced Accuracy Rate')
    ax.legend(loc='lower right', frameon=True)
    ax.grid(False)

    return ax


def plot_hex_map(ra, dec, origin=180, title=None, projection='mollweide', gridsize=100,
    milky_way=True, C=None, reduce_C_function=np.mean, vmin=0, vmax=1500, mincnt=1,
    cmap=plt.cm.bone_r, axisbg='white', colorbar=True, labels=False, ax=None):
    """ Plot the density of objects on a hex map.

        Parameters
        ----------
        ra : array
            The array containing the ra coordinates.
            
        dec : array
            The array containing the dec coordinates.

        origin : int
            The ra value in the middle of the map.
            
        title : str
            The title of the plot.

        projection : str
            The projection mode to be used. Default is 'mollweide'.

        gridsize : int
            The number of hexagons in the *x*-direction, default is
            100. The corresponding number of hexagons in the
            *y*-direction is chosen such that the hexagons are
            approximately regular. Alternatively, gridsize can be a
            tuple with two elements specifying the number of hexagons
            in the *x*-direction and the *y*-direction.

        milky_way : boolean
            Whether the plane of the Milky Way is plotted. Default is True.

        C : array
            If C is specified, it specifies values at the coordinate (ra[i],dec[i]).
            These values are accumulated for each hexagonal bin and then reduced according
            to reduce_C_function, which defaults to numpy’s mean function (np.mean).
            (If C is specified, it must also be a 1-D sequence of the same length as ra and dec.)

        reduce_C_function : function
            The function to be applied to the C values (or the count values) on each
            hexagon bin.

        vmin : scalar
            vmin is the value that sits at the bottom end of the colour bar.
            If None, the min of array C is used.

        vmax : scalar
            vmax is the value that sits at the top end of the colour bar.
            If None, the max of array C is used.

        mincnt : int
            If not None, only display cells with more than mincnt number of points in the cell.

        cmap : Colormap
            a matplotlib.colors.Colormap instance.

        axisbg : str
            The background colour of the map. Default is 'white'.

        colorbar : boolean
            Whether to render the color bar (i.e. legend). Default is True.

        labels : boolean
            Whether to render the axis labels. Default to False (to avoid clutter).

        Parameters
        ----------
        ax : matplotlib Axes
            Returns the Axes object with the hex map drawn onto it.

    """
    
    # shift ra values to range [-180, +180]
    ra = np.remainder(ra + 360 - origin, 360)
    ra[ra > 180] -= 360
    
    # reverse scale so that East is to the left
    ra = -ra
    
    # set tick labels to correct values
    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels + 360 + origin, 360)
    
    # plot data on map
    if not ax:
        ax = plt.gca(projection=projection, axisbg=axisbg)
    hex_quasar = ax.hexbin(np.radians(ra), np.radians(dec), gridsize=gridsize, cmap=cmap, mincnt=mincnt,
                           zorder=-1, vmin=vmin, vmax=vmax, C=C, reduce_C_function=reduce_C_function)
    if colorbar:
        plt.gcf().colorbar(hex_quasar)
    
    if title:
        ax.set_title(title)

    if labels:
        ax.set_xlabel('ra')
        ax.set_ylabel('dec')
        ax.set_xticklabels(tick_labels)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
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
    
    return ax

def plot_recall_maps(coords_test, y_test, y_pred_test, class_names, output,
    correct_boolean, vmin=0, vmax=1, mincnt=None, cmap=plt.cm.YlGn):
    """ Plot the recall map.

        Parameters
        ----------
        coords_test : array
            The ra and dec coordinates

        y_test : array
            The column of predicted values.

        y_pred_test : array
            The column of predicted values.

        class_names = array
            Names of the target (e.g. Galaxy, Star, Quasar)

        output : str
            The suffix on the saved figure.

        correct_boolean : array
            A boolean array indicating whehter a test exmaple was correctly predicted.

        vmin : scalar
            vmin is the value that sits at the bottom end of the colour bar.
            If None, the min of array C is used.

        vmax : scalar
            vmax is the value that sits at the top end of the colour bar.
            If None, the max of array C is used.

        mincnt : int
            If not None, only display cells with more than mincnt number of points in the cell.

        cmap : Matplotlib ColorMap object
            The color scheme to be used.
    """

    
    C_func = lambda c: np.sum(c) / len(c) if c else 0

    is_class = {}
    for class_name in class_names:
        is_class[class_name] = y_test == class_name
        ra = coords_test[:,0][is_class[class_name]]
        dec = coords_test[:,1][is_class[class_name]]
        C = correct_boolean[is_class[class_name]]

        fig = plt.figure(figsize=(10,5))
        ax = plot_hex_map(ra, dec, C=C, reduce_C_function=C_func,
             vmin=vmin, vmax=vmax, mincnt=mincnt, cmap=cmap)

        file_name = r'plots/map_recall_' + output + r'_' + class_name + r'.png'
        fig.savefig(file_name, bbox_inches='tight', dpi=300)


def plot_filters_and_spectrum(filter_url, spectrum_url, ax=None):
    """ Plot ugriz filters and spectrum in the same figure.

        filter_url : str
            The url where the ugriz filters can be obtained.

        spectrum_url : str
            The url where the spectrum data can be obtained.

        ax : Matplotlib Axes object
            A matplotlib Axes instance.

        Returns
        -------
        ax : Matplotlib Axes object
            The matplotlib Axes instance where the figure is drawn.
    """
    
    if not ax:
        ax = plt.gca()
        
    Xref = mclearn.photometry.fetch_spectrum(spectrum_url)
    Xref[:, 1] /= 2.1 * Xref[:, 1].max()
    
    ax.plot(Xref[:, 0], Xref[:, 1], '-k', lw=1)

    for f,c in zip('ugriz', 'bgrmk'):
        X = mclearn.photometry.fetch_filter(f, filter_url)
        plt.fill(X[:, 0], X[:, 1], ec=c, fc=c, alpha=0.4)

    kwargs = dict(fontsize=20, ha='center', va='center', alpha=0.5)
    ax.text(3500, 0.02, 'u', color='b', **kwargs)
    ax.text(4600, 0.02, 'g', color='g', **kwargs)
    ax.text(6100, 0.02, 'r', color='r', **kwargs)
    ax.text(7500, 0.02, 'i', color='m', **kwargs)
    ax.text(8800, 0.02, 'z', color='k', **kwargs)

    ax.set_xlim(3000, 11000)

    #ax1a.set_title('SDSS Filters and Reference Spectrum')
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Normalised Flux / Filter Transmission')
    ax.tick_params(top='off', right='off')
    ax.grid(False)

    return ax

def plot_scatter_with_classes(data, targets, classes, size=2, alpha=0.01,
    scatterpoints=1000, ax=None):
    """ Plot a scater plot of the classes.

        data : array
            The target array.

        targets : array
            The list of class names used in the target array.

        ax : Matplotlib Axes object
            A matplotlib Axes instance.

        Returns
        -------
        ax : Matplotlib Axes object
            The matplotlib Axes instance where the figure is drawn.
    """

    if not ax:
        ax = plt.gca()

    class_data = {}
    cls_scatters = []
    for i, cls in enumerate(classes):
        class_data[cls] = data[targets == cls]
        cls_scatter = ax.scatter(class_data[cls][:,0], class_data[cls][:,1], s=size,
            alpha=alpha, c=tableau10[i], label=cls)
        cls_scatters.append(cls_scatter)

    ax.legend(cls_scatters, classes, scatterpoints=scatterpoints, loc='upper right',
        frameon=True, ncol=1)
    ax.grid(False)

    return ax


def reshape_grid_socres(grid_scores, row_length, col_length, transpose=False):
    """ Reshape the scores to be used as input for the heathap.

        grid_scores : array
            The grid scores obtain from the GridSearch insteance.

        row_length : int
            The width of the heatmap.

        col_length : int
            The height of the heatmap.

        transpose : boolean
            Whether to tranpose the heatmap (e.g. for easier viewing).

        Returns
        -------
        scores : array
            The array of scores, shaped appropriately.
    """

    scores = [x[1] for x in grid_scores]
    scores = np.array(scores).reshape(row_length, col_length)
    if transpose:
        scores = scores.transpose()
    return scores



def plot_validation_accuracy_heatmap(scores, x_range=None, y_range=None,
    x_label=None, y_label=None, power10='both', ax=None):
    """ Plot heatmap of the validation accuracy from a grid search.

        Parameters
        ----------
        scores : array
            List of scores that has been shaped appropriately.

        x_range : array or None
            The range on the x-axis which will replace the default numbering.

        y_range : array or None
            The range on the y-axis which will replace the default numbering.

        x_label : str
            Label of the x-axis

        y_label : str
            Label of the y-axis

        power10 : 'x' or 'y' or 'both'
            Whether to format the numbering on the axes as powers of 10.

        ax : Matplotlib Axes object
            A matplotlib Axes instance.

        Returns
        -------
        ax : Matplotlib Axes object
            The matplotlib Axes instance where the figure is drawn.
    """
    
    if not ax:
        ax = plt.gca()
    
    heat_ax = ax.imshow(scores, interpolation='nearest', cmap=plt.cm.summer)
    plt.colorbar(heat_ax)

    format_power = lambda x, pos, p_range: "$10^{%d}$" % int(np.log10(p_range[pos]))

    if power10 == 'x' or power10 == 'both':
        plt.xticks(np.arange(len(x_range)), x_range, rotation=45)
        formatter = FuncFormatter(lambda x, pos: format_power(x, pos, x_range))
        ax.xaxis.set_major_formatter(formatter)

    if power10 == 'y' or power10 == 'both':
        plt.yticks(np.arange(len(y_range)), y_range)
        formatter = FuncFormatter(lambda x, pos: format_power(x, pos, y_range))
        ax.yaxis.set_major_formatter(formatter)
    
    if x_label:
        ax.set_xlabel(x_label)
    
    if y_label:
        ax.set_ylabel(y_label)
    
    ax.grid(False)

    return ax

