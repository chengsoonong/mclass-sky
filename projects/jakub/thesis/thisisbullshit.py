import sys, os
sys.path.insert(1, '/usr/local/texlive/2016/bin/universal-darwin/')
if '/usr/local/texlive/2016/bin/universal-darwin/' not in os.environ['PATH']:
    os.environ['PATH'] = '/usr/local/texlive/2016/bin/universal-darwin/:' + os.environ['PATH']

import matplotlib
import numpy

# http://bkanuka.com/articles/native-latex-plots/
def figsize(scale, ratio=(numpy.sqrt(5.0)-1.0)/2.0):
    fig_width_pt = 418.37225
    inches_per_pt = 1.0/72.27
    fig_width = fig_width_pt*inches_per_pt*scale
    fig_height = fig_width*ratio
    fig_size = [fig_width,fig_height]
    return fig_size

def do_this_bullshit(scale, ratio=None, top=.01, right=.01, left=.15, bottom=.15):
    pgf_with_latex = {
        "pgf.texsystem": "xelatex",
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ['Palatino'],
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.figsize": figsize(scale) if ratio is None else figsize(scale, ratio),
        "figure.subplot.top": 1 - (top / scale),
        "figure.subplot.right": 1 - (right / scale),
        "figure.subplot.bottom": bottom / scale,
        "figure.subplot.left": left / scale,
        "pgf.preamble": []
    }
    matplotlib.rcParams.update(pgf_with_latex)
