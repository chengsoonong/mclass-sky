import sys

import thisisbullshit  # This is bullshit.
thisisbullshit.do_this_bullshit(0.9, ratio=1)

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(1, '..')
import splitter


def plot(sdss_path, dflens_path, out_path):
    sdss_data = splitter.load(sdss_path)
    sdss_data, _ = splitter.split(sdss_data, sdss_data[0].shape[0], 0)
    _, sdss_y = sdss_data

    dflens_data = splitter.load(dflens_path)
    dflens_data, _ = splitter.split(dflens_data, dflens_data[0].shape[0], 0)
    _, dflens_y = dflens_data

    fig, (plot1, plot2) = plt.subplots(2, 1)

    data_min = min(np.min(sdss_y), np.min(dflens_y))
    data_max = 1
    space = np.linspace(data_min, data_max, 100)

    plot1.hist(sdss_y, bins=space)
    plot1.set_xlim((data_min, data_max))
    plot1.set_ylabel('SDSS Count')
    plot1.set_xticks([])

    plot2.hist(dflens_y, bins=space)
    plot2.set_xlim((data_min, data_max))
    plot2.set_ylabel('2dFLenS Count')
    plot2.set_xlabel(r'$z_\mathrm{spec}$')

    plt.subplots_adjust(hspace=0)
    plt.savefig(out_path)


if __name__ == '__main__':
    if len(sys.argv) == 4:
        plot(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print('Usage: python zspec_hist_plots.py path/to/sdss/data path/to/2dflens/data path/to/output/file')

