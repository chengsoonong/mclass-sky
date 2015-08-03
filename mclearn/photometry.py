""" Procedures specific to photometric data. """

import os
import numpy as np
from urllib.request import urlopen

def reddening_correction_sfd98(extinction_r):
    """ sf

    """

    E_BV = extinction_r / 2.751
    A_u = E_BV * 5.155
    A_g = E_BV * 3.793
    A_r = E_BV * 2.751
    A_i = E_BV * 2.086
    A_z = E_BV * 1.479

    return (A_u, A_g, A_r, A_i, A_z)


def reddening_correction_sf11(extinction_r):
    """ adf

    """

    E_BV = extinction_r / 2.751
    A_u = E_BV * 4.239
    A_g = E_BV * 3.303
    A_r = E_BV * 2.285
    A_i = E_BV * 1.698
    A_z = E_BV * 1.263

    return (A_u, A_g, A_r, A_i, A_z)

def reddening_correction_w14(extinction_r):
    """ adf

    """

    E_BV = extinction_r / 2.751

    region_2 = np.logical_and(E_BV >= 0.04, E_BV < 0.08)
    region_3 = E_BV >= 0.08

    E_BV[region_2] = E_BV[region_2] + 0.5 * (E_BV[region_2] - 0.04)
    E_BV[region_3] = E_BV[region_3] + 0.02

    A_u = E_BV * 4.305
    A_g = E_BV * 3.288
    A_r = E_BV * 2.261
    A_i = E_BV * 1.714
    A_z = E_BV * 1.263

    return (A_u, A_g, A_r, A_i, A_z)


def correct_magnitudes(data, magnitudes, corrections, suffix):
    """
    """

    for mag, cor in zip(magnitudes, corrections):
        data[mag + suffix] = data[mag] - cor


def compute_colours(data, colours, suffix):
    """
    """
    
    for colour in colours:
        prefix = 'psf' if colour[0].startswith('psf') else 'petro'
        colour_name = prefix + colour[0][-2:] + colour[1][-2:]
        data[colour_name + suffix] = data[colour[0] + suffix] - data[colour[1] + suffix]




def fetch_filter(filter, download_url):
    """
    """
    
    assert filter in 'ugriz'
    url = download_url % filter
    
    if not os.path.exists('data/filters'):
        os.makedirs('data/filters')

    loc = os.path.join('data/filters', '%s.dat' % filter)
    
    if not os.path.exists(loc):
        filter_file = urlopen(url)
        with open(loc, 'wb') as f:
            f.write(filter_file.read())

    with open(loc, 'rb') as f:
        data = np.loadtxt(f)

    return data



def fetch_spectrum(spectrum_url):
    """
    """

    if not os.path.exists('data/spectra'):
        os.makedirs('data/spectra')

    refspec_file = os.path.join('data/spectra', spectrum_url.split('/')[-1])

    if not os.path.exists(refspec_file):
        spectrum_file = urlopen(spectrum_url)
        with open(refspec_file, 'wb') as f:
            f.write(spectrum_file.read())

    with open(refspec_file, 'rb') as f:
        data = np.loadtxt(f)
    
    return data


def clean_up_subclasses(classes, subclasses):
    """
    """

    # remove null references
    subclasses.replace('null', '', inplace=True)

    # remove HD catalog number (stored in brackets)
    subclasses.replace(r'\s*\(\d+\)\s*', '', regex=True, inplace=True)

    # captialise only the first leter of some subclasses
    subclasses.replace('BROADLINE', 'Broadline', inplace=True)
    subclasses.replace('STARFORMING', 'Starforming', inplace=True)
    subclasses.replace('STARBURST', 'Starburst', inplace=True)
    subclasses.replace('STARBURST BROADLINE', 'Starburst Broadline', inplace=True)
    subclasses.replace('AGN BROADLINE', 'AGN Broadline', inplace=True)
    subclasses.replace('STARFORMING BROADLINE', 'Starforming Broadline', inplace=True)

    # remove other brackets
    subclasses.replace('F8V (G_243-63)', 'F8V', inplace=True)
    subclasses.replace('K5 (G_19-24)', 'K5', inplace=True)
    subclasses.replace('sd:F0 (G_84-29)', 'sd:F0', inplace=True)
    subclasses.replace('G0 (G_101-29)', 'G0', inplace=True)
    subclasses.replace('A4 (G_165-39)', 'A4', inplace=True)
    subclasses.replace('A4p (G_37-26)', 'A4p', inplace=True)

    not_empty = subclasses != ''
    subclasses.loc[not_empty] = classes[not_empty] + ' ' + subclasses[not_empty] 
