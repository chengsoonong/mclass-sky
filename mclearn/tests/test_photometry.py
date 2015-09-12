import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from mclearn.photometry import (reddening_correction_sfd98,
                                reddening_correction_sf11,
                                reddening_correction_w14,
                                correct_magnitudes)
from .datasets import Dataset


class TestPhotometry:
    @classmethod
    def setup_class(cls):
        cls.sdss = Dataset('sdss_tiny')


    def test_dust_extinction(self):
        # compute the three sets of reddening correction
        sfd98_corrections = reddening_correction_sfd98(self.sdss.data['extinction_r'])
        sf11_corrections = reddening_correction_sf11(self.sdss.data['extinction_r'])
        w14_corrections = reddening_correction_w14(self.sdss.data['extinction_r'])

        # column names of the magnitudes to be corrected
        psf_magnitudes = ['psfMag_u', 'psfMag_g', 'psfMag_r', 'psfMag_i', 'psfMag_z']
        petro_magnitudes = ['petroMag_u', 'petroMag_g', 'petroMag_r', 'petroMag_i', 'petroMag_z']

        # calculate the corrected magnitudes
        correct_magnitudes(self.sdss.data, psf_magnitudes, sfd98_corrections, '_sfd98')
        correct_magnitudes(self.sdss.data, petro_magnitudes, sfd98_corrections, '_sfd98')
        correct_magnitudes(self.sdss.data, psf_magnitudes, sf11_corrections, '_sf11')
        correct_magnitudes(self.sdss.data, petro_magnitudes, sf11_corrections, '_sf11')
        correct_magnitudes(self.sdss.data, psf_magnitudes, w14_corrections, '_w14')
        correct_magnitudes(self.sdss.data, petro_magnitudes, w14_corrections, '_w14')


    def test_plot_filters_and_spectrum(self):
        vega_url = 'http://www.astro.washington.edu/users/ivezic/DMbook/data/1732526_nic_002.ascii'
        ugriz_url = 'http://www.sdss.org/dr7/instruments/imager/filters/%s.dat'
        spectra_dir = 'mclearn/tests/data/spectra/'
        filter_dir = 'mclearn/tests/data/filters/'
    





