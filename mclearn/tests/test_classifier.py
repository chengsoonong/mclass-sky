import mclearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression


class TestNormalise(object):
    @classmethod
    def setup_class(cls):
        cls.sdss = pd.io.parsers.read_csv("mclearn/tests/data/sdss_tiny.csv")
        cls.uncorrected_features = ['psfMag_u', 'psfMag_g', 'psfMag_r', 'psfMag_i', 'psfMag_z', 'petroMag_u',
                                    'petroMag_g', 'petroMag_r', 'petroMag_i', 'petroMag_z', 'petroRad_r']
        cls.target_col = 'class'
 

    def test_plot_class_distribution(self):
        fig = plt.figure(figsize=(10, 5))
        ax = mclearn.viz.plot_class_distribution(self.sdss['class'])


    def test_plot_hex_map(self):
        fig = plt.figure(figsize=(10,5))
        ax = mclearn.viz.plot_hex_map(self.sdss['ra'], self.sdss['dec'], vmax=5)

    def test_dust_extinction(self):
        # compute the three sets of reddening correction
        sfd98_corrections = mclearn.photometry.reddening_correction_sfd98(self.sdss['extinction_r'])
        sf11_corrections = mclearn.photometry.reddening_correction_sf11(self.sdss['extinction_r'])
        w14_corrections = mclearn.photometry.reddening_correction_w14(self.sdss['extinction_r'])

        # column names of the magnitudes to be corrected
        psf_magnitudes = ['psfMag_u', 'psfMag_g', 'psfMag_r', 'psfMag_i', 'psfMag_z']
        petro_magnitudes = ['petroMag_u', 'petroMag_g', 'petroMag_r', 'petroMag_i', 'petroMag_z']

        # calculate the corrected magnitudes
        mclearn.photometry.correct_magnitudes(self.sdss, psf_magnitudes, sfd98_corrections, '_sfd98')
        mclearn.photometry.correct_magnitudes(self.sdss, petro_magnitudes, sfd98_corrections, '_sfd98')
        mclearn.photometry.correct_magnitudes(self.sdss, psf_magnitudes, sf11_corrections, '_sf11')
        mclearn.photometry.correct_magnitudes(self.sdss, petro_magnitudes, sf11_corrections, '_sf11')
        mclearn.photometry.correct_magnitudes(self.sdss, psf_magnitudes, w14_corrections, '_w14')
        mclearn.photometry.correct_magnitudes(self.sdss, petro_magnitudes, w14_corrections, '_w14')



