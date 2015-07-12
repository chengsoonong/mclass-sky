""" Procedures specific to photometric data. """

def dust_extinction_w14(sdss):
    """ Correct for dust extinction using the W14 set.
    """

    sdss['E_BV'] = sdss['extinction_r'] / 2.751
    E_region_2 = np.logical_and(sdss['E_BV'] >= 0.04, sdss['E_BV'] < 0.08)
    E_region_3 = sdss['E_BV'] >= 0.08

    sdss['Ecorr_BV'] = sdss['E_BV']
    sdss['Ecorr_BV'].loc[E_region_2] = sdss['E_BV'][E_region_2] + 0.5 * (sdss['E_BV'][E_region_2] - 0.04)
    sdss['Ecorr_BV'].loc[E_region_3] = sdss['E_BV'][E_region_3] + 0.02

    sdss['A_u_w14'] = sdss['Ecorr_BV'] * 4.305
    sdss['A_g_w14'] = sdss['Ecorr_BV'] * 3.288
    sdss['A_r_w14'] = sdss['Ecorr_BV'] * 2.261
    sdss['A_i_w14'] = sdss['Ecorr_BV'] * 1.714
    sdss['A_z_w14'] = sdss['Ecorr_BV'] * 1.263

    sdss['psfMag_u_w14'] = sdss['psfMag_u'] - sdss['A_u_w14']
    sdss['psfMag_g_w14'] = sdss['psfMag_g'] - sdss['A_g_w14']
    sdss['psfMag_r_w14'] = sdss['psfMag_r'] - sdss['A_r_w14']
    sdss['psfMag_i_w14'] = sdss['psfMag_i'] - sdss['A_i_w14']
    sdss['psfMag_z_w14'] = sdss['psfMag_z'] - sdss['A_z_w14']

    sdss['petroMag_u_w14'] = sdss['petroMag_u'] - sdss['A_u_w14']
    sdss['petroMag_g_w14'] = sdss['petroMag_g'] - sdss['A_g_w14']
    sdss['petroMag_r_w14'] = sdss['petroMag_r'] - sdss['A_r_w14']
    sdss['petroMag_i_w14'] = sdss['petroMag_i'] - sdss['A_i_w14']
    sdss['petroMag_z_w14'] = sdss['petroMag_z'] - sdss['A_z_w14']

    sdss['psf_u_g_w14'] = sdss['psfMag_u_w14'] - sdss['psfMag_g_w14']
    sdss['psf_g_r_w14'] = sdss['psfMag_g_w14'] - sdss['psfMag_r_w14']
    sdss['psf_r_i_w14'] = sdss['psfMag_r_w14'] - sdss['psfMag_i_w14']
    sdss['psf_i_z_w14'] = sdss['psfMag_i_w14'] - sdss['psfMag_z_w14']

    sdss['petro_u_g_w14'] = sdss['petroMag_u_w14'] - sdss['petroMag_g_w14']
    sdss['petro_g_r_w14'] = sdss['petroMag_g_w14'] - sdss['petroMag_r_w14']
    sdss['petro_r_i_w14'] = sdss['petroMag_r_w14'] - sdss['petroMag_i_w14']
    sdss['petro_i_z_w14'] = sdss['petroMag_i_w14'] - sdss['petroMag_z_w14']

