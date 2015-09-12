from pandas import read_csv

class Dataset:
    def __init__(self, name):
        self.data_folder = 'mclearn/tests/data/'

        if name == 'glass':
            self.feature_cols = ['ri', 'na', 'mg', 'ai', 'si', 'k', 'ca' , 'ba', 'fe']
            self.target_col = 'type'

        elif name == 'sdss_tiny':
            self.feature_cols = ['psfMag_u', 'psfMag_g', 'psfMag_r', 'psfMag_i', 'psfMag_z',
                                 'petroMag_u', 'petroMag_g', 'petroMag_r', 'petroMag_i',
                                 'petroMag_z', 'petroRad_r']
            self.target_col = 'class'
            
        elif name == 'wine':
            self.feature_cols = ['alcohol', 'malic_acid', 'ash', 'ash_alcalinity',
                                 'magnesium', 'phenols',  'flavanoids', 'nonfavanoid_phenols',
                                 'proanthocyanins', 'color', 'hue', 'od280', 'proline'] 
            self.target_col = 'class'

        else:
            raise ValueError('Dataset is not yet defined.')

        self.path = self.data_folder + name + '.csv'
        self.data = read_csv(self.path)
        self.features = self.data[self.feature_cols]
        self.target = self.data[self.target_col]