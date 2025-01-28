import pandas as pd
import numpy  as np

class Dataset:
    def __init__(self, root_dir, xlsx):
        self.df          = pd.read_excel(root_dir + xlsx)
        self.df.columns  = self.df.columns.str.replace(' ', '')
    
    def get_months(self):
        return self.df["Data"   ].to_numpy()
    
    def get_spei(self):
        return self.df["Series1"].to_numpy()
    
    def get_spei_normalized(self):
        spei = self.get_spei()
        
        return ( spei - np.min(spei) ) / ( np.max(spei) - np.min(spei) )