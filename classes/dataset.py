import pandas as pd

class Dataset:
    def __init__(self, root_dir, xlsx):
        self.df = pd.read_excel(root_dir + xlsx)
        self.df.rename(columns={'Series 1': 'SPEI Real'}, inplace=True)

    def get_months(self):
        return self.df.index.to_numpy()
    
    def get_spei(self):
        return self.df['SPEI Real'].to_numpy()
    
    def get_spei_normalized(self):
        spei = self.get_spei()
        return ( (spei - spei.min()) / (spei.max() - spei.min()) )