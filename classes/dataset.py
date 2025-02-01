import pandas as pd
from sklearn.model_selection import train_test_split

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

    def train_test_split(self, train_size):
        months          = self.get_months         ()
        spei_normalized = self.get_spei_normalized()

        (spei_for_training, spei_for_testing,
         months_for_training, months_for_testing) = train_test_split(spei_normalized, months, train_size=train_size, shuffle=False)
        
        split_position = len(spei_for_training)
        
        return (spei_for_training, spei_for_testing, months_for_training, months_for_testing, split_position)