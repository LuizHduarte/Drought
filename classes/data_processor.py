import json
import pandas     as pd
import numpy      as np
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, file_name):
        self.configs_dict   = self._load_config_file(file_name)
    
    def _load_config_file(self, file_name):
        with open(file_name) as file:
            return json.load(file)
        
    def _readXlsx(self, xlsx):
        df = pd.read_excel(xlsx)
        df.columns = df.columns.str.replace(' ', '')

        SpeiValues = df["Series1"].to_numpy()
        SpeiNormalizedValues = (SpeiValues-np.min(SpeiValues))/(np.max(SpeiValues)-np.min(SpeiValues))
        monthValues = df["Data"].to_numpy()

        return SpeiValues, SpeiNormalizedValues, monthValues
    
    def splitSpeiData(self, xlsx, train_size):
        
        SpeiValues, SpeiNormalizedValues, monthValues = self._readXlsx(xlsx)

        speiTrainData, speiTestData, monthTrainData, monthTestData = train_test_split(SpeiNormalizedValues, monthValues, train_size=train_size, shuffle=False)
        split = len(speiTrainData)
        
        return speiTrainData, speiTestData, monthTrainData, monthTestData, split
    
    def cria_IN_OUT(self, data, janela):
        OUT_indices = np.arange(janela, len(data), janela)
        OUT = data[OUT_indices]
        lin_x = len(OUT)
        IN = data[range(janela*lin_x)]
        IN = np.reshape(IN, (lin_x, janela, 1))    
        OUT_final = IN[:,-self.configs_dict['dense_units']:,0]
        IN_final = IN[:,:-self.configs_dict['dense_units'],:]
        return IN_final, OUT_final