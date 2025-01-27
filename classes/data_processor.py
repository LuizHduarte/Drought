import pandas     as pd
import numpy      as np
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self):
        pass
    
    def _readXlsx(self, xlsx):
        df = pd.read_excel(xlsx)
        df.columns = df.columns.str.replace(' ', '')

        SpeiValues = df["Series1"].to_numpy()
        SpeiNormalizedValues = (SpeiValues-np.min(SpeiValues))/(np.max(SpeiValues)-np.min(SpeiValues))
        monthValues = df["Data"].to_numpy()

        return SpeiValues, SpeiNormalizedValues, monthValues
    
    def _splitSpeiData(self, xlsx, train_size):
        
        SpeiValues, SpeiNormalizedValues, monthValues = self._readXlsx(xlsx)

        speiTrainData, speiTestData, monthTrainData, monthTestData = train_test_split(SpeiNormalizedValues, monthValues, train_size=train_size, shuffle=False)
        split = len(speiTrainData)
        
        return speiTrainData, speiTestData, monthTrainData, monthTestData, split
    
    def _cria_IN_OUT(self, data, janela, dense_units):
        OUT_indices = np.arange(janela, len(data), janela)
        OUT = data[OUT_indices]
        lin_x = len(OUT)
        IN = data[range(janela*lin_x)]
        IN = np.reshape(IN, (lin_x, janela, 1))    
        OUT_final = IN[:,-dense_units:,0]
        IN_final = IN[:,:-dense_units,:]
        return IN_final, OUT_final