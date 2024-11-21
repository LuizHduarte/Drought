import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

with open("./NeuralNetwork/modelConfig.json") as arquivo:
    dados_json = json.load(arquivo)

PARCEL_DATA_TRAIN = dados_json['parcelDataTrain' ]
PREDICTION_POINTS = dados_json['predictionPoints']

def readXlsx(XLSX):
    df = pd.read_excel(XLSX)
    df.columns = df.columns.str.replace(' ', '')

    SPEI_VALUES  = df['Series1'].to_numpy()
    MONTH_VALUES = df['Data'   ].to_numpy()

    SPEI_NORMALIZED_VALUES = (SPEI_VALUES-np.min(SPEI_VALUES))/(np.max(SPEI_VALUES)-np.min(SPEI_VALUES))

    return SPEI_VALUES, SPEI_NORMALIZED_VALUES, MONTH_VALUES

def splitSpeiData(XLSX): 
    SPEI_VALUES, SPEI_NORMALIZED_VALUES, MONTH_VALUES = readXLSX(XLSX)

    SPEI_TRAIN_DATA, SPEI_TEST_DATA, MONTH_TRAIN_DATA, MONTH_TEST_DATA = train_test_split(SPEI_NORMALIZED_VALUES, MONTH_VALUES, train_size=PARCEL_DATA_TRAIN, shuffle=False)
    split = len(SPEI_TRAIN_DATA)
    
    return SPEI_TRAIN_DATA, SPEI_TEST_DATA, MONTH_TRAIN_DATA, MONTH_TEST_DATA, split

def cria_IN_OUT(data, WINDOW_SIZE):
    NUM_WINDOWS = len(data)//WINDOW_SIZE
    
    data        = data[:(WINDOW_SIZE * NUM_WINDOWS)]
    data        = np.reshape(data, (NUM_WINDOWS, WINDOW_SIZE, 1))    

    INPUT       = data[:,:-PREDICTION_POINTS,:]
    OUTPUT      = data[:,-PREDICTION_POINTS:,0]
    
    return INPUT, OUTPUT
