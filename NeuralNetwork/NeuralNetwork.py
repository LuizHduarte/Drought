import tensorflow as tf
import matplotlib.pyplot as plt
import json

from NeuralNetwork.DataProcess import splitSpeiData, cria_IN_OUT
from NeuralNetwork.VisualRepresentation import showPredictionResults, showPredictionsDistribution, showSpeiData, showSpeiTest
from NeuralNetwork.Metrics import getError

# Abra o arquivo JSON
with open("./NeuralNetwork/modelConfig.json") as arquivo:
    dados_json = json.load(arquivo)

totalPoints      = dados_json['totalPoints']
predictionPoints = dados_json['predictionPoints']
numberOfEpochs   = dados_json['numberOfEpochs']
hiddenUnits      = dados_json['hiddenUnits']

def createNeuralNetwork(hidden_units, dense_units, input_shape, activation):
    model = tf.keras.Sequential()   
    model.add(tf.keras.Input(shape=input_shape))
    model.add(tf.keras.layers.LSTM(hidden_units,activation=activation[0]))
    model.add(tf.keras.layers.Dense(units=dense_units,activation=activation[1]))
    model.add(tf.keras.layers.Dense(units=dense_units,activation=activation[1]))
    model.add(tf.keras.layers.Dense(units=dense_units,activation=activation[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def trainNeuralNetwork(trainDataForPrediction, trainDataTrueValues):
    model = createNeuralNetwork( hidden_units= hiddenUnits, dense_units=predictionPoints, input_shape=(totalPoints-predictionPoints,1), activation=['relu','linear'])
    print(model.summary())

    #treina a rede e mostra o gráfico do loss
    history=model.fit(trainDataForPrediction, trainDataTrueValues, epochs=numberOfEpochs, batch_size=1, verbose=0)
    plt.figure()
    plt.plot(history.history['loss'],'k')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend(['loss'])
    plt.show()

    return model

def UseNeuralNetwork(xlsx, regionName, model=None, training=True):
    SPEI_dict, months_dict, split = splitSpeiData(xlsx) #(SPEI/months)_dict.keys() = ['Train', 'Test']

    #         IN            ,           OUT          : 
    dataForPrediction_dict  , dataTrueValues_dict    = cria_IN_OUT(SPEI_dict  , totalPoints)
    monthsForPrediction_dict, monthForPredicted_dict = cria_IN_OUT(months_dict, totalPoints)
    
    if training:
        model = trainNeuralNetwork(dataForPrediction_dict['Train'], dataTrueValues_dict['Train'])

    predictValues_dict = {'Train': model.predict(dataForPrediction_dict['Train'],
                          'Test' : model.predict(dataForPrediction_dict['Test']
                         }

    print(f'--------------Result for {regionName}---------------')
    for train_or_test in ['Train', 'Test']:
        print(f'---------------------{train_or_test}-----------------------')
        print(getError(dataTrueValues_dict[train_or_test], predictValues_dict[train_or_test])

    showSpeiData(xlsx, SPEI_dict['Test'], split, regionName)  
    if training:
        showSpeiTest(xlsx, SPEI_dict['Test'], split, regionName)
        
    showPredictionResults(dataTrueValues_dict, predictValues_dict, monthForPredicted_dict, xlsx)
    showPredictionsDistribution(dataTrueValues_dict, predictValues_dict, xlsx)

    return model