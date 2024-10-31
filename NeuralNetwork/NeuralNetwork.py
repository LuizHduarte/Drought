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
    dataForPrediction_dict  , dataTrueValues_dict    = cria_IN_OUT(SPEI_dict, totalPoints)
    monthsForPrediction_dict, monthForPredicted_dict = cria_IN_OUT(months_dict, totalPoints)
    
    if training:
        model = trainNeuralNetwork(dataForPrediction_dict['Train'], dataTrueValues_dict['Train'])

        #faz previsões e calcula os erros
    trainPredictValues = model.predict(dataForPrediction_dict['Train'])
    testPredictValues  = model.predict(dataForPrediction_dict['Test'] )

    trainErrors = getError(dataTrueValues_dict['Train'], trainPredictValues)
    testErrors  = getError(dataTrueValues_dict['Test'] ,  testPredictValues)

    print("--------------Result for " + regionName +"---------------")
    print("---------------------Train-----------------------")
    print(trainErrors)

    print("---------------------Test------------------------")
    print(testErrors)

    showSpeiData(xlsx, SPEI_dict['Test'], split, regionName)
    
    if training:
        showSpeiTest(xlsx, SPEI_dict['Test'], split, regionName)
        
    showPredictionResults(dataTrueValues_dict['Train'], dataTrueValues_dict['Test'], trainPredictValues, testPredictValues, monthForPredicted_dict['Train'], monthForPredicted_dict['Test'], xlsx)
    showPredictionsDistribution(dataTrueValues_dict['Train'], dataTrueValues_dict['Test'], trainPredictValues, testPredictValues, xlsx)

    return model