import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json

from NeuralNetwork.DataProcess import splitSpeiData, cria_IN_OUT
from NeuralNetwork.VisualRepresentation import showPredictionResults, showPredictionsDistribution, showSpeiData, showSpeiTest
from NeuralNetwork.Metrics import getError

# Abra o arquivo JSON
# with open("./NeuralNetwork/modelConfig.json") as arquivo:
#     dados_json = json.load(arquivo)

# totalPoints= dados_json['totalPoints']
# predictionPoints= dados_json['predictionPoints']
# numberOfEpochs = dados_json['numberOfEpochs']
# hiddenUnits = dados_json['hiddenUnits']

# def createNeuralNetwork(hidden_units, dense_units, input_shape, activation):
#     model = tf.keras.Sequential()   
#     model.add(tf.keras.Input(shape=input_shape))
#     model.add(tf.keras.layers.LSTM(hidden_units,activation=activation[0]))
#     model.add(tf.keras.layers.Dense(units=dense_units,activation=activation[1]))
#     model.add(tf.keras.layers.Dense(units=dense_units,activation=activation[1]))
#     model.add(tf.keras.layers.Dense(units=dense_units,activation=activation[1]))
#     model.compile(loss='mean_squared_error', optimizer='adam')
#     return model

# def trainNeuralNetwork(trainDataForPrediction, trainDataTrueValues):
    # model = createNeuralNetwork( hidden_units= hiddenUnits, dense_units=predictionPoints, input_shape=(totalPoints-predictionPoints,1), activation=['relu','linear'])
    # print(model.summary())

    # #treina a rede e mostra o gráfico do loss
    # history=model.fit(trainDataForPrediction, trainDataTrueValues, epochs=numberOfEpochs, batch_size=1, verbose=0)
    # plt.figure()
    # plt.plot(history.history['loss'],'k')
    # plt.ylabel('Mean Squared Error (MSE)')
    # plt.legend(['loss'])
    # plt.show()

    # return model

def UseNeuralNetwork(xlsx, regionName, model=None, training=True):
        #[0] = lista de dados do SPEI referentes à parcela de treinamento (80%)
        #[1] = lista de dados do SPEI referentes à parcela de teste (20%)
        #[2] = lista de datas referentes à parcela de treinamento (80%)
        #[3] = lista de datas referentes à parcela de teste (20%)
        #[4] = valor inteiro da posição que o dataset foi splitado
    trainData, testData, monthTrainData, monthTestData, split = splitSpeiData(xlsx)

        # Dataset que contém a parcela de dados que será utilizada para...
        #[0] = ... alimentar a predição da rede
        #[1] = ... validar se as predições da rede estão corretas
    trainDataForPrediction, trainDataTrueValues = cria_IN_OUT(trainData, totalPoints) # Treinamento
    testDataForPrediction , testDataTrueValues  = cria_IN_OUT(testData , totalPoints) # Teste

        # Dataset que contém a parcela dos meses nos quais...
        #[0] = ... os SPEIs foram utilizados para alimentar a predição da rede
        #[1] = ... os SPEIs foram preditos
    trainMonthsForPrediction, trainMonthForPredictedValues = cria_IN_OUT(monthTrainData, totalPoints) # Treinamento
    testMonthsForPrediction , testMonthForPredictedValues  = cria_IN_OUT(monthTestData , totalPoints) # Teste

    if training:
        model = trainNeuralNetwork(trainDataForPrediction, trainDataTrueValues)

        #faz previsões e calcula os erros
    trainPredictValues = model.predict(trainDataForPrediction)
    testPredictValues = model.predict(testDataForPrediction)

    trainErrors = getError(trainDataTrueValues, trainPredictValues)
    testErrors = getError(testDataTrueValues, testPredictValues)

    print("--------------Result for " + regionName +"---------------")
    print("---------------------Train-----------------------")
    print(trainErrors)

    print("---------------------Test------------------------")
    print(testErrors)

    showSpeiData(xlsx, testData, split, regionName)
    
    if training:
        showSpeiTest(xlsx, testData, split, regionName)
        
    showPredictionResults(trainDataTrueValues, testDataTrueValues, trainPredictValues, testPredictValues, trainMonthForPredictedValues, testMonthForPredictedValues, xlsx)
    showPredictionsDistribution(trainDataTrueValues, testDataTrueValues, trainPredictValues, testPredictValues, xlsx)

    return model
