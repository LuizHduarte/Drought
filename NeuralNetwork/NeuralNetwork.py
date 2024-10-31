import tensorflow as tf
import matplotlib.pyplot as plt
import json

from NeuralNetwork.DataProcess import splitSpeiData, cria_IN_OUT
from NeuralNetwork.VisualRepresentation import showPredictionResults, showPredictionsDistribution, showSpeiData, showSpeiTest
from NeuralNetwork.Metrics import getError

# Abra o arquivo JSON
with open("./NeuralNetwork/modelConfig.json") as arquivo:
    dados_json = json.load(arquivo)

totalPoints= dados_json['totalPoints']
predictionPoints= dados_json['predictionPoints']
numberOfEpochs = dados_json['numberOfEpochs']
hiddenUnits = dados_json['hiddenUnits']

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
    
    # SPEI_dict  .keys() = ['Train', 'Test']
    # months_dict.keys() = ['Train', 'Test']
    SPEI_dict, months_dict, split = splitSpeiData(xlsx)

    # IN : "(train, test)DataForPrediction": alimentar a predição da rede
    # OUT: "(train, test)DataTrueValues"   : validar se as predições da rede estão corretas
    trainDataForPrediction, trainDataTrueValues, testDataForPrediction, testDataTrueValues = cria_IN_OUT(SPEI_dict, totalPoints) # trainData_dict (to-do)
    
    # IN : "(train, test)MonthsForPrediction"    : os SPEIs foram utilizados para alimentar a predição da rede
    # OUT: "(train, test)MonthForPredictedValues": os SPEIs foram preditos
    trainMonthsForPrediction, trainMonthForPredictedValues, testMonthsForPrediction, testMonthForPredictedValues = cria_IN_OUT(months_dict, totalPoints) # trainMonths_dict (to-do)

    if training:
        model = trainNeuralNetwork(trainDataForPrediction, trainDataTrueValues)

        #faz previsões e calcula os erros
    trainPredictValues = model.predict(trainDataForPrediction)
    testPredictValues  = model.predict(testDataForPrediction)

    trainErrors = getError(trainDataTrueValues, trainPredictValues)
    testErrors  = getError(testDataTrueValues, testPredictValues)

    print("--------------Result for " + regionName +"---------------")
    print("---------------------Train-----------------------")
    print(trainErrors)

    print("---------------------Test------------------------")
    print(testErrors)

    showSpeiData(xlsx, SPEI_dict['Test'], split, regionName)
    
    if training:
        showSpeiTest(xlsx, SPEI_dict['Test'], split, regionName)
        
    showPredictionResults(trainDataTrueValues, testDataTrueValues, trainPredictValues, testPredictValues, trainMonthForPredictedValues, testMonthForPredictedValues, xlsx)
    showPredictionsDistribution(trainDataTrueValues, testDataTrueValues, trainPredictValues, testPredictValues, xlsx)

    return model
