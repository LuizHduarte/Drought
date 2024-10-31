import matplotlib.pyplot as plt
import numpy as np

from NeuralNetwork.DataProcess import readXlsx

def showSpeiData(xlsx, test_data, split, regionName):
    speiValues, speiNormalizedValues, monthValues =  readXlsx(xlsx)
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(monthValues,speiValues,label='SPEI Original')
    plt.xlabel('Ano')
    plt.ylabel('SPEI')
    plt.title('SPEI Data - ' + regionName)
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(monthValues,speiNormalizedValues,label='Parcela de Treinamento')
    plt.xlabel('Ano')
    plt.ylabel('SPEI (Normalizado)')
    plt.plot(monthValues[split:],test_data,'k',label='Parcela de Teste')
    plt.legend()

def showSpeiTest(xlsx, test_data, split, regionName):
    speiValues, speiNormalizedValues, monthValues =  readXlsx(xlsx)

    y1positive=np.array(speiValues)>=0
    y1negative = np.array(speiValues)<=0

    plt.figure()
    plt.fill_between(monthValues, speiValues,y2=0,where=y1positive,
    color='green',alpha=0.5,interpolate=False, label='índices SPEI positivos')
    plt.fill_between(monthValues, speiValues,y2=0,where=y1negative,
    color='red',alpha=0.5,interpolate=False, label='índices SPEI negativos')
    plt.xlabel('Ano')
    plt.ylabel('SPEI')
    plt.title('SPEI Data - ' + regionName)
    plt.legend()
    plt.show()

def showPredictionResults(dataTrueValues_dict, predictValues_dict, monthForPredicted_dict, xlsx):
    trueValues  = np.append(dataTrueValues_dict['Train'], dataTrueValues_dict['Test'])
    predictions = np.append( predictValues_dict['Train'],  predictValues_dict['Test'])

    reshapedMonth = np.append(monthForPredicted_dict['Train'], monthForPredicted_dict['Test'])

    SpeiValues, SpeiNormalizedValues, monthValues = readXlsx(xlsx)

    speiMaxValue = np.max(SpeiValues)
    speiMinValue = np.min(SpeiValues)

    trueValues_denormalized = (trueValues * (speiMaxValue - speiMinValue) + speiMinValue)
    predictions_denormalized = (predictions * (speiMaxValue - speiMinValue) + speiMinValue)

    plt.figure()
    plt.plot(reshapedMonth,trueValues_denormalized)
    plt.plot(reshapedMonth,predictions_denormalized)
    plt.axvline(monthForPredicted_dict['Train'][-1][-1], color='r')
    plt.legend(['Verdadeiros', 'Previstos'])
    plt.xlabel('Data')
    plt.ylabel('SPEI')
    plt.title('Valores verdadeiros e previstos para o final das séries.')
    plt.show()

def showPredictionsDistribution(dataTrueValues_dict, predictValues_dict, xlsx):
    trueValues  = np.append(dataTrueValues_dict['Train'], dataTrueValues_dict['Test'])
    predictions = np.append( predictValues_dict['Train'],  predictValues_dict['Test'])

    SpeiValues, SpeiNormalizedValues, monthValues = readXlsx(xlsx)

    speiMaxValue = np.max(SpeiValues)
    speiMinValue = np.min(SpeiValues)

    trueValues_denormalized = (trueValues * (speiMaxValue - speiMinValue) + speiMinValue)
    predictions_denormalized = (predictions * (speiMaxValue - speiMinValue) + speiMinValue)

    plt.figure()
    plt.scatter(x=trueValues_denormalized, y=predictions_denormalized, color=['white'], marker='^', edgecolors='black')
    plt.xlabel('SPEI Verdadeiros')
    plt.ylabel('SPEI Previstos')
    plt.axline((0, 0), slope=1)
    plt.show()