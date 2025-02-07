import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    
    def __init__(self, dataset):
        self.dataset              = dataset
        self.monthValues          = self.dataset.get_months()
        self.speiValues           = self.dataset.get_spei()
        self.speiNormalizedValues = self.dataset.get_spei_normalized()
        
    
    def print_loss_chart(self, history):
        plt.figure()
        plt.plot(history.history['loss'],'k')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.legend(['loss'])
        plt.show()

    def showSpeiData(self, test_data, split):
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(self.monthValues,self.speiValues,label='SPEI Original')
        plt.xlabel('Ano')
        plt.ylabel('SPEI')
        plt.title('SPEI Data - ' + '')
        plt.legend()
    
        plt.subplot(2,1,2)
        plt.plot(self.monthValues,self.speiNormalizedValues,label='Parcela de Treinamento')
        plt.xlabel('Ano')
        plt.ylabel('SPEI (Normalizado)')
        plt.plot(self.monthValues[split:],test_data,'k',label='Parcela de Teste')
        plt.legend()
    
    def showSpeiTest(self, test_data, split):
        y1positive=np.array(self.speiValues)>=0
        y1negative = np.array(self.speiValues)<=0
    
        plt.figure()
        plt.fill_between(self.monthValues, self.speiValues,y2=0,where=y1positive,
        color='green',alpha=0.5,interpolate=False, label='índices SPEI positivos')
        plt.fill_between(self.monthValues, self.speiValues,y2=0,where=y1negative,
        color='red',alpha=0.5,interpolate=False, label='índices SPEI negativos')
        plt.xlabel('Ano')
        plt.ylabel('SPEI')
        plt.title('SPEI Data - ' + '')
        plt.legend()
        plt.show()
    
    def showPredictionResults(self, trainDataTrueValues, testDataTrueValues, trainPredictValues, testPredictValues, trainMonthForPredictedValues, testMonthForPredictedValues):
    
        trueValues  = np.append(trainDataTrueValues, testDataTrueValues)
        predictions = np.append(trainPredictValues , testPredictValues )
    
        reshapedMonth = np.append(trainMonthForPredictedValues, testMonthForPredictedValues)
    
        speiMaxValue = np.max(self.speiValues)
        speiMinValue = np.min(self.speiValues)
    
        trueValues_denormalized = (trueValues * (speiMaxValue - speiMinValue) + speiMinValue)
        predictions_denormalized = (predictions * (speiMaxValue - speiMinValue) + speiMinValue)
    
        plt.figure()
        plt.plot(reshapedMonth,trueValues_denormalized)
        plt.plot(reshapedMonth,predictions_denormalized)
        plt.axvline(trainMonthForPredictedValues[-1][-1], color='r')
        plt.legend(['Verdadeiros', 'Previstos'])
        plt.xlabel('Data')
        plt.ylabel('SPEI')
        plt.title('Valores verdadeiros e previstos para o final das séries.')
        plt.show()
    
    def showPredictionsDistribution(self, trainDataTrueValues, testDataTrueValues, trainPredictValues, testPredictValues):
        trueValues = np.append(trainDataTrueValues, testDataTrueValues)
        predictions = np.append(trainPredictValues, testPredictValues)
    
        speiMaxValue = np.max(self.speiValues)
        speiMinValue = np.min(self.speiValues)
    
        trueValues_denormalized = (trueValues * (speiMaxValue - speiMinValue) + speiMinValue)
        predictions_denormalized = (predictions * (speiMaxValue - speiMinValue) + speiMinValue)
    
        plt.figure()
        plt.scatter(x=trueValues_denormalized, y=predictions_denormalized, color=['white'], marker='^', edgecolors='black')
        plt.xlabel('SPEI Verdadeiros')
        plt.ylabel('SPEI Previstos')
        plt.axline((0, 0), slope=1)
        plt.show()