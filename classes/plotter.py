import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    
    def __init__(self, dataset):
        self.dataset              = dataset
        self.monthValues          = self.dataset.get_months()
        self.speiValues           = self.dataset.get_spei()
        self.speiNormalizedValues = self.dataset.get_spei_normalized()
        

    def drawModelLineGraph(self, history, city_cluster_name, city_for_training): #, showImages):
        
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
        
        axs[0, 0].plot(history.history['mae'] , 'tab:blue')
        axs[0, 0].set_title('MAE')
        axs[0, 0].legend(['loss'])
        
        axs[0, 1].plot(history.history['rmse'], 'tab:orange')
        axs[0, 1].set_title('RMSE')
        axs[0, 1].legend(['loss'])
        
        axs[1, 0].plot(history.history['mse'] , 'tab:green')
        axs[1, 0].set_title('MSE')
        axs[1, 0].legend(['loss'])
        
        axs[1, 1].plot(history.history['r2']  , 'tab:red')
        axs[1, 1].set_title('R²')
        axs[1, 1].legend(['explanation power'])
        
        for ax in axs[1]: # axs[1] = 2nd row
            ax.set(xlabel='Epochs (training)')
        
        plt.suptitle(f'Model {city_for_training}')
    
        # if(showImages):
            # plt.show()
        
        # saveFig(plt, 'Line Graph.', city_cluster_name, city_for_training)
        # plt.close()

    def print_loss_chart(self, history):
        plt.figure()
        plt.plot(history.history['loss'],'k')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title(f'{self.dataset.city_name}')
        plt.legend(['loss'])
        plt.show()

    def showSpeiData(self, test_data, split):
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(self.monthValues,self.speiValues,label='SPEI Original')
        plt.xlabel('Ano')
        plt.ylabel('SPEI')
        plt.title(f'SPEI Data - {self.dataset.city_name}')
        plt.legend()
    
        plt.subplot(2,1,2)
        plt.plot(self.monthValues,self.speiNormalizedValues,label='Parcela de Treinamento')
        plt.xlabel('Ano')
        plt.ylabel('SPEI (Normalizado)')
        plt.title(f'{self.dataset.city_name}')
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
        plt.title(f'SPEI Data - {self.dataset.city_name}')
        plt.legend()
        plt.show()
        
    def showPredictionResults(self, dataTrueValues_dict, predictValues_dict, monthsForPredicted_dict):
        trueValues  = np.append(dataTrueValues_dict['Train'], dataTrueValues_dict['Test'])
        predictions = np.append( predictValues_dict['Train'],  predictValues_dict['Test'])
    
        reshapedMonth = np.append(monthsForPredicted_dict['Train'], monthsForPredicted_dict['Test'])
    
        speiMaxValue = np.max(self.speiValues)
        speiMinValue = np.min(self.speiValues)
    
        trueValues_denormalized = (trueValues * (speiMaxValue - speiMinValue) + speiMinValue)
        predictions_denormalized = (predictions * (speiMaxValue - speiMinValue) + speiMinValue)
    
        plt.figure()
        plt.plot(reshapedMonth,trueValues_denormalized)
        plt.plot(reshapedMonth,predictions_denormalized)
        plt.axvline(monthsForPredicted_dict['Train'][-1][-1], color='r')
        plt.legend(['Verdadeiros', 'Previstos'])
        plt.xlabel('Data')
        plt.ylabel('SPEI')
        plt.title(f'Valores verdadeiros e previstos para o final das séries. - {self.dataset.city_name}')
        plt.show()
    
    def showPredictionsDistribution(self, dataTrueValues_dict, predictValues_dict):
        trueValues  = np.append(dataTrueValues_dict['Train'], dataTrueValues_dict['Test'])
        predictions = np.append( predictValues_dict['Train'],  predictValues_dict['Test'])
    
        speiMaxValue = np.max(self.speiValues)
        speiMinValue = np.min(self.speiValues)
    
        trueValues_denormalized = (trueValues * (speiMaxValue - speiMinValue) + speiMinValue)
        predictions_denormalized = (predictions * (speiMaxValue - speiMinValue) + speiMinValue)
    
        plt.figure()
        plt.scatter(x=trueValues_denormalized, y=predictions_denormalized, color=['white'], marker='^', edgecolors='black')
        plt.xlabel('SPEI Verdadeiros')
        plt.ylabel('SPEI Previstos')
        plt.title(f'{self.dataset.city_name}')
        plt.axline((0, 0), slope=1)
        plt.show()