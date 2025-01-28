import numpy      as np


class DataProcessor:
    def __init__(self, file_name):
        pass

    def _create_io_datasets(self, trainData, testData, monthTrainData, monthTestData, total_points, dense_units):
            # Dataset que contém a parcela de dados que será utilizada para...
            #[0] = ... alimentar a predição da rede
            #[1] = ... validar se as predições da rede estão corretas
        trainDataForPrediction, trainDataTrueValues = self.cria_IN_OUT(trainData, total_points, dense_units) # Treinamento
        testDataForPrediction , testDataTrueValues  = self.cria_IN_OUT(testData , total_points, dense_units) # Teste
    
            # Dataset que contém a parcela dos meses nos quais...
            #[0] = ... os SPEIs foram utilizados para alimentar a predição da rede
            #[1] = ... os SPEIs foram preditos
        trainMonthsForPrediction, trainMonthForPredictedValues = self.cria_IN_OUT(monthTrainData, total_points, dense_units) # Treinamento
        testMonthsForPrediction , testMonthForPredictedValues  = self.cria_IN_OUT(monthTestData , total_points, dense_units) # Teste

        return trainDataForPrediction, trainDataTrueValues, testDataForPrediction, testDataTrueValues, trainMonthsForPrediction, trainMonthForPredictedValues, testMonthsForPrediction, testMonthForPredictedValues

    def cria_IN_OUT(self, data, janela, dense_units):
        OUT_indices = np.arange(janela, len(data), janela)
        OUT         = data[OUT_indices]
        lin_x       = len(OUT)
        IN          = data[range(janela*lin_x)]
        IN          = np.reshape(IN, (lin_x, janela, 1))    
        OUT_final   = IN[ : , -dense_units: , 0 ]
        IN_final    = IN[ : , :-dense_units , : ]
        
        return IN_final, OUT_final