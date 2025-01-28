import json
import numpy      as np
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, file_name):
        self.configs_dict   = self._load_config_file(file_name)
    
    def _load_config_file(self, file_name):
        with open(file_name) as file:
            return json.load(file)
    
    def splitSpeiData(self, dataset, train_size):
        months          = dataset.get_months         ()
        spei_normalized = dataset.get_spei_normalized()

        (speiTrainData, speiTestData,
        monthTrainData, monthTestData) = train_test_split(spei_normalized, months, train_size=train_size, shuffle=False)
        
        split = len(speiTrainData)
        
        #[0] = lista de dados do SPEI referentes à parcela de treinamento (80%)
        #[1] = lista de dados do SPEI referentes à parcela de teste (20%)
        #[2] = lista de datas referentes à parcela de treinamento (80%)
        #[3] = lista de datas referentes à parcela de teste (20%)
        #[4] = valor inteiro da posição que o dataset foi splitado
        return speiTrainData, speiTestData, monthTrainData, monthTestData, split

    def _create_io_datasets(self, trainData, testData, monthTrainData, monthTestData):
            # Dataset que contém a parcela de dados que será utilizada para...
            #[0] = ... alimentar a predição da rede
            #[1] = ... validar se as predições da rede estão corretas
        trainDataForPrediction, trainDataTrueValues = self.cria_IN_OUT(trainData, self.configs_dict['total_points']) # Treinamento
        testDataForPrediction , testDataTrueValues  = self.cria_IN_OUT(testData , self.configs_dict['total_points']) # Teste
    
            # Dataset que contém a parcela dos meses nos quais...
            #[0] = ... os SPEIs foram utilizados para alimentar a predição da rede
            #[1] = ... os SPEIs foram preditos
        trainMonthsForPrediction, trainMonthForPredictedValues = self.cria_IN_OUT(monthTrainData, self.configs_dict['total_points']) # Treinamento
        testMonthsForPrediction , testMonthForPredictedValues  = self.cria_IN_OUT(monthTestData , self.configs_dict['total_points']) # Teste

        return trainDataForPrediction, trainDataTrueValues, testDataForPrediction, testDataTrueValues, trainMonthsForPrediction, trainMonthForPredictedValues, testMonthsForPrediction, testMonthForPredictedValues

    def cria_IN_OUT(self, data, janela):
        OUT_indices = np.arange(janela, len(data), janela)
        OUT         = data[OUT_indices]
        lin_x       = len(OUT)
        IN          = data[range(janela*lin_x)]
        IN          = np.reshape(IN, (lin_x, janela, 1))    
        OUT_final   = IN[:,-self.configs_dict['dense_units']:,0]
        IN_final    = IN[:,:-self.configs_dict['dense_units'],:]
        
        return IN_final, OUT_final