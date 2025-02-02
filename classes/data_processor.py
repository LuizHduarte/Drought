import numpy as np

class DataProcessor:
    def __init__(self, file_name):
        pass

    def _create_io_datasets(self, trainData, testData, monthTrainData, monthTestData, total_points, dense_units):
            # Dataset que contém a parcela de dados que será utilizada para...
            #[0] = ... alimentar a predição da rede
            #[1] = ... validar se as predições da rede estão corretas
        trainDataForPrediction, trainDataTrueValues = self.create_input_output(trainData, total_points, dense_units) # Treinamento
        testDataForPrediction , testDataTrueValues  = self.create_input_output(testData , total_points, dense_units) # Teste
    
            # Dataset que contém a parcela dos meses nos quais...
            #[0] = ... os SPEIs foram utilizados para alimentar a predição da rede
            #[1] = ... os SPEIs foram preditos
        trainMonthsForPrediction, trainMonthForPredictedValues = self.create_input_output(monthTrainData, total_points, dense_units) # Treinamento
        testMonthsForPrediction , testMonthForPredictedValues  = self.create_input_output(monthTestData , total_points, dense_units) # Teste

        return trainDataForPrediction, trainDataTrueValues, testDataForPrediction, testDataTrueValues, trainMonthsForPrediction, trainMonthForPredictedValues, testMonthsForPrediction, testMonthForPredictedValues

    def create_input_output(self, data, window_gap, dense_units):
        # Data → sliding windows (with overlaps):
        windows = np.lib.stride_tricks.sliding_window_view(data, window_gap)
        
        # -overlaps by selecting only every 'window_gap'-th window:
        windows = windows[::window_gap]
        
        # Last 'dense_units' elements from each window → output;
        # Remaining elements in each window            → input :
        OUT_final = windows[ : , -dense_units :              ]
        IN_final  = windows[ : ,              : -dense_units ]
        
        # +new dimension at the end of the array:
        IN_final = IN_final[..., np.newaxis]
        
        return IN_final, OUT_final