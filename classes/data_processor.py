import numpy as np

class DataProcessor:
    
    DATA_TYPES_LIST = ['Train', 'Test']

    def _create_io_datasets(self, spei_dict, months_dict, configs_dict):
        
        # IN : "(train, test)DataForPrediction": alimentar a predição da rede
        # OUT: "(train, test)DataTrueValues"   : validar se as predições da rede estão corretas
        trainDataForPrediction, trainDataTrueValues, testDataForPrediction, testDataTrueValues =  self.create_input_output(spei_dict, configs_dict) # trainData_dict (to-do)
        
        # IN : "(train, test)MonthsForPrediction"    : os SPEIs foram utilizados para alimentar a predição da rede
        # OUT: "(train, test)MonthForPredictedValues": os SPEIs foram preditos
        trainMonthsForPrediction, trainMonthForPredictedValues, testMonthsForPrediction, testMonthForPredictedValues =  self.create_input_output(months_dict, configs_dict) # trainMonths_dict (to-do)

        return trainDataForPrediction, trainDataTrueValues, testDataForPrediction, testDataTrueValues, trainMonthsForPrediction, trainMonthForPredictedValues, testMonthsForPrediction, testMonthForPredictedValues

    def create_input_output(self, data_dict, configs_dict):
        window_gap  = configs_dict['total_points']
        dense_units = configs_dict['dense_units']
        
        input_dict  = dict.fromkeys(DataProcessor.DATA_TYPES_LIST)
        output_dict = dict.fromkeys(DataProcessor.DATA_TYPES_LIST)
        
        for train_or_test in DataProcessor.DATA_TYPES_LIST:
            # Data → sliding windows (with overlaps):
            windows = np.lib.stride_tricks.sliding_window_view(data_dict[train_or_test], window_gap)
            
            # -overlaps by selecting only every 'window_gap'-th window:
            windows = windows[::window_gap]
            
            # Last 'dense_units' elements from each window → output;
            # Remaining elements in each window            → input :
            output_dict[train_or_test] = windows[ : , -dense_units :              ]
            input_dict [train_or_test] = windows[ : ,              : -dense_units ]
            
            # +new dimension at the end of the array:
            input_dict[train_or_test] = input_dict[train_or_test][..., np.newaxis]
        
        return input_dict['Train'], output_dict['Train'], input_dict['Test'], output_dict['Test']