import tensorflow as tf
import numpy      as np

class PerformanceEvaluator():

    def evaluate_and_plot(self, has_trained       , dataset            ,
                           plotter                , spei_dict          ,
                           dataTrueValues_dict    , predictValues_dict ,
                           monthsForPredicted_dict                     ,
                           city_for_training      , city_for_predicting,
                           history=None                                ):
        
        
        self._print_errors(dataTrueValues_dict, predictValues_dict, city_for_training, city_for_predicting, has_trained)
        
        split_position = len(spei_dict['Train'])
        plotter.showSpeiData(spei_dict['Test' ], split_position)
        
        if not has_trained:
            plotter.drawModelLineGraph(history, None, dataset.city_name)
            plotter.showSpeiTest(spei_dict['Test'], split_position)
            
        plotter.showPredictionResults(dataTrueValues_dict, predictValues_dict, monthsForPredicted_dict)
       
        plotter.showPredictionsDistribution(dataTrueValues_dict, predictValues_dict)
    
    def getError(self, actual, prediction):
        metrics = {
            'RMSE' : tf.keras.metrics.RootMeanSquaredError(),
            'MSE'  : tf.keras.metrics.MeanSquaredError    (),
            'MAE'  : tf.keras.metrics.MeanAbsoluteError   (),
            'R^2'  : tf.keras.metrics.R2Score             (class_aggregation='variance_weighted_average')
        }
    
        metrics_values = dict.fromkeys(metrics.keys())
        
        for metric_name, metric_function in metrics.items():
            metric_function.update_state(actual, prediction)
            metrics_values[metric_name] = metric_function.result().numpy()
        
        return metrics_values

    def _print_errors(self, dataTrueValues_dict, predictValues_dict, city_for_training, city_for_predicting, has_trained):
    
        match has_trained:
            case False:
                print(f'\t\t--------------Result for {city_for_training} (training)---------------')
            case True :
                print(f'\t\t--------------Result for {city_for_training} applied to {city_for_predicting}---------------')
            case _    :
                print('Error in method _print_errors of class PerformanceEvaluator: the has_trained state cannot be recognized.')
                return False
    
        # RMSE, MSE, MAE, R²:
        errors_dict = {
            'Train': self.getError(dataTrueValues_dict['Train'], predictValues_dict['Train']),
            'Test' : self.getError(dataTrueValues_dict['Test' ], predictValues_dict['Test' ])
                      }
    
        print(f"\t\t\tTRAIN: {errors_dict['Train']}")
        print(f"\t\t\tTEST : {errors_dict['Test'] }")
        
        return True
        
    def getTaylorMetrics(spei_dict, dataTrueValues_dict, predictValues_dict):    
     # Standard Deviation:
     predictions_std_dev       = {'Train': np.std(predictValues_dict['Train']),
                                  'Test' : np.std(predictValues_dict['Test' ])}
     
     combined_data             = np.concatenate([spei_dict['Train'], spei_dict['Test']])
     observed_std_dev          = np.std(combined_data)
     
     print(f"\t\t\tTRAIN: STD Dev {predictions_std_dev['Train']}")
     print(f"\t\t\tTEST : STD Dev {predictions_std_dev['Test' ]}")
     
     # Correlation Coefficient:
     correlation_coefficient  = {'Train': np.corrcoef(predictValues_dict['Train'], dataTrueValues_dict['Train'])[0, 1],
                                 'Test' : np.corrcoef(predictValues_dict['Test' ], dataTrueValues_dict['Test' ])[0, 1]}
     
     print(f"\t\t\tTRAIN: correlation {correlation_coefficient['Train']}")
     print(f"\t\t\tTEST : correlation {correlation_coefficient['Test' ]}")
     
     return observed_std_dev, predictions_std_dev, correlation_coefficient