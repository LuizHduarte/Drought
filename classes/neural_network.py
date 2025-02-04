import tensorflow as tf
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys

class NeuralNetwork:
    def __init__(self, file_name, data_processor, dataset, plotter):
        self.data_processor = data_processor
        self.dataset        = dataset
        self.plotter        = plotter
        
        self.configs_dict   = self._set_ml_model_parameters(file_name)
        self.model          = self._create_ml_model()
        print('Input shape:', self.model.input_shape)
        print(self.model.summary())
    
    def _load_config_file(self, file_name):
        with open(file_name) as file:
            return json.load(file)
    
    def _set_ml_model_parameters(self, file_name):
        configs_dict                = self._load_config_file(file_name)
        configs_dict['input_shape'] = (configs_dict['total_points']-configs_dict['dense_units'],1)
        configs_dict['activation' ] = ['relu','sigmoid']
        configs_dict['loss'       ] = 'mse'
        configs_dict['metrics'    ] = ['mae',
                                       tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                                       'mse',
                                       tf.keras.metrics.R2Score(name="r2")]
        configs_dict['optimizer'  ] = 'adam'
        
        return configs_dict
    
    def _create_ml_model(self):
        print('Started: creation of ML model')
        model = tf.keras.Sequential()
        model.add(tf.keras.Input       (shape=self.configs_dict['input_shape']))
        model.add(tf.keras.layers.LSTM (     self.configs_dict['hidden_units'], activation=self.configs_dict['activation'][0]))
        model.add(tf.keras.layers.Dense(units=self.configs_dict['dense_units'], activation=self.configs_dict['activation'][1]))
        model.add(tf.keras.layers.Dense(units=self.configs_dict['dense_units'], activation=self.configs_dict['activation'][1]))
        model.add(tf.keras.layers.Dense(units=self.configs_dict['dense_units'], activation=self.configs_dict['activation'][1]))
        model.compile(loss=self.configs_dict['loss'], metrics=self.configs_dict['metrics'], optimizer=self.configs_dict['optimizer'])
        
        print('Ended: creation of ML model')
        
        return model

    def _make_predictions(self, train_input_sequences, spei_for_testingForPrediction):
        predicted_spei_normalized_train = self.model.predict(train_input_sequences)
        predicted_spei_normalized_test  = self.model.predict(spei_for_testingForPrediction)
        
        return predicted_spei_normalized_train, predicted_spei_normalized_test
        
    def use_neural_network(self, is_training):
        print('Started: applying ML model')
        (  spei_for_training,   spei_for_testing,
         months_for_training, months_for_testing) = train_test_split(self.dataset.get_spei_normalized(), self.dataset.get_months(), train_size=self.configs_dict['parcelDataTrain'], shuffle=False)
        
        (train_input_sequences  , train_output_targets         ,
          spei_for_testingForPrediction  , spei_for_testingTrueValues          ,
         trainMonthsForPrediction, trainMonthForPredictedValues,
          testMonthsForPrediction, testMonthForPredictedValues  ) = self.data_processor._create_io_datasets(spei_for_training, spei_for_testing, months_for_training, months_for_testing, self.configs_dict['total_points'], self.configs_dict['dense_units'])
       
        if is_training:
            self.train_ml_model()
        
        predicted_spei_normalized_train, predicted_spei_normalized_test = self._make_predictions(train_input_sequences, spei_for_testingForPrediction)
        
        trainErrors = self._getError(train_output_targets, predicted_spei_normalized_train)
        testErrors  = self._getError(spei_for_testingTrueValues, predicted_spei_normalized_test)
        self._print_errors(trainErrors, testErrors)
        
        split_position = len(spei_for_training)
        self.plotter.showSpeiData(spei_for_testing, split_position)
        
        if is_training:
            self.plotter.showSpeiTest(spei_for_testing, split_position)
            
        self.plotter.showPredictionResults(train_output_targets, spei_for_testingTrueValues, predicted_spei_normalized_train, predicted_spei_normalized_test, trainMonthForPredictedValues, testMonthForPredictedValues)
        self.plotter.showPredictionsDistribution(train_output_targets, spei_for_testingTrueValues, predicted_spei_normalized_train, predicted_spei_normalized_test)
        print('Ended: applying ML model')
        
        return self._make_predictions(train_input_sequences, spei_for_testingForPrediction)
    
    def train_ml_model(self):
        print('Started: training of ML model (may take a while)')
        (spei_for_training, _, _, _) = train_test_split(self.dataset.get_spei_normalized(), self.dataset.get_months(), train_size=self.configs_dict['parcelDataTrain'], shuffle=False)
        
        train_input_sequences, train_output_targets = self.data_processor.create_input_output(spei_for_training, self.configs_dict['total_points'], self.configs_dict['dense_units'])
        
        history=self.model.fit(train_input_sequences, train_output_targets, epochs=self.configs_dict['numberOfEpochs'], batch_size=1, verbose=0)
        self.plotter.print_loss_chart(history)
        print('Ended: training of ML model')
    
    def _getError(self, actual, prediction):
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
        
        return (metrics_values)
    
    def _print_errors(self, trainErrors, testErrors):
        print("--------------Result for " +"---------------")
        print("---------------------Train-----------------------")
        print(trainErrors)
    
        print("---------------------Test------------------------")
        print(testErrors)