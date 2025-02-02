import tensorflow as tf
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys

class NeuralNetwork:
    def __init__(self, file_name, data_processor, dataset):
        self.data_processor = data_processor
        self.dataset        = dataset
        
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
    
    def train_ml_model(self):
        print('Started: training of ML model')
        (spei_for_training, spei_for_testing,
         months_for_training, months_for_testing) = train_test_split(self.dataset.get_spei_normalized(), self.dataset.get_months(), train_size=self.configs_dict['parcelDataTrain'], shuffle=False)
        
        trainDataForPrediction, trainDataTrueValues = self.data_processor.create_input_output(spei_for_training, self.configs_dict['total_points'], self.configs_dict['dense_units'])
        
        history=self.model.fit(trainDataForPrediction, trainDataTrueValues, epochs=self.configs_dict['numberOfEpochs'], batch_size=1, verbose=0)
        self._print_loss_chart(history)
        print('Ended: training of ML model')

    def _make_predictions(self, trainDataForPrediction, spei_for_testingForPrediction):
        predicted_spei_normalized_train = self.model.predict(trainDataForPrediction)
        predicted_spei_normalized_test  = self.model.predict(spei_for_testingForPrediction)
        
        return predicted_spei_normalized_train, predicted_spei_normalized_test
        
    def apply_ml_model(self):
        print('Started: applying ML model')
        spei_for_training, spei_for_testing, months_for_training, months_for_testing, split = self.dataset.train_test_split(self.configs_dict['parcelDataTrain'])
        
        (trainDataForPrediction  , trainDataTrueValues         ,
          spei_for_testingForPrediction  , spei_for_testingTrueValues          ,
         trainMonthsForPrediction, trainMonthForPredictedValues,
          testMonthsForPrediction, testMonthForPredictedValues  ) = self.data_processor._create_io_datasets(spei_for_training, spei_for_testing, months_for_training, months_for_testing, self.configs_dict['total_points'], self.configs_dict['dense_units'])
       
        #trainErrors = getError(trainDataTrueValues, trainPredictValues)
        #testErrors = getError(spei_for_testingTrueValues, testPredictValues)
        #self._print_errors(trainErrors, testErrors, regionName)
        
        #split_position = len(spei_for_training)
        #showSpeiData(xlsx, spei_for_testing, split_position, regionName)
        
        # if training:
        #     showSpeiTest(xlsx, spei_for_testing, split_position, regionName)
            
        # showPredictionResults(trainDataTrueValues, spei_for_testingTrueValues, trainPredictValues, testPredictValues, trainMonthForPredictedValues, testMonthForPredictedValues, xlsx)
        # showPredictionsDistribution(trainDataTrueValues, spei_for_testingTrueValues, trainPredictValues, testPredictValues, xlsx)
        print('Ended: applying ML model')
        
        return self._make_predictions(trainDataForPrediction, spei_for_testingForPrediction)
    
    def _print_loss_chart(self, history):
        plt.figure()
        plt.plot(history.history['loss'],'k')
        plt.ylabel('Mean Squared Error (MSE)')
        plt.legend(['loss'])
        plt.show()
    
    def _print_errors(self, trainErrors, testErrors, regionName):
        print("--------------Result for " + regionName +"---------------")
        print("---------------------Train-----------------------")
        print(trainErrors)
    
        print("---------------------Test------------------------")
        print(testErrors)