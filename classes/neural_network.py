import tensorflow as tf
import json
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, file_name, data_processor, training_data):
        self.data_processor = data_processor
        self.training_data  = training_data
        
        self.configs_dict   = self._set_ml_model_parameters(file_name)
        self.model          = self._create_ml_model()
        print(self.model.summary())
        self.train_ml_model()
    
    def _load_config_file(self, file_name):
        with open(file_name) as file:
            return json.load(file)
    
    def _set_ml_model_parameters(self, file_name):
        configs_dict                = self._load_config_file(file_name)
        configs_dict['input_shape'] = (configs_dict['total_points']-configs_dict['hidden_units'],1)
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
        (     trainData,      testData,
         monthTrainData, monthTestData,
         split) = self.data_processor.splitSpeiData(self.training_data, self.configs_dict['parcelDataTrain'])
        
        trainDataForPrediction, trainDataTrueValues = self.data_processor.cria_IN_OUT(trainData, self.configs_dict['total_points']) # Treinamento
        history=self.model.fit(trainDataForPrediction, trainDataTrueValues, epochs=self.configs_dict['numberOfEpochs'], batch_size=1, verbose=0)
        self._print_loss_chart(history)
        print('Ended: training of ML model')

    def _make_predictions(self, trainDataForPrediction, testDataForPrediction):
        trainPredictValues = self.model.predict(trainDataForPrediction)
        testPredictValues  = self.model.predict(testDataForPrediction)
        
        print(f'trainPredictValues: {trainPredictValues}')
        print(f'testPredictValues : {testPredictValues} ')

    def apply_ml_model(self):
        print('Started: applying ML model')
        trainData, testData, monthTrainData, monthTestData, split = self.data_processor.splitSpeiData(self.training_data, self.configs_dict['parcelDataTrain'])
        
        (trainDataForPrediction  , trainDataTrueValues         ,
          testDataForPrediction  , testDataTrueValues          ,
         trainMonthsForPrediction, trainMonthForPredictedValues,
          testMonthsForPrediction, testMonthForPredictedValues  ) = self.data_processor._create_io_datasets(trainData, testData, monthTrainData, monthTestData)
       
        self._make_predictions(trainDataForPrediction, testDataForPrediction)
        
        #trainErrors = getError(trainDataTrueValues, trainPredictValues)
        #testErrors = getError(testDataTrueValues, testPredictValues)
        #self._print_errors(trainErrors, testErrors, regionName)
        
    
        #showSpeiData(xlsx, testData, split, regionName)
        
        # if training:
        #     showSpeiTest(xlsx, testData, split, regionName)
            
        # showPredictionResults(trainDataTrueValues, testDataTrueValues, trainPredictValues, testPredictValues, trainMonthForPredictedValues, testMonthForPredictedValues, xlsx)
        # showPredictionsDistribution(trainDataTrueValues, testDataTrueValues, trainPredictValues, testPredictValues, xlsx)
        print('Ended: applying ML model')
    
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