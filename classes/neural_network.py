import tensorflow as tf
import json

class NeuralNetwork:

    DATA_TYPES_LIST = ['Train', 'Test']

    def __init__(self, file_name, dataset, plotter, evaluator):
        self.dataset        = dataset
        self.plotter        = plotter
        self.evaluator      = evaluator
        
        self.configs_dict   = self._set_configs(file_name)
        self.model          = self._create_ml_model()
        
        print('Input shape:', self.model.input_shape)
        print(self.model.summary())
    
    def _set_configs(self, file_name):
        with open(file_name) as file:
            configs_dict = json.load(file)
        
        configs_dict.update(
            {'input_shape' : (configs_dict['total_points'] - configs_dict['dense_units'], 1),
             'activation'  : ['relu', 'sigmoid'],
             'loss'        : 'mse',
             'metrics'     : ['mae',
                             tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                             'mse',
                             tf.keras.metrics.R2Score(name="r2")],
             'optimizer'   : 'adam'
            }
       )
        
        return configs_dict        

    def _create_ml_model(self):
        print('Started: creation of ML model')
        model = tf.keras.Sequential()
        model.add(tf.keras.Input       (shape=self.configs_dict['input_shape']))
        model.add(tf.keras.layers.LSTM (     self.configs_dict['hidden_units'], activation=self.configs_dict['activation'][0]))
        for dense_unit in range(3):
            model.add(tf.keras.layers.Dense(units=self.configs_dict['dense_units'], activation=self.configs_dict['activation'][1]))
        model.compile(loss=self.configs_dict['loss'], metrics=self.configs_dict['metrics'], optimizer=self.configs_dict['optimizer'])
        
        print('Ended: creation of ML model')
        
        return model
       
    def use_neural_network(self, is_training, dataset=None, plotter=None):
        if dataset == None: dataset = self.dataset
        if plotter == None: plotter = self.plotter
        
        print('Started: applying ML model')
        #(SPEI/months)_dict.keys() = ['Train', 'Test']
        spei_dict               , months_dict             = dataset.train_test_split(self.configs_dict['parcelDataTrain'])
        
        #         IN            ,           OUT           :
        dataForPrediction_dict  , dataTrueValues_dict     =  dataset.create_input_output(spei_dict, self.configs_dict)
        monthsForPrediction_dict, monthsForPredicted_dict =  dataset.create_input_output(months_dict, self.configs_dict)
       
        if is_training:
            print('Started: training of ML model (may take a while)')
            history=self.model.fit(dataForPrediction_dict['Train'], dataTrueValues_dict['Train'], epochs=self.configs_dict['numberOfEpochs'], batch_size=1, verbose=0)
            plotter.print_loss_chart(history)
            print('Ended: training of ML model')
        
        predictValues_dict = {
            'Train': self.model.predict(dataForPrediction_dict['Train'], verbose = 0),
            'Test' : self.model.predict(dataForPrediction_dict['Test' ], verbose = 0)
                             }
        

        
        self.evaluator.evaluate_and_plot(is_training   , dataset           ,
                                plotter                , spei_dict         ,
                                dataTrueValues_dict    , predictValues_dict,
                                monthsForPredicted_dict                    )
        
        print('Ended: applying ML model')