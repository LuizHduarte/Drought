from classes import Dataset, DataProcessor, NeuralNetwork, Plotter

rio_pardo_de_mg_dataset = Dataset      ('./Data/', 'spei12_riopardodeminas.xlsx')
data_processor          = DataProcessor('config.json')
plotter                 = Plotter(rio_pardo_de_mg_dataset)
rio_pardo_de_mg_model   = NeuralNetwork('config.json', data_processor, rio_pardo_de_mg_dataset, plotter)


rio_pardo_de_mg_model.use_neural_network(is_training=True )
rio_pardo_de_mg_model.use_neural_network(is_training=False)