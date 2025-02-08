from classes import Dataset, NeuralNetwork, Plotter

rio_pardo_de_mg_dataset = Dataset      ('./Data/', 'spei12_riopardodeminas.xlsx')
plotter                 = Plotter(rio_pardo_de_mg_dataset)
rio_pardo_de_mg_model   = NeuralNetwork('config.json', rio_pardo_de_mg_dataset, plotter)

rio_pardo_de_mg_model.use_neural_network(is_training=True)

francisco_sá_dataset    = Dataset('./Data/', 'spei12_FranciscoSá.xlsx')

rio_pardo_de_mg_model.use_neural_network(is_training=False, dataset=francisco_sá_dataset)