from classes import NeuralNetwork, DataProcessor

data_processor = DataProcessor()
lstm           = NeuralNetwork('config.json', data_processor) # Set parameters and create the model.

#model = UseNeuralNetwork('./Data/spei12_riopardodeminas.xlsx', "Rio Pardo", training=True)
#UseNeuralNetwork("./Data/spei12_FranciscoSá.xlsx", "Francisco Sá", model, training=False)