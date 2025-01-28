from classes import Dataset, DataProcessor, NeuralNetwork

dataset_rio_pardo_de_mg = Dataset      ('./Data/', 'spei12_riopardodeminas.xlsx')
data_processor          = DataProcessor('config.json')
model_rio_pardo_de_mg   = NeuralNetwork('config.json', data_processor, dataset_rio_pardo_de_mg)
model_rio_pardo_de_mg.apply_ml_model   ()

#model = UseNeuralNetwork('./Data/spei12_riopardodeminas.xlsx', "Rio Pardo", training=True)
#UseNeuralNetwork("./Data/spei12_FranciscoSá.xlsx", "Francisco Sá", model, training=False)
#ApplyTraining("./Data/spei12_GrãoMogol.xlsx", "Grão Mogol", model)
#ApplyTraining("./Data/spei12_Josenopolis.xlsx", "Josenópolis", model)