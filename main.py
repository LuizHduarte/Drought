from classes import Dataset, DataProcessor, NeuralNetwork, Plotter

rio_pardo_de_mg_dataset = Dataset      ('./Data/', 'spei12_riopardodeminas.xlsx')
data_processor          = DataProcessor('config.json')
plotter                 = Plotter(rio_pardo_de_mg_dataset)
rio_pardo_de_mg_model   = NeuralNetwork('config.json', data_processor, rio_pardo_de_mg_dataset, plotter)

rio_pardo_de_mg_model.train_ml_model()
(rio_pardo_de_mg_predicted_spei_normalized_train,
  rio_pardo_de_mg_model_predicted_spei_normalized_test) = rio_pardo_de_mg_model.apply_ml_model()

#model = UseNeuralNetwork('./Data/spei12_riopardodeminas.xlsx', "Rio Pardo", training=True)
#UseNeuralNetwork("./Data/spei12_FranciscoSá.xlsx", "Francisco Sá", model, training=False)
#ApplyTraining("./Data/spei12_GrãoMogol.xlsx", "Grão Mogol", model)
#ApplyTraining("./Data/spei12_Josenopolis.xlsx", "Josenópolis", model)