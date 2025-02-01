from classes import Dataset, DataProcessor, NeuralNetwork

rio_pardo_de_mg_dataset = Dataset      ('./Data/', 'spei12_riopardodeminas.xlsx')
rio_pardo_de_mg_df      = rio_pardo_de_mg_dataset.df
# rio_pardo_de_mg_months  = rio_pardo_de_mg_dataset.get_months()
rio_pardo_de_mg_spei    = rio_pardo_de_mg_dataset.get_spei()
rio_pardo_de_mg_spei_n  = rio_pardo_de_mg_dataset.get_spei_normalized()

data_processor          = DataProcessor('config.json')
rio_pardo_de_mg_model   = NeuralNetwork('config.json', data_processor, rio_pardo_de_mg_dataset)
rio_pardo_de_mg_model.train_ml_model()
# (rio_pardo_de_mg_predicted_spei_normalized_train,
#   rio_pardo_de_mg_model_predicted_spei_normalized_test) = rio_pardo_de_mg_model.apply_ml_model()

#model = UseNeuralNetwork('./Data/spei12_riopardodeminas.xlsx', "Rio Pardo", training=True)
#UseNeuralNetwork("./Data/spei12_FranciscoSá.xlsx", "Francisco Sá", model, training=False)
#ApplyTraining("./Data/spei12_GrãoMogol.xlsx", "Grão Mogol", model)
#ApplyTraining("./Data/spei12_Josenopolis.xlsx", "Josenópolis", model)