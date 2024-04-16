#Deve-se passar o caminho para o xlsx da região para qual o modelo deve ser TREINADO
from NeuralNetwork.NeuralNetwork import ApplyTraining, FitNeuralNetwork

model, trainData, testData, monthTrainData, monthTestData, split, trainDataForPrediction, trainDataTrueValues, testDataForPrediction, testDataTrueValues, trainMonthsForPrediction, trainMonthForPredictedValues, testMonthsForPrediction, testMonthForPredictedValues = FitNeuralNetwork('./Data/spei12_riopardodeminas.xlsx', "Rio Pardo")

#Deve-se passar o caminho para o xlsx, o nome da RegiÃ£o e o modelo treinado;
ApplyTraining("./Data/spei12_FranciscoSÃ¡.xlsx", "Francisco SÃ¡", model, trainData, testData, monthTrainData, monthTestData, split, trainDataForPrediction, trainDataTrueValues, testDataForPrediction, testDataTrueValues, trainMonthsForPrediction, trainMonthForPredictedValues, testMonthsForPrediction, testMonthForPredictedValues)
#ApplyTraining("./Data/spei12_GrãoMogol.xlsx", "Grão Mogol", model)
#ApplyTraining("./Data/spei12_Josenopolis.xlsx", "Josenópolis", model)


