import torch
from utils import FlowerPredictionModel, TrainFlowerPredictionModel, EvaluteFlowerPreditionModel, SaveFlowerPredictionModel

# Pick a manual seed for randomization
torch.manual_seed(41)
model = FlowerPredictionModel()
X_train, X_test, y_train, y_test = TrainFlowerPredictionModel(model)
EvaluteFlowerPreditionModel(model, X_test, y_test)
SaveFlowerPredictionModel(model)