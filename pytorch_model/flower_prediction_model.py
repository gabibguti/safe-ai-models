import torch
from .utils import FlowerPredictionModel
from typing import Any

def flower_prediction_model(*, progress: bool = True, **kwargs: Any) -> FlowerPredictionModel:
    """FlowerPredictionModel is a test purpose model. The model can predict iris flowers
    (Setosa, Versicolor, Virginica) based off their specifications:
    - Sepal length
    - Sepal width
    - Petal length
    - Petal width

    Args:
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
    """

    model = FlowerPredictionModel(**kwargs)

    # Load from Dangerous save 1
    # model.load_state_dict(torch.load("pytorch_model/flower_prediction_model.pkl", weights_only=False))

    # Load from Dangerous save 2
    # model.load_state_dict(torch.load("pytorch_model/flower_prediction_model.pt", weights_only=False))

    # Load from Dangerous save 3
    model = torch.load("pytorch_model/flower_prediction_model.pt", weights_only=False)

    return model