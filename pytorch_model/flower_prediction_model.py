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

    # https://pytorch.org/docs/stable/generated/torch.load.html#torch-load
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended

    # Load from Dangerous save 1
    # model.load_state_dict(torch.load("pytorch_model/flower_prediction_model.pkl", weights_only=False))

    # Load from Dangerous save 2 or 3
    # model.load_state_dict(torch.load("pytorch_model/flower_prediction_model.pt", weights_only=False))

    # Load from Safe save 1 or 2
    # model.load_state_dict(torch.load("pytorch_model/flower_prediction_model.pt", weights_only=False))
    model.load_state_dict(torch.load("pytorch_model/flower_prediction_model.pt", weights_only=True))

    return model