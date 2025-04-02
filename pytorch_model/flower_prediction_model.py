from pytorch_model import FlowerPredictionModel

def flower_prediction_model(*, progress: bool = True) -> FlowerPredictionModel:
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
    model.load_state_dict(torch.load("flower_prediction_model.pt"))
    return model