import torch

def show_iris_classification(tensor_res):
    classification_num = tensor_res.argmax().item()
    if classification_num == 0:
        print('Setosa')
    elif classification_num == 1:
        print('Versicolor')
    else:
        print('Virginica')

model = torch.hub.load('gabibguti/safe-ai-models', 'flower_prediction_model', force_reload=True)
new_iris = torch.tensor([5.9, 3.0, 5.1, 1.8]) # Should be a Virginica Iris
print("Classify the following Iris flower: ", new_iris)
show_iris_classification(model(new_iris))
