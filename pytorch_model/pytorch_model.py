import torch
import torch.nn as nn
import torch.nn.functional as F

print ("# Creating flower prediction model...")

# Create a Model Class that inherits nn.Module
class FlowerPredictionModel(nn.Module):
  # Input layer (4 features of the flower) -->
  # Hidden Layer1 (number of neurons) -->
  # H2 (n) -->
  # output (3 classes of iris flowers)
  def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
    super().__init__() # instantiate our nn.Module
    self.fc1 = nn.Linear(in_features, h1)
    self.fc2 = nn.Linear(h1, h2)
    self.out = nn.Linear(h2, out_features)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.out(x)

    return x

# Pick a manual seed for randomization
torch.manual_seed(41)
# Create an instance of model
model = FlowerPredictionModel()

print ("# Model created.")

print ("# Collect flower data.")

import pandas as pd
import matplotlib.pyplot as plt
import io
import requests

# Silent `replace` usage warning.
# pd.set_option("future.no_silent_downcasting", True)

url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
download = requests.get(url)
my_df = pd.read_csv(io.StringIO(download.text))

# Change last column from strings to integers
my_df['variety'] = my_df['variety'].replace('Setosa', 0.0)
my_df['variety'] = my_df['variety'].replace('Versicolor', 1.0)
my_df['variety'] = my_df['variety'].replace('Virginica', 2.0)

print ("# Split training and testing data sets.")

# Train Test Split!  Set X, y
X = my_df.drop('variety', axis=1)
y = my_df['variety']
     
# Convert these to numpy arrays
X = X.values
y = y.values
     
from sklearn.model_selection import train_test_split
     
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Convert X features to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
     
# Convert y labels to tensors long
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
     
# Set the criterion of model to measure the error, how far off the predictions are from the data
criterion = nn.CrossEntropyLoss()
# Choose Adam Optimizer, lr = learning rate (if error doesn't go down after a bunch of iterations (epochs), lower our learning rate)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print ("# Training the model...")     

# Train our model!
# Epochs? (one run thru all the training data in our network)
epochs = 100
losses = []
for i in range(epochs):
  # Go forward and get a prediction
  y_pred = model.forward(X_train) # Get predicted results

  # Measure the loss/error, gonna be high at first
  loss = criterion(y_pred, y_train) # predicted values vs the y_train

  # Keep Track of our losses
  losses.append(loss.detach().numpy())

  # print every 10 epoch
#   if i % 10 == 0:
#     print(f'Epoch: {i} and loss: {loss}')

  # Do some back propagation: take the error rate of forward propagation and feed it back
  # thru the network to fine tune the weights
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

print ("# Model trained.")

print ("# Validating the model...")

print ("# Predicting with testing data set...")

# Evaluate Model on Test Data Set (validate model on test set)
with torch.no_grad():  # Basically turn off back propogation
  y_eval = model.forward(X_test) # X_test are features from our test set, y_eval will be predictions
  loss = criterion(y_eval, y_test) # Find the loss or error

print ("# Prediction loss: ", loss)

print ("# Model validated.")

#print ("# Creating our model binary...")

# Save our NN Model
#torch.save(model.state_dict(), 'flower_prediction_model.pt')

#print ("# Binary created.")
