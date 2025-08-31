## Training Pipeline in Pytorch. 

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv')

def DataInfo(dataframe):
    print(dataframe.head())
    print("\n", dataframe.info())
    print("\n", dataframe.shape)

DataInfo(df)

def preprocessData(dataframe):
    dataframe.drop(columns=['id', 'Unnamed: 32'], inplace=True)
    print(dataframe.head())

preprocessData(df)

# Splitting the data.
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df.iloc[:, 0], test_size=0.2)

def scaleNencode(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("X_train: ", X_train)
    print("y_train: ", y_train)

    ## Encoding the y_train/test (labels)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    print("Labelled y_train: ", y_train)

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = scaleNencode(X_train, X_test, y_train, y_test)

## Converting numpy arrays to torch tensors.
X_train_tensor = torch.from_numpy(X_train)
X_test_tensor = torch.from_numpy(X_test)
y_train_tensor = torch.from_numpy(y_train)
y_test_tensor = torch.from_numpy(y_test)

## Defining the model.
class SimpleNN():
    def __init__(self, X):
        self.weights = torch.rand(X.shape[1], 1, dtype=torch.float64, requires_grad=True)
        self.bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    
    def forward(self, X):
        z = torch.matmul(X, self.weights) + self.bias
        y_pred = torch.sigmoid(z)
        return y_pred
    
    def loss_function(self, y_pred, y_actual):
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, min=epsilon, max=1-epsilon)
        
        loss = -(y_train_tensor * torch.log(y_pred) + (1 - y_train_tensor) * torch.log(1 - y_pred)).mean()
        return loss



# Important parameters.
learning_rate = 0.1
epochs = 25

## Training Pipeline.
# Create model.
model = SimpleNN(X_train_tensor)
#print(model.weights)

#Inside the loop
for epoch in range(epochs):
    #Forward pass
    y_pred = model.forward(X_train_tensor)
    #print(y_pred)

    ## Compute loss
    loss = model.loss_function(y_pred, y_train_tensor)
    #print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    ## Backward pass
    loss.backward()

    #Update weights and bias.
    with torch.no_grad():
        model.weights -= learning_rate * model.weights.grad
        model.bias -= learning_rate * model.bias.grad
    
    # zero gradients
    model.weights.grad.zero_()
    model.bias.grad.zero_()

    #Print loss in each epoch
    print(f"Epoch {epoch+1}, loss: {loss.item()}")


#Evaluate the model
with torch.no_grad():
        y_pred = model.forward(X_test_tensor)
        # Original y_test values are 0 and 1. We need to convert the predicted values to 0 and 1.
        y_pred = (y_pred > 0.65).float()

        accuracy = (y_pred == y_test_tensor).float().mean()
        print(f"Test Accruacy: {accuracy.item()}")    