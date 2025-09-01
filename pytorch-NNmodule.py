## Sample PyTorch code using nn.Module facility.

'''
import torch
import torch.nn as nn
from torchinfo import summary

class Model(nn.Module):
    def __init__(self, features):
        super().__init__()

    #Structure of the model.
        self.network = nn.Sequential(
            nn.Linear(features, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )
    
    # Forward pass
    def forward(self, features):
        output = self.network(features)
        return output

#Create a random dataset
dataset = torch.rand(10, 5)

#Create the model
myModel = Model(dataset.shape[1])

#Calling the model
#myModel(dataset)
print(myModel(dataset))

#Summary
summary(myModel, input_size=(10, 5))
'''



## Training Pipeline in Pytorch using nn.Module facility.

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchinfo import summary
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv")

def DataInfo(dataframe):
    print(dataframe.head())
    print("\n", dataframe.info())
    print("\n", dataframe.shape)

DataInfo(df)

def preprocessData(dataframe):
    dataframe.drop(columns=['id', 'Unnamed: 32'], inplace=True)
    print("\nAfter preprocessing:\n", dataframe.head())

preprocessData(df)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df.iloc[:, 0], test_size=0.2, random_state=42)

def scaleNencode(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = scaleNencode(X_train, X_test, y_train, y_test)

# Convert numpy arrays to torch tensors
X_train_tensor = torch.from_numpy(X_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1)
y_test_tensor = torch.from_numpy(y_test).float().view(-1, 1)

# Define the model
class SimpleNN(nn.Module):
    def __init__(self, features):
        super().__init__()
        # self.linear = nn.Linear(features, 1)
        # self.sigmoid = nn.Sigmoid()
        self.network = nn.Sequential(
            nn.Linear(features, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        y_pred = self.network(features)
        return y_pred

# Important parameters
learning_rate = 0.01
epochs = 50

# Create model
model = SimpleNN(X_train_tensor.shape[1])
summary(model, input_size=(64, X_train_tensor.shape[1]))

# Define loss and optimizer
loss_function = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    # Forward pass
    y_pred = model(X_train_tensor)

    # Compute loss
    loss = loss_function(y_pred, y_train_tensor)

    # Backward pass
    optimizer.zero_grad() #clear grads before backward pass.
    loss.backward()
    optimizer.step()


    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluation
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_classes = (y_pred > 0.5).float()
    accuracy = (y_pred_classes == y_test_tensor).float().mean()
    print(f"\nTest Accuracy: {accuracy.item()*100:.2f}%")
