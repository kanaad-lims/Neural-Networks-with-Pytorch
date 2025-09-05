## Building an ANN using pytorch.

## Dataset - Using Kaggle - fashion MNIST (70000 images)
## We will be drawing a subset of 6000 images.

## Workflow
    # Dataloader objects for training and testing data.
    # Training loop.
    # Evaluation of model.


import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(42)

df = pd.read_csv("fmnist_small.csv")
print(df.head)
print("Dataset loaded successfully")

# Create a 4x4 grid of images
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle("First 16 Images", fontsize=16)

'''
# Plot the first 16 images from the dataset
for i, ax in enumerate(axes.flat):
    img = df.iloc[i, 1:].values.reshape(28, 28)  # Reshape to 28x28
    ax.imshow(img)  # Display in grayscale
    ax.axis('off')  # Remove axis for a cleaner look
    ax.set_title(f"Label: {df.iloc[i, 0]}")  # Show the label

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
plt.show()
'''

# getting features and labels
X = df.iloc[:, 1:].values #col 1st and next
y = df.iloc[:, 0].values #col 0th

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scaling the features
X_train = X_train/255.0
X_test = X_test/255.0

print("X_train", X_train)

#Creating a custom dataset class

class customDataset(Dataset):

    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    

#Create dataset objects.
train_dataset_object = customDataset(X_train, y_train)
test_dataset_object = customDataset(X_test, y_test)

#Creating train and test loader.
train_loader = DataLoader(train_dataset_object, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset_object, batch_size=32, shuffle=False)

# 80% of 6000 = 4800 and 32 images in each batch.
# Hence total number of batches will be 150, with 32 images each for training data.
# For testing purpose, number of batches will be 38, with 32 images in each batch.


#Creating custom neural network class.
class NeuralANN(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.structure = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            #nn.Softmax() -> already done in Cross Entropy Loss
        )

    def forward(self, X):
        return self.structure(X)
    

#Setting learning rate and epochs
learning_rate = 0.1
epochs = 200

#Instantiate the model
ANNModel = NeuralANN(X_train.shape[1])

#Loss function
loss_function = nn.CrossEntropyLoss()

#Optimizer
optimizer = optim.SGD(ANNModel.parameters(), lr=learning_rate)


#Training loop

for epoch in range(epochs):
    total_epoch_loss = 0
    for batch_features, batch_labels in train_loader:

        #Forward pass
        y_pred = ANNModel(batch_features)

        #Loss calculation
        loss = loss_function(y_pred, batch_labels)

        #Backward pass
        optimizer.zero_grad()
        loss.backward()

        #Update grads
        optimizer.step()

        total_epoch_loss += loss.item()
    
    avg_loss = total_epoch_loss/len(train_loader)
    if(epoch%10 == 0):
        print(f"Epoch: {epoch} Loss: {avg_loss}")
print("Model trained successfully\n")


#Evaluation
# Setting the model into evaluation mode.
print(ANNModel.eval())
    
total = 0
correct = 0

with torch.no_grad():

    for batch_features, batch_labels in test_loader:
         
        output = ANNModel(batch_features)
        

        _, predicted = torch.max(output, 1) #Extracting the label for the maximum probabilistic value returned by the model.
        
        total = total + batch_labels.shape[0]
        correct = correct + (predicted == batch_labels).sum().item()
    

accuracy = correct/total
print(f"Test Accuracy: {accuracy * 100:.4f}%")

#improving the accuracy.