from sklearn.datasets import make_classification
import torch
from torch.utils.data import Dataset, DataLoader

#Creating synthetic classification dataset

X, y = make_classification(
    n_samples=15,
    n_features = 2,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

print(X, X.shape)

#Converting numpy arrays into tensors.
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)


## Creating custom dataset class

class customdataset(Dataset):
    
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self): #Returns number of rows in the dataset.
        return self.features.shape[0]
    
    def __getitem__(self, index): #Returns the actual row in the dataset present at the given index
        return self.features[index], self.labels[index]
    

dataset = customdataset(X, y)
print("Custom dataset class created")

print(len(dataset))

print(dataset[0])

dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
print("Dataloader class created \n")

for batch_features, batch_labels in dataloader:
    print(batch_features)
    print(batch_labels)
    print("-"*50)