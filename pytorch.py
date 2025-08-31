import torch
print(torch.__version__)

if torch.cuda.is_available():
    print("CUDA is present: ", torch.cuda.get_device_name(0))
else:
    print("CUDA not available.")


## Creating a tensor.

## 1. Using empty
tensor1 = torch.empty(2,3)
print(tensor1)
print(tensor1.type)

## 2. Using zeros
tensor2 = torch.zeros(2,3)
print(tensor2)
print(tensor2.type)

## 3. Using rand
tensor3 = torch.rand(2,3)
print(tensor3)
print(tensor2.type)

# Rand function using seed
print("Torch rand using seed.\n")
torch.manual_seed(100)
tensor4 = torch.rand(2,3)
print(tensor4)

## 5. Using tensor
tensor5 = torch.tensor([[1,2,3], [4,5,6]])
print(tensor5)

## Identity matrix
tensor6 = torch.eye(5)
print(tensor6)

# 6. Using full
tensor7 = torch.full((3,2), 10)
print(tensor7)


## Tensor Shapes

x = torch.tensor([[1,2,3], [4,5,6]])
print(x.shape)

y = torch.empty_like(x)
print(y, y.shape)


## Tensor Datatypes

print(x.dtype)

## Creating tensor with specific datatype
x1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype = torch.int32)
print(x1.dtype)

## Changing datatype of tensor
x2 = x1.type(torch.float32)
print(x2.dtype)




## Mathematical operations on tensors
#Scalar operations.
a = torch.tensor([[1,2,3], [4,5,6]])
b = torch.tensor([[7,8,9], [10,11,12]])

# Addition
a = a + 2
print(a)

#Subtraction
b = b - 2
print(b)

#Multiplication
a = a * 100
print(a)

#Division
a = a // 7
print(a)

# Modulus
b = b % 5
print(b)

# Power
a = a * 2
print(a)

## Element-wise operations
print("Element-wise operations\n")
a = torch.rand(2,3)
b = torch.rand(2,3)
# Addition
addTensor = a + b
print(addTensor)

#Subtraction
subTensor = a - b
print(subTensor)

#Multiplication
mulTensor = a * b
print(mulTensor)

#Division
divTensor = a // b
print(divTensor)