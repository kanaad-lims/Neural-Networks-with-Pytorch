import torch
import torch.nn.functional as F
print(torch.__version__)


## Autograd for Differentiation in Pytorch
# dy/dx for f(x) = y = x^2

'''
x = torch.tensor(3.0, requires_grad=True)
y = x**2

print(x)
print(y)

#dy/dx
y.backward()
print(x.grad)
'''

'''
# y = x^2 and z = sin(y)
x = torch.tensor(3.0, requires_grad=True)
y = x**2
z = torch.sin(y)

print(x)
print(y)
print(z)

z.backward()
print(x.grad)
'''

## Neural Network (simple perceptron) using autograd.
# CGPA and Placement chance.
#X -> CGPA, Y -> Placement chance

x = torch.tensor(7.8, requires_grad=False)
y = torch.tensor(0.0, requires_grad=False)

w = torch.tensor(1.0, requires_grad=True) #Weight
b = torch.tensor(0.0, requires_grad=True) #Bias

#Forward Pass
z = w*x + b
print("Z: ", z)

y_pred = torch.sigmoid(z)
print("Y_pred: ", y_pred)


# Loss calculation
loss = F.binary_cross_entropy(y_pred, y)
print("Loss: ", loss)

#Backpropogation
loss.backward()
print("dL/dw: ", w.grad)
print("dL/db: ", b.grad)

## Evaluation, though important, is out of scope right now.
