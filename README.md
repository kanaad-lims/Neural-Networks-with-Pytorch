## Neural Networks with PyTorch.
- Using Pytorch for building simple to complex NNs including ANNs, RNNs, CNNs, LSTMs and Transfer Learning.
- We will also be optimizing the neural networks using hyperparameter tuning using `Optuna`.

1. pytorch.py - Consists all the basic and important PyTorch functions, essential to begin with.
2. pytorch-autograd.py - PyTorch's state of the art differentiation feature (gradient calculations) and it's implementation.
3. pytorch-Tpipeline.py - Emulating a typical Training Pipeline for Deep Learning projects by building a very simple neural network. No use of standard functions, self written model, forward, loss, backprop and evaluation functions.
4. pytorch-NNmodule.py - Uses the PyTorch's Neural Network module `nn.Module` along with loss function and optimizers `torch.optim...` and imporving the previously written NN in `pytorch-Tpipeline.py`.
5. pytorch-allData - Used the Dataset and Dataloader classes in pytorch to demonstrate how data can be divided and arranged into different batches using a sample dataset `from sklearn.datasets import make_classification`.
