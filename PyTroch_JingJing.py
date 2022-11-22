#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html


# 
# [Learn the Basics](intro.html) ||
# **Quickstart** ||
# [Tensors](tensorqs_tutorial.html) ||
# [Datasets & DataLoaders](data_tutorial.html) ||
# [Transforms](transforms_tutorial.html) ||
# [Build Model](buildmodel_tutorial.html) ||
# [Autograd](autogradqs_tutorial.html) ||
# [Optimization](optimization_tutorial.html) ||
# [Save & Load Model](saveloadrun_tutorial.html)
# 
# # Quickstart
# This section runs through the API for common tasks in machine learning. Refer to the links in each section to dive deeper.
# 
# ## Working with data
# PyTorch has two [primitives to work with data](https://pytorch.org/docs/stable/data.html):
# ``torch.utils.data.DataLoader`` and ``torch.utils.data.Dataset``.
# ``Dataset`` stores the samples and their corresponding labels, and ``DataLoader`` wraps an iterable around
# the ``Dataset``.
# 

# In[3]:


import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# PyTorch offers domain-specific libraries such as [TorchText](https://pytorch.org/text/stable/index.html),
# [TorchVision](https://pytorch.org/vision/stable/index.html), and [TorchAudio](https://pytorch.org/audio/stable/index.html),
# all of which include datasets. For this tutorial, we  will be using a TorchVision dataset.
# 
# The ``torchvision.datasets`` module contains ``Dataset`` objects for many real-world vision data like
# CIFAR, COCO ([full list here](https://pytorch.org/vision/stable/datasets.html)). In this tutorial, we
# use the FashionMNIST dataset. Every TorchVision ``Dataset`` includes two arguments: ``transform`` and
# ``target_transform`` to modify the samples and labels respectively.
# 
# 

# In[4]:


#Import Jingjing's data
import pickle
filename = 'Cluster_selected_ML.pkl'
unpickleFile = open(filename, 'rb')
new_dict = pickle.load(unpickleFile)
print(new_dict)

#The parameter that we wish to predict is f_true;
#haloid_clu and lambda_true is not observable, so should not be used.


# In[5]:


#Visualise data
classes= ['cluster_id', 'lambda_true', 'lambda_cluster', 'f_true','e1z','e2z','cos_theta', 'tf_min_value']
y = new_dict["f_true"]
for class_now in classes:
    x = new_dict[class_now]
    plt.hist(x)
    plt.xlabel(class_now)
    plt.show()
    plt.clf()


# In[6]:


#Put data into arrays
y = new_dict["f_true"]

X1 = new_dict['lambda_true']
#X1 = new_dict['lambda_cluster']
X1 = X1-np.mean(X1)#normalisation
X1 = X1/np.std(X1)

#X2 = new_dict['f_true']
X2 = new_dict['e1z']
X2 = X2-np.mean(X2)#normalisation
X2 = X2/np.std(X2)

X3 = new_dict['e2z']
X3 = X3-np.mean(X3)#normalisation
X3 = X3/np.std(X3)

X4 = new_dict['cos_theta']
X4 = X4-np.mean(X4)#normalisation
X4 = X4/np.std(X4)

X5 = new_dict['tf_min_value']
X5 = X5-np.mean(X5)#normalisation
X5 = X5/np.std(X5)


# In[7]:


from torch.utils.data import TensorDataset, DataLoader
X=[]
for i in range(len(X1)):
    X.append([X1[i],X2[i],X3[i],X4[i],X5[i]])

y = np.reshape(y,[11542,1])
    
#Help from here: https://stackoverflow.com/questions/65017261/how-to-input-a-numpy-array-to-a-neural-network-in-pytorch
tensor_X = torch.Tensor(X) # transform to torch tensor
tensor_y = torch.Tensor(y)


# We pass the ``Dataset`` as an argument to ``DataLoader``. This wraps an iterable over our dataset, and supports
# automatic batching, sampling, shuffling and multiprocess data loading. Here we define a batch size of 64, i.e. each element
# in the dataloader iterable will return a batch of 64 features and labels.
# 
# 

# In[8]:


print(np.shape(tensor_X))
print(np.shape(tensor_y))
Ntot = 11542


# Read more about [loading data in PyTorch](data_tutorial.html).
# 
# 
# 

# --------------
# 
# 
# 

# ## Creating Models
# To define a neural network in PyTorch, we create a class that inherits
# from [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html). We define the layers of the network
# in the ``__init__`` function and specify how data will pass through the network in the ``forward`` function. To accelerate
# operations in the neural network, we move it to the GPU if available.
# 
# 

# In[9]:


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
#            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


# Read more about [building neural networks in PyTorch](buildmodel_tutorial.html).
# 
# 
# 

# --------------
# 
# 
# 

# ## Optimizing the Model Parameters
# To train a model, we need a [loss function](https://pytorch.org/docs/stable/nn.html#loss-functions)
# and an [optimizer](https://pytorch.org/docs/stable/optim.html).
# 
# 

# In[10]:


loss_fn = nn.MSELoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# In a single training loop, the model makes predictions on the training dataset (fed to it in batches), and
# backpropagates the prediction error to adjust the model's parameters.
# 
# 

# In[11]:


def train(tensor_X, tensor_y, model, loss_fn, optimizer):
    size = Ntot #len(dataloader.dataset)
    model.train()
#    for batch, (X, y) in enumerate(dataloader):
    for i in [0]:#need loop later?
        X, y = tensor_X.to(device), tensor_y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
#        optimizer.Adam()
        loss.backward()
        optimizer.step()

        if True:
#        if batch % 100 == 0:
            loss =  loss.item()
            print("loss:",loss)


# We also check the model's performance against the test dataset to ensure it is learning.
# 
# 

# In[12]:


def test(tensor_X, tensor_y, model, loss_fn):
    size = Ntot#len(dataloader.dataset)
    num_batches = 1#len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for i in [0]:
            X, y = tensor_X.to(device), tensor_y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
#            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    loss_arr.append(test_loss)
    plt.scatter(tensor_y.to("cpu").numpy(),pred.to("cpu").numpy())
    plt.xlabel("Ground Truth")
    plt.ylabel("Predicted")
    plt.plot([0,1],[0,1],"k",linestyle="dashed")
    plt.show()


# The training process is conducted over several iterations (*epochs*). During each epoch, the model learns
# parameters to make better predictions. We print the model's accuracy and loss at each epoch; we'd like to see the
# accuracy increase and the loss decrease with every epoch.
# 
# 

# In[13]:


loss_arr = []

epochs = 128
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(tensor_X, tensor_y, model, loss_fn, optimizer)
    test(tensor_X, tensor_y, model, loss_fn)
print("Done!")


# In[14]:


plt.plot(loss_arr)
#plt.yscale("log")


# In[ ]:




