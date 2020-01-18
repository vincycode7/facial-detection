## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Sequential, MaxPool2d, Conv2d, BatchNorm2d, ReLU
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
from collections import OrderedDict

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        #Conv layer
        
        self.feature1 = Sequential(OrderedDict([('batch_norm1', BatchNorm2d(1)),
                                                   ('conv1_0', Conv2d(1, 32, 5)),
                                                   ('relu1_0', ReLU()),
                                                   ('batch_norm2', BatchNorm2d(32)),
                                                   ('avg1_0',  MaxPool2d((3,3), stride=3)),
                                                   ('conv1_1', Conv2d(32, 64, 5)),
                                                   ('relu1_1', ReLU()),
                                                   ('batch_norm3', BatchNorm2d(64)),
                                                   ('avg1_1',  MaxPool2d((2,2), stride=2)),
                                                   ('conv1_2',  Conv2d(64, 150, 5)),
                                                   ('relu1_2', ReLU()),
                                                   ('batch_norm4', BatchNorm2d(150)),
                                                   ('maxp1_2',  MaxPool2d((2,2), stride=2)),
                                                   ('conv2_0',  Conv2d(150, 148, 3)),
                                                   ('relu2_0', ReLU()),
                                                   ('batch_norm5', BatchNorm2d(148)),
                                                   ('avg2_0',  MaxPool2d((3,3), stride=3)),
                                                   ('conv2_1',  Conv2d(148, 140, 3)),
                                                   ('relu2_1', ReLU()),
                                                   #('avg2_1',  nn.MaxPool2d((2,2), stride=1)),
                                                   ('batch_norm6', BatchNorm2d(140)),
                                                   ('conv2_2',  Conv2d(140, 136, 2)),
                                                   ]))

        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        
        # first feature pooling
        x = self.feature1(x).view(x.shape[0],-1)
        return x
