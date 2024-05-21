import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, in_dim:int) -> None:
        """Structure of our FCN

        Args:
            in_dim (int): Number of input variables
        """
        super().__init__()
        self.hidden1 = nn.Linear(in_dim,32) # 32 hidden neurons
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(32,16) # 16 hidden neurons
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(16,8) # 8 hidden neurons
        self.act3 = nn.ReLU()
        self.out = nn.Linear(8,1)
        self.act_out = nn.Sigmoid() # to get probability
    
    def forward(self, x:np.ndarray)-> np.ndarray:
        """One forward pass of input through our FCN

        Args:
            x (np.ndarray): Input features

        Returns:
            np.ndarray: Output probabilities for the batch
        """
        x1 = self.act1(self.hidden1(x))
        x2 = self.act2(self.hidden2(x1))
        x3 = self.act3(self.hidden3(x2))
        y = self.act_out(self.out(x3))

        return y
