import sys
import os
import numpy as np

import torch
import torch.nn as nn
import torch.Functional as f

def MLP(nn.Module):
	def __init__(self, n_in, n_out, mlp_hidden_size = 200, dropout = 0.33, activation = True):
		super(MLP, self).__init__()

		self.linear1 = nn.Linear(n_in, mlp_hidden_size)
		self.linear2 = nn.Linear(mlp_hidden_size, mlp_hidden_size)
		self.linear3 = nn.Linear(mlp_hidden_size, n_out)

		self.activation = nn.LeakyReLU(negative_slop = 0.1) if activation else nn.Identity()
		self.dropout = nn.Dropout(dropout)

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.orthogonal_(self.linear1.weight)
		nn.init.zeros_(self.linear1.bias)
		nn.init.orthogonal_(self.linear2.weight)
		nn.init.zeros_(self.linear2.bias)		
		nn.init.orthogonal_(self.linear2.weight)
		nn.init.zeros_(self.linear3.bias)

	def forward(self, x):
		x = self.linear1(x)
		x = self.linear2(x)
		x = self.linear3(x)
		x = self.activation(x)
		x = self.dropout(x)

		return x