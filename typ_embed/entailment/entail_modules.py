import torch
import torch.nn as nn
import torch.functional as F 
import torch.autograd as autograd

class MLP(nn.Module):
	def __init__(self, n_in, n_out, dropout = 0.33, activation = True):
		super(MLP, self).__init__()

		self.linear1 = nn.Linear(n_in, hidden_size)
		self.linear2 = nn.Linear(hidden_size, n_out)
		self.activation = nn.LeakyReLU(negative_slope = 0.1) if activation else nn.Identity()
		self.dropout = nn.Dropout(dropout)

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.orthogonal_(self.linear1.weight)
		nn.init.zeros_(self.linear1.bias)
		nn.init.orthogonal_(self.linear2.weight)
		nn.init.zeros_(self.linear2.bias)

	def forward(self, x1, x2):
		x = torch.concat([x1, x2], dim = 1)
		x = self.linear1(x)
		x = self.activation(x)
		x = self.linear2(x)
		x = self.dropout(x)

		return x

class Bilinear(nn.Module):
	def __init__(self, n_in, n_out, dropout = 0.33, activation = True):
		super(Bilinear, self).__init__()

		self.bilinear = nn.Bilinear(n_in, n_in, n_out)
		self.activation = nn.LeakyReLU(negative_slop = 0.1) if activation else nn.Identity()
		self.dropout = nn.Dropout(dropout)

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.orthogonal_(self.bilinear.weight)
		nn.init.zeros_(self.bilinear.bias)

	def forward(self, x):
		x = self.bilinear(x)
		x = self.activation(x)
		x = self.dropout(x)

		return x