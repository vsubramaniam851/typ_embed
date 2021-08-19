# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.functional as F 
import torch.autograd as autograd

class MLP(nn.Module):
	def __init__(self, n_in, n_out, dropout = 0.33, activation = True):
		super(MLP, self).__init__()

		self.linear1 = nn.Linear(n_in, 200)
		self.linear2 = nn.Linear(200, n_out)
		self.activation = nn.LeakyReLU(negative_slope = 0.1) if activation else nn.Identity()
		self.dropout = nn.Dropout(dropout)

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.orthogonal_(self.linear1.weight)
		nn.init.zeros_(self.linear1.bias)
		nn.init.orthogonal_(self.linear2.weight)
		nn.init.zeros_(self.linear2.bias)

	def forward(self, x):
		x = self.linear1(x)
		x = self.linear2(x)
		x = self.activation(x)
		x = self.dropout(x)

		return x

class Biaffine(nn.Module):
	def __init__(self, n_in, n_out = 1, scale = 0, bias_x = True, bias_y = True):
		super(Biaffine, self).__init__()

		self.n_in = n_in
		self.n_out = n_out
		self.scale = scale
		self.bias_x = bias_x
		self.bias_y = bias_y
		self.weight = nn.Parameter(torch.Tensor(n_out, n_in + bias_x, n_in + bias_y))

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.zeros_(self.weight)

	def forward(self, x, y):
		if self.bias_x:
			x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
		if self.bias_y:
			y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
		s = torch.einsum('bxi, oij, byj->boxy', x, self.weight, y) / self.n_in**self.scale
		s = s.squeeze(1)

		return s