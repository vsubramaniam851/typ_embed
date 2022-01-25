import sys
import os
import numpy as np
import pandas as pd

import torch
import torch.utils.data as data
import transformers

class EntailmentData(data.Dataset):
	def __init__(self, en_df):
		self.en_df = en_df
		self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
	def __len__(self):
		return len(self.en_df.index)
	def __getitem__(self, idx):
		return {
			'sentence_A': self.en_df.loc[idx, 'sentence_A'],
			'sentence_B': self.en_df.loc[idx, 'sentence_B'],
			'sentence_A_tokens': self.tokenizer.encode(self.en_df.loc[idx, 'sentence_A'], return_tensor = 'pt'),
			'sentence_B_tokens': self.tokenizer.encode(self.en_df.loc[idx, 'sentence_B'], return_tensor = 'pt'),
			'entailment_label': torch.tensor([self.en_df.loc[idx, 'entailment_label']])
		}

def entail_dataloaders(data_path, train_split, val_split):
	assert train_split+val_split < 1

	en_df = pd.read_csv(os.path.join(data_path, 'SICK.txt'), delimiter = '\t')
	en_dataset = EntailmentData(en_df)

	num_train, num_val = int(train_split*len(en_dataset)), int(val_split*len(en_dataset))
	num_test = len(en_dataset) - (num_train+num_val)

	train_en_dataset, val_en_dataset, test_en_dataset = data.random_split(en_dataset, lengths = [num_train, num_val, num_test])
	en_train_loader = data.DataLoader(train_en_dataset, batch_size = 1, shuffle = True)
	en_val_loader = data.DataLoader(val_en_dataset, batch_size = 1, shuffle = True)
	en_test_loader = data.DataLoader(test_en_dataset, batch_size = 1, shuffle = True)

	return en_train_loader, en_val_loader, en_test_loader