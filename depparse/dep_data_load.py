# -*- coding: utf-8 -*-

import sys
import os
import numpy as np 
import csv
from conll_df import conll_df
from types import SimpleNamespace
import pandas as pd

import torch
from torch.utils import data

class LoadConllu(object):
	def __init__(self, data_path, filename, mode = 'train', vocab_dict = None, pos_dict = None, label_dict = None, input_type = 'lemma'):
		file_path = os.path.join(data_path, filename)
		word_df = conll_df(file_path, file_index = False, skip_morph = True)
		self.sent_parses, self.vocab_dict, self.label_dict, self.pos_dict = self.process_corpus(word_df, mode = mode, input_type = input_type, vocab_dict = vocab_dict, pos_dict = pos_dict, label_dict = label_dict)

	def process_corpus(self, word_df, mode = 'train', input_type = 'form', vocab_dict = None, pos_dict = None, label_dict = None):
		if not vocab_dict:
			vocab_dict = {'UNK': 0, 'ROOT': 1}
		if not pos_dict:
			pos_dict = {'UNK': 0, 'ROOT': 1}
		if not label_dict:
			label_dict = {'UNK': 0, 'NO-REL': 1}
		sent_parses = []
		prev_s = None
		for s, i in word_df.index:
			if prev_s == s:
				continue
			prev_s = s
			deprels = ['NO-REL'] + word_df.loc[s, 'f'].tolist()
			lemmas = ['ROOT'] + word_df.loc[s, 'l'].tolist()
			pos = ['ROOT'] + word_df.loc[s, 'x'].tolist()
			heads = [0] + word_df.loc[s, 'g'].tolist()
			words = ['ROOT'] + word_df.loc[s, 'w'].tolist()
			lemma_ids, word_ids, pos_ids, deprel_ids = [vocab_dict['ROOT']], [vocab_dict['ROOT']], [pos_dict['ROOT']], [label_dict['NO-REL']]

			for w, l, p, d in zip(words[1:], lemmas[1:], pos[1:], deprels[1:]):
				if input_type == 'lemma':
					if l in vocab_dict:
						lemma_ids.append(vocab_dict[l])
					elif mode == 'train':
						lemma_ids.append(len(vocab_dict))
						vocab_dict[l] = len(vocab_dict)
					else:
						lemma_ids.append(vocab_dict['UNK'])

				elif input_type == 'form':
					if w in vocab_dict:
						word_ids.append(vocab_dict[w])
					elif mode == 'train':
						word_ids.append(len(vocab_dict))
						vocab_dict[w] = len(vocab_dict)
					else:
						word_ids.append(vocab_dict['UNK'])

				if d in label_dict:
					deprel_ids.append(label_dict[d])
				elif mode == 'train':
					deprel_ids.append(len(label_dict))
					label_dict[d] = len(label_dict)
				else:
					deprel_ids.append(label_dict['UNK'])

				if p in pos_dict:
					pos_ids.append(pos_dict[p])
				elif mode == 'train':
					pos_ids.append(len(pos_dict))
					pos_dict[p] = len(pos_dict)
				else:
					pos_ids.append(pos_dict['UNK'])

			sentence = ' '.join(words)

			assert(len(words) == (len(word_ids) if len(word_ids) > len(lemma_ids) else len(lemma_ids)) == len(deprel_ids) == len(heads)), 'Sizes are not the same'
			if len(word_ids) > len(lemma_ids):
				input_data = word_ids
			else:
				input_data = lemma_ids

			sent_parse = {'sent': words, 'input_data': input_data, 'deprel_ids': deprel_ids, 'heads': heads, 'pos_ids': pos_ids}
			sent_parses.append(pd.DataFrame.from_dict(sent_parse))
		return pd.concat(sent_parses, keys = [i for i in range(len(sent_parses))]), vocab_dict, label_dict, pos_dict

class DepData(data.Dataset):
	def __init__(self, dep_df):
		self.dep_df = dep_df 
	def __len__(self):
		return len(self.dep_df.index.levels[0])
	def __getitem__(self, idx):
		return {
			'words': self.dep_df.loc[idx, 'sent'].tolist(),
			'input_data': torch.tensor(self.dep_df.loc[idx, 'input_data'].tolist()),
			'pos_ids': torch.tensor(self.dep_df.loc[idx, 'pos_ids'].tolist()),
			'deprel_ids': torch.tensor(self.dep_df.loc[idx, 'deprel_ids'].tolist()).squeeze(),
			'heads': torch.tensor(self.dep_df.loc[idx, 'heads'].tolist()).squeeze()		
		}

def dep_data_loaders(args, train_filename, valid_filename, test_filename):
	train_conllu = LoadConllu(args.data_path, train_filename, mode = 'train')
	valid_conllu = LoadConllu(args.data_path, valid_filename, mode = 'valid', vocab_dict = train_conllu.vocab_dict, pos_dict = train_conllu.pos_dict, label_dict = train_conllu.label_dict)
	test_conllu = LoadConllu(args.data_path, test_filename, mode = 'test', vocab_dict = train_conllu.vocab_dict, pos_dict = train_conllu.pos_dict, label_dict = train_conllu.label_dict)

	train_dep_data, valid_dep_data, test_dep_data = DepData(train_conllu.sent_parses), DepData(valid_conllu.sent_parses), DepData(test_conllu.sent_parses)
	train_data_loader, valid_data_loader, test_data_loader = data.DataLoader(train_dep_data, batch_size = 1, shuffle = args.shuffle), data.DataLoader(valid_dep_data, batch_size = 1, shuffle = args.shuffle), data.DataLoader(test_dep_data, batch_size = 1, shuffle = args.shuffle)
	return train_data_loader, valid_data_loader, test_data_loader, train_conllu.vocab_dict, train_conllu.pos_dict, train_conllu.label_dict

if __name__ == '__main__':
	args = SimpleNamespace()
	args.data_path = '../datasets/UD_English-EWT'
	train_filename = 'en_ewt-ud-train.conllu'
	valid_filename = 'en_ewt-ud-dev.conllu'
	test_filename = 'en_ewt-ud-test.conllu'
	args.shuffle = False

	train_data_loader, valid_data_loader, test_data_loader, _, _, _ = dep_data_loaders(args, train_filename = train_filename, valid_filename = valid_filename, test_filename = test_filename)
	print(len(train_data_loader))
	for i in train_data_loader:
		line = i
		break