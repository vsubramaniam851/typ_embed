# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import math
from types import SimpleNamespace

import torch
import torch.cuda as cuda
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import transformers

from pos_data_load import *
from pos_train import *

class MDL_Probing(object):
	def __init__(self, args, filenames, timesteps, device):
		self.full_conllu = LoadConllu(data_path = args.data_path, filenames = filenames, mode = 'train', input_type = 'form')
		dataloaders = self.data_loaders(self.full_conllu.sent_parses, timesteps)
		self.K = len(self.full_conllu.label_dict)
		self.online_codelength = self.mdl_probing(args, dataloaders, device)
	def data_loaders(self, sent_parses, timesteps):
		pos_dataset = PosData(sent_parses)
		index_pairs, dataloaders = [], []
		for t in timesteps:
			data_idx = int(t*0.01*len(pos_dataset))
			train_idx = round(0.5*data_idx)
			index_pairs.append((train_idx, data_idx))
		for train_idx, test_idx in index_pairs:
			train_pos_dataset, test_pos_dataset = data.Subset(en_dataset, range(0, train_idx)), data.Subset(en_dataset, range(train_idx, test_idx))
			dataloaders.append((data.DataLoader(train_pos_dataset, batch_size = 1, shuffle = False), data.DataLoader(test_pos_dataset, batch_size = 1, shuffle = False)))
		return dataloaders
	def mdl_training(self, args, train_loader, classifier, optimizer, device):
		print('Beginning Training')
		for epoch in range(args.num_epochs):
			total_loss = 0
			classifier.train()
			classifier = classifier.to(device)
			for i, batch in enumerate(train_loader):
				if args.encoder == 'lstm':
					word_batch = batch['input_data'].to(device)

					pos_preds = classifier.forward(words = word_batch, lang = args.lang, typ_feature = args.typ_feature, device = device)

				else:
					sentence = ' '.join([x[0] for x in batch['words']])
					input_ids = args.tokenizer.encode(sentence, return_tensors = 'pt')
					input_ids = input_ids.to(device)
					word_batch = batch['input_data'].to(device)

					pos_preds = classifier.forward(words = word_batch, input_ids = input_ids, lang = args.lang, typ_feature = args.typ_feature, sentence = batch['words'], device = device)

				pos_tags = batch['pos_ids'].squeeze(0).to(device)
				loss = classifier.loss(pred_tags = pos_preds, tags = pos_tags)
				loss.backward()
				optimizer.step()
				optimizer.zero_grad()
				total_loss = total_loss + loss.item()

			print('Epoch {}, train loss={}'.format(epoch, total_loss / len(train_loader)))

		return classifier
	def mdl_eval(self, args, classifier, test_loader, device):
		print('Beginning Evaluation')
		codelength = 0
		classifier.eval()
		classifier.to(device)
		for i, batch in enumerate(test_loader):
			if args.encoder == 'lstm':
				word_batch = batch['input_data'].to(device)

				pos_preds = classifier.forward(words = word_batch, lang = args.lang, typ_feature = args.typ_feature, device = device)

			else:
				sentence = ' '.join([x[0] for x in batch['words']])
				input_ids = args.tokenizer.encode(sentence, return_tensors = 'pt')
				input_ids = input_ids.to(device)
				word_batch = batch['input_data'].to(device)

				s_arc, s_rel, mask = classifier.forward(words = word_batch, input_ids = input_ids, lang = args.lang, typ_feature = args.typ_feature, sentence = batch['words'], device = device)
			
			pos_tags = batch['pos_ids']
			pos_logits = pos_preds.argmax(-1)

			pos_probs = nn.functional.softmax(pos_logits, dim = 1)

			for i, pos in enumerate(pos_tags.squeeze(0)):
				prob_val = pos_probs[i, rel].item()
				codelength += np.log2(prob_val)
		return codelength
	def mdl_probing(self, args, dataloaders, device):
		term1 = 0.001*np.log2(self.K)
		classifier = POSTaggingModel(n_words = num_words, n_tags = num_labels, word_embed_size = args.word_embed_size, lstm_hidden_size = args.lstm_hidden_size, encoder = args.encoder, lstm_layers = args.lstm_layers,
			lm_model_name = args.lm_model_name, tokenizer = args.tokenizer, dropout = dropout, n_lm_layer = args.lm_layer, mlp_hidden_size = args.mlp_hidden_size, typological = args.typological, typ_embed_size = args.typ_embed_size, num_typ_features = args.num_typ_features, 
			typ_encode = args.typ_encode, attention_hidden_size = args.attention_hidden_size, fine_tune = args.fine_tune)
		optimizer = optim.Adam(classifier.parameters(), lr = args.lr)
		classifier = classifier.double()
		term2 = 0
		for i, (train_loader, test_loader) in enumerate(dataloaders):
			print('TIMESTEP {}'.format(i))
			trained_classifier = self.mdl_training(args, train_loader, classifier, optimizer, device)
			term2 += self.mdl_eval(args, trained_classifier, test_loader, device)
		return term1 - term2

if __name__ == '__main__':
	args = SimpleNamespace()
	args.lang = 'en'
	args.data_path = '../datasets/UD_English-EWT'
	args.lm_model_name = 'bert-base-uncased'
	args.word_embed_size = 200
	args.pos_embed_size = 200
	args.lstm_hidden_size = 400
	args.encoder = 'lm'
	args.lstm_layers = 3
	args.tokenizer = transformers.BertTokenizer.from_pretrained(args.lm_model_name)
	args.lm_layer = 8
	args.scale = 0
	args.dropout = 0.33
	args.typ_encode = 'add_att'
	args.typ_embed_size = 32
	args.typological = True
	args.num_typ_features = 289
	args.attention_hidden_size = 200
	args.num_epochs = 50
	args.typ_feature=  'syntax_knn+phonology_knn+inventory_knn'
	args.fine_tune = False
	args.lr = 0.000005
	timesteps = [0.2, 0.4, 0.8, 1.6, 3.2, 6.25, 12.5, 25, 50, 100]
	filenames = ['en_ewt-ud-train.conllu', 'en_ewt-ud-dev.conllu', 'en_ewt-ud-test.conllu']
	device = 'cuda' if cuda.is_available() else 'cpu'

	typ_str = 'with ' + args.typ_feature if args.typological else 'without'
	print('Beginning MDL Evaluation on POS Tagging on device {} {} typological features on language {} using encoder {}'.format(device, typ_str, args.lang, args.encoder))

	mdl_probing = MDL_Probing(args, filenames, timesteps, device)
	print(mdl_probing.online_codelength)