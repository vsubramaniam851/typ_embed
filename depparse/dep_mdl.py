import os
import sys
import numpy as np
import pandas as pd
import math
from types import Simplenamespace

import torch
import torch.cuda as cuda
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import transformers

from dep_data_load import *
from dep_train import *

class MDL_Probing(object):
	def __init__(self, args, filenames, timesteps, device):
		self.full_conllu = LoadConllu(data_path = args.data_path, filenames = filenames, mode = 'train', input_type = 'form')
		dataloaders = self.data_loaders(self.full_conllu.sent_parses, timesteps)
		self.K = len(full_conllu.label_dict)
		self.online_codelength = self.mdl_probing(args, dataloaders, device)
	def data_loaders(self, sent_parses, timesteps):
		dataloaders = []
		for t in timesteps:
			data_idx = int(t*0.01*len(sent_parses.index.levels[0]))
			part_conllu = sent_parses.iloc[:data_idx+1]
			half_idx = 0.5*len(part_conllu.index.levels[0])
			part_dep_train_dataset, part_dep_test_dataset = DepData(part_conllu.iloc[:half_idx]), DepData(part_conllu.iloc[half_idx:])
			dataloaders.append((data.DataLoader(part_dep_train_dataset, batch_size = 1, shuffle = False), data.DataLoader(part_dep_test_dataset, batch_size = 1, shuffle = False)))
		return dataloaders
	def mdl_training(self, args, train_loader, classifier, optimizer, device):
		for epoch in range(args.num_epochs):
			total_loss = 0
			classifier.train()
			for i, batch in enumerate(train_loader):
				if i % 1000 == 0:
					print('Epoch {} Train Batch {}'.format(epoch, i))

				if args.encoder == 'lstm':
					word_batch = batch['input_data'].to(device)
					pos_batch = batch['pos_ids'].to(device)
					arcs = batch['heads'].to(device)
					rels = batch['deprel_ids'].to(device)

					s_arc, s_rel, mask = classifier.forward(words = word_batch, lang = args.lang, typ_feature = args.typ_feature, pos_tags = pos_batch, device = device)

				else:
					sentence = ' '.join([x[0] for x in batch['words']])
					input_ids = args.tokenizer.encode(sentence, return_tensors = 'pt')
					input_ids = input_ids.to(device)
					word_batch = batch['input_data'].to(device)
					arcs = batch['heads'].to(device)
					rels = batch['deprel_ids'].to(device)

					s_arc, s_rel, mask = classifier.forward(words = word_batch, input_ids = input_ids, lang = args.lang, typ_feature = args.typ_feature, sentence = batch['words'], device = device)

				loss = classifier.loss(s_arc = s_arc, s_rel = s_rel, arcs = arcs, rels = rels, mask = mask)
				loss.backward()
				optimizer.step()
				optimizer.zero_grad()
				total_loss = total_loss + loss.item()

			print('Epoch {}, train loss={}'.format(epoch, total_loss / len(train_loader)))

		return classifier
	def mdl_eval(self, args, classifier, test_loader, device):
		codelength = 0
		classifier.eval()
		classifier.to(device)
		for i, batch in enumerate(test_loader):
			if args.encoder == 'lstm':
				word_batch = batch['input_data'].to(device)
				pos_batch = batch['pos_ids'].to(device)

				s_arc, s_rel, mask = classifier.forward(words = word_batch, lang = args.lang, typ_feature = args.typ_feature, pos_tags = pos_batch, device = device)

			else:
				sentence = ' '.join([x[0] for x in batch['words']])
				input_ids = args.tokenizer.encode(sentence, return_tensors = 'pt')
				input_ids = input_ids.to(device)
				word_batch = batch['input_data'].to(device)

				s_arc, s_rel, mask = classifier.forward(words = word_batch, input_ids = input_ids, lang = args.lang, typ_feature = args.typ_feature, sentence = batch['words'], device = device)
			
			arcs = batch['heads'].to(device)
			rels = batch['deprel_ids'].to(device)

			arc_preds = s_arc.argmax(-1)
			rel_dists = []
			for i in range(len(arc_preds[0])):
				idx = arc_preds[0][i]
				rel_dists.append(s_rel[:, i, x, :])
			rel_logits = torch.stack(rel_dists, dim = 0)
			rel_probs = F.softmax(rel_logits, dim = 1)

			for i, rel in enumerate(rels.squeeze(0)):
				prob_val = rel_probs[i, rel].item()
				codelength += np.log2(prob_val)
		return codelength
	def mdl_probing(self, args, dataloaders, device):
		term1 = 0.001*np.log2(self.K)
		classifier = BiaffineDependencyModel(n_words = len(self.full_conllu.vocab_dict), n_pos = len(self.full_conllu.pos_dict), n_rels = self.K, word_embed_size = args.word_embed_size, pos_embed_size = args.pos_embed_size, lstm_hidden_size = args.lstm_hidden_size, encoder = args.encoder, lstm_layers = args.lstm_layers, 
			lm_model_name = args.lm_model_name, tokenizer = args.tokenizer, dropout = args.dropout, n_lm_layer = args.lm_layer, n_arc_mlp = 500, n_rel_mlp = 100, scale = args.scale, pad_index = len(full_conllu.vocab_dict), 
			unk_index = 0, typological = args.typological, typ_embed_size = args.typ_embed_size, num_typ_features = args.num_typ_features, 
			typ_encode = args.typ_encode, attention_hidden_size = args.attention_hidden_size, fine_tune = args.fine_tune)
		optimizer = optim.Adam(classifier.parameters(), lr = args.lr)
		term2 = 0
		for train_loader, test_loader in dataloaders:
			trained_classifier = self.mdl_train(args, train_loader, classifier, optimizer, device)
			term2 += self.mdl_eval(args, train_classifier, test_loader, device)
		return term1 - term2

if __name__ == '__main__':
	args = Simplenamespace()
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
	args.typ_encode = 'add_att'
	args.typ_embed_size = 32
	args.typological = False
	args.num_typ_features = 289
	args.attention_hidden_size = 200
	args.typ_feature=  'syntax_knn+phonology_knn+inventory_knn'
	args.fine_tune = False
	args.lr = 0.000005
	timesteps = [0.2, 0.4, 0.8, 1.6, 3.2, 6.25, 12.5, 25, 50, 100]
	filenames = ['en_ewt-ud-train.conllu', 'en_ewt-ud-dev.conllu', 'en_ewt-ud-test.conllu']
	device = 'cuda' is cuda.is_available() else 'cpu'

	typ_str = 'with' + args.typ_feature if args.typological else 'without'
	print('Beginning MDL Evaluation on Dependency Parsing on device {} {} typological features on language {} using encoder {}'.format(device, typ_str, args.lang, args.encoder))

	mdl_probing = MDL_Probing(args, filenames, timesteps, device)
	print(mdl_probing.online_codelength)