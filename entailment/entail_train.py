# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '../')
import os
import random
import numpy as np 
from types import SimpleNamespace

import torch
from torch import cuda
import torch.nn as nn
import torch.optim as optim 

from entail_model import *
from entail_data_load import *

import transformers

def entail_train(args, train_loader, valid_loader, num_labels, device):
	classifier = EnMLP(n_rels = num_labels, tokenizer = args.tokenizer, lm_model_name = args.lm_model_name, model_type = args.model_type, typological = args.typological, 
		typ_embed_size = args.typ_embed_size, num_typ_features = args.num_typ_features, typ_encode = args.typ_encode, n_lm_layer = args.n_lm_layer, attention_hidden_size = args.attention_hidden_size,
		fine_tune = args.fine_tune, extract_cls = args.extract_cls, average_sen = args.average_sen, mlp_hidden_size = args.mlp_hidden_size, dropout = args.dropout)

	optimizer = optim.Adam(classifier.parameters(), lr = args.lr)

	classifier = classifier.to(device)
	classifier = classifier.double()
	classifier.train()

	if args.typological:
		typ_str = 'with'
	else:
		typ_str = 'without'

	print('Beginning training on {} with encoder {} and {} typological features'.format(args.input_type, args.encoder, typ_str))

	for epoch in range(num_epochs):
		total_loss = 0
		classifier.train()
		for i, batch in enumerate(train_loader):
			if i%1000 == 0:
				print('Epoch {} Train Batch {}'.format(epoch, i))

			sent1 = batch['sentence_A_tokens'].to(device)
			sent2 = batch['sentence_B_tokens'].to(device)
			label = batch['entailment_label'].to(device)

			entail_pred = classifier.forward(sent1, sent2, lang = args.lang, typ_feature = arg.typ_feature, device = args.device)

			loss = classifier.loss(pred_label = entail_pred, label = label)
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			total_loss = total_loss + loss.item()

		print('Epoch {}, train loss={}'.format(epoch, total_loss / len(train_loader)))
		total_loss = 0
		classifier.eval()

		for i, batch in enumerate(valid_loader):
			if i%1000 == 0:
				print('Epoch {} Valid Batch {}'.format(epoch, i))

			sent1 = batch['sentence_A_tokens'].to(device)
			sent2 = batch['sentence_B_tokens'].to(device)
			label = batch['entailment_label'].to(device)

			entail_pred = classifier.forward(sent1, sent2, lang = args.lang, typ_feature = arg.typ_feature, device = device)

			loss = classifier.loss(pred_label = entail_pred, label = label)
			total_loss = total_loss + loss.item()
		print('Epoch {}, valid loss = {}'.format(epoch, total_loss / len(valid_corpus)))
	print('TRAINING IS FINISHED')

	if args.save_model:
		save_path = os.path.join(args.base_path, 'saved_models', args.modelname)
		print('Saving model to {}'.format(save_path))
		torch.save(classifier.state_dict(), save_path)

	return classifier

if __name__ == '__main__':
	args = SimpleNamespace()
	args.base_path = './'
	args.data_path = '../datasets/SICK'
	args.lm_model_name = 'bert-base-uncased'
	args.model_type = 'concat'
	args.n_lm_layer = -1
	args.dropout = 0.33
	args.typological = False
	args.typ_embed_size = 32
	args.num_typ_features = 289
	args.typ_encode = 'concat'
	args.attention_hidden_size = 200
	args.typ_feature = 'syntax_knn+phonology_knn+inventory_knn'
	args.lang = 'en'
	args.save_model = False
	args.modelname = 'mlp_model_1.pt'
	args.lr = 0.005
	args.tokenizer = transformers.BertTokenizer.from_pretrained(args.lm_model_name) if 'bert' in args.lm_model_name else transformers.GPT2Model.from_pretrained(args.lm_model_name)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	args.fine_tune = False	
	args.extract_cls = True 
	args.average_sen = False
	args.mlp_hidden_size = 200
	args.save_model = True

	train_loader, val_loader, test_loader = entail_dataloaders(data_path = args.data_path, train_split = 0.8, val_split = 0.1)
	classifier = entail_train(args, train_loader, val_loader, 3, device = device)