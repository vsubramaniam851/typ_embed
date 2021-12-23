# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, '../')
import os
import random
import numpy as np 

import torch
from torch import cuda
import torch.nn as nn
import torch.optim as optim 

from postag_model import *
from postag_data_load import *

import transformers

def pos_train(args, train_loader, valid_loader, num_words, num_labels, device):
	classifier = POSTaggingModel(n_words = num_words, n_tags = num_labels, word_embed_size = args.word_embed_size, lstm_hidden_size = args.lstm_hidden_size, encoder = args.encoder, lstm_layers = args.lstm_layers,
		lm_model_name = args.lm_model_name, dropout = dropout, n_lm_layer = args.lm_layer, mlp_hidden_size = args.mlp_hidden_size, typological = args.typological, typ_embed_size = args.typ_embed_size, num_typ_features = args.num_typ_features, 
		typ_encode = args.typ_encode, attention_hidden_size = args.attention_hidden_size, fine_tune = args.fine_tune)

	optimizer = optim.Adam(classifier.parameters(), lr = args.lr)

	classifier = classifier.to(device)
	classifier = classifier.double()
	classifier.train()

	if typological:
		typ_str = 'with'
	else:
		typ_str = 'without'

	print('Beginning training on {} with encoder {} and {} typological features'.format(args.input_type, args.encoder, args.typ_str))
	for epoch in range(num_epochs):
		total_loss = 0
		classifier.train()
		for i, batch in enumerate(train_loader):
			if i % 1000 == 0:
				print('Epoch {} Train Batch {}'.format(epoch, i))
			if encoder == 'lstm':
				word_batch = batch['input_data'].to(device)
				pos_batch = batch['pos_ids'].squeeze(0).to(device)

				pos_preds = classifier.forward(words = word_batch, lang = args.lang, typ_feature = args.typ_feature, pos_tags = pos_batch, device = device)
			else:
				sentence = ' '.join([x[0] for x in batch['words']])
				input_ids = args.tokenizer.encode(sentence, return_tensors = 'pt')
				input_ids = input_ids.to(device)
				word_batch = batch['input_data'].to(device)
				pos_batch = batch['pos_ids'].squeeze(0).to(device)

				pos_preds = classifier.forward(words = word_batch, input_ids = input_ids, lang = args.lang, typ_feature = args.typ_feature, sentence = batch['words'], device = device)

			loss = classifier.loss(pred_tags = pos_preds, tags = pos_tags)
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
			if encoder == 'lstm':
				word_batch = batch['input_data'].to(device)
				pos_batch = batch['pos_ids'].squeeze(0).to(device)

				pos_preds = classifier.forward(words = word_batch, lang = args.lang, typ_feature = args.typ_feature, pos_tags = pos_batch, device = device)
			else:
				sentence = ' '.join([x[0] for x in batch['words']])
				input_ids = args.tokenizer.encode(sentence, return_tensors = 'pt')
				input_ids = input_ids.to(device)
				word_batch = batch['input_data'].to(device)
				pos_batch = batch['pos_ids'].squeeze(0).to(device)

				pos_preds = classifier.forward(words = word_batch, input_ids = input_ids, lang = args.lang, typ_feature = args.typ_feature, sentence = batch['words'], device = device)

			loss = classifier.loss(pred_tags = pos_preds, tags = pos_tags)
			total_loss = total_loss + loss.item()
		print('Epoch {}, valid loss = {}'.format(epoch, total_loss / len(valid_corpus)))
	print('TRAINING IS FINISHED')

	save_path = os.path.join(base_path, 'saved_models', modelname)
	torch.save(classifier.state_dict(), save_path)

if __name__ == '__main__':
	args = SimpleNamespace()
	args.base_path = './'
	args.data_path = '../datasets/UD_English-EWT'
	train_filename = 'en_ewt-ud-train.conllu'
	valid_filename = 'en_ewt-ud-dev.conllu'
	test_filename = 'en_ewt-ud-test.conllu'
	args.shuffle = False
	args.word_embed_size = 100
	args.pos_embed_size = 100
	args.encoder = 'lstm'
	args.lstm_hidden_size = 400
	args.lstm_layers = 3
	args.lm_model_name = 'bert-base-uncased'
	args.lm_layer = 4
	args.dropout = 0.33
	args.typological = False
	args.typ_embed_size = 32
	args.num_typ_features = 289
	args.typ_encode = 'concat'
	args.attention_hidden_size = 200
	args.typ_feature = 'syntax_knn+phonology_knn+inventory_knn'
	args.lang = 'en'
	args.save_model = False
	args.modelname = 'lstm_model_1.pt'
	args.lr = 0.005
	args.tokenizer = transformers.BertTokenizer.from_pretrained(args.lm_model_name) if 'bert' in args.lm_model_name else transformers.GPT2Model.from_pretrained(args.lm_model_name)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	train_data_loader, valid_data_loader, test_data_loader, vocab_dict, pos_dict, label_dict = dep_data_loaders(args, train_filename = train_filename, valid_filename = valid_filename, test_filename = test_filename)
	classifier = arc_train(args, train_loader, valid_loader, num_words = len(vocab_dict), num_pos = len(pos_dict), num_labels = len(label_dict))
	pass