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

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if cuda.is_available():
	device = 'cuda'
	torch.cuda.manual_seed_all(seed)
else:
	print('WARNING, this program is running on CPU')
	device = 'cpu'

# lm_pretrained = transformers.BertModel.from_pretrained('bert-base-uncased').to(device)

base_path = '/storage/vsub851/typ_embed/postag'
data_path = '/storage/vsub851/typ_embed/datasets'
train_filename = 'en_ewt-ud-train.conllu'
valid_filename = 'en_ewt-ud-dev.conllu'

def pos_train(base_path,
	train_corpus,
	valid_corpus,
	train_type,
	num_words,
	num_labels,
	modelname,
	word_embed_size = 100,
	encoder = 'lstm',
	lstm_hidden_size = 400,
	mlp_hidden_size = 200,
	lr = 0.0005,
	dropout = 0.33,
	num_epochs = 3,
	lstm_layers = 3,
	batch_size = 1,
	bert = 'bert-base-uncased',
	bert_layer = 4,
	typological = False,
	typ_embed_size = 32,
	num_typ_features = 289,
	typ_feature = 'syntax_knn+phonology_knn+inventory_knn',
	typ_encode = 'concat',
	attention_hidden_size = 200,
	lang = 'en',
	device = 'cpu'):

	classifier = POSTaggingModel(n_words = num_words, n_tags = num_labels, word_embed_size = word_embed_size, lstm_hidden_size = lstm_hidden_size, encoder = encoder, lstm_layers = lstm_layers,
		bert = bert, dropout = dropout, n_bert_layer = bert_layer, mlp_hidden_size = mlp_hidden_size, typological = typological, typ_embed_size = typ_embed_size, num_typ_features = num_typ_features, 
		typ_encode = typ_encode, attention_hidden_size = attention_hidden_size)

	optimizer = optim.Adam(classifier.parameters(), lr = lr)

	classifier = classifier.to(device)
	classifier = classifier.double()
	classifier.train()

	if typological:
		typ_str = 'with'
	else:
		typ_str = 'without'

	print('Beginning training on {} with encoder {} and {} typological features'.format(train_type, encoder, typ_str))
	for epoch in range(num_epochs):
		total_loss = 0
		classifier.train()
		for i in range(0, len(train_corpus), batch_size):
			if i % 1000 == 0:
				print('Epoch {} Train Batch {}'.format(epoch, i))
			batch = train_corpus[i:i+batch_size]
			if encoder == 'lstm':
				word_batch = []
				pos_tags = []
				for sent in batch:
					word_batch.append(torch.tensor(sent[train_type]).long().to(device))
					pos_tags.append(torch.tensor(sent['pos_tags']).long().to(device))
				word_batch = torch.stack(word_batch).to(device)
				pos_tags = torch.stack(pos_tags).to(device)

				pos_tags = pos_tags.squeeze(0)

				pos_preds = classifier.forward(words = word_batch, lang = lang, typ_feature = typ_feature, device = device)
			else:
				input_ids = []
				attention_mask = []
				pos_tags = []
				for sent in batch:
					input_ids.append(torch.tensor(sent['input_ids']).long().to(device))
					attention_mask.append(torch.tensor(sent['attention_mask']).long().to(device))
					pos_tags.append(torch.tensor(sent['pos_tags']).long().to(device))
				input_ids = torch.stack(input_ids).to(device)
				attention_mask = torch.stack(attention_mask).to(device)
				pos_tags = torch.stack(pos_tags).to(device)

				pos_tags = pos_tags.squeeze(0)

				pos_preds = classifier.forward(input_ids = input_ids, attention_mask = attention_mask, lang = lang, typ_feature = typ_feature, device = device)

			loss = classifier.loss(pred_tags = pos_preds, tags = pos_tags)
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			total_loss = total_loss + loss.item()

		print('Epoch {}, train loss={}'.format(epoch, total_loss / len(train_corpus)))
		total_loss = 0
		classifier.eval()

		for i in range(0, len(valid_corpus), batch_size):
			if i%1000 == 0:
				print('Epoch {} Valid Batch {}'.format(epoch, i))
			batch = train_corpus[i:i+batch_size]
			if encoder == 'lstm':
				word_batch = []
				pos_tags = []
				for sent in batch:
					word_batch.append(torch.tensor(sent[train_type]).long().to(device))
					pos_tags.append(torch.tensor(sent['pos_tags']).long().to(device))
				word_batch = torch.stack(word_batch).to(device)
				pos_tags = torch.stack(pos_tags).to(device)

				pos_tags = pos_tags.squeeze(0)

				pos_preds = classifier.forward(words = word_batch, lang = lang, typ_feature = typ_feature, device = device)
			else:
				input_ids = []
				attention_mask = []
				pos_tags = []
				for sent in batch:
					input_ids.append(torch.tensor(sent['input_ids']).long().to(device))
					attention_mask.append(torch.tensor(sent['attention_mask']).long().to(device))
					pos_tags.append(torch.tensor(sent['pos_tags']).long().to(device))
				input_ids = torch.stack(input_ids).to(device)
				attention_mask = torch.stack(attention_mask).to(device)
				pos_tags = torch.stack(pos_tags).to(device)

				pos_tags = pos_tags.squeeze(0)

				pos_preds = classifier.forward(input_ids = input_ids, attention_mask = attention_mask, lang = lang, typ_feature = typ_feature, device = device)

			loss = classifier.loss(pred_tags = pos_preds, tags = pos_tags)
			total_loss = total_loss + loss.item()
		print('Epoch {}, valid loss = {}'.format(epoch, total_loss / len(valid_corpus)))
	print('TRAINING IS FINISHED')

	save_path = os.path.join(base_path, 'saved_models', modelname)
	torch.save(classifier.state_dict(), save_path)

def test_train(base_path,
	data_path,
	train_filename,
	valid_filename,
	train_type,
	modelname,
	word_embed_size = 100,
	encoder = 'lstm',
	lstm_hidden_size = 400,
	mlp_hidden_size = 200,
	lr = 0.0005,
	dropout = 0.33,
	num_epochs = 3,
	lstm_layers = 3,
	batch_size = 1,
	bert = 'bert-base-uncased',
	bert_layer = 4,
	typological = False,
	typ_embed_size = 32,
	num_typ_features = 289,
	typ_feature = 'syntax_knn+phonology_knn+inventory_knn',
	typ_encode = 'concat',
	attention_hidden_size = 200,
	lang = 'en',
	device = 'cpu'):

	#Load data
	print('Loading data from training file {} and validation file {}'.format(train_filename, valid_filename))

	file_path = os.path.join(data_path, 'UD_English-EWT')
	train_lines = preproc_conllu(file_path, filename = train_filename)
	train_sent_collection = sentence_collection(train_lines)
	if train_type == 'lemma_ids':
		input_type = 'lemma'
	else:
		input_type = 'form'

	train_corpus, vocab_dict, label_dict = process_corpus(train_sent_collection, mode = 'train', input_type = input_type)

	valid_lines = preproc_conllu(file_path, filename = valid_filename)
	valid_sent_collection = sentence_collection(valid_lines)
	valid_corpus, _, _ = process_corpus(valid_sent_collection, mode = 'valid', vocab_dict = vocab_dict, label_dict = label_dict, input_type = input_type)

	if encoder == 'bert':
		train_corpus = bert_tokenizer(train_corpus)
		valid_corpus = bert_tokenizer(valid_corpus)

	print('Data loading complete')

	pos_train(base_path = base_path, train_corpus = train_corpus, valid_corpus = valid_corpus, train_type = train_type, num_words = len(vocab_dict), num_labels = len(label_dict),
		modelname = modelname, word_embed_size = word_embed_size, encoder = encoder, lstm_hidden_size = lstm_hidden_size, mlp_hidden_size = mlp_hidden_size, lr = lr, dropout = dropout,
		num_epochs = num_epochs, lstm_layers = lstm_layers, batch_size = batch_size, bert = bert, bert_layer = bert_layer, typological = typological, typ_embed_size = typ_embed_size,
		num_typ_features = num_typ_features, typ_feature = typ_feature, typ_encode = typ_encode, attention_hidden_size = attention_hidden_size, lang = lang, device = device)

# test_train(base_path = base_path, data_path = data_path, train_filename = train_filename, valid_filename = valid_filename, train_type = 'lemma_ids', modelname = 'pos1_lstm.pt', device = device)