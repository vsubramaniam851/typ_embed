import sys
sys.path.insert(1, '/storage/vsub851/typ_embed')
import os
import random
import numpy as np 

import torch
from torch import cuda
import torch.nn as nn
import torch.optim as optim

from dep_model import *
from embedding_models import *
from dep_data_load import *

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

base_path = '/storage/vsub851/typ_embed/depparse'
train_filename = 'en_ewt-ud-train.conllu'
valid_filename = 'en_ewt-ud-dev.conllu'

print('Using device: {}'.format(device)) #Ensure on GPU!

lm_pretrained = transformers.BertModel.from_pretrained('bert-base-uncased').to(device)

def arc_train(base_path,
	train_corpus,
	valid_corpus,
	train_type, 
	num_words, 
	num_pos,
	num_labels, 
	modelname, 
	word_embed_size = 100, 
	pos_embed_size = 100,
	encoder = 'lstm', 
	lstm_hidden_size = 400,
	lr = 0.005, 
	dropout = 0.33, 
	num_epochs = 3, 
	lstm_layers = 3, 
	batch_size = 1, 
	bert = 'bert-base-uncased', 
	bert_layer = 4, 
	scale = 0,
	typological = False,
	typ_size = 200,
	typ_features = 289,
	typ_feature_vec = None):
	'''Train the model. Specify the model type and hyperparameters. The LSTM model takes in lemmas in a sentence and predicts its heads and dependencies
	The LM model uses the input ids and attention masks instead. 

	Pass in the training type, either on lemma ids or word ids.
	'''
	pad_index = num_words
	classifier = BiaffineDependencyModel(n_words = num_words, n_pos = num_pos, n_rels = num_labels, word_embed_size = word_embed_size, pos_embed_size = pos_embed_size, lstm_hidden_size = lstm_hidden_size, encoder = encoder, lstm_layers = lstm_layers, 
		bert = bert, bert_pad_index = 0, dropout = dropout, n_bert_layer = bert_layer, 
		n_arc_mlp = 500, n_rel_mlp = 100, scale = scale, pad_index = pad_index, 
		unk_index = 0, typological = typological, typ_size = typ_size, typ_features = typ_features)

	optimizer = optim.Adam(classifier.parameters(), lr = lr)

	classifier = classifier.to(device)
	classifier.train()

	#Collect correct label types
	label1 = 'heads'
	label2 = 'deprel_ids'

	#Training loop
	print('Beginning training on {} with encoder {}'.format(train_type, encoder))
	for epoch in range(num_epochs):
		total_loss = 0
		classifier.train()
		for i in range(0, len(train_corpus), batch_size):
			if i % 1000 == 0:
				print('Epoch {} Train Batch {}'.format(epoch, i))
			batch = train_corpus[i:i+batch_size]
			if encoder == 'lstm':
				word_batch = []
				pos_batch = []
				arcs = []
				rels = []
				for sent in batch:
					word_batch.append(torch.tensor(sent[train_type]).long().to(device))
					pos_batch.append(torch.tensor(sent['pos_ids']).long().to(device))
					arcs.append(torch.tensor(sent[label1]).long().to(device))
					rels.append(torch.tensor(sent[label2]).long().to(device))
				word_batch = torch.stack(word_batch).to(device)
				pos_batch = torch.stack(pos_batch).to(device)
				arcs = torch.stack(arcs).to(device)
				# arcs = arcs.squeeze(0)
				rels = torch.stack(rels).to(device)
				# rels = rels.squeeze(0)

				s_arc, s_rel, mask = classifier.forward(words = word_batch, lang_typ = typ_feature_vec, pos_tags = pos_batch)
			else:
				input_ids = []
				attention_mask = []
				word_batch = []
				arcs = []
				rels = []
				for sent in batch:
					word_batch.append(torch.tensor(sent[train_type]).long().to(device))
					input_ids.append(torch.tensor(sent['input_ids']).long().to(device))
					attention_mask.append(torch.tensor(sent['attention_mask']).long().to(device))
					arcs.append(torch.tensor(sent[label1]).long().to(device))
					rels.append(torch.tensor(sent[label2]).long().to(device))
				word_batch = torch.stack(word_batch).to(device)
				input_ids = torch.stack(input_ids).to(device)
				attention_mask = torch.stack(attention_mask).to(device)
				arcs = torch.stack(arcs).to(device)
				# arcs = arcs.squeeze(0)
				rels = torch.stack(rels).to(device)
				# rels = rels.squeeze(0)
		
				s_arc, s_rel, mask = classifier.forward(words = word_batch, input_ids = input_ids, attention_mask = attention_mask)

			#Calculate loss and step backwards through the model.
			loss = classifier.loss(s_arc = s_arc, s_rel = s_rel, arcs = arcs, rels = rels, mask = mask)
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			total_loss = total_loss + loss.item()

		print('Epoch {}, train loss={}'.format(epoch, total_loss / len(train_corpus)))
		total_loss = 0
		classifier.eval()

		for i in range(0, len(valid_corpus), batch_size):
			if i % 1000 == 0:
				print('Epoch {} Valid Batch {}'.format(epoch, i))
			batch = valid_corpus[i:i+batch_size]
			if encoder == 'lstm':
				word_batch = []
				pos_batch = []
				arcs = []
				rels = []
				for sent in batch:
					word_batch.append(torch.tensor(sent[train_type]).long().to(device))
					pos_batch.append(torch.tensor(sent['pos_ids']).long().to(device))
					arcs.append(torch.tensor(sent[label1]).long().to(device))
					rels.append(torch.tensor(sent[label2]).long().to(device))
				word_batch = torch.stack(word_batch).to(device)
				pos_batch = torch.stack(pos_batch).to(device)
				arcs = torch.stack(arcs).to(device)
				# arcs = arcs.squeeze(0)
				rels = torch.stack(rels).to(device)
				# rels = rels.squeeze(0)

				s_arc, s_rel, mask = classifier.forward(words = word_batch, lang_typ = typ_feature_vec, pos_tags = pos_batch)
			else:
				input_ids = []
				attention_mask = []
				word_batch = []
				arcs = []
				rels = []
				for sent in batch:
					word_batch.append(torch.tensor(sent[train_type]).long().to(device))
					input_ids.append(torch.tensor(sent['input_ids']).long().to(device))
					attention_mask.append(torch.tensor(sent['attention_mask']).long().to(device))
					arcs.append(torch.tensor(sent[label1]).long().to(device))
					rels.append(torch.tensor(sent[label2]).long().to(device))
				word_batch = torch.stack(word_batch).to(device)
				input_ids = torch.stack(input_ids).to(device)
				attention_mask = torch.stack(attention_mask).to(device)
				arcs = torch.stack(arcs).to(device)
				# arcs = arcs.squeeze(0)
				rels = torch.stack(rels).to(device)
				# rels = rels.squeeze(0)
		
				s_arc, s_rel, mask = classifier.forward(words = word_batch, input_ids = input_ids, attention_mask = attention_mask, lang_typ = typ_feature_vec)

			#Calculate loss and add to total loss.
			loss = classifier.loss(s_arc = s_arc, s_rel = s_rel, arcs = arcs, rels = rels, mask = mask)
			total_loss += loss.item()
		print('Epoch {}, valid loss={}'.format(epoch, total_loss / len(valid_corpus)))
	print('TRAINING IS FINISHED')
	#Save model using modelname passed in as parameter
	save_path = os.path.join(base_path, 'checkpoints', modelname)
	torch.save(classifier.state_dict(), save_path)

def test_train(base_path, 
	train_filename, 
	valid_filename,
	modelname, 
	train_type, 
	word_embed_size = 100, 
	pos_embed_size = 100,
	encoder = 'lstm', 
	lstm_hidden_size = 400,
	lr = 0.0005, 
	dropout = 0.33, 
	num_epochs = 3, 
	lstm_layers = 3, 
	batch_size = 1,
	bert = 'bert-base-uncased',
	bert_layer = 4,
	scale = 0,
	typological = False,
	typ_size = 200,
	typ_features = 289,
	typ_feature_vec = None):
	
	#Load data
	print('Loading data from training file {} and validation file {}'.format(train_filename, valid_filename))

	file_path = os.path.join(base_path, 'UD_English-EWT')
	train_lines = preproc_conllu(file_path, filename = train_filename)
	train_sent_collection = sentence_collection(train_lines)
	if train_type == 'lemma_ids':
		input_type = 'lemma'
	else:
		input_type = 'form'
	train_corpus, vocab_dict, label_dict, pos_dict = process_corpus(train_sent_collection, mode = 'train', input_type = input_type)

	# print(len(train_corpus))

	valid_lines = preproc_conllu(file_path, filename = valid_filename)
	valid_sent_collection = sentence_collection(valid_lines)
	valid_corpus, _, _, _= process_corpus(valid_sent_collection, mode = 'valid', vocab_dict = vocab_dict, label_dict = label_dict, pos_dict = pos_dict, input_type = input_type)

	if encoder == 'bert':
		train_corpus = bert_tokenizer(train_corpus)
		valid_corpus = bert_tokenizer(valid_corpus)

	print('Data loading complete')

	arc_train(base_path = base_path, train_corpus = train_corpus, valid_corpus = valid_corpus, train_type = train_type ,num_words = len(vocab_dict), num_pos = len(pos_dict), num_labels = len(label_dict), modelname = modelname, word_embed_size = word_embed_size, 
		pos_embed_size = pos_embed_size, encoder = encoder, lstm_hidden_size = lstm_hidden_size, lr = lr, dropout = dropout, num_epochs = num_epochs, lstm_layers = lstm_layers, batch_size = batch_size, bert = bert, bert_layer = bert_layer, scale = scale, 
		typological = typological, typ_size = typ_size, typ_features = typ_features, typ_feature_vec = typ_feature_vec)

test_train(base_path = base_path, train_filename = train_filename, valid_filename = valid_filename, modelname = 'dep1_lstm.pt', train_type = 'word_ids', num_epochs = 10, 
	encoder = 'lstm', dropout = 0.33)