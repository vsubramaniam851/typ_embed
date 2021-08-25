import os
import sys
sys.path.insert(1, '../')
import numpy as np 
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.functional as F 
import torch.cuda as cuda

from postag_model import *
from embedding_models import *
from postag_data_load import *

# base_path = '/storage/vsub851/typ_embed/postag'
# data_path = '/storage/vsub851/typ_embed/datasets'
# train_filename = 'en_ewt-ud-train.conllu'
# test_filename = 'en_ewt-ud-test.conllu'

# seed = 0

if cuda.is_available():
	device = 'cuda'
	# torch.cuda.manual_seed_all(seed)
else:
	print('WARNING, this program is running on CPU')
	device = 'cpu'

def po_eval(base_path,
	test_corpus,
	eval_input,
	num_words,
	num_labels,
	modelname,
	word_embed_size = 100,
	encoder = 'lstm',
	lstm_hidden_size = 400,
	dropout = 0.25,
	lstm_layers = 3,
	bert = 'bert-based-uncased',
	bert_layer = 4,
	typological = False,
	typ_embed_size = 32,
	num_typ_features = 103,
	typ_feature = 'syntax_knn',
	typ_encode = 'concat',
	attention_hidden_size = 200,
	lang = 'en',
	device = 'cpu'):
	
	classifier = POSTaggingModel(n_words = num_words, n_tags = num_labels, word_embed_size = word_embed_size,  
		encoder = encoder, lstm_hidden_size = lstm_hidden_size, lstm_layers = lstm_layers, bert = bert, bert_pad_index = 0, dropout = dropout, n_bert_layer = bert_layer, 
		typological = typological, typ_embed_size = typ_embed_size, num_typ_features = num_typ_features, typ_encode = typ_encode, attention_hidden_size = 200)

	model_loc = os.path.join(base_path, 'saved_models', modelname)
	classifier.load_state_dict(torch.load(model_loc))
	
	classifier = classifier.to(device)
	classifier = classifier.double()
	classifier.eval()

	pos_total_examples = 0
	pos_total_correct = 0

	if typological:
		typ_str = 'with'
	else:
		typ_str = 'without'

	print('Evaluating {} model {} typological features'.format(modelname, typ_str))

	for i in tqdm(range(0, len(test_corpus))):
		sent_dict = test_corpus[i]
		if encoder == 'lstm':
			word_batch = []
			word_batch.append(torch.tensor(sent_dict[eval_input]).long().to(device))
			word_batch = torch.stack(word_batch).to(device)

			pos_preds= classifier.forward(words = word_batch, lang = lang, typ_feature = typ_feature, device = device)
		else:
			input_ids = []
			attention_mask = []
			word_batch = []
			input_ids.append(torch.tensor(sent_dict['input_ids']).long().to(device))
			attention_mask.append(torch.tensor(sent_dict['attention_mask']).long().to(device))

			input_ids = torch.stack(input_ids).to(device)
			attention_mask = torch.stack(attention_mask).to(device)

			pos_preds = classifier.forward(input_ids = input_ids, attention_mask = attention_mask, lang = lang, typ_feature = typ_feature, device = device)

		pos_tags = sent_dict['pos_tags']
		
		pos_preds = classifier.decode(pos_preds)

		pos_preds = pos_preds.squeeze(0)

		pos_tags = pos_tags.tolist()

		for i in range(1, len(pos_preds)):
			pos_pred = pos_preds[i]
			pos = pos_tags[i]

			if pos_pred == pos:
				pos_total_correct += 1
			pos_total_examples += 1

	return 'POS Accuracy {}'.format(pos_total_correct/pos_total_examples)

def test_eval(base_path,
	data_path,
	train_filename,
	test_filename,
	eval_input,
	modelname,
	word_embed_size = 100,
	pos_embed_size = 100,
	encoder = 'lstm',
	lstm_hidden_size = 400,
	dropout = 0.25,
	lstm_layers = 3,
	bert = 'bert-base-uncased',
	bert_layer = 7,
	typological = True,
	typ_embed_size = 32,
	num_typ_features = 103,
	typ_feature = 'syntax_knn',
	typ_encode = 'concat',
	attention_hidden_size = 200,
	lang = 'en',
	device = 'cpu'):
	
	file_path = os.path.join(data_path, 'UD_English-EWT')
	print('Loading data from training file {} and testing file {}'.format(train_filename, test_filename))
	train_lines = preproc_conllu(file_path, filename = train_filename)
	train_sent_collection = sentence_collection(train_lines)
	if eval_input == 'lemma_ids':
		input_type = 'lemma'
	else:
		input_type = 'form'
	train_corpus, vocab_dict, label_dict, pos_dict = process_corpus(train_sent_collection, mode = 'train', input_type = input_type)

	test_lines = preproc_conllu(file_path, filename = test_filename)
	test_sent_collection = sentence_collection(test_lines)
	test_corpus, _, _, _ = process_corpus(test_sent_collection, mode = 'test', vocab_dict = vocab_dict, label_dict = label_dict, pos_dict = pos_dict, input_type = input_type)
	print('Finished loading data')

	num_words = len(vocab_dict)
	num_labels = len(label_dict)
	num_pos = len(pos_dict)

	if encoder == 'bert':
		test_corpus = bert_tokenizer(test_corpus)

	print(arc_eval(base_path = base_path, test_corpus = test_corpus, eval_input = eval_input, num_words = num_words, num_pos = num_pos, num_labels = num_labels, modelname = modelname, word_embed_size = word_embed_size, pos_embed_size = pos_embed_size, encoder = encoder, lstm_hidden_size = lstm_hidden_size, dropout = dropout, 
		lstm_layers = lstm_layers, bert = bert, bert_layer = bert_layer, typological = typological, typ_embed_size = typ_embed_size, typ_feature = typ_feature, num_typ_features = num_typ_features, typ_encode = typ_encode, attention_hidden_size = attention_hidden_size, lang = lang, device = device))

# test_eval(base_path = base_path, train_filename = train_filename, test_filename = test_filename, eval_input = 'lemma_ids', modelname = 'dep5_lstm_typ.pt', dropout = 0.33, device = device,
# 	encoder = 'lstm')