import os
import sys
sys.path.insert(1, '/storage/vsub851/typ_embed')
import numpy as np 
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.functional as F 
import torch.cuda as cuda

from dep_model import *
from embedding_models import *
from dep_data_load import *

base_path = '/storage/vsub851/typ_embed/depparse'
train_filename = 'en_ewt-ud-train.conllu'
test_filename = 'en_ewt-ud-test.conllu'

seed = 0

if cuda.is_available():
	device = 'cuda'
	torch.cuda.manual_seed_all(seed)
else:
	print('WARNING, this program is running on CPU')
	device = 'cpu'

def arc_eval(base_path,
	test_corpus,
	eval_input,
	num_words,
	num_pos,
	num_labels,
	modelname,
	word_embed_size = 100,
	pos_embed_size = 100,
	encoder = 'lstm',
	lstm_hidden_size = 400,
	dropout = 0.25,
	lstm_layers = 4,
	bert = 'bert-based-uncased',
	bert_layer = 4,
	scale = 0,
	typological = False,
	typ_embed_size = 32,
	num_typ_features = 289,
	typ_feature = 'syntax_knn+phonology_knn+inventory_knn',
	lang = 'en',
	device = 'cpu'):
	
	pad_index = num_words
	classifier = BiaffineDependencyModel(n_words = num_words, n_pos = num_pos, n_rels = num_labels, word_embed_size = word_embed_size, pos_embed_size = pos_embed_size,  
		encoder = encoder, lstm_hidden_size = lstm_hidden_size, lstm_layers = lstm_layers, bert = bert, bert_pad_index = 0, dropout = dropout, n_bert_layer = bert_layer, 
		n_arc_mlp = 500, n_rel_mlp = 100, scale = scale, pad_index = pad_index, unk_index = 0, typological = typological, typ_embed_size = typ_size, num_typ_features = num_typ_features)

	model_loc = os.path.join(base_path, 'checkpoints', modelname)
	classifier.load_state_dict(torch.load(model_loc))
	
	classifier = classifier.to(device)
	classifier.eval()

	uas_total_examples = 0
	uas_total_correct = 0
	las_total_examples = 0
	las_total_correct = 0

	for i in tqdm(range(0, len(test_corpus))):
		sent_dict = test_corpus[i]
		if encoder == 'lstm':
			word_batch = []
			pos_batch = []
			word_batch.append(torch.tensor(sent_dict[eval_input]).long().to(device))
			pos_batch.append(torch.tensor(sent_dict['pos_ids']).long().to(device))
			word_batch = torch.stack(word_batch).to(device)
			pos_batch = torch.stack(pos_batch).to(device)

			s_arc, s_rel, mask = classifier.forward(words = word_batch, lang = lang, typ_feature = typ_feature, pos_tags = pos_batch, device = device)
		else:
			input_ids = []
			attention_mask = []
			word_batch = []
			input_ids.append(torch.tensor(sent['input_ids']).long().to(device))
			attention_mask.append(torch.tensor(sent['attention_mask']).long().to(device))
			word_batch.append(torch.tensor(sent_dict[eval_input]).long().to(device))

			input_ids = torch.stack(input_ids).to(device)
			attention_mask = torch.stack(attention_mask).to(device)
			word_batch = torch.stack(word_batch).to(device)

			s_arc, s_rel, mask = classifier.forward(words = word_batch, input_ids = input_ids, attention_mask = attention_mask, lang = lang, typ_feature = typ_feature, device = device)

		arcs = sent_dict['heads']
		rels = sent_dict['deprel_ids']

		arc_preds, rel_preds = classifier.decode(s_arc = s_arc, s_rel = s_rel, mask = mask)

		arc_preds = arc_preds.squeeze(0)
		rel_preds = rel_preds.squeeze(0)

		arc_preds = arc_preds.tolist()
		rel_preds = rel_preds.tolist()

		for i in range(1, len(arc_preds)):
			arc_pred = arc_preds[i]
			rel_pred = rel_preds[i]
			arc = arcs[i]
			rel = rels[i]

			if arc_pred == arc:
				uas_total_correct += 1
				if rel_pred == rel:
					las_total_correct += 1
			uas_total_examples += 1
			las_total_examples += 1

	return 'UAS Score {}, LAS Score {}'.format(uas_total_correct/uas_total_examples, las_total_correct/las_total_examples)

def test_eval(base_path,
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
	bert = 'bert-based-uncased',
	bert_layer = 4,
	scale = 0,
	typological = False,
	typ_embed_size = 32,
	num_typ_features = 289,
	typ_feature = None,
	device = 'cpu'):
	
	file_path = os.path.join(base_path, 'UD_English-EWT')
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

	print(arc_eval(base_path = base_path, test_corpus = test_corpus, eval_input = eval_input, num_words = num_words, num_pos = num_pos, num_labels = num_labels, modelname = modelname, word_embed_size = word_embed_size, pos_embed_size = pos_embed_size, encoder = encoder, lstm_hidden_size = lstm_hidden_size, dropout = dropout, 
		lstm_layers = lstm_layers, bert = bert, bert_layer = bert_layer, scale = scale, typological = typological, typ_size = typ_size, typ_features = typ_features, typ_feature_vec = typ_feature_vec))

test_eval(base_path = base_path, train_filename = train_filename, test_filename = test_filename, eval_input = 'word_ids', modelname = 'dep1_lstm.pt', dropout = 0.33, device = device)