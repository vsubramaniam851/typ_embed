# -*- coding: utf-8 -*-

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

from dep_model import *
from embedding_models import *
from dep_data_load import *

def arc_eval(args, classifier, test_loader, device):
	classifier = classifier.to(device)
	classifier = classifier.double()
	classifier.eval()

	uas_total_examples = 0
	uas_total_correct = 0
	las_total_examples = 0
	las_total_correct = 0

	if args.typological:
		typ_str = 'with'
	else:
		typ_str = 'without'

	print('Evaluating {} model {} typological features'.format(args.modelname, typ_str))

	for i, batch in tqdm(enumerate(test_loader)):
		if args.encoder == 'lstm':
			word_batch = batch['input_data'].to(device)
			pos_batch = batch['pos_ids'].to(device)

			s_arc, s_rel, mask = classifier.forward(words = word_batch, lang = lang, typ_feature = typ_feature, pos_tags = pos_batch, device = device)
		else:
			sentence = ' '.join([x[0] for x in batch['words']])
			input_ids = args.tokenizer.encode(sentence, return_tensors = 'pt')
			input_ids = input_ids.to(device)
			word_batch = batch['input_data'].to(device)

			s_arc, s_rel, mask = classifier.forward(words = word_batch, input_ids = input_ids, lang = args.lang, typ_feature = args.typ_feature, sentence = batch['words'], device = device)
		
		arcs = batch['heads'].squeeze(0)
		rels = batch['deprel_ids'].squeeze(0)

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

if __name__ == '__main__':
	args = SimpleNamespace()
	args.base_path = './'
	args.data_path = '../datasets'
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
	args.bert_layer = 4
	args.modelname = 'lstm_model_1.pt'
	args.scale = 0
	args.dropout = 0.33
	args.typological = False
	args.typ_embed_size = 32
	args.num_typ_features = 289
	args.typ_encode = 'concat'
	args.attention_hidden_size = 200
	args.typ_feature = 'syntax_knn+phonology_knn+inventory_knn'
	args.lang = 'en'
	args.save_model = False
	args.lr = 0.005
	args.tokenizer = transformers.BertTokenizer.from_pretrained(args.lm_model_name) if 'bert' in args.lm_model_name else transformers.GPT2Model.from_pretrained(args.lm_model_name)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	args.fine_tune = True

	train_data_loader, valid_data_loader, test_data_loader, vocab_dict, pos_dict, label_dict = dep_data_loaders(args, train_filename = train_filename, valid_filename = valid_filename, test_filename = test_filename)
	classifier = BiaffineDependencyModel(n_words = num_words, n_pos = num_pos, n_rels = num_labels, word_embed_size = args.word_embed_size, pos_embed_size = args.pos_embed_size, lstm_hidden_size = args.lstm_hidden_size, encoder = args.encoder, lstm_layers = args.lstm_layers, 
		lm_model_name = args.lm_model_name, dropout = args.dropout, n_lm_layer = args.lm_layer, n_arc_mlp = 500, n_rel_mlp = 100, scale = args.scale, pad_index = num_words, 
		unk_index = 0, typological = args.typological, typ_embed_size = args.typ_embed_size, num_typ_features = args.num_typ_features, 
		typ_encode = args.typ_encode, attention_hidden_size = args.attention_hidden_size, fine_tune = args.fine_tune)

	classifier.load_state_dict(torch.load(os.path.join(args.base_path, 'saved_models', args.modelname)))
	print(arc_eval(args, classifier, test_loader))
	pass
