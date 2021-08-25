import sys
import os
import numpy as np
import pandas as pd 
import random
import argparse
import unittest
import lang2vec.lang2vec as l2v

import torch
import torch.cuda as cuda

from dep_train import *
from dep_eval import *

seed = 0
if cuda.is_available():
	device = 'cuda'
	torch.cuda.manual_seed_all(seed)
else:
	print('WARNING, this program is running on CPU')
	device = 'cpu'

def get_cmd_arguments_dep():
	ap = argparse.ArgumentParser()

	ap.add_argument('-p', '--path', action = 'store', type = str, dest = 'base_path', default = '/storage/vsub851/typ_embed/depparse',
		help = 'Base path to all Dependency Parsing models and data')
	ap.add_argument('-d', '--data', action = 'store', type = str, dest = 'data_path', default = '/storage/vsub851/typ_embed/datasets', 
		help = 'Dataset location')
	ap.add_argument('-t', '--train', action = 'store', type = bool, dest = 'train_model', default = False,
		help = 'Train a new model, saved in saved_models directory in depparse directory')
	ap.add_argument('-m', '--model', action = 'store', type = str, dest = 'modelname', default = 'dep_model.pt', 
		help = 'Name of saved model that is either being trained or being evaluated. Most be stored in saved_models directory')
	ap.add_argument('-l', '--lang', action = 'store', type = str, dest = 'lang', default = 'en',
		help = 'Language to run dependency parsing model on')
	ap.add_argument('-i', '--input', action = 'store', type = str, dest = 'input_type', default = 'form',
		help = 'Type of input to run through LSTM, either form or lemma')
	ap.add_argument('-ty', '--typological', action = 'store', type = bool, dest = 'typological', default = False,
		help = 'Include typological features in training')
	ap.add_argument('-tf', '--typfeatures', action = 'store', type = str, dest = 'typ_feature', default = 'syntax_knn',
		help = 'Which typological features to extract from the typological database')
	ap.add_argument('-e', '--encoder', action = 'store', type = str, dest = 'encoder', default = 'lstm',
		help = 'Word Embedding model, either BERT or LSTM')
	ap.add_argument('-te', '--typencode', action = 'store', type = str, dest = 'typ_encode', default = 'concat',
		help = 'Method to use for incorporating typological features. Choose from [concat, add_att, mul_att] to decide to either use a concatentation or attention method')

	#Model Hyperparameters
	ap.add_argument('-wes', '--wordsize', action = 'store', dest = 'word_embed_size', type = int, default = 100, 
		help = 'Word Embedding Size for model')
	ap.add_argument('-pes', '--possize', action = 'store', dest = 'pos_embed_size', type = int, default = 100,
		help = 'POS Embedding Size for model')
	ap.add_argument('-lhs', '--lstmsize', action = 'store', dest = 'lstm_hidden_size', type = int, default = 400, 
		help = 'LSTM Hidden size when using encoder LSTM')
	ap.add_argument('-ahs', '--attentionsize', action = 'store', dest = 'attention_hidden_size', type = int, default = 200,
		help = 'Multiplicative Attention Hidden Size')
	ap.add_argument('-ll', '--lstmlayers', action = 'store', dest = 'lstm_layers', type = int, default = 3,
		help = 'Number of LSTM Layers in LSTM encoder')
	ap.add_argument('-dr', '--dropout', action = 'store', dest = 'dropout', type = float, default = 0.33,
		help = 'Dropout probability to be used in all components of model')
	ap.add_argument('-b', '--bert', action = 'store', dest = 'bert', type = str, default = 'bert-base-uncased',
		help = 'BERT Model to use when using BERT as encoder')
	ap.add_argument('-tes', '--typsize', action = 'store', dest = 'typ_embed_size', type = int, default = 32,
		help = 'Embedding size for typological embedding vector')
	ap.add_argument('-nt', '--numtyp', action = 'store', type = int, dest = 'num_typ_features', default = 103,
		help = 'Number of typological features in the typological features extracted.')
	ap.add_argument('-bl', '--bertlayer', action = 'store', dest = 'bert_layer', type = int, default = 8,
		help = 'Layer to obtain BERT representations from')
	ap.add_argument('-sc', '--scale', action = 'store', dest = 'scale', type = float, default = 0,
		help = 'Scaling factor for biaffine attention')
	ap.add_argument('-lr', '--learningrate', action = 'store', dest = 'lr', type = float, default = 0.0005, 
		help = 'Learning rate for optimization')
	ap.add_argument('-ep', '--numepochs', action = 'store', dest = 'num_epochs', type = int, default = 10,
		help = 'Number of epochs of training')

	return ap.parse_args()

def dep_main(train_filename,
	valid_filename,
	test_filename,
	lang,
	base_path,
	data_path,
	data_directory,
	train_model,
	input_type,
	word_embed_size,
	pos_embed_size,
	modelname,
	encoder,
	lstm_hidden_size,
	lr,
	dropout,
	num_epochs,
	lstm_layers,
	batch_size,
	bert,
	bert_layer,
	scale,
	typological,
	typ_embed_size,
	num_typ_features,
	typ_feature,
	typ_encode,
	attention_hidden_size,
	device):
	print('Starting Dependency Parsing')

	print('Loading data in language {} from training file {}, validation file {}, and testing file {}'.format(lang, train_filename, valid_filename, test_filename))

	file_path = os.path.join(data_path, data_directory)
	train_lines = preproc_conllu(file_path, filename = train_filename)
	train_sent_collection = sentence_collection(train_lines)
	train_corpus, vocab_dict, label_dict, pos_dict = process_corpus(train_sent_collection, mode = 'train', input_type = input_type)

	if train_model:
		valid_lines = preproc_conllu(file_path, filename = valid_filename)
		valid_sent_collection = sentence_collection(valid_lines)
		valid_corpus, _, _, _ = process_corpus(valid_sent_collection, mode = 'valid', vocab_dict = vocab_dict, label_dict = label_dict,
			pos_dict = pos_dict, input_type = input_type)

		if encoder == 'bert':
			train_corpus = bert_tokenizer(train_corpus)
			valid_corpus = bert_tokenizer(valid_corpus)

		print('Data Loading Complete')

		if input_type == 'form':
			input_type = 'word_ids'
		else:
			input_type = 'lemma_ids'

		arc_train(base_path = base_path, train_corpus = train_corpus, valid_corpus = valid_corpus, train_type = input_type, num_words = len(vocab_dict), 
			num_pos = len(pos_dict), num_labels = len(label_dict), modelname = modelname, word_embed_size = word_embed_size, pos_embed_size = pos_embed_size, encoder = encoder,
			lstm_hidden_size = lstm_hidden_size, lr = lr, dropout = dropout, num_epochs = num_epochs, lstm_layers = lstm_layers, 
			batch_size = batch_size, bert = bert, bert_layer = bert_layer, scale = scale, typological = typological, typ_embed_size = typ_embed_size,
			num_typ_features = num_typ_features, typ_feature = typ_feature, typ_encode = typ_encode, attention_hidden_size = attention_hidden_size, lang = lang, device = device)
	else:
		test_lines = preproc_conllu(file_path, filename = test_filename)
		test_sent_collection = sentence_collection(test_lines)
		test_corpus, _, _, _ = process_corpus(test_sent_collection, mode = 'test', vocab_dict = vocab_dict, label_dict = label_dict,
			pos_dict = pos_dict, input_type = args.input_type)

		if encoder == 'bert':
			test_corpus = bert_tokenizer(test_corpus)

		print('Data Loading Complete')

		if input_type == 'form':
			input_type = 'word_ids'
		else:
			input_type = 'lemma_ids'

		print(arc_eval(base_path = base_path, test_corpus = test_corpus, eval_input = input_type, num_words = len(vocab_dict), num_pos = len(pos_dict), 
			num_labels = len(label_dict), modelname = modelname, word_embed_size = word_embed_size, pos_embed_size = pos_embed_size, encoder = encoder,
			lstm_hidden_size = lstm_hidden_size, dropout = dropout, lstm_layers = lstm_layers, bert = bert, bert_layer = bert_layer,
			scale = scale, typological = typological, typ_embed_size = typ_embed_size, typ_feature = typ_feature, num_typ_features = num_typ_features,
			typ_encode = typ_encode, attention_hidden_size = attention_hidden_size, lang = lang, device = device))

if __name__ == '__main__':
	print('Using device: {}'.format(device)) #Ensure on GPU!

	args = get_cmd_arguments_dep()

	seed = 0
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	if args.lang == 'en':
		directory = 'UD_English-EWT'
		train_filename = 'en_ewt-ud-train.conllu'
		valid_filename = 'en_ewt-ud-dev.conllu'
		test_filename = 'en_ewt-ud-test.conllu'
	else:
		directory = None
		train_filename = None
		valid_filename = None
		test_filename = None
		raise AssertionError('Please enter a valid language')

	debug = unittest.TestCase()
	debug.assertTrue(os.path.exists(args.base_path), msg = 'Base path does not exist')	
	debug.assertTrue(os.path.exists(args.data_path), msg = 'Data path does not exist')

	assert(args.encoder in ['bert', 'lstm']), 'Please choose either BERT or LSTM to build word embeddings'
	assert(args.input_type in ['lemma', 'form']), 'Please choose an input type of form or lemma'
	assert(args.typ_encode in ['concat', 'add_att', 'mul_att']), 'Please use attention or concatention for encoding typological features'

	dep_main(train_filename = train_filename,
		valid_filename = valid_filename,
		test_filename = test_filename,
		lang = args.lang,
		base_path = args.base_path,
		data_path = args.data_path,
		data_directory = directory,
		train_model = args.train_model,
		input_type = args.input_type,
		word_embed_size = args.word_embed_size,
		pos_embed_size = args.pos_embed_size,
		modelname = args.modelname,
		encoder = args.encoder,
		lstm_hidden_size = args.lstm_hidden_size,
		lr = args.lr,
		dropout = args.dropout,
		num_epochs = args.num_epochs,
		lstm_layers = args.lstm_layers,
		batch_size = 1,
		bert = args.bert,
		bert_layer = args.bert_layer,
		scale = args.scale,
		typological = args.typological,
		typ_embed_size = args.typ_embed_size,
		num_typ_features = args.num_typ_features,
		typ_feature = args.typ_feature,
		typ_encode = args.typ_encode,
		attention_hidden_size = args.attention_hidden_size,
		device = device)