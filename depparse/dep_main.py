# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd 
import random
import argparse
import lang2vec.lang2vec as l2v

import torch
import torch.cuda as cuda

from dep_data_load import *
from dep_train import *
from dep_eval import *

def get_cmd_arguments_dep():
	ap = argparse.ArgumentParser()

	ap.add_argument('-p', '--path', action = 'store', type = str, dest = 'base_path', default = './',
		help = 'Base path to all Dependency Parsing models and data')
	ap.add_argument('-d', '--data', action = 'store', type = str, dest = 'data_path', default = '../datasets', 
		help = 'Dataset location')
	ap.add_argument('-t', '--train', action = 'store_true', dest = 'train_model',
		help = 'Train a new model, saved in saved_models directory in depparse directory')
	ap.add_argument('-ev', '--eval', action = 'store_false', dest = 'train_model', 
		help = 'Evaluate a pre-existing model, saved in saved_models directory in the depparse directory')
	ap.set_defaults(train_model = False)
	ap.add_argument('-m', '--model', action = 'store', type = str, dest = 'modelname', default = 'dep_model.pt', 
		help = 'Name of saved model that is either being trained or being evaluated. Most be stored in saved_models directory')
	ap.add_argument('-l', '--lang', action = 'store', type = str, dest = 'lang', default = 'en',
		help = 'Language to run dependency parsing model on')
	ap.add_argument('-i', '--input', action = 'store', type = str, dest = 'input_type', default = 'form',
		help = 'Type of input to run through LSTM, either form or lemma')
	ap.add_argument('-ty', '--typological', action = 'store_true', dest = 'typological',
		help = 'Include typological features in training')
	ap.add_argument('-nty', '--notypological', action = 'store_false', dest = 'typological',
		help = 'Do not include typological features in training')
	ap.set_defaults(typological = False)
	ap.add_argument('-tf', '--typfeatures', action = 'store', type = str, dest = 'typ_feature', default = 'syntax_knn',
		help = 'Which typological features to extract from the typological database')
	ap.add_argument('-e', '--encoder', action = 'store', type = str, dest = 'encoder', default = 'lstm',
		help = 'Word Embedding model, either LM or LSTM')
	ap.add_argument('-te', '--typencode', action = 'store', type = str, dest = 'typ_encode', default = 'concat',
		help = 'Method to use for incorporating typological features. Choose from [concat, add_att, mul_att] to decide to either use a concatentation or attention method')
	ap.add_argument('-ft', '--fine_tune', action = 'store_true', dest = 'fine_tune',
		help = 'Fine tune language model')
	ap.add_argument('-nft', '--no_fine_tune', action = 'store_false', dest = 'fine_tune',
		help = 'Use frozen representations')
	ap.set_defaults(fine_tune = True)
	ap.add_argument('-sh', '--shuffle', action = 'store_true', dest = 'shuffle',
		help = 'Shuffle data in data loaders')
	ap.add_argument('-nsh', '--no_shuffle', action = 'store_false', dest = 'shuffle',
		help = 'Keep data order the same')
	ap.set_defaults(shuffle = False)
	ap.add_argument('-sm', '--save_model', action = 'store_true', dest = 'save_model', 
		help = 'Save a trained model')
	ap.add_argument('-nsm', '--no_save', action = 'store_false', dest = 'save_model',
		help = 'Don\'t save a run')
	ap.set_defaults(save_model = True)

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
	ap.add_argument('-lm', '--lm_model_name', action = 'store', dest = 'lm_model_name', type = str, default = 'bert-base-uncased',
		help = 'BERT Model to use when using BERT as encoder')
	ap.add_argument('-tes', '--typsize', action = 'store', dest = 'typ_embed_size', type = int, default = 32,
		help = 'Embedding size for typological embedding vector')
	ap.add_argument('-nt', '--numtyp', action = 'store', type = int, dest = 'num_typ_features', default = 103,
		help = 'Number of typological features in the typological features extracted.')
	ap.add_argument('-lml', '--lmlayer', action = 'store', dest = 'lm_layer', type = int, default = 8,
		help = 'Layer to obtain BERT representations from')
	ap.add_argument('-sc', '--scale', action = 'store', dest = 'scale', type = float, default = 0,
		help = 'Scaling factor for biaffine attention')
	ap.add_argument('-lr', '--learningrate', action = 'store', dest = 'lr', type = float, default = 0.0005, 
		help = 'Learning rate for optimization')
	ap.add_argument('-ep', '--numepochs', action = 'store', dest = 'num_epochs', type = int, default = 10,
		help = 'Number of epochs of training')

	return ap.parse_args()

def dep_main(args, device):
	print('Starting Dependency Parsing')

	print('Loading data in language {} from training file {}, validation file {}, and testing file {}'.format(args.lang, args.train_filename, args.valid_filename, args.test_filename))
	train_loader, valid_loader, test_loader, vocab_dict, pos_dict, label_dict = dep_data_loaders(args, args.train_filename, args.valid_filename, args.test_filename)
	print('Data loading complete')
	if args.train_model:
		classifier = arc_train(args, train_loader, valid_loader, len(vocab_dict), len(pos_dict), len(label_dict), device)
	else:
		classifier = BiaffineDependencyModel(n_words = len(vocab_dict), n_pos = len(pos_dict), n_rels = len(label_dict), word_embed_size = args.word_embed_size, pos_embed_size = args.pos_embed_size, lstm_hidden_size = args.lstm_hidden_size, encoder = args.encoder, lstm_layers = args.lstm_layers, 
			lm_model_name = args.lm_model_name, dropout = args.dropout, n_lm_layer = args.lm_layer, n_arc_mlp = 500, n_rel_mlp = 100, scale = args.scale, pad_index = pad_index, 
			unk_index = 0, typological = args.typological, typ_embed_size = args.typ_embed_size, num_typ_features = args.num_typ_features, 
			typ_encode = args.typ_encode, attention_hidden_size = args.attention_hidden_size, fine_tune = args.fine_tune)
		model_path = os.path.join(args.base_path, 'saved_models', args.modelname)
		classifier.load_state_dict(torch.load(model_path))

	print(arc_eval(args, test_loader, device))

if __name__ == '__main__':
	if cuda.is_available():
		device = 'cuda'
		torch.cuda.manual_seed_all(seed)
	else:
		device = 'cpu'
	print('Using device: {}'.format(device)) #Ensure on GPU!

	args = get_cmd_arguments_dep()

	seed = 0
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	if args.lang == 'en':
		data_directory = 'UD_English-EWT'
		args.data_path = os.path.join(args.data_path, data_directory)
		args.train_filename = 'en_ewt-ud-train.conllu'
		args.valid_filename = 'en_ewt-ud-dev.conllu'
		args.test_filename = 'en_ewt-ud-test.conllu'
	else:
		raise AssertionError('Please enter a valid language')

	assert(os.path.exists(args.base_path)), 'Base path does not exist'	
	assert(os.path.exists(args.data_path)), 'Data path does not exist'
	assert(args.encoder in ['lstm', 'lm']), 'Please choose either BERT or LSTM to build word embeddings'
	assert(args.input_type in ['lemma', 'form']), 'Please choose an input type of form or lemma'
	if args.typological:
		assert(args.typ_encode in ['concat', 'add_att', 'mul_att']), 'Please use attention or concatention for encoding typological features'
	if args.encoder == 'lm':
		assert('bert' in args.lm_model_name or 'gpt2' in args.lm_model_name), 'Please choose BERT or GPT2 as the LM'
		args.tokenizer = transformers.BertTokenizer.from_pretrained(args.lm_model_name) if 'bert' in args.lm_model_name else transformers.GPT2Tokenizer.from_pretrained(args.lm_model_name)
	dep_main(args, device)