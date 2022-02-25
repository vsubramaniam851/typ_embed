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

from entail_data_load import *
from entail_train import *
from entail_eval import *

def get_cmd_arguments_entail():
	ap = argparse.ArgumentParser()

	ap.add_argument('-p', '--path', action = 'store', type = str, dest = 'base_path', default = './',
		help = 'Base path to all Dependency Parsing models and data')
	ap.add_argument('-d', '--data', action = 'store', type = str, dest = 'data_path', default = '../datasets', 
		help = 'Dataset location')
	ap.add_argument('-t', '--train', action = 'store_true', dest = 'train_model',
		help = 'Train a new model, saved in saved_models directory in entailment directory')
	ap.add_argument('-ev', '--eval', action = 'store_false', dest = 'train_model', 
		help = 'Evaluate a pre-existing model, saved in saved_models directory in the entailment directory')
	ap.set_defaults(train_model = False)
	ap.add_argument('-m', '--model', action = 'store', type = str, dest = 'modelname', default = 'en_model.pt', 
		help = 'Name of saved model that is either being trained or being evaluated. Most be stored in saved_models directory')
	ap.add_argument('-l', '--lang', action = 'store', type = str, dest = 'lang', default = 'en',
		help = 'Language to run entailment model on')
	ap.add_argument('-ty', '--typological', action = 'store_true', dest = 'typological',
		help = 'Include typological features in training')
	ap.add_argument('-nty', '--notypological', action = 'store_false', dest = 'typological',
		help = 'Do not include typological features in training')
	ap.set_defaults(typological = False)
	ap.add_argument('-tf', '--typfeatures', action = 'store', type = str, dest = 'typ_feature', default = 'syntax_knn',
		help = 'Which typological features to extract from the typological database')
	ap.add_argument('-te', '--typencode', action = 'store', type = str, dest = 'typ_encode', default = 'concat',
		help = 'Method to use for incorporating typological features. Choose from [concat, add_att, mul_att] to decide to either use a concatentation or attention method')
	ap.add_argument('-ft', '--fine_tune', action = 'store_true', dest = 'fine_tune',
		help = 'Fine tune language model')
	ap.add_argument('-nft', '--no_fine_tune', action = 'store_false', dest = 'fine_tune',
		help = 'Use frozen representations')
	ap.set_defaults(fine_tune = True)
	ap.add_argument('-sm', '--save_model', action = 'store_true', dest = 'save_model', 
		help = 'Save a trained model')
	ap.add_argument('-nsm', '--no_save', action = 'store_false', dest = 'save_model',
		help = 'Don\'t save a run')
	ap.set_defaults(save_model = True)

	#Model Hyperparameters
	ap.add_argument('-ahs', '--attentionsize', action = 'store', dest = 'attention_hidden_size', type = int, default = 200,
		help = 'Multiplicative Attention Hidden Size')
	ap.add_argument('-dr', '--dropout', action = 'store', dest = 'dropout', type = float, default = 0.33,
		help = 'Dropout probability to be used in all components of model')
	ap.add_argument('-lm', '--lm_model_name', action = 'store', dest = 'lm_model_name', type = str, default = 'bert-base-uncased',
		help = 'BERT Model to use when using BERT as encoder')
	ap.add_argument('-tes', '--typsize', action = 'store', dest = 'typ_embed_size', type = int, default = 32,
		help = 'Embedding size for typological embedding vector')
	ap.add_argument('-nt', '--numtyp', action = 'store', type = int, dest = 'num_typ_features', default = 103,
		help = 'Number of typological features in the typological features extracted.')
	ap.add_argument('-lml', '--lmlayer', action = 'store', dest = 'lm_layer', type = int, default = -1,
		help = 'Layer to obtain BERT representations from')
	ap.add_argument('-lr', '--learningrate', action = 'store', dest = 'lr', type = float, default = 0.0005, 
		help = 'Learning rate for optimization')
	ap.add_argument('-ep', '--numepochs', action = 'store', dest = 'num_epochs', type = int, default = 10,
		help = 'Number of epochs of training')

	return ap.parse_args()

def en_main(args, device):
	print('Starting Entailment Identification')

	print('Loading data')
	en_train_loader, en_val_loader, en_test_loader = entail_dataloaders(args.data_path, args.train_split, args.val_split)
	print('Data loading complete')
	if args.train_model:
		classifier = entail_train(args, train_loader, valid_loader, device)
	else:
		classifier = EnMLP(n_rels = num_labels, tokenizer = args.tokenizer, lm_model_name = args.lm_model_name, model_type = args.model_type, typological = args.typological, 
			typ_embed_size = args.typ_embed_size, num_typ_features = args.num_typ_features, typ_encode = args.typ_encode, n_lm_layer = args.n_lm_layer, attention_hidden_size = args.attention_hidden_size,
			fine_tune = args.fine_tune, extract_cls = args.extract_cls, average_sen = args.average_sen, mlp_hidden_size = args.mlp_hidden_size, dropout = args.dropout)
		model_path = os.path.join(args.base_path, 'saved_models', args.modelname)
		classifier.load_state_dict(torch.load(model_path))

	print(entail_eval(args, classifier, test_loader, device))

if __name__ == '__main__':
	if cuda.is_available():
		device = 'cuda'
	else:
		device = 'cpu'
	print('Using device: {}'.format(device)) #Ensure on GPU!

	args = get_cmd_arguments_en()

	seed = 0
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	if args.lang != 'en':
		raise AssertionError('Please enter a valid language')

	assert(os.path.exists(args.base_path)), 'Base path does not exist'	
	assert(os.path.exists(args.data_path)), 'Data path does not exist'
	assert(args.encoder in ['lm']), 'Please choose either BERT or LSTM to build word embeddings'
	assert('bert' in args.lm_model_name or 'gpt2' in args.lm_model_name), 'Please choose BERT or GPT2 as the LM'
	args.tokenizer = transformers.BertTokenizer.from_pretrained(args.lm_model_name) if 'bert' in args.lm_model_name else transformers.GPT2Tokenizer.from_pretrained(args.lm_model_name)
	en_main(args, device)