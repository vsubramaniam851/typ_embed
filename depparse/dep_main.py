import sys
import os
import numpy as np
import pandas as pd 
import random
import argparse
import lang2vec.lang2vec as l2v

import torch
import torch.cuda as cuda

from dep_train import *
from dep_eval import *

if cuda.is_available():
	device = 'cuda'
	torch.cuda.manual_seed_all(seed)
else:
	print('WARNING, this program is running on CPU')
	device = 'cpu'

print('Using device: {}'.format(device)) #Ensure on GPU!

def get_cmd_arguments():
	ap = argparse.ArgumentParser()

	ap.add_argument('-p', '--path', action = 'store', type = str, dest = 'base_path', default = '/storage/vsub851/typ_embed/depparse',
		help = 'Base path to all Dependency Parsing models and data')
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
	ap.add_argument('-tf', '--typfeatures', action = 'store', typ = str, dest = 'typ_feature', default = 'syntax_knn',
		help = 'Which typological features to extract from the typological database')
	ap.add_argument('nt', '--numtyp', action = 'store', typ = int, dest = 'num_typ_features', default = 103,
		help = 'Number of typological features in the typological features extracted.')
	ap.add_argument('-e', '--encoder', action = 'store', type = str, dest = 'encoder', default = 'lstm',
		help = 'Word Embedding model, either BERT or LSTM')

	#Model Hyperparameters
	ap.add_argument('-wes', '--wordsize', action = 'store', dest = 'word_embed_size', type = int, default = 100, 
		help = 'Word Embedding Size for model')
	ap.add_argument('-pes', '--possize', action = 'store', dest = 'pos_embed_size', type = int, default = 100,
		help = 'POS Embedding Size for model')
	ap.add_argument('-lhs', '--lstmsize', action = 'store', dest = 'lstm_hidden_size', type = int, default = 400, 
		help = 'LSTM Hidden size when using encoder LSTM')
	ap.add_argument('-ll', '--lstmlayers', action = 'store', dest = 'lstm_layers', type = int, default = 3,
		help = 'Number of LSTM Layers in LSTM encoder')
	ap.add_argument('-d', '--dropout', action = 'store', dest = 'dropout', type = float, default = 0.33,
		help = 'Dropout probability to be used in all components of model')
	ap.add_argument('-b', '--bert', action = 'store', dest = 'bert', type = str, default = 'bert-base-uncased',
		help = 'BERT Model to use when using BERT as encoder')
	ap.add_argument('-tes', '--typsize', action = 'store', dest = 'typ_embed_size', type = int, default = 32,
		help = 'Embedding size for typological embedding vector')
	ap.add_argument('-bl', '--bertlayer', action = 'store', dest = 'bert_layer', type = int, default = 8,
		help = 'Layer to obtain BERT representations from')
	ap.add_argument('-sc', '--scale', action = 'store', dest = 'scale', type = float, default = 0,
		help = 'Scaling factor for biaffine attention')
	ap.add_argument('-lr', '--learningrate', action = 'store', dest = 'lr', type = float, default = 0.0005, 
		help = 'Learning rate for optimization')
	ap.add_argument('-ep', '--numepochs', action = 'store', dest = 'num_epochs', type = int, default = 10,
		help = 'Number of epochs of training')

	return ap.parse_args()

def dep_main():
	args = get_cmd_argument()

	seed = 0
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	if args.lang == 'en':
		directory = 'UD_English-EWT'
		train_filename = 'en_ewt-ud-train.conllu'
		valid_filename = 'en_ewt-ud-dev.conllu'
		test_filename = 'en_ewt-ud-test.conllu'

	#Testing for right encoder type
	if args.encoder not in ['bert', 'lstm']:
		raise NameError('Please choose an encoder in either BERT or LSTM')
	if args.input_type not in ['lemma', 'form']:
		raise NameError('Please choose an input type of form or lemma')

	print('Loading data in language {} from training file {}, validation file {}, and testing file {}'.format(args.lang, train_filename, valid_filename, test_filename))

	file_path = os.path.join(args.base_path, directory)
	train_lines = preproc_conllu(file_path, filename = train_filename)
	train_sent_collection = sentence_collection(train_lines)
	train_corpus, vocab_dict, label_dict, pos_dict = process_corpus(train_sent_collection, mode = 'train', input_type = args.input_type)

	if args.train_model:
		valid_lines = preproc_conllu(file_path, filename = valid_filename)
		valid_sent_collection = sentence_collection(valid_lines)
		valid_corpus, _, _, _ = process_corpus(valid_sent_collection, mode = 'valid', vocab_dict = vocab_dict, label_dict = label_dict,
			pos_dict = pos_dict, input_type = args.input_type)

		if args.encoder == 'bert':
			train_corpus = bert_tokenizer(train_corpus)
			valid_corpus = bert_tokenizer(valid_corpus)

		print('Data Loading Complete')

		if args.input_type == 'form':
			input_type = 'word_ids'
		else:
			input_type = 'lemma_ids'

		arc_train(base_path = args.base_path, train_corpus = train_corpus, valid_corpus = valid_corpus, train_type = input_type, num_words = len(vocab_dict), 
			num_pos = len(pos_dict), num_labels = len(label_dict), modelname = modelname, word_embed_size = args.word_embed_size, encoder = args.encoder,
			lstm_hidden_size = args.lstm_hidden_size, lr = args.lr, dropout = args.dropout, num_epochs = args.num_epochs, lstm_layers = args.lstm_layers, 
			batch_size = 1, bert = args.bert, bert_layer = args.bert_layer, scale = args.scale, typological = args.typological, typ_embed_size = args.typ_embed_size,
			num_typ_features = args.num_typ_features, typ_feature = args.typ_feature, lang = lang, device = device)
	else:
		test_lines = preproc_conllu(file_path, filename = test_filename)
		test_sent_collection = sentence_collection(test_lines)
		test_corpus, _, _, _ = process_corpus(test_sent_collection, mode = 'test', vocab_dict = vocab_dict, label_dict = label_dict,
			pos_dict = pos_dict, input_type = args.input_type)

		print('Data Loading Complete')

		if args.input_type == 'form':
			input_type = 'word_ids'
		else:
			input_type = 'lemma_ids'

		print(arc_eval(base_path = args.base_path, test_corpus = test_corpus, eval_input = input_type, num_words = len(vocab_dict), num_pos = len(pos_dict), 
			num_labels = len(label_dict), modelname = args.modelname, word_embed_size = args.word_embed_size, pos_embed_size = pos_embed_size, encoder = args.encoder,
			lstm_hidden_size = args.lstm_hidden_size, dropout = args.dropout, lstm_layers = args.lstm_layers, bert = args.bert, bert_layer = args.bert_layer,
			scale = args.scale, typological = args.typological, typ_embed_size = args.typ_embed_size, typ_feature = args.typ_feature, num_typ_features = args.num_typ_features,
			lang = args.lang, device = device))