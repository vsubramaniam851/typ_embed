import sys
import os
import argparse
import unittest

import torch
import torch.cuda as cuda

sys.path.insert(1, './depparse')
sys.path.insert(1, './postag')
from depparse.dep_main import *
from postag.postag_main import *

eed = 0
if cuda.is_available():
	device = 'cuda'
	torch.cuda.manual_seed_all(seed)
else:
	print('WARNING, this program is running on CPU')
	device = 'cpu'

print('Using device: {}'.format(device)) #Ensure on GPU!

def get_cmd_arguments():
	ap = argparse.ArgumentParser()
	subparsers = ap.add_subparsers(dest = 'task_name')
	subparsers.required = True
	dep_parser = subparsers.add_parser('dep')
	pos_parser = subparsers.add_parser('pos')

	#DEPENDENCY PARSER ARGUMENTS

	dep_parser.add_argument('-p', '--path', action = 'store', type = str, dest = 'base_path', default = './depparse',
		help = 'Base path to all Dependency Parsing models and data')
	dep_parser.add_argument('-d', '--data', action = 'store', type = str, dest = 'data_path', default = './datasets', 
		help = 'Dataset location')
	dep_parser.add_argument('-t', '--train', action = 'store', type = bool, dest = 'train_model', default = False,
		help = 'Train a new model, saved in saved_models directory in depparse directory')
	dep_parser.add_argument('-m', '--model', action = 'store', type = str, dest = 'modelname', default = 'dep_model.pt', 
		help = 'Name of saved model that is either being trained or being evaluated. Most be stored in saved_models directory')
	dep_parser.add_argument('-l', '--lang', action = 'store', type = str, dest = 'lang', default = 'en',
		help = 'Language to run dependency parsing model on')
	dep_parser.add_argument('-i', '--input', action = 'store', type = str, dest = 'input_type', default = 'form',
		help = 'Type of input to run through LSTM, either form or lemma')
	dep_parser.add_argument('-ty', '--typological', action = 'store', type = bool, dest = 'typological', default = False,
		help = 'Include typological features in training')
	dep_parser.add_argument('-tf', '--typfeatures', action = 'store', type = str, dest = 'typ_feature', default = 'syntax_knn',
		help = 'Which typological features to extract from the typological database')
	dep_parser.add_argument('-e', '--encoder', action = 'store', type = str, dest = 'encoder', default = 'lstm',
		help = 'Word Embedding model, either BERT or LSTM')
	dep_parser.add_argument('-te', '--typencode', action = 'store', type = str, dest = 'typ_encode', default = 'concat',
		help = 'Method to use for incorporating typological features. Choose from [concat, add_att, mul_att] to decide to either use a concatentation or attention method')

	# Dependency Parsing Model Hyperparameters
	dep_parser.add_argument('-wes', '--wordsize', action = 'store', dest = 'word_embed_size', type = int, default = 100, 
		help = 'Word Embedding Size for model')
	dep_parser.add_argument('-pes', '--possize', action = 'store', dest = 'pos_embed_size', type = int, default = 100,
		help = 'POS Embedding Size for model')
	dep_parser.add_argument('-lhs', '--lstmsize', action = 'store', dest = 'lstm_hidden_size', type = int, default = 400, 
		help = 'LSTM Hidden size when using encoder LSTM')
	dep_parser.add_argument('-ahs', '--attentionsize', action = 'store', dest = 'attention_hidden_size', type = int, default = 200,
		help = 'Multiplicative Attention Hidden Size')
	dep_parser.add_argument('-ll', '--lstmlayers', action = 'store', dest = 'lstm_layers', type = int, default = 3,
		help = 'Number of LSTM Layers in LSTM encoder')
	dep_parser.add_argument('-dr', '--dropout', action = 'store', dest = 'dropout', type = float, default = 0.33,
		help = 'Dropout probability to be used in all components of model')
	dep_parser.add_argument('-b', '--bert', action = 'store', dest = 'bert', type = str, default = 'bert-base-uncased',
		help = 'BERT Model to use when using BERT as encoder')
	dep_parser.add_argument('-tes', '--typsize', action = 'store', dest = 'typ_embed_size', type = int, default = 32,
		help = 'Embedding size for typological embedding vector')
	dep_parser.add_argument('-nt', '--numtyp', action = 'store', type = int, dest = 'num_typ_features', default = 103,
		help = 'Number of typological features in the typological features extracted.')
	dep_parser.add_argument('-bl', '--bertlayer', action = 'store', dest = 'bert_layer', type = int, default = 8,
		help = 'Layer to obtain BERT representations from')
	dep_parser.add_argument('-sc', '--scale', action = 'store', dest = 'scale', type = float, default = 0,
		help = 'Scaling factor for biaffine attention')
	dep_parser.add_argument('-lr', '--learningrate', action = 'store', dest = 'lr', type = float, default = 0.0005, 
		help = 'Learning rate for optimization')
	dep_parser.add_argument('-ep', '--numepochs', action = 'store', dest = 'num_epochs', type = int, default = 10,
		help = 'Number of epochs of training')

	#POS Tagger Arguments

	pos_parser.add_argument('-p', '--path', action = 'store', type = str, dest = 'base_path', default = './postag',
		help = 'Base path to all Dependency Parsing models and data')
	pos_parser.add_argument('-d', '--data', action = 'store', type = str, dest = 'data_path', default = './datasets', 
		help = 'Dataset location')
	pos_parser.add_argument('-t', '--train', action = 'store', type = bool, dest = 'train_model', default = False,
		help = 'Train a new model, saved in saved_models directory in postag directory')
	pos_parser.add_argument('-m', '--model', action = 'store', type = str, dest = 'modelname', default = 'dep_model.pt', 
		help = 'Name of saved model that is either being trained or being evaluated. Most be stored in saved_models directory')
	pos_parser.add_argument('-l', '--lang', action = 'store', type = str, dest = 'lang', default = 'en',
		help = 'Language to run dependency parsing model on')
	pos_parser.add_argument('-i', '--input', action = 'store', type = str, dest = 'input_type', default = 'form',
		help = 'Type of input to run through LSTM, either form or lemma')
	pos_parser.add_argument('-ty', '--typological', action = 'store', type = bool, dest = 'typological', default = False,
		help = 'Include typological features in training')
	pos_parser.add_argument('-tf', '--typfeatures', action = 'store', type = str, dest = 'typ_feature', default = 'syntax_knn',
		help = 'Which typological features to extract from the typological database')
	pos_parser.add_argument('-e', '--encoder', action = 'store', type = str, dest = 'encoder', default = 'lstm',
		help = 'Word Embedding model, either BERT or LSTM')
	pos_parser.add_argument('-te', '--typencode', action = 'store', type = str, dest = 'typ_encode', default = 'concat',
		help = 'Method to use for incorporating typological features. Choose from /[concat, add_att, mul_att]/ to decide to either use a concatentation or attention method')

	#POS Tagging Model Hyperparameters
	pos_parser.add_argument('-wes', '--wordsize', action = 'store', dest = 'word_embed_size', type = int, default = 100, 
		help = 'Word Embedding Size for model')
	pos_parser.add_argument('-lhs', '--lstmsize', action = 'store', dest = 'lstm_hidden_size', type = int, default = 400, 
		help = 'LSTM Hidden size when using encoder LSTM')
	pos_parser.add_argument('-mhs', '--mlpsize', action = 'store', dest = 'mlp_hidden_size', type = int, default = 200,
		help = 'Hidden size in MLP')
	pos_parser.add_argument('-ahs', '--attentionsize', action = 'store', dest = 'attention_hidden_size', type = int, default = 200,
		help = 'Multiplicative Attention Hidden Size')
	pos_parser.add_argument('-ll', '--lstmlayers', action = 'store', dest = 'lstm_layers', type = int, default = 3,
		help = 'Number of LSTM Layers in LSTM encoder')
	pos_parser.add_argument('-dr', '--dropout', action = 'store', dest = 'dropout', type = float, default = 0.33,
		help = 'Dropout probability to be used in all components of model')
	pos_parser.add_argument('-b', '--bert', action = 'store', dest = 'bert', type = str, default = 'bert-base-uncased',
		help = 'BERT Model to use when using BERT as encoder')
	pos_parser.add_argument('-tes', '--typsize', action = 'store', dest = 'typ_embed_size', type = int, default = 32,
		help = 'Embedding size for typological embedding vector')
	pos_parser.add_argument('-nt', '--numtyp', action = 'store', type = int, dest = 'num_typ_features', default = 103,
		help = 'Number of typological features in the typological features extracted.')
	pos_parser.add_argument('-bl', '--bertlayer', action = 'store', dest = 'bert_layer', type = int, default = 8,
		help = 'Layer to obtain BERT representations from')
	pos_parser.add_argument('-lr', '--learningrate', action = 'store', dest = 'lr', type = float, default = 0.0005, 
		help = 'Learning rate for optimization')
	pos_parser.add_argument('-ep', '--numepochs', action = 'store', dest = 'num_epochs', type = int, default = 10,
		help = 'Number of epochs of training')

	return ap.parse_args()

def main():
	args = get_cmd_arguments()

	assert(args.task_name in ['dep', 'pos']), 'Task must be either Dependency Parsing or POS Tagging'

	debug = unittest.TestCase()
	debug.assertTrue(os.path.exists(args.base_path), msg = 'Base path does not exist')	
	debug.assertTrue(os.path.exists(args.data_path), msg = 'Data path does not exist')

	assert(args.encoder in ['bert', 'lstm']), 'Please choose either BERT or LSTM to build word embeddings'
	assert(args.input_type in ['lemma', 'form']), 'Please choose an input type of form or lemma'
	assert(args.typ_encode in ['concat', 'add_att', 'mul_att']), 'Please use attention or concatention for encoding typological features'

	if args.task_name == 'dep':
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

		dep_main(train_filename = train_filename, valid_filename = valid_filename, test_filename = test_filename, lang = args.lang, base_path = args.base_path,
			data_path = data_path, data_directory = directory, train_model = args.train_model, input_type = args.input_type, word_embed_size = args.word_embed_size, pos_embed_size = args.pos_embed_size,
			modelname = args.modelname, encoder = args.encoder, lstm_hidden_size = args.lstm_hidden_size, lr = args.lr, dropout = args.dropout, num_epochs = args.num_epochs, 
			lstm_layers = args.lstm_layers, batch_size = 1, bert = args.bert, bert_layer = args.bert_layer, scale = args.scale, typological = args.typological, 
			typ_embed_size = args.typ_embed_size, num_typ_features = args.num_typ_features, typ_feature = args.typ_feature, typ_encode = args.typ_encode, attention_hidden_size = args.attention_hidden_size,
			device = device)

	else:
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

		pos_main(train_filename = train_filename, valid_filename = valid_filename, test_filename = test_filename, lang = args.lang, base_path = args.base_path,
			data_path = args.data_path, data_directory = directory, train_model = args.train_model, input_type = args.input_type, word_embed_size = args.word_embed_size, modelname = args.modelname,
			encoder = args.encoder, lstm_hidden_size = args.lstm_hidden_size, mlp_hidden_size = args.mlp_hidden_size, lr = args.lr, dropout = args.dropout, num_epochs = args.num_epochs,
			lstm_layers = args.lstm_layers, batch_size = 1, bert = args.bert, bert_layer = args.bert_layer, typological = args.typological, typ_embed_size = args.typ_embed_size,
			num_typ_features = args.num_typ_features, typ_feature = args.typ_feature, typ_encode = args.typ_encode, attention_hidden_size = args.attention_hidden_size,
			device = device)

if __name__ == '__main__':
	main()