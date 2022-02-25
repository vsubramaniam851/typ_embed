# -*- coding: utf-8 -*-

import sys
import os
import argparse

import torch
import torch.cuda as cuda

sys.path.insert(1, './depparse')
sys.path.insert(1, './postag')
sys.path.insert(1, './entailment')
sys.path.insert(1, './summarization')
from depparse.dep_main import *
from postag.postag_main import *
from entailment.entail_main import *
from summarization.sum_main import *

def get_cmd_arguments():
	ap = argparse.ArgumentParser()
	subparsers = ap.add_subparsers(dest = 'task_name')
	subparsers.required = True
	dep_parser = subparsers.add_parser('dep')
	pos_parser = subparsers.add_parser('pos')
	entail_parser = subparsers.add_parser('entail')
	sum_parser = subparsers.add_parser('sum')

	#DEPENDENCY PARSER ARGUMENTS

	dep_parser.add_argument('-p', '--path', action = 'store', type = str, dest = 'base_path', default = './',
		help = 'Base path to all Dependency Parsing models and data')
	dep_parser.add_argument('-d', '--data', action = 'store', type = str, dest = 'data_path', default = '../datasets', 
		help = 'Dataset location')
	dep_parser.add_argument('-t', '--train', action = 'store_true', dest = 'train_model',
		help = 'Train a new model, saved in saved_models directory in depparse directory')
	dep_parser.add_argument('-ev', '--eval', action = 'store_false', dest = 'train_model', 
		help = 'Evaluate a pre-existing model, saved in saved_models directory in the depparse directory')
	dep_parser.set_defaults(train_model = False)
	dep_parser.add_argument('-m', '--model', action = 'store', type = str, dest = 'modelname', default = 'dep_model.pt', 
		help = 'Name of saved model that is either being trained or being evaluated. Most be stored in saved_models directory')
	dep_parser.add_argument('-l', '--lang', action = 'store', type = str, dest = 'lang', default = 'en',
		help = 'Language to run dependency parsing model on')
	dep_parser.add_argument('-i', '--input', action = 'store', type = str, dest = 'input_type', default = 'form',
		help = 'Type of input to run through LSTM, either form or lemma')
	dep_parser.add_argument('-ty', '--typological', action = 'store_true', dest = 'typological',
		help = 'Include typological features in training')
	dep_parser.add_argument('-nty', '--notypological', action = 'store_false', dest = 'typological',
		help = 'Do not include typological features in training')
	dep_parser.set_defaults(typological = False)
	dep_parser.add_argument('-tf', '--typfeatures', action = 'store', type = str, dest = 'typ_feature', default = 'syntax_knn',
		help = 'Which typological features to extract from the typological database')
	dep_parser.add_argument('-e', '--encoder', action = 'store', type = str, dest = 'encoder', default = 'lstm',
		help = 'Word Embedding model, either LM or LSTM')
	dep_parser.add_argument('-te', '--typencode', action = 'store', type = str, dest = 'typ_encode', default = 'concat',
		help = 'Method to use for incorporating typological features. Choose from [concat, add_att, mul_att] to decide to either use a concatentation or attention method')
	dep_parser.add_argument('-ft', '--fine_tune', action = 'store_true', dest = 'fine_tune',
		help = 'Fine tune language model')
	dep_parser.add_argument('-nft', '--no_fine_tune', action = 'store_false', dest = 'fine_tune',
		help = 'Use frozen representations')
	dep_parser.set_defaults(fine_tune = True)
	dep_parser.add_argument('-sh', '--shuffle', action = 'store_true', dest = 'shuffle',
		help = 'Shuffle data in data loaders')
	dep_parser.add_argument('-nsh', '--no_shuffle', action = 'store_false', dest = 'shuffle',
		help = 'Keep data order the same')
	dep_parser.set_defaults(shuffle = False)
	dep_parser.add_argument('-sm', '--save_model', action = 'store_true', dest = 'save_model', 
		help = 'Save a trained model')
	dep_parser.add_argument('-nsm', '--no_save', action = 'store_false', dest = 'save_model',
		help = 'Don\'t save a run')
	dep_parser.set_defaults(save_model = True)

	#Dependency Parsing Model Hyperparameters
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
	dep_parser.add_argument('-lm', '--lm_model_name', action = 'store', dest = 'lm_model_name', type = str, default = 'bert-base-uncased',
		help = 'BERT Model to use when using BERT as encoder')
	dep_parser.add_argument('-tes', '--typsize', action = 'store', dest = 'typ_embed_size', type = int, default = 32,
		help = 'Embedding size for typological embedding vector')
	dep_parser.add_argument('-nt', '--numtyp', action = 'store', type = int, dest = 'num_typ_features', default = 103,
		help = 'Number of typological features in the typological features extracted.')
	dep_parser.add_argument('-lml', '--lmlayer', action = 'store', dest = 'lm_layer', type = int, default = 8,
		help = 'Layer to obtain BERT representations from')
	dep_parser.add_argument('-sc', '--scale', action = 'store', dest = 'scale', type = float, default = 0,
		help = 'Scaling factor for biaffine attention')
	dep_parser.add_argument('-lr', '--learningrate', action = 'store', dest = 'lr', type = float, default = 0.0005, 
		help = 'Learning rate for optimization')
	dep_parser.add_argument('-ep', '--numepochs', action = 'store', dest = 'num_epochs', type = int, default = 10,
		help = 'Number of epochs of training')

	#POS TAGGER ARGUMENTS

	pos_parser.add_argument('-p', '--path', action = 'store', type = str, dest = 'base_path', default = './',
		help = 'Base path to all Dependency Parsing models and data')
	pos_parser.add_argument('-d', '--data', action = 'store', type = str, dest = 'data_path', default = '../datasets', 
		help = 'Dataset location')
	pos_parser.add_argument('-t', '--train', action = 'store_true', dest = 'train_model',
		help = 'Train a new model, saved in saved_models directory in depparse directory')
	pos_parser.add_argument('-ev', '--eval', action = 'store_false', dest = 'train_model', 
		help = 'Evaluate a pre-existing model, saved in saved_models directory in the depparse directory')
	pos_parser.set_defaults(train_model = False)
	pos_parser.add_argument('-m', '--model', action = 'store', type = str, dest = 'modelname', default = 'dep_model.pt', 
		help = 'Name of saved model that is either being trained or being evaluated. Most be stored in saved_models directory')
	pos_parser.add_argument('-l', '--lang', action = 'store', type = str, dest = 'lang', default = 'en',
		help = 'Language to run dependency parsing model on')
	pos_parser.add_argument('-i', '--input', action = 'store', type = str, dest = 'input_type', default = 'form',
		help = 'Type of input to run through LSTM, either form or lemma')
	pos_parser.add_argument('-ty', '--typological', action = 'store_true', dest = 'typological',
		help = 'Include typological features in training')
	pos_parser.add_argument('-nty', '--notypological', action = 'store_false', dest = 'typological',
		help = 'Do not include typological features in training')
	pos_parser.set_defaults(typological = False)
	pos_parser.add_argument('-tf', '--typfeatures', action = 'store', type = str, dest = 'typ_feature', default = 'syntax_knn',
		help = 'Which typological features to extract from the typological database')
	pos_parser.add_argument('-e', '--encoder', action = 'store', type = str, dest = 'encoder', default = 'lstm',
		help = 'Word Embedding model, either LM or LSTM')
	pos_parser.add_argument('-te', '--typencode', action = 'store', type = str, dest = 'typ_encode', default = 'concat',
		help = 'Method to use for incorporating typological features. Choose from [concat, add_att, mul_att] to decide to either use a concatentation or attention method')
	pos_parser.add_argument('-ft', '--fine_tune', action = 'store_true', dest = 'fine_tune',
		help = 'Fine tune language model')
	pos_parser.add_argument('-nft', '--no_fine_tune', action = 'store_false', dest = 'fine_tune',
		help = 'Use frozen representations')
	pos_parser.set_defaults(fine_tune = True)
	pos_parser.add_argument('-sh', '--shuffle', action = 'store_true', dest = 'shuffle',
		help = 'Shuffle data in data loaders')
	pos_parser.add_argument('-nsh', '--no_shuffle', action = 'store_false', dest = 'shuffle',
		help = 'Keep data order the same')
	pos_parser.set_defaults(shuffle = False)
	pos_parser.add_argument('-sm', '--save_model', action = 'store_true', dest = 'save_model', 
		help = 'Save a trained model')
	pos_parser.add_argument('-nsm', '--no_save', action = 'store_false', dest = 'save_model',
		help = 'Don\'t save a run')
	pos_parser.set_defaults(save_model = True)

	#POS Tagging Model Hyperparameters
	pos_parser.add_argument('-wes', '--wordsize', action = 'store', dest = 'word_embed_size', type = int, default = 100, 
		help = 'Word Embedding Size for model')
	pos_parser.add_argument('-pes', '--possize', action = 'store', dest = 'pos_embed_size', type = int, default = 100,
		help = 'POS Embedding Size for model')
	pos_parser.add_argument('-lhs', '--lstmsize', action = 'store', dest = 'lstm_hidden_size', type = int, default = 400, 
		help = 'LSTM Hidden size when using encoder LSTM')
	pos_parser.add_argument('-ahs', '--attentionsize', action = 'store', dest = 'attention_hidden_size', type = int, default = 200,
		help = 'Multiplicative Attention Hidden Size')
	pos_parser.add_argument('-ll', '--lstmlayers', action = 'store', dest = 'lstm_layers', type = int, default = 3,
		help = 'Number of LSTM Layers in LSTM encoder')
	pos_parser.add_argument('-dr', '--dropout', action = 'store', dest = 'dropout', type = float, default = 0.33,
		help = 'Dropout probability to be used in all components of model')
	pos_parser.add_argument('-lm', '--lm_model_name', action = 'store', dest = 'lm_model_name', type = str, default = 'bert-base-uncased',
		help = 'BERT Model to use when using BERT as encoder')
	pos_parser.add_argument('-tes', '--typsize', action = 'store', dest = 'typ_embed_size', type = int, default = 32,
		help = 'Embedding size for typological embedding vector')
	pos_parser.add_argument('-nt', '--numtyp', action = 'store', type = int, dest = 'num_typ_features', default = 103,
		help = 'Number of typological features in the typological features extracted.')
	pos_parser.add_argument('-lml', '--lmlayer', action = 'store', dest = 'lm_layer', type = int, default = 8,
		help = 'Layer to obtain BERT representations from')
	pos_parser.add_argument('-sc', '--scale', action = 'store', dest = 'scale', type = float, default = 0,
		help = 'Scaling factor for biaffine attention')
	pos_parser.add_argument('-lr', '--learningrate', action = 'store', dest = 'lr', type = float, default = 0.0005, 
		help = 'Learning rate for optimization')
	pos_parser.add_argument('-ep', '--numepochs', action = 'store', dest = 'num_epochs', type = int, default = 10,
		help = 'Number of epochs of training')

	#ENTAILMENT CLASSIFICATION ARGUMENTS

	entail_parser.add_argument('-p', '--path', action = 'store', type = str, dest = 'base_path', default = './',
		help = 'Base path to all Dependency Parsing models and data')
	entail_parser.add_argument('-d', '--data', action = 'store', type = str, dest = 'data_path', default = '../datasets', 
		help = 'Dataset location')
	entail_parser.add_argument('-t', '--train', action = 'store_true', dest = 'train_model',
		help = 'Train a new model, saved in saved_models directory in entailment directory')
	entail_parser.add_argument('-ev', '--eval', action = 'store_false', dest = 'train_model', 
		help = 'Evaluate a pre-existing model, saved in saved_models directory in the entailment directory')
	entail_parser.set_defaults(train_model = False)
	entail_parser.add_argument('-m', '--model', action = 'store', type = str, dest = 'modelname', default = 'en_model.pt', 
		help = 'Name of saved model that is either being trained or being evaluated. Most be stored in saved_models directory')
	entail_parser.add_argument('-l', '--lang', action = 'store', type = str, dest = 'lang', default = 'en',
		help = 'Language to run entailment model on')
	entail_parser.add_argument('-ty', '--typological', action = 'store_true', dest = 'typological',
		help = 'Include typological features in training')
	entail_parser.add_argument('-nty', '--notypological', action = 'store_false', dest = 'typological',
		help = 'Do not include typological features in training')
	entail_parser.set_defaults(typological = False)
	entail_parser.add_argument('-tf', '--typfeatures', action = 'store', type = str, dest = 'typ_feature', default = 'syntax_knn',
		help = 'Which typological features to extract from the typological database')
	entail_parser.add_argument('-te', '--typencode', action = 'store', type = str, dest = 'typ_encode', default = 'concat',
		help = 'Method to use for incorporating typological features. Choose from [concat, add_att, mul_att] to decide to either use a concatentation or attention method')
	entail_parser.add_argument('-ft', '--fine_tune', action = 'store_true', dest = 'fine_tune',
		help = 'Fine tune language model')
	entail_parser.add_argument('-nft', '--no_fine_tune', action = 'store_false', dest = 'fine_tune',
		help = 'Use frozen representations')
	entail_parser.set_defaults(fine_tune = True)
	entail_parser.add_argument('-sm', '--save_model', action = 'store_true', dest = 'save_model', 
		help = 'Save a trained model')
	entail_parser.add_argument('-nsm', '--no_save', action = 'store_false', dest = 'save_model',
		help = 'Don\'t save a run')
	entail_parser.set_defaults(save_model = True)

	#Entailment Classification Model Hyperparameters
	entail_parser.add_argument('-ahs', '--attentionsize', action = 'store', dest = 'attention_hidden_size', type = int, default = 200,
		help = 'Multiplicative Attention Hidden Size')
	entail_parser.add_argument('-dr', '--dropout', action = 'store', dest = 'dropout', type = float, default = 0.33,
		help = 'Dropout probability to be used in all components of model')
	entail_parser.add_argument('-lm', '--lm_model_name', action = 'store', dest = 'lm_model_name', type = str, default = 'bert-base-uncased',
		help = 'BERT Model to use when using BERT as encoder')
	entail_parser.add_argument('-tes', '--typsize', action = 'store', dest = 'typ_embed_size', type = int, default = 32,
		help = 'Embedding size for typological embedding vector')
	entail_parser.add_argument('-nt', '--numtyp', action = 'store', type = int, dest = 'num_typ_features', default = 103,
		help = 'Number of typological features in the typological features extracted.')
	entail_parser.add_argument('-lml', '--lmlayer', action = 'store', dest = 'lm_layer', type = int, default = -1,
		help = 'Layer to obtain BERT representations from')
	entail_parser.add_argument('-lr', '--learningrate', action = 'store', dest = 'lr', type = float, default = 0.0005, 
		help = 'Learning rate for optimization')
	entail_parser.add_argument('-ep', '--numepochs', action = 'store', dest = 'num_epochs', type = int, default = 10,
		help = 'Number of epochs of training')

	#TEXT SUMMARIZATION ARGUMENTS

	sum_parser.add_argument('-p', '--path', action = 'store', type = str, dest = 'base_path', default = './',
		help = 'Base path to all Dependency Parsing models and data')
	sum_parser.add_argument('-d', '--data', action = 'store', type = str, dest = 'data_path', default = '../datasets', 
		help = 'Dataset location')
	sum_parser.add_argument('-t', '--train', action = 'store_true', dest = 'train_model',
		help = 'Train a new model, saved in saved_models directory in entailment directory')
	sum_parser.add_argument('-ev', '--eval', action = 'store_false', dest = 'train_model', 
		help = 'Evaluate a pre-existing model, saved in saved_models directory in the entailment directory')
	sum_parser.set_defaults(train_model = False)
	sum_parser.add_argument('-m', '--model', action = 'store', type = str, dest = 'modelname', default = 'sum_model.pt', 
		help = 'Name of saved model that is either being trained or being evaluated. Most be stored in saved_models directory')
	sum_parser.add_argument('-l', '--lang', action = 'store', type = str, dest = 'lang', default = 'en',
		help = 'Language to run summarization model on')
	sum_parser.add_argument('-ty', '--typological', action = 'store_true', dest = 'typological',
		help = 'Include typological features in training')
	sum_parser.add_argument('-nty', '--notypological', action = 'store_false', dest = 'typological',
		help = 'Do not include typological features in training')
	sum_parser.set_defaults(typological = False)
	sum_parser.add_argument('-tf', '--typfeatures', action = 'store', type = str, dest = 'typ_feature', default = 'syntax_knn',
		help = 'Which typological features to extract from the typological database')
	sum_parser.add_argument('-te', '--typencode', action = 'store', type = str, dest = 'typ_encode', default = 'concat',
		help = 'Method to use for incorporating typological features. Choose from [concat, add_att, mul_att] to decide to either use a concatentation or attention method')
	sum_parser.add_argument('-ft', '--fine_tune', action = 'store_true', dest = 'fine_tune',
		help = 'Fine tune language model')
	sum_parser.add_argument('-nft', '--no_fine_tune', action = 'store_false', dest = 'fine_tune',
		help = 'Use frozen representations')
	sum_parser.set_defaults(fine_tune = True)
	sum_parser.add_argument('-sm', '--save_model', action = 'store_true', dest = 'save_model', 
		help = 'Save a trained model')
	sum_parser.add_argument('-nsm', '--no_save', action = 'store_false', dest = 'save_model',
		help = 'Don\'t save a run')
	sum_parser.set_defaults(save_model = True)

	#Model Hyperparameters
	sum_parser.add_argument('-lhs', '--lstmsize', action = 'store', dest = 'lstm_hidden_size', type = int, default = 400, 
		help = 'LSTM Hidden size when using encoder LSTM')
	sum_parser.add_argument('-ahs', '--attentionsize', action = 'store', dest = 'attention_hidden_size', type = int, default = 200,
		help = 'Multiplicative Attention Hidden Size')
	sum_parser.add_argument('-ll', '--lstmlayers', action = 'store', dest = 'lstm_layers', type = int, default = 3,
		help = 'Number of LSTM Layers in LSTM encoder')
	sum_parser.add_argument('-dr', '--dropout', action = 'store', dest = 'dropout', type = float, default = 0.33,
		help = 'Dropout probability to be used in all components of model')
	sum_parser.add_argument('-lm', '--lm_model_name', action = 'store', dest = 'lm_model_name', type = str, default = 'bert-base-uncased',
		help = 'BERT Model to use when using BERT as encoder')
	sum_parser.add_argument('-tes', '--typsize', action = 'store', dest = 'typ_embed_size', type = int, default = 32,
		help = 'Embedding size for typological embedding vector')
	sum_parser.add_argument('-nt', '--numtyp', action = 'store', type = int, dest = 'num_typ_features', default = 103,
		help = 'Number of typological features in the typological features extracted.')
	sum_parser.add_argument('-lml', '--lmlayer', action = 'store', dest = 'lm_layer', type = int, default = -1,
		help = 'Layer to obtain BERT representations from')
	sum_parser.add_argument('-lr', '--learningrate', action = 'store', dest = 'lr', type = float, default = 0.0005, 
		help = 'Learning rate for optimization')
	sum_parser.add_argument('-ep', '--numepochs', action = 'store', dest = 'num_epochs', type = int, default = 10,
		help = 'Number of epochs of training')

	return ap.parse_args()

def main():
	args = get_cmd_arguments()

	assert(args.task_name in ['dep', 'pos', 'entail', 'sum']), 'Task must be either Dependency Parsing (dep), POS Tagging (pos), Entailment Classification (entail), or Text Summarization (sum)'

	seed = 0
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	if cuda.is_available():
		device = 'cuda'
		torch.cuda.manual_seed_all(seed)
	else:
		device = 'cpu'

	print('Using device: {}'.format(device)) #Ensure on GPU!

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

	if args.task_name == 'dep':
		dep_main(args, device)
	elif args.task_name == 'pos':
		pos_main(args, device)
	elif args.task_name == 'entail':
		en_main(args, device)
	else:
		sum_main(args, device)

if __name__ == '__main__':
	main()