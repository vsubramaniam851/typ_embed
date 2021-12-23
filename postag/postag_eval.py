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

base_path = '/storage/vsub851/typ_embed/postag'
data_path = '/storage/vsub851/typ_embed/datasets'
train_filename = 'en_ewt-ud-train.conllu'
test_filename = 'en_ewt-ud-test.conllu'

seed = 0

if cuda.is_available():
	device = 'cuda'
	torch.cuda.manual_seed_all(seed)
else:
	print('WARNING, this program is running on CPU')
	device = 'cpu'

def pos_eval(args, classifier, test_loader, device):
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
			word_batch = batch['input_data'].to(device)
			pos_batch = batch['pos_ids'].squeeze(0).to(device)

			pos_preds = classifier.forward(words = word_batch, lang = args.lang, typ_feature = args.typ_feature, pos_tags = pos_batch, device = device)
		else:
			sentence = ' '.join([x[0] for x in batch['words']])
			input_ids = args.tokenizer.encode(sentence, return_tensors = 'pt')
			input_ids = input_ids.to(device)
			word_batch = batch['input_data'].to(device)
			pos_batch = batch['pos_ids'].squeeze(0).to(device)

			pos_preds = classifier.forward(words = word_batch, input_ids = input_ids, lang = args.lang, typ_feature = args.typ_feature, sentence = batch['words'], device = device)

		pos_tags = sent_dict['pos_tags']
		
		pos_preds = classifier.decode(pos_preds)

		for i in range(1, len(pos_preds)):
			pos_pred = pos_preds[i]
			pos = pos_tags[i]

			if pos_pred == pos:
				pos_total_correct += 1
			pos_total_examples += 1

	return 'POS Accuracy {}'.format(pos_total_correct/pos_total_examples)

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

	train_data_loader, valid_data_loader, test_data_loader, vocab_dict, pos_dict, label_dict = dep_data_loaders(args, train_filename = train_filename, valid_filename = valid_filename, test_filename = test_filename)
	classifier = BiaffineDependencyModel(n_words = len(vocab_dict), n_pos = len(pos_dict), n_rels = len(label_dict), word_embed_size = word_embed_size, pos_embed_size = pos_embed_size,  
	encoder = encoder, lstm_hidden_size = lstm_hidden_size, lstm_layers = lstm_layers, bert = bert, bert_pad_index = 0, dropout = dropout, n_bert_layer = bert_layer, 
	n_arc_mlp = 500, n_rel_mlp = 100, scale = scale, pad_index = pad_index, unk_index = 0, typological = typological, typ_embed_size = typ_embed_size, 
	num_typ_features = num_typ_features, typ_encode = typ_encode, attention_hidden_size = 200)

	classifier.load_state_dict(torch.load(os.path.join(args.base_path, 'saved_models', args.modelname)))
	print(arc_eval(args, classifier, test_loader))
	pass