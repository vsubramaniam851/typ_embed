# -*- coding: utf-8 -*-

import sys
import torch
import torch.nn as nn
import torch.functional as F 

from postag_modules import *
sys.path.insert(1, '../')
from embedding_models import *

class POSTaggingModel(nn.Module):
	def __init__(self, 
		n_words,
		n_tags,
		word_embed_size = 100,
		lstm_hidden_size = 400,
		encoder = 'lstm',
		lstm_layers = 3,
		bert = None,
		bert_pad_index = 0,
		dropout = 0.33,
		n_bert_layer = 4,
		mlp_hidden_size = 200,
		typological = False,
		typ_embed_size = 32,
		num_typ_features = 289,
		typ_encode = 'concat',
		attention_hidden_size = 200):

		super(POSTaggingModel, self).__init__()

		self.encoder = encoder
		if encoder == 'lstm':
			self.encode = LSTMEmbedding(num_words = n_words, num_pos = 0, lstm_layers = lstm_layers, word_embed_size = word_embed_size, pos_embed_size = 0, 
				lstm_hidden_size = lstm_hidden_size, dropout = dropout, typological = typological, typ_embed_size = typ_embed_size, num_typ_features = num_typ_features,
				typ_encode = typ_encode, attention_hidden_size = attention_hidden_size)
			n_embed = lstm_hidden_size * 2 
		elif encoder == 'bert':
			self.encode = BERTEmbedding(bert = bert, typological = typological, bert_pad_index = bert_pad_index, bert_hidden_size = 768, typ_embed_size = typ_embed_size,
				num_typ_features = num_typ_features, bert_layer = n_bert_layer, typ_encode = typ_encode, attention_hidden_size = attention_hidden_size)
			n_embed = 768
		else:
			n_embed = 0
			raise AssertionError('Please choose either LSTM or BERT Embedding')

		if typological and typ_encode == 'concat':
			n_embed = n_embed + typ_embed_size

		self.n_embed = n_embed

		self.mlp = MLP(n_in = n_embed, n_out = n_tags, mlp_hidden_size = mlp_hidden_size, dropout = dropout)
		self.criterion = nn.CrossEntropyLoss()
	
	def forward(self, words = None, lang = 'en', typ_feature = 'syntax_knn+phonology_knn+inventory_knn', input_ids = None, attention_mask = None, device = 'cpu'):
		if self.encoder == 'lstm':
			x = self.encode(words = words, pos_tags = None, lang = lang, typ_feature = typ_feature, device = device)
		else:
			x = self.encode(input_ids = input_ids, attention_mask = attention_mask, lang=  lang, typ_feature = typ_feature, device = device)

		assert(x.size(2) == self.n_embed), 'Check if size of encoding is correct'

		output = self.mlp(x)

		output = output.squeeze(0)

		assert(len(x.shape) == 2), 'Check that size of encoding is not 3 and has been squeezed'

		return output

	def loss(self, pred_tags, tags):
		assert(pred_tags.size(0) == tags.size(0)), 'Length does not match between tags and predicted tags'
		loss = self.criterion(pred_tags, tags)
		return loss

	def decode(self, pred_tags):
		return pred_tags.argmax(-1)