import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(1, '../')
from embedding_models import *
from sum_modules import *

class SumModel(nn.Module):
	def __init__(self, 		
		lm_model_name = 'bert-base-uncased', 
		typological = False, 
		typ_embed_size = 32, 
		num_typ_features = 289,
		typ_encode = 'concat',
		num_rnn_layers = 3,
		n_lm_layer = -1,
		attention_hidden_size = 200,
		summarization = True,
		rnn_hidden_size = 200,
		dropout = 0.33):

		super(SumModel, self).__init__()
		self.encode = LMEmbedding(lm_model_name = lm_model_name, tokenizer = tokenizer, typological = typological, bert_hidden_size = 768, typ_embed_size = typ_embed_size,  num_typ_features = num_typ_features, lm_layer = n_lm_layer, typ_encode = typ_encode, 
			attention_hidden_size = attention_hidden_size, fine_tune = fine_tune, summarization = summarization)

		self.rnn_model = RNNEncoder(bidirectional = True, input_size = 768, num_layers = num_rnn_layers, hidden_size = rnn_hidden_size, dropout = dropout)

		self.criterion = nn.BCELoss()

	def forward(self, input_ids, cls_idxs, sentence = None, lang = 'en', typ_feature = 'syntax_knn+phonology_knn+inventory_knn', device = 'cpu'):
		x = self.encode(input_ids, sentence = sentence, lang = lang, typ_feature = typ_feature, device = device)
		x = x[:, cls_idxs]
		return self.rnn_model(x)

	def loss(self, preds, targets):
		return self.criterion(preds, targets)