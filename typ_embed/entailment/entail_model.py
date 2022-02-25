import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from entail_modules import *
sys.path.insert(1, '../')
from embedding_models import *

class EnMLP(nn.Module):
	def __init__(self, 
		n_rels,
		tokenizer,
		lm_model_name = 'bert-base-uncased', 
		model_type = 'concat',		
		typological = False, 
		typ_embed_size = 32, 
		num_typ_features = 289,
		typ_encode = 'concat',
		n_lm_layer = -1,
		attention_hidden_size = 200,
		fine_tune = True,
		extract_cls = True,
		average_sen = False,
		mlp_hidden_size = 200,
		dropout = 0.33):

		super(EnMLP, self).__init__()

		self.encode = LMEmbedding(lm_model_name = lm_model_name, tokenizer = tokenizer, typological = typological, bert_hidden_size = 768, typ_embed_size = typ_embed_size, 
			num_typ_features = num_typ_features, lm_layer = n_lm_layer, typ_encode = typ_encode, attention_hidden_size = attention_hidden_size, fine_tune = fine_tune, 
			extract_cls = extract_cls, average_sen = average_sen)
		n_embed = 768
		if typological and typ_encode == 'concat':
			n_embed = n_embed + typ_embed_size

		self.n_embed = n_embed

		if model_type == 'concat':
			self.model = MLP(n_in = self.n_embed*2, hidden_size = mlp_hidden_size, n_out = n_rels, dropout = dropout)
		elif model_type == 'bilinear':
			self.model = Bilinear(n_in = self.n_embed, n_out = n_rels, dropout = dropout)

		self.criterion = nn.CrossEntropyLoss()

	def forward(self, sent1, sent2, lang = 'en', typ_feature = 'syntax_knn+phonology_knn+inventory_knn', device = 'cpu'):
		sent1_embed = self.encode(sent1, sentence = None, typ_feature = typ_feature, lang = lang, device = device)
		sent2_embed = self.encode(sent1, sentence = None, typ_feature = typ_feature, lang = lang, device = device)

		sent1_embed = sent1_embed.squeeze(0)
		sent2_embed = sent2_embed.squeeze(0)

		output = self.model(sent1_embed, sent2_embed)

	def loss(self, pred_label, label):
		loss = self.criterion(pred_labels, labels)
		return loss

	def decode(self, pred_labels):
		return pred_labels.argmax(-1)