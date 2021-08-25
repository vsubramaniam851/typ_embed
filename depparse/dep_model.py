# -*- coding: utf-8 -*-

import sys
import torch
import torch.nn as nn
import torch.functional as F 

from dep_modules import *
sys.path.insert(1, '../')
from embedding_models import *

class BiaffineDependencyModel(nn.Module):
	def __init__(self, 
		n_words, 
		n_pos,
		n_rels, 
		word_embed_size = 100,
		pos_embed_size = 100,
		lstm_hidden_size = 400, 
		encoder = 'lstm', 
		lstm_layers = 3, 
		bert = None, 
		bert_pad_index = 0, 
		dropout = 0.33, 
		n_bert_layer = 4, 
		n_arc_mlp = 500, 
		n_rel_mlp = 100, 
		scale = 0, 
		pad_index = -1, 
		unk_index = 0, 
		typological = False, 
		typ_embed_size = 32, 
		num_typ_features = 289,
		typ_encode = 'concat',
		attention_hidden_size = 200):

		super(BiaffineDependencyModel, self).__init__()

		self.encoder = encoder
		self.pad_index = pad_index 
		if encoder == 'lstm':
			self.encode = LSTMEmbedding(num_words = n_words, num_pos = n_pos, lstm_layers = lstm_layers, word_embed_size = word_embed_size, pos_embed_size = pos_embed_size, 
				lstm_hidden_size = lstm_hidden_size, dropout = dropout, typological = typological, typ_embed_size = typ_embed_size, num_typ_features = num_typ_features, typ_encode = typ_encode,
				attention_hidden_size = attention_hidden_size)
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

		self.arc_mlp_d = MLP(n_in = n_embed, n_out = n_arc_mlp, dropout = dropout)
		self.arc_mlp_h = MLP(n_in = n_embed, n_out = n_arc_mlp, dropout = dropout)
		self.rel_mlp_d = MLP(n_in = n_embed, n_out = n_rel_mlp, dropout = dropout)
		self.rel_mlp_h = MLP(n_in = n_embed, n_out = n_rel_mlp, dropout = dropout)

		self.arc_attn = Biaffine(n_in = n_arc_mlp, scale = scale, bias_x = True, bias_y = False)
		self.rel_attn = Biaffine(n_in = n_rel_mlp, n_out = n_rels, bias_x = True, bias_y = True)
		self.criterion = nn.CrossEntropyLoss()

	def forward(self, words, lang = 'en', typ_feature = 'syntax_knn+phonology_knn+inventory_knn', pos_tags = None, input_ids = None, attention_mask = None, device = 'cpu'):
		if self.encoder == 'lstm':
			x = self.encode(words = words, pos_tags = pos_tags, lang = lang, typ_feature = typ_feature, device = device)
		else:
			x = self.encode(input_ids = input_ids, attention_mask = attention_mask, lang = lang, typ_feature = typ_feature, device = device)
		assert(x.size(2) == self.n_embed), 'Check if size of encoding is correct'

		mask = words.ne(self.pad_index) if len(words.shape) < 3 else words.ne(self.pad_index).any(-1)

		arc_d = self.arc_mlp_d(x)
		arc_h = self.arc_mlp_h(x)
		rel_d = self.rel_mlp_d(x)
		rel_h = self.rel_mlp_h(x)

		s_arc = self.arc_attn(arc_d, arc_h).masked_fill_(~mask.unsqueeze(1), float('-inf'))
		s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

		return s_arc, s_rel, mask

	def loss(self, s_arc, s_rel, arcs, rels, mask):
		s_arc, arcs = s_arc[mask], arcs[mask]
		s_rel, rels = s_rel[mask], rels[mask]
		s_rel = s_rel[torch.arange(len(arcs)), arcs]
		arc_loss = self.criterion(s_arc, arcs)
		rel_loss = self.criterion(s_rel, rels)

		return arc_loss + rel_loss 

	def decode(self, s_arc, s_rel, mask):
		lens = mask.sum(1)
		arc_preds = s_arc.argmax(-1)
		rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

		return arc_preds, rel_preds