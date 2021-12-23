# -*- coding: utf-8 -*-

import os
import numpy as np 
import torch
import torch.nn as nn
import transformers
from transformers import BertModel, BertConfig
import lang2vec.lang2vec as l2v

class TypologicalLanguageEmbed(nn.Module):
	def __init__(self, num_typ_features, typ_embed_size, hidden_size, dropout = 0.25):
		super(TypologicalLanguageEmbed, self).__init__()
		self.linear1 = nn.Linear(num_typ_features, hidden_size)
		self.linear2 = nn.Linear(hidden_size, typ_embed_size)
		self.activation = nn.ReLU()
		self.dropout = nn.Dropout(dropout)

	def forward(self, lang, typ_feature, device = 'cpu'):
		f_vec = torch.tensor(l2v.get_features(l2v.LETTER_CODES[lang], typ_feature)[l2v.LETTER_CODES[lang]]).double().to(device)
		x = self.linear1(f_vec)
		x = self.linear2(x)
		x = self.activation(x)
		x = self.dropout(x)

		return x

class Attention(nn.Module):
	def __init__(self, d1, d2, d3):
		super(Attention, self).__init__()
		self.d1 = d1
		self.d2 = d2
		self.d3 = d3

	def forward(self, typ_embed, word_embed):
		e_scores = self._get_weights(typ_embed, word_embed)
		a_scores = nn.functional.softmax(e_scores, dim = 0)
		return torch.mul(word_embed.T, a_scores).T

class AdditiveAttention(Attention):
	def __init__(self, d1, d2, d3 = 200):
		assert(d3 is not None), 'Enter a numerical d3 for Additive Attention'
		super().__init__(d1, d2, d3)

		self.W_3 = nn.Parameter(torch.FloatTensor(self.d3).uniform_(-0.1, 0.1))
		self.W_1 = nn.Linear(self.d2, self.d3)
		self.W_2 = nn.Linear(self.d1, self.d3)

	def _get_weights(self, typ_embed, word_embed):
		typ_embed = typ_embed.repeat(word_embed.size(0), 1)
		weights = self.W_1(word_embed) + self.W_2(typ_embed)
		return torch.tanh(weights)@self.W_3

class MultiplicativeAttention(Attention):
	def __init__(self, d1, d2, d3 = None):
		assert(d3 is None), 'Multiplicative only requires one dimension'
		super().__init__(d1, d2, d3)

		self.W = nn.Parameter(torch.FloatTensor(self.d1, self.d2).uniform_(-0.1, 0.1))

	def _get_weights(self, typ_embed, word_embed):
		weights = (typ_embed@self.W@word_embed.T).squeeze(0)
		return weights/np.sqrt(weights.size(0))

class LSTMEmbedding(nn.Module):
	def __init__(self, 
		num_words, 
		num_pos, 
		lstm_layers, 
		word_embed_size = 100, 
		pos_embed_size = 100, 
		lstm_hidden_size = 400, 
		typological = True, 
		typ_embed_size = 32, 
		dropout = 0.33, 
		num_typ_features = 289,
		typ_encode = 'concat',
		attention_hidden_size = 200):

		super(LSTMEmbedding, self).__init__()
		self.word_embed = nn.Embedding(num_embeddings = num_words, embedding_dim = word_embed_size)
		if num_pos != 0:
			self.pos_embed = nn.Embedding(num_embeddings = num_pos, embedding_dim = pos_embed_size)
		self.lstm = nn.LSTM(word_embed_size+pos_embed_size, lstm_hidden_size, num_layers = lstm_layers, dropout = dropout, batch_first = True, bidirectional = True)
		self.dropout = nn.Dropout(dropout)
		if typological:
			self.typ = TypologicalLanguageEmbed(num_typ_features = num_typ_features, typ_embed_size = typ_embed_size, hidden_size = num_typ_features, dropout = dropout)
			if typ_encode == 'add_att':
				self.attention = AdditiveAttention(d1 = typ_embed_size, d2 = lstm_hidden_size * 2, d3 = attention_hidden_size)
			elif typ_encode == 'mul_att':
				self.attention = MultiplicativeAttention(d1 = typ_embed_size, d2 = lstm_hidden_size*2)
		else:
			self.typ = None
		self.typological = typological
		self.typ_encode = typ_encode

	def forward(self, words, pos_tags, lang = 'en', typ_feature = 'syntax_knn+phonology_knn+inventory_knn', device = 'cpu'):
		word_embeddings = self.word_embed(words)
		word_embeddings = self.dropout(word_embeddings)
		if pos_tags is not None:
			pos_embeddings = self.pos_embed(pos_tags)
			pos_embeddings = self.dropout(pos_embeddings)
			embeddings = torch.cat([word_embeddings, pos_embeddings], dim = 2)
		else:
			embeddings = word_embeddings
		outputs, _ = self.lstm(embeddings)

		if self.typological:
			typ_embed = self.typ(lang = lang, typ_feature = typ_feature, device = device)
			if self.typ_encode == 'concat':
				typ_embed = typ_embed.repeat(outputs.size(1), 1).unsqueeze(0)
				outputs = torch.cat([outputs, typ_embed], dim = 2)
			elif self.typ_encode == 'add_att' or self.typ_encode == 'mul_att':
				outputs = outputs.squeeze(0)
				outputs = self.attention.forward(typ_embed = typ_embed, word_embed = outputs)
				outputs = outputs.unsqueeze(dim = 0)

		outputs = self.dropout(outputs)

		return outputs

class LMEmbedding(nn.Module):
	def __init__(self, 
		lm_model_name, 
		typological, 
		bert_pad_index = 0, 
		bert_hidden_size = 768, 
		typ_embed_size = 32, 
		num_typ_features = 289, 
		lm_layer = 4, 
		dropout = 0.25,
		typ_encode = 'concat',
		attention_hidden_size = 200,
		fine_tune = True):

		super(LMEmbedding, self).__init__()
		self.lm_model_name = lm_model_name
		if 'bert' in self.lm_model_name:
			self.lm = transformers.BertModel.from_pretrained(self.lm_model_name)
		else:
			self.lm = transformers.GPT2Model.from_pretrained(self.lm_model_name)
		if not fine_tune:
			self.lm.eval()
			for param in self.lm.base_model.parameters():
				param.requires_grad = False

		if typological:
			self.typ = TypologicalLanguageEmbed(num_typ_features = num_typ_features, typ_embed_size = typ_embed_size, hidden_size = num_typ_features, dropout = dropout)
			if typ_encode == 'add_att':
				self.attention = AdditiveAttention(d1 = typ_embed_size, d2 = 768, d3 = attention_hidden_size)
			elif typ_encode == 'mul_att':
				self.attention = MultiplicativeAttention(d1 = typ_embed_size, d2 = 768)
		else: 
			self.typ = None

		self.dropout = nn.Dropout(dropout)

		self.typological = typological 
		self.lm_layer = lm_layer
		self.typ_encode = typ_encode

	def forward(self, input_ids, sentence, lang = 'en', typ_feature = 'syntax_knn+phonology_knn+inventory_knn', device = 'cpu'):
		lm_output = self.lm(input_ids = input_ids, output_hidden_states = True)

		hidden_state = lm_output.hidden_states[self.lm_layer]
		if 'bert' in self.lm_model_name:
			hidden_state = hidden_state[:, 1:-1, :]
		else:
			hidden_state = hidden_state[:, :-1, :]
		outputs = self.average_subwords(hidden_state, input_ids, sentence)

		if self.typological:
			typ_embed = self.typ(lang = lang, typ_feature = typ_feature, device = device)
			if self.typ_encode == 'concat':
				typ_embed = typ_embed.repeat(outputs.size(1), 1).unsqueeze(0)
				outputs = torch.cat([outputs, typ_embed], dim = 2)
			elif self.typ_encode == 'add_att' or self.typ_encode == 'mul_att':
				outputs = outputs.squeeze(0)
				outputs = self.attention.forward(typ_embed = typ_embed, word_embed = outputs)
				outputs = outputs.unsqueeze(0)

		outputs = self.dropout(outputs)
		return outputs

	def average_subwords(self, embeddings, input_ids, sentence):
		tokenizer = transformers.BertTokenizer.from_pretrained(self.lm_model_name) if 'bert' in self.lm_model_name else transformers.GPT2Tokenizer.from_pretrained(self.lm_model_name)
		idx = 0
		word_embeddings = []
		for word, in sentence:
			subwords = tokenizer.tokenize(word)
			word_embedding = torch.mean(embeddings[:, idx:idx+len(subwords), :], dim = 1)
			word_embeddings.append(word_embedding)
			idx = idx + len(subwords)
		return torch.stack(word_embeddings, dim = 1)