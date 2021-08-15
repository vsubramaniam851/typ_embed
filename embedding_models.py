# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import transformers
from transformers import BertModel, BertConfig

class TypologicalLanguageEmbed(nn.Module):
	def __init__(self, typ_features, typ_embed_size, hidden_size = 100, dropout = 0.25):
		suoer(TypologicalLanguageEmbed, self).__init__()
		self.linear1 = nn.Linear(typ_features, hidden_size)
		self.linear2 = nn.Linear(hidden_size, typ_embed_size)
		self.activation = nn.ReLU()
		self.dropout = nn.Dropout(dropout)

	def forward(f_vec):
		x = self.linear1(f_vec)
		x = self.linear2(x)
		x = self.activation(x)
		x = self.dropout(x)

		return x

class LSTMEmbedding(nn.Module):
	def __init__(self, num_words, num_pos, lstm_layers, word_embed_size = 100, pos_embed_size = 100, lstm_hidden_size = 400, typological = True, typ_size = 200, dropout = 0.33, typ_features = 289):
		super(LSTMEmbedding, self).__init__()
		self.word_embed = nn.Embedding(num_embeddings = num_words, embedding_dim = word_embed_size)
		self.pos_embed = nn.Embedding(num_embeddings = num_pos, embedding_dim = pos_embed_size)
		self.lstm = nn.LSTM(word_embed_size+pos_embed_size, lstm_hidden_size, num_layers = lstm_layers, dropout = dropout, batch_first = True, bidirectional = True)
		self.dropout = nn.Dropout(dropout)
		if typological:
			self.typ = TypologicalLanguageEmbed(typ_features, typ_embed_size = typ_size, dropout = dropout)
		else:
			self.typ = None
		self.typological = typological

	def forward(self, words, pos_tags, f_vec = None):
		word_embeddings = self.word_embed(words)
		pos_embeddings = self.pos_embed(pos_tags)
		# if word_embeddings.size()[0] != pos_embeddings.size()[0]:
		# 	print(words.size())
		# 	print(pos_tags.size())
		word_embeddings = self.dropout(word_embeddings)
		pos_embeddings = self.dropout(pos_embeddings)
		embeddings = torch.cat([word_embeddings, pos_embeddings], dim = 2)
		outputs, _ = self.lstm(embeddings)
		outputs = outputs.squeeze(0)

		if self.typological:
			lstm_embeddings = []
			typ_embeds = self.typ(f_vec)
			for i in range(len(outputs)):
				word_vec = outputs[i]
				typ_word_vec = torch.cat([word_vec, typ_embeds], dim = 1)
				lstm_embeddings.append(typ_word_vec)
			outputs = torch.stack(lstm_embeddings, dim = 0)
		outputs = outputs.unsqueeze(0)
		outputs = self.dropout(outputs)

		return outputs

class BERTEmbedding(nn.Module):
	def __init__(self, bert, typological, bert_pad_index = 0, bert_hidden_size = 768, typ_size = 200, typ_features = 289, bert_layer = 4, dropout = 0.25):
		super(BERTEmbedding, self).__init__()
		self.lm = transformers.BertModel.from_pretrained(bert)
		if typological:
			self.typ = TypologicalLanguageEmbed(typ_features, typ_embed_size = typ_size, dropout = dropout)
		else: 
			self.typ = None
		self.dropout = nn.Dropout(dropout)

		self.typological = typological 
		self.bert_layer = bert_layer 

	def forward(self, input_ids, attention_mask, f_vec = None):
		lm_output = self.lm(input_ids = input_ids, attention_mask = attention_mask, output_hidden_states = True)

		hidden_state = lm_output.hidden_states[self.bert_layer]
		hidden_state = hidden_state[:, 1:-1, :]

		hidden_state = hidden_state.squeeze(0)
		if self.typological:
			bert_embeddings = []
			typ_embeds = self.typ(f_vec)
			for i in range(len(hidden_state)):
				word_vec = hidden_state[i]
				typ_word_vec = torch.cat([word_vec, typ_embeds], dim = 1)
				bert_embeddings.append(typ_word_vec)
			hidden_state = torch.stack(bert_embeddings)
		hidden_state = hidden_state.unsqueeze(0)
		outputs = self.dropout(hidden_state)

		return outputs