# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd

import stanza
import torch
import torch.utils.data as data

import transformers

class PreProcessText(object):
	def __init__(self, data_path):
		#Use stanza sentence tokenizer to tokenize the sentences
		self.nlp = stanza.Pipeline(lang='en', processors='tokenize')
		self.art2labels=  self.process_text(data_path)

	def split_sents(self, text):
		doc = self.nlp(text)
		return [sentence.text for sentence in doc.sentences]

	def process_text(self, data_path):
		text_types = ['business', 'entertainment', 'politics', 'sport', 'tech']
		art2sum = {}

		article_path = os.path.join(data_path, 'News Articles')
		summary_path = os.path.join(data_path, 'Summaries')

		for ty in text_types:
			articles = os.path.join(article_path, ty)
			summaries = os.path.join(summary_path, ty)

			for a, s in zip(os.listdir(articles), os.listdir(summaries)):
				with open(os.path.join(articles), a) as f:
					article = f.read()
					article = tuple(self.split_sents(article))
				with open(os.path.join(summaries), s) as f:
					summary = f.read()
					summary = self.split_sents(summary)
				art2sum[article] = summary

		art2labels = []

		#Split the article and map each sentence in the article to a binary label
		for article, summary in art2sum:
			labels = [0 for i in range(len(article))]
			for i, sent in enumerate(article):
				if sent in summary:
					labels[i] = 1
			art2labels.append((article, labels))

		return art2labels

class SummarizationDataset(data.Dataset):
	def __init__(self, art2labels):
		self.art2labels = art2labels
		self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
	def __len__(self):
		return len(self.art2labels)
	def __getitem__(self, idx):
		document = self.art2labels[idx]
		article_tokens =  torch.concat([self.tokenizer.encode(sentence) for sentence in document], dim = 1)
		#Track indexes for CLS token
		cls_idxs = []
		for i, token in enumerate(article_tokens[0]):
			if token == 101:
				cls_idxs.append(i)
		return {
			'article': document
			'article_tokens': article_tokens,
			'cls_idxs': cls_idxs,
			'labels': torch.tensor(document[1])
		}

def sum_dataloaders(data_path, train_split, val_split):
	art2labels = PreProcessText(data_path)
	sum_dataset = SummarizationDataset(art2labels)

	assert train_split+val_split < 1

	#Split by number of documents
	num_train, num_val = int(train_split*len(sum_dataset)), int(val_split*len(sum_dataset))
	num_test = len(sum_dataset) - num_train - num_val

	train_dataset, val_dataset, test_dataset = data.random_split(sum_dataset, [num_train, num_val, num_test])
	sum_train_dataloader, sum_val_dataloader, sum_test_dataloader = data.DataLoader(train_dataset, batch_size = 1, shuffle = True), data.DataLoader(val_dataset, batch_size = 1, shuffle = True), data.DataLoader(test_dataset, batch_size = 1, shuffle = True)
	return sum_train_dataloader, sum_val_dataloader, sum_test_dataloader