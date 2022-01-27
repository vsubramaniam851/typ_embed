import sys
import os
import numpy as np
import random

import torch
import torch.optim as optim

import transformers

def sum_train(args, train_loader, val_loader, device):
	classifier = SumModel(lm_model_name = args.lm_model_name, typological = args.typological, typ_embed_size = args.typ_embed_size, 
		num_typ_features = args.num_typ_features, typ_encode = args.typ_encode, num_rnn_layers = args.num_rnn_layers, n_lm_layer = args.n_lm_layer, 
		attention_hidden_size = args.attention_hidden_size, summarization = args.summarization, rnn_hidden_size = args.rnn_hidden_size, 
		dropout = args.dropout)
	classifier = classifier.to(device)
	classifier = classifier.double()

	optimizer = optim.Adam(classifier.parameters(), lr = args.lr)

	if args.typological:
		typ_str = 'with'
	else:
		typ_str = 'without'

	print('Beginning training for summarization {} typological features'.format(typ_str))

	for epoch in range(args.num_epochs):
		total_loss = 0
		classifier.train()

		for i,batch in enumerate(train_loader):
			if i % 1000 == 0:
				print('Epoch {} Train Batch {}'.format(epoch, i))
			article_tokens = batch['article_tokens'].to(device)
			cls_idxs = batch['cls_idxs']
			labels = batch['labels'].to(device)

			preds = classifier.forward(article_tokens, cls_idxs, lang = args.lang, typ_feature = args.typ_feature, device = device)
			loss = classifier.loss(preds, labels)
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			total_loss = total_loss + loss.item()

		print('Epoch {}, train loss={}'.format(epoch, total_loss / len(train_loader)))
		total_loss = 0
		classifier.eval()

		for i, batch in enumerate(val_loader):
			if i%1000 == 0:
				print('Epoch {} Train Batch {}'.format(epoch, i))
			article_tokens = batch['article_tokens'].to(device)
			cls_idxs = batch['cls_idxs']
			labels = batch['labels'].to(device)

			preds = classifier.forward(article_tokens, cls_idxs, lang = args.lang, typ_feature = args.typ_feature, device = device)
			loss = classifier.loss(preds, labels)
			total_loss = total_loss + loss.item()
		print('Epoch {}, valid loss={}'.format(epoch, total_loss/len(val_loader)))

	print('Training is finished')
	if args.save_model:
		save_path = os.path.join(args.base_path, 'saved_models', args.modelname)
		print('Saving model to {}'.format(save_path))
		torch.save(classifier.state_dict(), save_path)

	return classifier