import os
import sys
import numpy as np
import pandas as pd
import math
from types import SimpleNamespace

import torch
import torch.cuda as cuda
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import transformers

from entail_data_load import *
from entail_train import *

class MDL_Probing(object):
	def __init__(self, args, filenames, timesteps, device):
		en_df = pd.read_csv(os.path.join(args.data_path, 'SICK.txt'), delimiter = '\t')
		dataloaders = self.data_loaders(en_df, timesteps)
		self.K = 3
		self.online_codelength = self.mdl_probing(args, dataloaders, device)
	def data_loaders(self, sent_parses, timesteps):
		en_dataset = EntailmentData(sent_parses)
		index_pairs, dataloaders = [], []
		for t in timesteps:
			data_idx = int(t*0.01*len(en_dataset))
			train_idx = round(0.5*data_idx)
			index_pairs.append((train_idx, data_idx))
		for train_idx, test_idx in index_pairs:
			train_en_dataset, test_en_dataset = data.Subset(en_dataset, range(0, train_idx)), data.Subset(en_dataset, range(train_idx, test_idx))
			dataloaders.append((data.DataLoader(train_en_dataset, batch_size = 1, shuffle = False), data.DataLoader(test_en_dataset, batch_size = 1, shuffle = False)))
		return dataloaders
	def mdl_training(self, args, train_loader, classifier, optimizer, device):
		print('Beginning Training')
		for epoch in range(args.num_epochs):
			total_loss = 0
			classifier.train()
			classifier = classifier.to(device)
			for i, batch in enumerate(train_loader):

				sent1 = batch['sentence_A_tokens'].to(device)
				sent2 = batch['sentence_B_tokens'].to(device)
				label = batch['entailment_label'].to(device)

				entail_pred = classifier.forward(sent1, sent2, lang = args.lang, typ_feature = arg.typ_feature, device = args.device)

				loss = classifier.loss(pred_label = entail_pred, label = label)
				loss.backward()
				optimizer.step()
				optimizer.zero_grad()
				total_loss = total_loss + loss.item()

				print('Epoch {}, train loss={}'.format(epoch, total_loss / len(train_loader)))

		return classifier
	def mdl_eval(self, args, classifier, test_loader, device):
		print('Beginning Evaluation')
		codelength = 0
		classifier.eval()
		classifier.to(device)
		for i, batch in enumerate(test_loader):
			sent1 = batch['sentence_A_tokens'].to(device)
			sent2 = batch['sentence_B_tokens'].to(device)

			entail_pred = classifier.forward(sent1, sent2, lang = args.lang, typ_feature = arg.typ_feature, device = device)
			
			label = batch['entailment_label'].to(device)
			en_logits = entail_pred.argmax(-1)

			pos_probs = nn.functional.softmax(en_logits, dim = 1)

			for i, pos in enumerate(label.squeeze(0)):
				prob_val = pos_probs[i, label].item()
				codelength += np.log2(prob_val)
		return codelength
	def mdl_probing(self, args, dataloaders, device):
		term1 = 0.001*np.log2(self.K)
		classifier = EnMLP(n_rels = num_labels, tokenizer = args.tokenizer, lm_model_name = args.lm_model_name, model_type = args.model_type, typological = args.typological, 
			typ_embed_size = args.typ_embed_size, num_typ_features = args.num_typ_features, typ_encode = args.typ_encode, n_lm_layer = args.n_lm_layer, attention_hidden_size = args.attention_hidden_size,
			fine_tune = args.fine_tune, extract_cls = args.extract_cls, average_sen = args.average_sen, mlp_hidden_size = args.mlp_hidden_size, dropout = args.dropout)
		optimizer = optim.Adam(classifier.parameters(), lr = args.lr)
		classifier = classifier.double()
		term2 = 0
		for i, (train_loader, test_loader) in enumerate(dataloaders):
			print('TIMESTEP {}'.format(i))
			trained_classifier = self.mdl_training(args, train_loader, classifier, optimizer, device)
			term2 += self.mdl_eval(args, trained_classifier, test_loader, device)
		return term1 - term2

if __name__ == '__main__':
	args = SimpleNamespace()
	args.base_path = './'
	args.data_path = '../datasets/SICK'
	args.lm_model_name = 'bert-base-uncased'
	args.model_type = 'concat'
	args.n_lm_layer = -1
	args.dropout = 0.33
	args.typological = False
	args.typ_embed_size = 32
	args.num_typ_features = 289
	args.typ_encode = 'concat'
	args.attention_hidden_size = 200
	args.typ_feature = 'syntax_knn+phonology_knn+inventory_knn'
	args.lang = 'en'
	args.save_model = False
	args.modelname = 'mlp_model_1.pt'
	args.lr = 0.005
	args.tokenizer = transformers.BertTokenizer.from_pretrained(args.lm_model_name) if 'bert' in args.lm_model_name else transformers.GPT2Model.from_pretrained(args.lm_model_name)
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	args.fine_tune = False	
	args.extract_cls = True 
	args.average_sen = False
	args.mlp_hidden_size = 200
	args.save_model = True
	timesteps = [0.2, 0.4, 0.8, 1.6, 3.2, 6.25, 12.5, 25, 50, 100]
	device = 'cuda' if cuda.is_available() else 'cpu'

	typ_str = 'with ' + args.typ_feature if args.typological else 'without'
	print('Beginning MDL Evaluation on Entailment on device {} {} typological features on language {}'.format(device, typ_str, args.lang))

	mdl_probing = MDL_Probing(args, filenames, timesteps, device)
	print(mdl_probing.online_codelength)