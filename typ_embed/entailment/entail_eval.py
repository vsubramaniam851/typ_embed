# -*- coding: utf-8 -*-

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

from entail_model import *
from embedding_models import *
from entail_data_load import *

def entail_eval(args, classifier, test_loader, device):
	classifier = classifier.to(device)
	classifier = classifier.double()
	classifier.eval()

	total_correct = 0

	if args.typological:
		typ_str = 'with'
	else:
		typ_str = 'without'

	print('Evaluating {} model {} typological features'.format(args.modelname, typ_str))

	for batch in tqdm(test_loader):
		sent1 = batch['sentence_A_tokens'].to(device)
		sent2 = batch['sentence_B_tokens'].to(device)
		label = batch['entailment_label'].to(device)

		entail_pred = classifier.forward(sent1, sent2, lang = args.lang, typ_feature = arg.typ_feature, device = device)
		
		entail_pred = classifier.decode(entail_pred)
		entail_pred = entail_pred.squeeze(0)
		entail_pred = entail_preds.tolist()

		if entail_pred[0] == label[0]:
			total_correct += 1

	return f'Entailment Accuracy {total_correct/len(test_loader)}'