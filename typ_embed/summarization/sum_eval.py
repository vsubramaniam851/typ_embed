# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from tqdm import tqdm

import torch
from rouge_scorer import rouge_scorer

def calc_rouge_scores(pred_summaries, gold_summaries, 
                                 keys=['rouge1', 'rougeL'], use_stemmer=True):
    #Calculate rouge scores
    scorer = rouge_scorer.RougeScorer(keys, use_stemmer= use_stemmer)
    
    n = len(pred_summaries)
    
    scores = [scorer.score(pred_summaries[j], gold_summaries[j]) for 
              j in range(n)] 
    
    dict_scores={}                                                            
    for key in keys:
        dict_scores.update({key: {}})
        
    for key in keys:
        
        precision_list = [scores[j][key][0] for j in range(len(scores))]
        recall_list = [scores[j][key][1] for j in range(len(scores))]
        f1_list = [scores[j][key][2] for j in range(len(scores))]

        precision = np.mean(precision_list)
        recall = np.mean(recall_list)
        f1 = np.mean(f1_list)
        
        dict_results = {'recall': recall, 'precision': precision, 'f1': f1}
        
        dict_scores[key] = dict_results
        
    return dict_scores

def sum_eval(args, test_loader, classifier, device):
	classifier = classifier.to(device)
	classifier = classifier.double()
	classifier.eval()

	pred_summaries = []
	gold_summaries = []

	for batch in tqdm(test_loader):
		article = np.array(batch['article'])
		article_tokens = batch['article_tokens'].to(device)
		labels = batch['labels'].tolist()[0]
		cls_idxs = batch['cls_idxs']

		gold_summary = ' '.join(article[cls_idxs].tolist())
		gold_summaries.append(gold_summary)

		pred_idxs = classifier.forward(article_tokens, cls_idxs, lang = args.lang, typ_feature = args.typ_feature, device = device)
		pred_idxs = pred_idxs.squeeze(1).cpu().tolist()
		pred_summary = article[pred_idxs]
		pred_summaries.append(pred_summary)

	print(calc_rouge_scores(pred_summaries, gold_summaries))