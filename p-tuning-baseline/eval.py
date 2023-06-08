from transformers import AutoModelForMaskedLM,AutoTokenizer
import os
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
import argparse
from data_util.dataset import load_files, LAMADataset
import sys
sys.path.append('./model')
from model.counter_modeling import CounterPTuneForLAMA
import random
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
pretrained_model = AutoModelForMaskedLM.from_pretrained(model_name)

def construct_generation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relation_id", type=str, default="P19")
    parser.add_argument("--seed", type=int, default=34)
    args = parser.parse_args()
    return args
args = construct_generation_args()


data_template_path = '../../data/LAMA/relations.jsonl'
data_pairs_path = '../../data/LAMA/fact-retrieval/original'
relations_path = '../../data/LAMA/all_relations.txt'

train_data = load_files(data_pairs_path,
                             map(lambda relation: f'{relation}/train.jsonl', [args.relation_id]))
dev_data = load_files(data_pairs_path,
                             map(lambda relation: f'{relation}/dev.jsonl', [args.relation_id]))
test_data = load_files(data_pairs_path,
                             map(lambda relation: f'{relation}/test.jsonl', [args.relation_id]))

train_set = LAMADataset('train', data_template_path, train_data, tokenizer, args)
dev_set = LAMADataset('dev', data_template_path, dev_data, tokenizer, args)
test_set = LAMADataset('test', data_template_path, test_data, tokenizer, args)


train_loader = DataLoader(train_set, batch_size=8, shuffle=True, drop_last=True)
dev_loader = DataLoader(dev_set, batch_size=8, shuffle=False, drop_last=False)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False, drop_last=False)

def set_seed(seed=34):
    np.random.seed(seed)
    torch.manual_seed(seed)
 

set_seed(args.seed)



def eval():
    model = CounterPTuneForLAMA(device='cuda:0', pretrained_model=pretrained_model,
                             tokenizer=tokenizer, template=(3, 3, 3))
    checkpoint = torch.load(f'./saved_model/{args.relation_id}.pt')
    model.prompt_encoder.load_state_dict(checkpoint['embedding'])
    model.eval()
    top_predicted_words = []
    top_predicted_logits = []
    counter_top_predicted_words = []
    counter_top_predicted_logits = []
    all_x_hs, all_x_ts = [], []
    with torch.no_grad():
        model.eval()
        for batch_idx, (subjs, objs, predicate_id, template) in tqdm(enumerate(test_loader)):
            _, _top_predicted_words, _top_predicted_logits, \
            _counter_top_predicted_words, _counter_top_predicted_logits, \
            hit1, batch_matrics = \
                model(subjs, objs, predicate_id, template, return_candidates=True)
            top_predicted_words += _top_predicted_words
            top_predicted_logits += _top_predicted_logits
            counter_top_predicted_words += _counter_top_predicted_words
            counter_top_predicted_logits += _counter_top_predicted_logits
            all_x_hs += subjs
            all_x_ts += objs

        if not os.path.exists(f'./result/'):
            os.makedirs(f'./result/')
        result_file = open(f'./result/test_{args.relation_id}.json', 'w',encoding='utf-8')
        json_objs = []
        for idx in range(len(all_x_hs)):
            json_objs.append({'subj': all_x_hs[idx],
                              'obj': all_x_ts[idx],
                              'predicted_words': [(top_predicted_words[idx][idxx], top_predicted_logits[idx][idxx])
                                                  for idxx in range(len(top_predicted_words[idx]))],
                              'subject_mask_predicted_words': [
                                  (counter_top_predicted_words[idx][idxx], counter_top_predicted_logits[idx][idxx])
                                  for idxx in range(len(counter_top_predicted_words[idx]))]
                              })
        json.dump(json_objs, result_file, indent=4)
eval()

