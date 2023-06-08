from transformers import AutoModelForMaskedLM,AutoTokenizer
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
import argparse
from data_util.dataset import load_files, LAMADataset

import sys
sys.path.append('./model')
from model.modeling import PTuneForLAMA

model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
pretrained_model = AutoModelForMaskedLM.from_pretrained(model_name)

def construct_generation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relation_id", type=str, default="P19")
    parser.add_argument("--seed", type=int, default=34)
    parser.add_argument("--alpha_1", type=float, default=0.2)
    parser.add_argument("--alpha_2", type=float, default=0.2)

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


prompt_model_path = f"../p-tuning-baseline/saved_model/{args.relation_id}.pt"

def set_seed(seed=34):
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(args.seed)

def train():
    if os.path.exists(f"./saved_model/{args.alpha_1}_{args.alpha_2}/{args.relation_id}.pt"):
        print('relation OK')
        exit(0)
    model = \
        PTuneForLAMA(device='cuda:0', pretrained_model =pretrained_model,
                         tokenizer=tokenizer, template=(3,3,3))
    checkpoint = torch.load(prompt_model_path)
    model.prompt_encoder.load_state_dict(checkpoint['embedding'])
    optima_params = [{'params': model.prompt_encoder.parameters()},
                     {'params': model.gate.parameters()}]
    optimizer = torch.optim.Adam(optima_params, lr=1e-5, weight_decay=0.0005)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)
    total_loss = 0
    total_mlm_loss = 0
    total_entropy_loss = 0
    total_cl_loss = 0
    hit = 0
    best_dev = 0.0
    best_test = 0.0
    early_stop = 20
    for epoch_idx in range(50):
        for batch_idx, (subjs, objs, predicate_id, template) in tqdm(enumerate(train_loader)):
            model.train()
            mlm_loss, entropy_loss, cl_loss, batch_hit = model(subjs, objs, predicate_id, template)
            loss = mlm_loss + args.alpha_1 * entropy_loss + args.alpha_2 * cl_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            total_mlm_loss += mlm_loss.item()
            total_cl_loss += cl_loss.item()
            total_entropy_loss += entropy_loss.item()
            hit += batch_hit
        print("epoch: {}, loss: {}, mlm_loss: {}, entropy_loss: {}, cl_loss: {}, hit@1: {} ".format(epoch_idx,
                                                      total_loss / (batch_idx + 1),
                                                      total_mlm_loss / (batch_idx + 1),
                                                      total_entropy_loss / (batch_idx + 1),
                                                      total_cl_loss / (batch_idx + 1),
                                                      float(hit)/ len(train_set)))

        total_loss = 0
        total_mlm_loss = 0
        total_cl_loss = 0
        hit = 0
        my_lr_scheduler.step()
        # testing
        model.eval()
        dev_hit, test_hit = 0, 0
        for batch_idx, (subjs, objs, predicate_id, template) in tqdm(enumerate(dev_loader)):
            mlm_loss, entropy_loss, cl_loss, batch_hit = model(subjs, objs, predicate_id, template)
            dev_hit += batch_hit
        if float(dev_hit) / len(dev_set) >= best_dev:
            best_dev = float(dev_hit) / len(dev_set)
            print("epoch: {}, dev_hit@1: {}".format(epoch_idx, float(dev_hit) / len(dev_set)))
            best_ckpt = {'embedding': model.prompt_encoder.state_dict(),
                         'gate': model.gate.state_dict(),
                         'dev_hit@1': best_dev}
            if not os.path.exists(f'./saved_model/{args.alpha_1}_{args.alpha_2}/'):
                os.makedirs(f'./saved_model/{args.alpha_1}_{args.alpha_2}/')
            torch.save(best_ckpt, f'./saved_model/{args.alpha_1}_{args.alpha_2}/{args.relation_id}.pt')

            for batch_idx, (subjs, objs, predicate_id, template) in tqdm(enumerate(test_loader)):
                mlm_loss, entropy_loss, cl_loss, batch_hit = model(subjs, objs, predicate_id, template)
                test_hit += batch_hit
            if float(test_hit) / len(test_set) >= best_test:
                best_test = float(test_hit) / len(test_set)
                print("epoch: {}, test_hit@1: {}".format(epoch_idx, float(test_hit) / len(test_set)))
            early_stop = 20
        else:
            early_stop = early_stop - 1
        if early_stop == 0:
            break
    return best_dev, best_test
best_dev, best_test = train()
print(str(args.relation_id), best_dev, best_test)
