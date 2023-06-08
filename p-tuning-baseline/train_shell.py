import os
import json

def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

relation_data = load_file('../../data/LAMA/relations.jsonl')
relation_ids = [line['relation'] for line in relation_data]

for relation_id in relation_ids:
    os.system("CUDA_VISIBLE_DEVICES=0 python train.py --relation_id={} --seed={}".
              format(relation_id, 34))