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

for relation_id in relation_ids[25:]:
    os.system("CUDA_VISIBLE_DEVICES=6 python train.py --relation_id={} --seed={} --alpha_1={} --alpha_2={}".
              format(relation_id, 34, 0.2, 0.1))

