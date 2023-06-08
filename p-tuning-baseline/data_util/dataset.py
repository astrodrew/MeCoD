import os
from torch.utils.data import Dataset
import sys
sys.path.append('../data_util')
from data_util.vocab import get_vocab_by_strategy,token_wrapper
import json

def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

def load_json_file(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data

def load_files(basedir, relations, jsonl=True):
    data = []
    for relation in relations:
        try:
            if jsonl:
                data.extend(load_file(os.path.join(basedir, relation)))
            else:
                data.extend(load_json_file(os.path.join(basedir, relation)))
        except FileNotFoundError:
            print(f"Cannot load relation: {relation}")
    return data

def load_relations_templates(path):
    ext = os.path.splitext(path)[1]
    if ext == ".jsonl":
        return dict((d['relation'], [d['template']]) for d in load_file(path))
    elif ext == ".json":
        rel_map = {rel: [t['template'] for t in templates] for rel, templates in load_json_file(path).items()}
        return rel_map


class LAMADataset(Dataset):
    def __init__(self, dataset_type,data_template_path,data, tokenizer, args):
        super().__init__()
        self.args = args
        self.data = list()
        self.dataset_type = dataset_type
        self.relations = load_relations_templates(data_template_path)
        self.x_hs, self.x_ts = [], []

        vocab = get_vocab_by_strategy()
        for d in data:
            if token_wrapper(d['obj_label']) not in vocab:
                continue
            for template in self.relations[d['predicate_id']]:
                self.data.append((
                    d['sub_label'],
                    d['obj_label'],
                    d['predicate_id'],
                    template,
                ))
                self.x_ts.append(d['obj_label'])
                self.x_hs.append(d['sub_label'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


