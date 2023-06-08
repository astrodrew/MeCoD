import json
def token_wrapper( token):
    return token
def get_vocab_by_strategy():
    return json.load(open('../../data/LAMA/29k-vocab.json'))['bert-base-cased']