import json
import os
import numpy as np
from torch.distributions import Categorical, kl
import torch
import torch.nn as nn
softmax = nn.Softmax(dim=-1)
path = f'./result/'
file_name_list = os.listdir(path)
p_1_list, p_5_list, p_10_list, mrr_list = [],[],[],[]
entropy_p_list=[]
valid_relation_id = []
for file_name in file_name_list:
    if 'test' not in file_name:
        continue
    data = json.load(open(path+file_name,'r',encoding='utf-8'))
    p_1 = 0.0
    p_5 = 0.0
    mrr = 0.0
    for line in data:
        predict_words = [ele[0] for ele in line['predicted_words']]
        if line['obj'] == predict_words[0]:
            p_1 += 1
        if line['obj'] in predict_words[:5]:
            p_5 += 1
        if line['obj'] in predict_words[:100]:
            mrr += 1.0 / (predict_words[:100].index(line['obj']) + 1)
    counter_predict_logits = [ele[1] for ele in data[0]['subject_mask_predicted_words']][:10]
    entropy = Categorical(softmax(torch.tensor(counter_predict_logits))).entropy().item()
    entropy_p_list.append(entropy)
    p_1 = p_1/len(data)
    p_5 = p_5/len(data)
    mrr = mrr/len(data)
    p_1_list.append(p_1)
    p_5_list.append(p_5)
    mrr_list.append(mrr)
    valid_relation_id.append(file_name.replace('test_','').replace('.json',''))
    print(file_name, p_1, mrr)

print(len(p_1_list))
print(np.average(p_1_list))
print(np.average(p_5_list))
print(np.average(mrr_list))
print(np.average(entropy_p_list))




