import torch
from torch.nn.utils.rnn import pad_sequence
import sys
sys.path.append('../model')
from model.prompt_encoder import PromptEncoder
sys.path.append('../data_util')
from data_util.vocab import get_vocab_by_strategy,token_wrapper
import numpy as np
import torch.nn as nn

def get_embedding_layer(model):
    embeddings = model.get_input_embeddings()
    return embeddings

class PTuneForLAMA(torch.nn.Module):

    def __init__(self, device, pretrained_model, tokenizer, template):
        super().__init__()
        self.device = device

        # load tokenizer
        self.tokenizer = tokenizer

        # load pre-trained model
        self.model = pretrained_model
        self.model = self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.embeddings = get_embedding_layer(self.model)

        # set allowed vocab set
        self.vocab = self.tokenizer.get_vocab()
        self.allowed_vocab_ids = set(self.vocab[k] for k in get_vocab_by_strategy())

        self.template = template

        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[PROMPT]']})
        self.pseudo_token_id = self.tokenizer.get_vocab()['[PROMPT]']
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id

        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, self.tokenizer, self.device)
        self.prompt_encoder = self.prompt_encoder.to(self.device)

        self.dense = self.model.cls.predictions.transform
        self.output_embedding = self.model.get_output_embeddings().weight
        self.bias_vec = self.model.cls.predictions.decoder.bias

        self.neg_sample_num = 64
        self.cre = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)
        self.gate = nn.Linear(self.hidden_size, 2).to(self.device)

    def embed_input(self, queries):
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding)
        blocked_indices = (queries == self.pseudo_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
        replace_embeds = self.prompt_encoder()
        for bidx in range(bz):
            for i in range(self.prompt_encoder.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        return raw_embeds


    def get_query(self, x_h, prompt_tokens, x_t=None):
     
        # For P-tuning

        return [[self.tokenizer.cls_token_id]  # [CLS]
                + prompt_tokens * self.template[0]
                + [self.tokenizer.mask_token_id]  # head entity
                + prompt_tokens * self.template[1]
                + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + x_h))  # [MASK] (tail entity)
                + (prompt_tokens * self.template[2] if self.template[
                                                           2] > 0 else self.tokenizer.convert_tokens_to_ids(['.']))
                + [self.tokenizer.sep_token_id]
                ]

    def get_counterfactual_query(self, x_h, prompt_tokens):
        return [[self.tokenizer.cls_token_id]  # [CLS]
                + prompt_tokens * self.template[0]
                + [self.tokenizer.mask_token_id]  # head entity
                + prompt_tokens * self.template[1]
                + [self.tokenizer.mask_token_id] * len(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + x_h)))# [MASK] (tail entity)
                + (prompt_tokens * self.template[2] if self.template[
                                                           2] > 0 else self.tokenizer.convert_tokens_to_ids(['.']))
                + [self.tokenizer.sep_token_id]
                ]

    def forward(self, x_hs, x_ts, predicate_id, template, return_candidates=False):
        bz = len(x_hs)
        # construct query ids
        prompt_tokens = [self.pseudo_token_id]
        x_ts = [token_wrapper(x_t) for x_t in x_ts]
        queries = [torch.LongTensor(self.get_query(x_hs[i], prompt_tokens)).squeeze(0) for i in range(bz)]
        queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)

        counter_queries = [torch.LongTensor(self.get_counterfactual_query(x_hs[i], prompt_tokens)).squeeze(0) for i in
                           range(bz)]
        counter_queries = pad_sequence(counter_queries, True, padding_value=self.pad_token_id).long().to(self.device)

        # construct label ids
        label_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(x_ts)).reshape(
            (bz, -1)).to(self.device)
        attention_mask = queries != self.pad_token_id
        label_mask = (queries == self.tokenizer.mask_token_id).nonzero().reshape(bz, -1)[:, 1].unsqueeze(
            1).to(self.device)  # bz * 1
        labels = torch.empty_like(queries).fill_(-100).long().to(self.device)  # bz * seq_len
        labels = labels.scatter_(1, label_mask, label_ids)
        # get embedded input
        inputs_embeds = self.embed_input(queries)
        counter_inputs_embeds = self.embed_input(counter_queries)
        counter_query_output = self.model(inputs_embeds=counter_inputs_embeds.to(self.device),
                                          attention_mask=attention_mask.to(self.device).bool(),
                                          output_hidden_states=True,
                                          return_dict=True)
        counter_logits = counter_query_output['logits']
        query_output = self.model(inputs_embeds=inputs_embeds.to(self.device),
                            attention_mask=attention_mask.to(self.device).bool(),
                            labels=labels.to(self.device),
                            output_hidden_states=True,
                            return_dict=True)
        mlm_loss, logits = query_output['loss'], query_output['logits']
        hidden_state = query_output['hidden_states'][-1]
        hit1 = 0
        loss_entropy = 0
        loss_cl = 0
        pred_ids = torch.argsort(logits, dim=2, descending=True)
        counter_pred_logits, counter_pred_ids = torch.sort(counter_logits, dim=2, descending=True)

        for i in range(bz):
            mask_hidden_state = hidden_state[i, label_mask[i, 0], :].unsqueeze(0)
            mask_proj_emb = self.dense(mask_hidden_state)

            counter_candidates = counter_pred_ids[i, label_mask[i, 0], :300]
            counter_candidates_embeddings = \
                [self.output_embedding[idx].detach().cpu().numpy() for idx in counter_candidates]

            counter_candidates_embeddings = torch.tensor(np.array(counter_candidates_embeddings)).to(self.device)
            entropy_mask = torch.zeros(counter_pred_ids[i, label_mask[i, 0]].shape).to(self.device)
            gate = torch.nn.functional.gumbel_softmax(self.gate(counter_candidates_embeddings), tau=1, hard=True)
            neg_object_ids = []
            for idx in range(len(gate)):
                if gate[idx][0] == 1:
                    neg_object_ids.append(int(counter_candidates[idx]))
                    entropy_mask[counter_candidates[idx]] = 1
            log_probs = torch.nn.functional.log_softmax(counter_logits[i, label_mask[i, 0]], dim=-1)
            probs = self.softmax(counter_logits[i, label_mask[i, 0]])

            loss_entropy += 1 * torch.sum(probs.mul(log_probs).mul(entropy_mask))
            if int(label_ids[i]) in neg_object_ids:
                neg_object_ids.remove(label_ids[i])
            neg_object_embeddings = [self.output_embedding[idx].detach().cpu().numpy()
                                     for idx in neg_object_ids]
            neg_object_proj_emb = torch.tensor(np.array(neg_object_embeddings)).cuda()

            pos_object_id = self.tokenizer.convert_tokens_to_ids(x_ts[i])
            pos_object_embeddings = self.output_embedding[pos_object_id].unsqueeze(0)

            pos_bias = self.bias_vec[pos_object_id]
            neg_bias = [self.bias_vec[neg_object_id].item() for neg_object_id in neg_object_ids]

            neg_score = neg_object_proj_emb.mm(mask_proj_emb.T).squeeze(1) + torch.tensor(neg_bias).to(self.device)
            neg_score = torch.sum(torch.exp(neg_score))
            pos_score = pos_object_embeddings.mm(mask_proj_emb.T).squeeze(1) + pos_bias
            postive = torch.exp(pos_score)
            log_val = torch.log((neg_score + postive) / postive)
            loss_cl += log_val
            pred_seq = pred_ids[i, label_mask[i, 0]].tolist()
            for pred in pred_seq:
                if pred in self.allowed_vocab_ids:
                    break
            if pred == label_ids[i, 0]:
                hit1 += 1
        loss_entropy = loss_entropy / bz
        loss_cl = loss_cl / bz
        if return_candidates:
            return mlm_loss, loss_entropy, loss_cl, hit1
        return mlm_loss, loss_entropy, loss_cl, hit1


