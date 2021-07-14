# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : EvoGAN_G.py
# @Time         : Created at 2019-07-09
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from models.generator import LSTMGenerator
from models.relational_rnn_general import RelationalMemory
from utils.text_process import build_word2vec_embedding_matrix


class KeyGAN_G(LSTMGenerator):
    def __init__(self, mem_slots, num_heads, head_size, embedding_dim, hidden_dim, vocab_size, max_seq_len, max_key_len, padding_idx,
                 dataset,gpu=False,load_model=None):
        super(KeyGAN_G, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
        self.max_key_len = max_key_len
        self.name = 'keygan'
        self.gpu=gpu
        if not load_model:
            self.temperature = nn.Parameter(torch.Tensor([1.0]), requires_grad=False)  # init value is 1.0

        #self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embeddings.from_pretrained(build_word2vec_embedding_matrix(dataset,embedding_dim))
        if cfg.model_type == 'LSTM':
            # LSTM
            self.hidden_dim = hidden_dim
            #encoder lstm
            self.encoderlstm = nn.LSTM(embedding_dim, self.hidden_dim, batch_first=True)
            #decoder lstm
            self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, batch_first=True)
            self.lstm2out = nn.Linear(self.hidden_dim, vocab_size)
        else:
            # RMC
            self.hidden_dim = mem_slots * num_heads * head_size
            #encoder RMC
            self.encoderlstm = RelationalMemory(mem_slots=mem_slots, head_size=head_size, input_size=embedding_dim,
                                        num_heads=num_heads, return_all_outputs=True)
            #decoder RMC
            self.lstm = RelationalMemory(mem_slots=mem_slots, head_size=head_size, input_size=embedding_dim,
                                        num_heads=num_heads, return_all_outputs=True)
            self.lstm2out = nn.Linear(self.hidden_dim, vocab_size)

        self.init_params()

        if load_model:
            print("load_model:",load_model)
            self.load_state_dict(torch.load(load_model,map_location='cpu'),strict=False)


    def init_hidden(self, batch_size=cfg.batch_size):
        if cfg.model_type == 'LSTM':
            h = torch.zeros(1, batch_size, self.hidden_dim)
            c = torch.zeros(1, batch_size, self.hidden_dim)

            if self.gpu:
                return h.cuda(), c.cuda()
            else:
                return h, c
        else:
            """init RMC memory"""
            memory = self.lstm.initial_state(batch_size)
            memory = self.lstm.repackage_hidden(memory)  # detch memory at first
            return memory.cuda() if self.gpu else memory


    def step_decoder(self, inp, hidden,CUDA=True, ulist=None):
        """
        RelGAN step forward
        :param inp: [batch_size]
        :param hidden: memory size
        :return: pred, hidden, next_token, next_token_onehot, next_o
            - pred: batch_size * vocab_size, use for adversarial training backward
            - hidden: next hidden
            - next_token: [batch_size], next sentence token
            - next_token_onehot: batch_size * vocab_size, not used yet
            - next_o: batch_size * vocab_size, not used yet
        """
        emb = self.embeddings(inp).unsqueeze(1)
        out, hidden = self.lstm(emb, hidden)
        #gumbel_t=self.lstm2out(out.squeeze(1))#こうすると全部同じ文章になる
        gumbel_t = self.add_gumbel(self.lstm2out(out.squeeze(1)),gpu=CUDA,ulist=ulist)
        next_token = torch.argmax(gumbel_t, dim=1).detach()
        # next_token_onehot = F.one_hot(next_token, cfg.vocab_size).float()  # not used yet
        next_token_onehot = None

        pred = F.softmax(gumbel_t * self.temperature, dim=-1)  # batch_size * vocab_size
        # next_o = torch.sum(next_token_onehot * pred, dim=1)  # not used yet
        next_o = None

        return pred, hidden, next_token, next_token_onehot, next_o

    def step_encoder(self, inp, hidden,CUDA=True, ulist=None):
        """
        Encoder step forward
        :param inp: [batch_size]
        :param hidden: memory size
        :return: out,hidden
            - out: batch_size * vocab_size, use for adversarial training backward
            - hidden: next hidden
        """
        emb = self.embeddings(inp).unsqueeze(1)
        out, hidden = self.encoderlstm(emb, hidden)

        return out, hidden

    def forward(self, inp, enchidden,key=None,need_hidden=False):
        """
        Embeds input and applies LSTM
        :param inp: batch_size * seq_len
        :param key: batch_size * key_len
        :param hidden: (h, c)
        :param need_hidden: if return hidden, use for sampling
        """

        
        #encoder
        if key is not None:
            keyword_emb = self.embeddings(key) # batch_size * max_key_len * embedding_dim
            if len(key.size()) == 1:
                keyword_emb = keyword_emb.unsqueeze(1)  # batch_size * 1 * embedding_dim
            encout,enchidden = self.encoderlstm(keyword_emb, enchidden) 
            #enchidden=enchidden[0] #????

        #decoder
        emb = self.embeddings(inp)
        if len(inp.size()) == 1:
            emb = emb.unsqueeze(1)  # batch_size * 1 * embedding_dim
        #encoderの隠れ層を用いてforward
        out, hidden = self.lstm(emb, enchidden)  # out: batch_size * seq_len * hidden_dim
        out = out.contiguous().view(-1, self.hidden_dim)  # out: (batch_size * len) * hidden_dim
        out = self.lstm2out(out)  # (batch_size * seq_len) * vocab_size
        # out = self.temperature * out  # temperature
        pred = self.softmax(out)

        if need_hidden:
            return pred, hidden
        else:
            return pred

    def sample_from_keyword(self, keywords, num_samples, batch_size, one_hot=False, start_letter=cfg.start_letter, CUDA=True, ulist=None):
        """
        Sample from RelGAN Generator
        - keyword: [int] * keywordnum of keyword
        - one_hot: if return pred of RelGAN, used for adversarial training
        :return:
            - all_preds: batch_size * seq_len * vocab_size, only use for a batch
            - samples: all samples
        """
        global all_preds
        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        samples = torch.zeros(num_batch * batch_size, self.max_seq_len).long()
        if one_hot:
            all_preds = torch.zeros(batch_size, self.max_seq_len, self.vocab_size)
            if self.gpu:
                all_preds = all_preds.cuda()

        # encoder
        encoder_hidden=self.init_hidden(batch_size)
        for keyword in keywords:
            inp = torch.LongTensor([keyword] * batch_size)
            if self.gpu:
                inp = inp.cuda()
            encoder_out,encoder_hidden=self.step_encoder(inp,encoder_hidden)
        # decoder
        for b in range(num_batch):
            hidden = encoder_hidden
            inp = torch.LongTensor([start_letter] * batch_size)
            if self.gpu:
                inp = inp.cuda()

            for i in range(self.max_seq_len):
                pred, hidden, next_token, _, _ = self.step_decoder(inp, hidden,CUDA=CUDA,ulist=ulist)
                samples[b * batch_size:(b + 1) * batch_size, i] = next_token
                if one_hot:
                    all_preds[:, i] = pred
                inp = next_token
        samples = samples[:num_samples]  # num_samples * seq_len

        if one_hot:
            return all_preds  # batch_size * seq_len * vocab_size
        return samples

    @staticmethod
    def add_gumbel(o_t, eps=1e-10, gpu=cfg.CUDA, ulist=None):
        """Add o_t by a vector sampled from Gumbel(0,1)"""
        u = torch.zeros(o_t.size())
        if gpu:
            u = u.cuda()

        u.uniform_(0, 1)
        if ulist:
            ulist[0]=u
        g_t = -torch.log(-torch.log(u + eps) + eps)
        gumbel_t = o_t + g_t
        return gumbel_t
