# -*- coding: utf-8 -*-
# @Author       : William
# @Project      : TextGAN-william
# @FileName     : data_loader.py
# @Time         : Created at 2019-05-31
# @Blog         : http://zhiweil.ml/
# @Description  : 
# Copyrights (C) 2018. All Rights Reserved.

import random
from torch.utils.data import Dataset, DataLoader
import config as cfg
from utils.text_process import *


class GANDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class KeyGenDataIter:
    def __init__(self, samples, keywords=None,if_test_data=False, shuffle=None):
        self.batch_size = cfg.batch_size
        self.max_seq_len = cfg.max_seq_len
        self.max_key_len = cfg.max_key_len
        self.start_letter = cfg.start_letter
        self.shuffle = cfg.data_shuffle if not shuffle else shuffle
        if cfg.if_real_data:
            self.word2idx_dict, self.idx2word_dict = load_dict(cfg.dataset)
        if if_test_data:  # used for the classifier
            self.word2idx_dict, self.idx2word_dict = load_test_dict(cfg.dataset)

        self.loader = DataLoader(
            dataset=GANDataset(self.__read_data__(samples,keywords)),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True)

        self.input = self._all_data_('input')
        self.keyword = self._all_data_('keyword')
        self.target = self._all_data_('target')

    def __read_data__(self, samples,keywords):
        """
        input: same as target, but start with start_letter.
        """
        # global all_data
        if isinstance(samples, torch.Tensor):  # Tensor
            inp, keyword,target = self.prepare(samples,keywords)
            all_data = [{'input': i, 'keyword':k ,'target': t} for (i,k,t) in zip(inp,keyword,target)]
        else:#if isinstance(samples, str):  # filename
            inp, keyword, target = self.load_data(samples,keywords)
            all_data = [{'input': i, 'keyword':k ,'target': t} for (i,k,t) in zip(inp,keyword,target)]
        return all_data

    def random_batch(self):
        """Randomly choose a batch from loader, please note that the data should not be shuffled."""
        idx = random.randint(0, len(self.loader) - 1)
        return list(self.loader)[idx]

    def _all_data_(self, col):
        return torch.cat([data[col].unsqueeze(0) for data in self.loader.dataset.data], 0)

    @staticmethod
    def prepare(samples,keywords,gpu=False):
        """Add start_letter to samples as inp, target same as samples"""
        inp = torch.zeros(samples.size()).long()
        target = samples
        inp[:, 0] = cfg.start_letter
        inp[:, 1:cfg.max_seq_len] = target[:, 0:cfg.max_seq_len - 1]
        if keywords is None:
            keyword = torch.zeros(samples.size()).long() #これでいいのか？
            keyword = keyword[:,:cfg.max_key_len]
        else:
            keyword=keywords[:, :cfg.max_key_len]

        if gpu:
            return inp.cuda(),keyword.cuda(),target.cuda()
        return inp, keyword,target

    def load_data(self, filename,keywordfilename):
        """Load real data from local file"""
        self.tokens = get_tokenlized(filename)
        if keywordfilename and (not os.path.exists(keywordfilename)):
            extract_keyword(cfg.dataset)
        if keywordfilename:
            self.keywords = get_tokenlized(keywordfilename)
            keywords_index = tokens_to_tensor(self.keywords, self.word2idx_dict,max_len=cfg.max_key_len)
        else:
            keywords_index=None
        samples_index = tokens_to_tensor(self.tokens, self.word2idx_dict)
        return self.prepare(samples_index,keywords_index)


class DisDataIter:
    def __init__(self, pos_samples, neg_samples, shuffle=None):
        self.batch_size = cfg.batch_size
        self.max_seq_len = cfg.max_seq_len
        self.start_letter = cfg.start_letter
        self.shuffle = cfg.data_shuffle if not shuffle else shuffle

        self.loader = DataLoader(
            dataset=GANDataset(self.__read_data__(pos_samples, neg_samples)),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True)

    def __read_data__(self, pos_samples, neg_samples):
        """
        input: same as target, but start with start_letter.
        """
        inp, target = self.prepare(pos_samples, neg_samples)
        all_data = [{'input': i, 'target': t} for (i, t) in zip(inp, target)]
        return all_data

    def random_batch(self):
        idx = random.randint(0, len(self.loader) - 1)
        return list(self.loader)[idx]

    def prepare(self, pos_samples, neg_samples, gpu=False):
        """Build inp and target"""
        inp = torch.cat((pos_samples, neg_samples), dim=0).long().detach()  # !!!need .detach()
        target = torch.ones(inp.size(0)).long()
        target[pos_samples.size(0):] = 0

        # shuffle
        perm = torch.randperm(inp.size(0))
        inp = inp[perm]
        target = target[perm]

        if gpu:
            return inp.cuda(), target.cuda()
        return inp, target
