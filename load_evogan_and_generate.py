import config as cfg
import torch
from models.EvoGAN_D import EvoGAN_D
from models.EvoGAN_G import EvoGAN_G

gen_path = "./save/20210626/haiku_wakati/evogan_vanilla_dt-Ra_lt-ragan_mt-ra+rs_et-Ra_sl15_temp100_lfd0.001_T0626_0653_25/models/gen_ADV_00000.pt"
dis_path = "./save/20210626/haiku_wakati/evogan_vanilla_dt-Ra_lt-ragan_mt-ra+rs_et-Ra_sl15_temp100_lfd0.001_T0626_0653_25/models/dis_ADV_00000.pt"

import argparse
from utils.text_process import load_test_dict, text_process
from utils.text_process import  load_dict,tensor_to_tokens


cfg.if_test = int(False)
cfg.run_model = 'evogan'
cfg.k_label = 2
cfg.CUDA = int(False)
cfg.ora_pretrain = int(True)
cfg.gen_pretrain = int(True)
cfg.dis_pretrain = int(False)
cfg.LE_train_epoch = 150
cfg.clas_pre_epoch = 5
cfg.ADV_train_epoch = 2000
cfg.tips = '{} experiments'

# ===Oracle or Real===
cfg.if_real_data = int(True) #param
cfg.dataset = 'haiku_wakati' #param
cfg.vocab_size = 0

# ===CatGAN Param===
cfg.n_parent = 1
cfg.loss_type = 'ragan'
cfg.mu_type = 'ragan rsgan'
cfg.eval_type = 'Ra'
cfg.temp_adpt = 'exp'
cfg.temperature = 100 #param
cfg.d_out_mean = int(True)
cfg.lambda_fq = 1.0
cfg.lambda_fd = 0.001
cfg.eval_b_num = 8

# === Basic Param ===
cfg.data_shuffle = int(False)
cfg.model_type = 'vanilla'
cfg.gen_init = 'truncated_normal'
cfg.dis_init = 'uniform'
cfg.samples_num = 10000
cfg.batch_size = 64
cfg.max_seq_len = 20
cfg.gen_lr = 0.01
cfg.gen_adv_lr = 1e-4
cfg.dis_lr = 1e-4
cfg.pre_log_step = 10
cfg.adv_log_step = 20

# ===Generator===
cfg.ADV_g_step = 1
cfg.gen_embed_dim = 32
cfg.gen_hidden_dim = 32
cfg.mem_slots = 1
cfg.num_heads = 2
cfg.head_size = 256 #param

# ===Discriminator===
cfg.ADV_d_step = 3
cfg.dis_embed_dim = 64
cfg.dis_hidden_dim = 64
cfg.num_rep = 64

# ===Metrics===
cfg.use_nll_oracle = int(True)
cfg.use_nll_gen = int(True)
cfg.use_nll_div = int(True)
cfg.use_bleu = int(True)
cfg.use_self_bleu = int(True)
cfg.use_clas_acc = int(True)
cfg.use_ppl = int(False)

# MAIN
if __name__ == '__main__':
    # Hyper Parameters
    if cfg.if_real_data:
        cfg.max_seq_len, cfg.vocab_size = text_process('dataset/' + cfg.dataset + '.txt')
        cfg.extend_vocab_size = len(load_test_dict(cfg.dataset)[0])  # init classifier vocab_size

    gen_model=EvoGAN_G(cfg.mem_slots, cfg.num_heads, cfg.head_size, cfg.gen_embed_dim, cfg.gen_hidden_dim,
                            cfg.vocab_size, cfg.max_seq_len, cfg.padding_idx,gpu=False,load_model=gen_path)
    word2idx_dict, idx2word_dict = load_dict(cfg.dataset)
    print( tensor_to_tokens(gen_model.sample(64,64,CUDA=False), idx2word_dict) )