import config as cfg
import torch
from models.KeyGAN_D import KeyGAN_D
from models.KeyGAN_G import KeyGAN_G
from utils.key_data_loader import KeyGenDataIter
from metrics.bleu import BLEU
from metrics.clas_acc import ACC
from metrics.nll import NLL
from metrics.ppl import PPL
import gensim

#全部
#gen_path = "./save/20210628/haiku_wakati/keygan_vanilla_dt-Ra_lt-ragan_mt-ra+rs_et-Ra_sl15_temp100_lfd0.001_T0628_0725_17/models/gen_ADV_01999.pt"
#最初の単語
#gen_path = "./save/20210629/haiku_wakati/keygan_vanilla_dt-Ra_lt-ragan_mt-ra+rs_et-Ra_sl15_temp100_lfd0.001_T0629_0148_03/models/gen_ADV_01999.pt"
# mr15/最初の単語/word2vec
#gen_path = "save/20210701/mr15/keygan_vanilla_dt-Ra_lt-ragan_mt-ra+rs_et-Ra_sl19_temp100_lfd0.001_T0701_0800_28/models/gen_ADV_01999.pt"
#mr15/word2vec/名詞と形容詞
#gen_path = "save/20210714/mr15/keygan_vanilla_dt-Ra_lt-ragan_mt-ra+rs_et-Ra_sl19_temp100_lfd0.001_T0714_1222_24/models/gen_ADV_01999.pt"
#emnlp_mini/word2vec/名詞と形容詞
gen_path = "./save/20210721/emnlp_news_mini/keygan_vanilla_dt-Ra_lt-ragan_mt-ra+rs_et-Ra_sl51_temp100_lfd0.001_T0721_1234_23/models/gen_ADV_01440.pt"
#image_coco/word2vec/名詞と形容詞
#gen_path = "./save/20210726/image_coco/keygan_vanilla_dt-Ra_lt-ragan_mt-ra+rs_et-Ra_sl37_temp100_lfd0.001_T0726_0926_02/models/gen_ADV_01080.pt"
#dis_path = "./save/20210626/haiku_wakati/evogan_vanilla_dt-Ra_lt-ragan_mt-ra+rs_et-Ra_sl15_temp100_lfd0.001_T0626_0653_25/models/dis_ADV_01999.pt"
#mr15/word2vec/GPによるキーワード
#gen_path="./save/20220104/mr15/keygan_vanilla_dt-Ra_lt-ragan_mt-ra+rs_et-Ra_sl19_temp100_lfd0.001_T0104_0823_28/models/gen_ADV_01999.pt"
#emnlp_news/word2vec/GPによるキーワード
#gen_path="./save/20220105/emnlp_news_mini/keygan_vanilla_dt-Ra_lt-ragan_mt-ra+rs_et-Ra_sl51_temp100_lfd0.001_T0105_0243_46/models/gen_ADV_01640.pt"
#image_coco/word2vec/GPによるキーワード
#gen_path="./save/20220107/image_coco/keygan_vanilla_dt-Ra_lt-ragan_mt-ra+rs_et-Ra_sl37_temp100_lfd0.001_T0107_1119_21/models/gen_ADV_01999.pt"
import argparse
from utils.text_process import load_test_dict, text_process
from utils.text_process import   write_tokens,load_dict,tensor_to_tokens,tokens_to_tensor


cfg.if_test = int(False)
cfg.run_model = 'keygan'
cfg.k_label = 2
cfg.CUDA = int(True)
cfg.ora_pretrain = int(True)
cfg.gen_pretrain = int(True)
cfg.dis_pretrain = int(False)
cfg.LE_train_epoch = 150
cfg.clas_pre_epoch = 5
cfg.ADV_train_epoch = 2000
cfg.tips = '{} experiments'

# ===Oracle or Real===
cfg.if_real_data = int(True) #param
cfg.dataset = 'emnlp_news_mini' #param change here!
cfg.vocab_size = 0

# ===CatGAN Param===
cfg.n_parent = 1
cfg.loss_type = 'ragan'
cfg.mu_type = 'ragan rsgan'
cfg.eval_type = 'Ra'
cfg.temp_adpt = 'exp'
cfg.temperature = 100 #param change here!
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
cfg.max_seq_len = 51 #param change here! oracle: 20, coco: 37, emnlp: 51, amazon_app_book: 40
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
cfg.head_size = 256 #param change here!
#dataset = ['oracle', 'mr15', 'amazon_app_book', 'oracle', 'image_coco', 'emnlp_news', 'haiku_wakati','aozora_moriougai','mr15', "emnlp_news_mini"]
#head_size = [512, 512, 512, 256, 256, 256, 256, 512, 256, 256]

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

    gen_model=KeyGAN_G(cfg.mem_slots, cfg.num_heads, cfg.head_size, cfg.gen_embed_dim, cfg.gen_hidden_dim,
                            cfg.vocab_size, cfg.max_seq_len,cfg.max_key_len, cfg.padding_idx,cfg.dataset,gpu=True,load_model=gen_path)
    word2idx_dict, idx2word_dict = load_dict(cfg.dataset)

    keywords= [1260]
    # haiku:桜4571 さみしい1204　冬枯れ21767 
    # mr15:awesome3517 suspenseful 3244 it5600 boring 3894 bad1037 good;5873 movie:1443
    # enmlp_news_mini bad:1778 good:1260 america:3793 enjoy:3165 water:781 report:756 better:877
    # image_coco  man:1682 toilet:3102
    #generate_size=cfg.batch_size
    #samples=gen_model.sample_from_keyword_with_ES(keywords,generate_size,cfg.batch_size,idx2word_dict,CUDA=True)
    #生成
    #samples=gen_model.sample_from_keyword(keywords,generate_size,cfg.batch_size,generate_size,CUDA=True)
    #評価
    train_data=KeyGenDataIter(cfg.train_data, keywords=cfg.keyword_data)
    test_data=KeyGenDataIter(cfg.test_data,if_test_data=True)
    bleu = BLEU('BLEU', gram=[2, 3, 4, 5], if_use=cfg.use_bleu)
    nll_gen = NLL('NLL_gen', if_use=cfg.use_nll_gen, gpu=cfg.CUDA)
    nll_div = NLL('NLL_div', if_use=cfg.use_nll_div, gpu=cfg.CUDA)
    self_bleu = BLEU('Self-BLEU', gram=[2, 3, 4], if_use=cfg.use_self_bleu)
    clas_acc = ACC(if_use=cfg.use_clas_acc)
    ppl = PPL(train_data, test_data, n_gram=5, if_use=cfg.use_ppl)
    all_metrics = [bleu, nll_gen, nll_div, self_bleu, ppl]
    eval_samples = gen_model.sample_from_keyword(keywords,cfg.samples_num, 4 * cfg.batch_size)
    gen_data = KeyGenDataIter(eval_samples)
    gen_tokens = tensor_to_tokens(eval_samples, idx2word_dict)
    gen_tokens_s = tensor_to_tokens(gen_model.sample_from_keyword(keywords,200, 200),idx2word_dict)
    # Reset metrics
    bleu.reset(test_text=gen_tokens, real_text=test_data.tokens)
    nll_gen.reset(gen_model, train_data.loader)
    nll_div.reset(gen_model, gen_data.loader)
    self_bleu.reset(test_text=gen_tokens_s, real_text=gen_tokens)
    ppl.reset(gen_tokens)
    print( ', '.join(['%s = %s' % (metric.get_name(), metric.get_score()) for metric in all_metrics]) )
    tokens=tensor_to_tokens(eval_samples, idx2word_dict) 
    keyword_tokens=tensor_to_tokens(torch.LongTensor([keywords]), idx2word_dict) 
    write_tokens("keywords.txt",keyword_tokens)
    write_tokens("output_enmini_good.txt",tokens)