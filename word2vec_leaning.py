from gensim.models import word2vec
import logging
import config as cfg
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="file to make word2vec",type=str,default="mr15")
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus('./dataset/'+args.file+'.txt')

    #https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec
    model = word2vec.Word2Vec(sentences, vector_size=cfg.gen_embed_dim,min_count=1, epochs=1000)
    model.wv.save_word2vec_format('./word2vec_models/'+args.file+'.vec.pt', binary=True)