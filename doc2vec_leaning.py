from gensim.models import word2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import logging
import config as cfg
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="file to make doc2vec",type=str,default="mr15")
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = []
    with open('./dataset/'+args.file+'.txt','r') as f:
        word_list=f.readline().split(' ')
        sentences.append(word_list)
    documents=[TaggedDocument(doc, [i]) for i, doc in enumerate(sentences)]
    model = Doc2Vec(documents, vector_size=cfg.gen_embed_dim, window=5, min_count=1, workers=4, epochs=100000)
    model.save("word2vec_models/doc2vec_"+args.file+".model")