from gensim.models import KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

"""
word_vectors = KeyedVectors.load_word2vec_format('./word2vec_models/mr15_32d.vec.pt', binary=True)
print(word_vectors.index_to_key[0])
results = word_vectors.most_similar(positive=['bad'])
for result in results:
    print(result)
print( word_vectors.similarity('boring', 'bad') )
"""
model = Doc2Vec.load('word2vec_models/doc2vec_mr15.model')
print( model.dv.most_similar(0) )
doc_words1 = ["funny","and","good","movie"]
doc_words2 = ["good","and","funny","movie"]
doc_words3 = ["good", "movie","ever"]
doc_words4 = ["worst","and","boring"]
sim_value = model.similarity_unseen_docs(doc_words1, doc_words2, alpha=1, min_alpha=0.0001, steps=100)
print(sim_value)
print(model.similarity_unseen_docs(doc_words1, doc_words3, alpha=1, min_alpha=0.0001, steps=100) )
print(model.similarity_unseen_docs(doc_words1, doc_words4, alpha=1, min_alpha=0.0001, steps=100) )
print(model.similarity_unseen_docs(doc_words2, doc_words3, alpha=1, min_alpha=0.0001, steps=100) )
newvec = model.infer_vector(doc_words1)
print(newvec)
