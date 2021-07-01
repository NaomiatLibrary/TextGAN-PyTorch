from gensim.models import KeyedVectors

word_vectors = KeyedVectors.load_word2vec_format('./word2vec_models/haiku_wakati.vec.pt', binary=True)
print(word_vectors.index_to_key[0])
results = word_vectors.most_similar(positive=['桜'])
for result in results:
    print(result)
