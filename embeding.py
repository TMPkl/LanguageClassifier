from gensim.models import Word2Vec
import json

def load_tokenized(path: str) -> list[tuple[str, list[str]]]:
    samples = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            lang, tokens_str = line.strip().split('\t', maxsplit=1)
            samples.append((lang, tokens_str.split(' ')))
    return samples

samples = load_tokenized('tokenized_into_indices.txt')
sentences = [tokens for _, tokens in samples]  # Word2Vec nie potrzebuje etykiet

print(sentences[0][:10])  # podgląd pierwszego przykładu

model = Word2Vec(
    sentences,
    vector_size=128,   # embed_dim — musi zgadzać się z LSTM
    window=4,          # kontekst: 8 tokenów w lewo i prawo
    min_count=10,       # ignoruj rzadkie tokeny (jak min_freq w vocab)
    workers=-1,
    epochs=10,

)


model.save('word2vec.model')
print(f"Słownik W2V: {len(model.wv)} tokenów")


# podgląd — najbliższe tokeny do "import"
print(model.wv.most_similar('10', topn=5))
