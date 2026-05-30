from collections import Counter
import json

def build_vocab_from_tokenized(path: str, max_vocab=10_000, min_freq=30) -> dict:
    counter = Counter()
    with open(path, encoding='utf-8') as f:
        for line in f:
            _, tokens_str = line.strip().split('\t', maxsplit=1)
            counter.update(tokens_str.split(' '))
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for token, freq in counter.most_common(max_vocab):
        if freq < min_freq:
            break
        vocab[token] = len(vocab)
    
    return vocab

def load_tokenized(path: str) -> list[tuple[str, list[str]]]:
    samples = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            lang, tokens_str = line.strip().split('\t', maxsplit=1)
            samples.append((lang, tokens_str.split(' ')))
    return samples

# użycie
samples = load_tokenized('tokenized.txt')
vocab   = build_vocab_from_tokenized('tokenized.txt')

# zapis słownika
with open('vocab.json', 'w') as f:
    json.dump(vocab, f)

# podgląd
lang, tokens = samples[0]
indices = [vocab.get(t, 1) for t in tokens]
print(f"[{lang}] {tokens[:10]} → {indices[:10]}")

with open('tokenized_into_indices.txt', 'w', encoding='utf-8') as f:
    for lang, tokens in samples:
        indices = [vocab.get(t, 1) for t in tokens]
        f.write(f"{lang}\t{' '.join(map(str, indices))}\n")