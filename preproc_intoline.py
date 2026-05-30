import re
import pandas as pd

def tokenize(code):
    return re.findall(r'[a-zA-Z_]\w*|\d+|[^\s\w]', code)

df = pd.read_csv('dataset.csv')

# usuń rekordy bez treści
df = df.dropna(subset=['content', 'language'])

with open('tokenized.txt', 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        tokens = tokenize(row['content'])
        tokens = tokens[:1000]

        if tokens:  # pomiń puste
            f.write(f"{row['language']}\t{' '.join(tokens)}\n")

