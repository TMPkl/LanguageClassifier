"""
Evaluate the trained classifier on validation and test splits.

Usage:
    python test_classifier.py
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── Must match main_train.py ─────────────────────────────────────────────────
DATA_PATH       = "tokenized_into_indices.txt"
W2V_PATH        = "word2vec.model"
CHECKPOINT_PATH = "best_model.pt"

MAX_SEQ_LEN  = 500
RANDOM_SEED  = 42

VOCAB_SIZE   = 10002
EMBED_DIM    = 128
HIDDEN_DIM   = 256
NUM_LAYERS   = 2
DROPOUT      = 0
NUM_CLASSES  = 8
FREEZE_EMBED = False

BATCH_SIZE   = 64
# ─────────────────────────────────────────────────────────────────────────────


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_data(path: str):
    sequences, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            lang, indices_str = line.split("\t", maxsplit=1)
            sequences.append(list(map(int, indices_str.split())))
            labels.append(lang)
    return sequences, labels


def encode_labels(labels):
    classes = sorted(set(labels))
    label2idx = {c: i for i, c in enumerate(classes)}
    return np.array([label2idx[l] for l in labels], dtype=np.int64), label2idx


def pad_or_truncate(seq, max_len: int, pad_idx: int = 0):
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_idx] * (max_len - len(seq))


def build_embedding_matrix(w2v, vocab_size: int, embed_dim: int) -> np.ndarray:
    matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    for i in range(vocab_size):
        key = str(i)
        if key in w2v.wv:
            matrix[i] = w2v.wv[key]
    return matrix


class CodeDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.X = torch.LongTensor(sequences)
        self.y = torch.LongTensor(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 num_classes, dropout, pretrained_weights):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            pretrained_weights, freeze=FREEZE_EMBED, padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        h = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(h))


def load_splits():
    sequences, labels_str = load_data(DATA_PATH)
    labels_int, label2idx = encode_labels(labels_str)
    idx2label = {v: k for k, v in label2idx.items()}

    X = np.array([pad_or_truncate(s, MAX_SEQ_LEN) for s in sequences], dtype=np.int32)

    # Stratified 80 / 10 / 10 split (same as main_train.py)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, labels_int, test_size=0.2, random_state=RANDOM_SEED, stratify=labels_int
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_tmp
    )

    splits = {
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
    return splits, idx2label


def evaluate_split(model, loader, device, idx2label, split_name: str):
    model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for X, y in loader:
            logits = model(X.to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y.numpy())

    label_names = [idx2label[i] for i in range(NUM_CLASSES)]
    acc = accuracy_score(all_true, all_preds)

    print(f"\n=== {split_name.upper()} SET ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nPer-class metrics (precision, sensitivity/recall, F1, support):")
    print(
        classification_report(
            all_true,
            all_preds,
            target_names=label_names,
            digits=4,
            zero_division=0,
        )
    )


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = get_device()
    print(f"Device: {device}")

    splits, idx2label = load_splits()

    print("Loading embeddings...")
    w2v = Word2Vec.load(W2V_PATH)
    emb_matrix = build_embedding_matrix(w2v, VOCAB_SIZE, EMBED_DIM)
    emb_tensor = torch.FloatTensor(emb_matrix)

    model = BiLSTMClassifier(
        VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT, emb_tensor
    ).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    for split_name in ("val", "test"):
        X_split, y_split = splits[split_name]
        dataset = CodeDataset(X_split, y_split)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        evaluate_split(model, loader, device, idx2label, split_name)


if __name__ == "__main__":
    main()
