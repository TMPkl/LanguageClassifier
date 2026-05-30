import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from tqdm import tqdm

# ── Hyperparameters ──────────────────────────────────────────────────────────
DATA_PATH       = "tokenized_into_indices.txt"
W2V_PATH        = "word2vec.model"
CHECKPOINT_PATH = "best_model.pt"

MAX_SEQ_LEN  = 500
RANDOM_SEED  = 42

VOCAB_SIZE   = 10002
EMBED_DIM    = 128
HIDDEN_DIM   = 256
NUM_LAYERS   = 2
DROPOUT      = 0.3
NUM_CLASSES  = 8
FREEZE_EMBED = False

BATCH_SIZE   = 64
NUM_EPOCHS   = 30
LR           = 1e-3
LR_PATIENCE  = 3
LR_FACTOR    = 0.5
EARLY_STOP   = 5
WEIGHT_DECAY = 1e-4
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
    hits = 0
    for i in range(vocab_size):
        key = str(i)
        if key in w2v.wv:
            matrix[i] = w2v.wv[key]
            hits += 1
    print(f"Embedding coverage: {hits}/{vocab_size} = {hits/vocab_size*100:.2f}%")
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
        embedded = self.dropout(self.embedding(x))       # (B, T, E)
        _, (hidden, _) = self.lstm(embedded)             # hidden: (layers*2, B, H)
        h = torch.cat([hidden[-2], hidden[-1]], dim=1)   # (B, 2H)
        return self.fc(self.dropout(h))                  # (B, C)


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for X, y in tqdm(loader, desc="Training"):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for X, y in tqdm(loader, desc="Evaluating"):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += criterion(logits, y).item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def train(model, train_loader, val_loader, optimizer, scheduler, criterion, device):
    best_val_loss = float("inf")
    patience_count = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            tag = " [saved]"
        else:
            patience_count += 1
            tag = ""

        print(
            f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}{tag}"
        )

        if patience_count >= EARLY_STOP:
            print(f"Early stopping after epoch {epoch}.")
            break


def evaluate_test(model, loader, device, idx2label):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X, y in loader:
            preds = model(X.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y.numpy())
    label_names = [idx2label[i] for i in range(NUM_CLASSES)]
    print(classification_report(all_true, all_preds, target_names=label_names, digits=4))


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = get_device()
    print(f"Device: {device}")

    print("Loading data...")
    sequences, labels_str = load_data(DATA_PATH)
    labels_int, label2idx = encode_labels(labels_str)
    idx2label = {v: k for k, v in label2idx.items()}
    print(f"Samples: {len(sequences)}, Classes: {list(label2idx)}")

    print("Padding sequences...")
    X = np.array([pad_or_truncate(s, MAX_SEQ_LEN) for s in sequences], dtype=np.int32)

    # Stratified 80 / 10 / 10 split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, labels_int, test_size=0.2, random_state=RANDOM_SEED, stratify=labels_int
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_tmp
    )
    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    print("Building embedding matrix...")
    w2v = Word2Vec.load(W2V_PATH)
    emb_matrix = build_embedding_matrix(w2v, VOCAB_SIZE, EMBED_DIM)
    emb_tensor = torch.FloatTensor(emb_matrix)

    train_ds = CodeDataset(X_train, y_train)
    val_ds   = CodeDataset(X_val,   y_val)
    test_ds  = CodeDataset(X_test,  y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))

    model = BiLSTMClassifier(
        VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT, emb_tensor
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE
    )

    print("\nTraining...")
    train(model, train_loader, val_loader, optimizer, scheduler, criterion, device)

    print("\n=== Test Set Evaluation ===")
    evaluate_test(model, test_loader, device, idx2label)


if __name__ == "__main__":
    main()
