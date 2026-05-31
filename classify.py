"""
Usage:
    python classify.py <file.txt>
"""
import sys
import re
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Must match main_train.py ─────────────────────────────────────────────────
VOCAB_PATH      = "vocab.json"
CHECKPOINT_PATH = "best_model.pt"
MAX_SEQ_LEN     = 500
VOCAB_SIZE      = 10002
EMBED_DIM       = 128
HIDDEN_DIM      = 256
NUM_LAYERS      = 2
DROPOUT         = 0.3
NUM_CLASSES     = 8
FREEZE_EMBED    = False

# Alphabetical order — must match encode_labels() in main_train.py
CLASSES = ["assembly", "c++", "c-sharp", "go", "java", "prolog", "python", "verilog"]
# ─────────────────────────────────────────────────────────────────────────────


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
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


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tokenize(code: str):
    return re.findall(r'[a-zA-Z_]\w*|\d+|[^\s\w]', code)


def load_model(device: torch.device) -> BiLSTMClassifier:
    model = BiLSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, DROPOUT)
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def classify(code: str, model: BiLSTMClassifier, vocab: dict, device: torch.device):
    tokens = tokenize(code)[:MAX_SEQ_LEN]
    indices = [vocab.get(t, 1) for t in tokens]  # 1 = <UNK>

    # pad to MAX_SEQ_LEN
    indices += [0] * (MAX_SEQ_LEN - len(indices))

    x = torch.LongTensor([indices]).to(device)  # (1, MAX_SEQ_LEN)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_idx = int(np.argmax(probs))
    return CLASSES[top_idx], probs


def main():
    if len(sys.argv) < 2:
        print("Usage: python classify.py <file.txt>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        code = f.read()

    device = get_device()

    with open(VOCAB_PATH, encoding="utf-8") as f:
        vocab = json.load(f)

    model = load_model(device)

    predicted, probs = classify(code, model, vocab, device)

    print(f"Predicted language: {predicted}\n")
    print("Confidence scores:")
    for lang, prob in sorted(zip(CLASSES, probs), key=lambda x: -x[1]):
        bar = "█" * int(prob * 40)
        print(f"  {lang:10s} {prob*100:6.2f}%  {bar}")


if __name__ == "__main__":
    main()
