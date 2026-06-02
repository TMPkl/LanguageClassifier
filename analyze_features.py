"""
Analiza feature importance dla BiLSTMClassifier — które tokeny/słowa
najbardziej determinują każdą klasę.

Metody:
  1. Integrated Gradients (IG) — matematycznie poprawna atrybucja
  2. Token Occlusion (perturbacja) — empiryczna weryfikacja
  3. Vocabulary-level aggregation — ranking słów w całym słowniku

Uruchomienie:
    python analyze_features.py

Wyniki zapisywane do:
    feature_importance/  (CSV-y + wykresy PNG)
"""

import argparse
import os
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Kopiuj hiperparametry z oryginalnego skryptu ────────────────────────────
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
BATCH_SIZE   = 512

# Analiza
IG_STEPS          = 35     # kroki całkowania w Integrated Gradients
OCCLUDE_SAMPLES   = 500    # ile próbek do occlusion analysis
TOP_K             = 30     # top-K słów w rankingu per klasa
OUTPUT_DIR        = Path("feature_importance")
# ────────────────────────────────────────────────────────────────────────────


# ── Model i dane (identyczne z treningiem) ──────────────────────────────────

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def load_data(path):
    sequences, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line: continue
            lang, idx_str = line.split("\t", maxsplit=1)
            sequences.append(list(map(int, idx_str.split())))
            labels.append(lang)
    return sequences, labels


def encode_labels(labels):
    classes = sorted(set(labels))
    label2idx = {c: i for i, c in enumerate(classes)}
    return np.array([label2idx[l] for l in labels], dtype=np.int64), label2idx


def pad_or_truncate(seq, max_len, pad_idx=0):
    if len(seq) >= max_len: return seq[:max_len]
    return seq + [pad_idx] * (max_len - len(seq))


def build_embedding_matrix(w2v, vocab_size, embed_dim):
    matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    for i in range(vocab_size):
        key = str(i)
        if key in w2v.wv:
            matrix[i] = w2v.wv[key]
    return matrix


class CodeDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = torch.LongTensor(sequences)
        self.y = torch.LongTensor(labels)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 num_classes, dropout, pretrained_weights):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            pretrained_weights, freeze=FREEZE_EMBED, padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0, bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        h = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(h))

    def forward_embed(self, embedded):
        """Forward pass od gotowych embeddingów (dla IG).
        Dropout celowo pominięty — deterministyczne gradienty.
        Musi być wywołany w model.train() ze względu na cuDNN backward.
        """
        _, (hidden, _) = self.lstm(embedded)
        h = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(h)


# ── Metoda 1: Integrated Gradients ──────────────────────────────────────────

def integrated_gradients(model, x_ids, target_class, device, steps=IG_STEPS):
    """
    Zwraca atrybucje IG kształtu (seq_len,) dla pojedynczej próbki.
    Baseline = zero embedding (padding token).

    cuDNN RNN backward wymaga train mode — dropout jest jednak wyłączony
    w forward_embed(), więc gradienty są deterministyczne.
    FP16 mixed precision przyspiesza obliczenia na GPU.
    """
    x_ids = x_ids.unsqueeze(0).to(device)   # (1, T)

    # Embeddingi pobieramy w eval (bez szumu dropout)
    model.eval()
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16):
            embed_orig = model.embedding(x_ids).detach()  # (1, T, E)
    baseline = torch.zeros_like(embed_orig)

    alphas = torch.linspace(0, 1, steps, device=device)          # (steps,)
    interpolated = baseline + alphas[:, None, None] * (embed_orig - baseline)
    # shape: (steps, T, E)

    # cuDNN wymaga train() do obliczenia gradientów przez LSTM
    model.train()
    grads = []
    for alpha_embed in interpolated:
        with torch.autocast("cuda", dtype=torch.float16):
            inp = alpha_embed.unsqueeze(0).requires_grad_(True)       # (1, T, E)
            logit = model.forward_embed(inp)[0, target_class]
        logit.backward()
        grads.append(inp.grad.detach().clone())

    avg_grads = torch.stack(grads).mean(dim=0)      # (1, T, E)
    ig = (embed_orig - baseline) * avg_grads        # (1, T, E)
    attribution = ig.squeeze(0).norm(dim=-1)        # (T,)  L2 per token

    return attribution.cpu().numpy()


def compute_ig_scores_per_class(model, dataset, idx2label, device,
                                max_samples_per_class=150):
    """
    Dla każdej klasy zbiera uśrednione atrybucje IG per token_id.
    Zwraca dict: class_name → {token_id: avg_attribution}
    """
    print("\n[1/2] Integrated Gradients — atrybucje per klasa...")

    agg = {c: defaultdict(float) for c in idx2label.values()}
    cnt = {c: defaultdict(int)   for c in idx2label.values()}
    samples_per_class = defaultdict(int)

    for x, y in tqdm(dataset, desc="  IG"):
        label = idx2label[y.item()]
        if samples_per_class[label] >= max_samples_per_class:
            continue

        attrs = integrated_gradients(model, x, y.item(), device)
        token_ids = x.numpy()

        for tok_id, attr in zip(token_ids, attrs):
            if tok_id == 0: continue    # padding
            agg[label][tok_id] += float(attr)
            cnt[label][tok_id] += 1

        samples_per_class[label] += 1

    # Normalizacja przez liczbę wystąpień
    return {
        label: {
            tok_id: agg[label][tok_id] / cnt[label][tok_id]
            for tok_id in agg[label]
        }
        for label in agg
    }


# ── Metoda 2: Token Occlusion ───────────────────────────────────────────────

def token_occlusion_scores(model, dataset, idx2label, device,
                           n_samples=OCCLUDE_SAMPLES):
    """
    Maskuje każdy unikalny token (→ 0) i mierzy spadek logitu.
    Zwraca dict: class_name → {token_id: avg_drop}
    FP16 mixed precision przyspiesza obliczenia na GPU.
    """
    print("\n[2/2] Token Occlusion — perturbacja tokenów...")

    agg = {c: defaultdict(float) for c in idx2label.values()}
    cnt = {c: defaultdict(int)   for c in idx2label.values()}

    indices = torch.randperm(len(dataset))[:n_samples]

    model.eval()
    with torch.no_grad():
        for idx in tqdm(indices, desc="  Occlusion"):
            x, y = dataset[idx.item()]
            label = idx2label[y.item()]

            with torch.autocast("cuda", dtype=torch.float16):
                x_dev = x.unsqueeze(0).to(device)
                base_logit = model(x_dev)[0, y.item()].item()

                for tok_id in set(x.numpy().tolist()) - {0}:
                    masked = x.clone()
                    masked[masked == tok_id] = 0
                    new_logit = model(masked.unsqueeze(0).to(device))[0, y.item()].item()
                    drop = base_logit - new_logit   # > 0 → token pomagał
                    agg[label][tok_id] += drop
                    cnt[label][tok_id] += 1

    return {
        label: {
            tok_id: agg[label][tok_id] / cnt[label][tok_id]
            for tok_id in agg[label]
        }
        for label in agg
    }


# ── Agregacja i ranking ──────────────────────────────────────────────────────

def build_idx2token(w2v):
    idx2tok = {}
    for key in w2v.wv.index_to_key:
        try:
            idx2tok[int(key)] = key
        except ValueError:
            pass
    return idx2tok


def top_tokens(scores_dict, idx2token, top_k=TOP_K):
    items = [(idx2token.get(k, f"<{k}>"), v) for k, v in scores_dict.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:top_k], items[-top_k:][::-1]


# ── Zapis wyników ───────────────────────────────────────────────────────────

def save_csv(rows, path):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "token", "score"])
        for i, (tok, sc) in enumerate(rows, 1):
            w.writerow([i, tok, f"{sc:.6f}"])


def plot_class(pos_items, neg_items, class_name, method, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"{class_name}  [{method}]", fontsize=14, fontweight="bold")

    for ax, items, color, title in [
        (axes[0], pos_items[:20], "#2ecc71", "Top 20 — pomocne (↑ logit)"),
        (axes[1], neg_items[:20], "#e74c3c", "Top 20 — szkodliwe (↓ logit)"),
    ]:
        tokens = [t for t, _ in items]
        scores = [s for _, s in items]
        bars = ax.barh(range(len(tokens)), scores, color=color, alpha=0.8)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Attribution score")
        ax.bar_label(bars, fmt="%.4f", fontsize=7, padding=2)

    plt.tight_layout()
    fname = out_dir / f"{class_name}_{method}.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    return fname


def print_ranking(pos_items, neg_items, class_name, top_n=15):
    print(f"\n{'─'*55}")
    print(f"  {class_name}")
    print(f"{'─'*55}")
    print(f"  {'Rank':<5} {'Token':<20} {'Score':>10}   DIR")
    print(f"  {'────':<5} {'─────':<20} {'─────':>10}   ───")
    for i, (tok, sc) in enumerate(pos_items[:top_n], 1):
        print(f"  {i:<5} {tok:<20} {sc:>10.4f}   [+]")
    print()
    for i, (tok, sc) in enumerate(neg_items[:top_n], 1):
        print(f"  {i:<5} {tok:<20} {sc:>10.4f}   [-]")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ig", "occlusion", "both"],
                        default="both")
    parser.add_argument("--ig-samples",  type=int, default=200)
    parser.add_argument("--occ-samples", type=int, default=500)
    parser.add_argument("--top-k",       type=int, default=TOP_K)
    parser.add_argument("--no-plots",    action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    print("Ładowanie danych...")
    sequences, labels_str = load_data(DATA_PATH)
    labels_int, label2idx = encode_labels(labels_str)
    idx2label = {v: k for k, v in label2idx.items()}

    X = np.array([pad_or_truncate(s, MAX_SEQ_LEN) for s in sequences], dtype=np.int32)
    _, X_tmp, _, y_tmp = train_test_split(
        X, labels_int, test_size=0.2, random_state=RANDOM_SEED, stratify=labels_int
    )
    _, X_test, _, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_tmp
    )
    print(f"Próbki testowe: {len(X_test)}")

    test_ds = CodeDataset(X_test, y_test)

    print("Ładowanie modelu...")
    w2v = Word2Vec.load(W2V_PATH)
    emb_matrix = build_embedding_matrix(w2v, VOCAB_SIZE, EMBED_DIM)
    emb_tensor = torch.FloatTensor(emb_matrix)
    idx2token = build_idx2token(w2v)

    model = BiLSTMClassifier(
        VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS,
        NUM_CLASSES, DROPOUT, emb_tensor
    ).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    print("Model załadowany.")

    all_results = {}

    if args.method in ("ig", "both"):
        all_results["IG"] = compute_ig_scores_per_class(
            model, test_ds, idx2label, device, args.ig_samples
        )

    if args.method in ("occlusion", "both"):
        all_results["Occlusion"] = token_occlusion_scores(
            model, test_ds, idx2label, device, args.occ_samples
        )

    print("\n\n══════════════════════════════════════════════════")
    print("  WYNIKI ANALIZY FEATURE IMPORTANCE")
    print("══════════════════════════════════════════════════")

    for method, class_scores in all_results.items():
        method_dir = OUTPUT_DIR / method.lower()
        method_dir.mkdir(exist_ok=True)
        print(f"\n{'═'*55}")
        print(f"  METODA: {method}")
        print(f"{'═'*55}")

        for class_name in sorted(class_scores):
            pos, neg = top_tokens(class_scores[class_name], idx2token, args.top_k)
            print_ranking(pos, neg, class_name)
            save_csv(pos, method_dir / f"{class_name}_positive.csv")
            save_csv(neg, method_dir / f"{class_name}_negative.csv")
            if not args.no_plots:
                out = plot_class(pos, neg, class_name, method, method_dir)
                if out:
                    print(f"  → wykres: {out}")

    if len(all_results) == 2:
        print("\n\n══════════════════════════════════════════════════")
        print("  ZGODNOŚĆ IG vs OCCLUSION (top-10 overlap)")
        print("══════════════════════════════════════════════════")
        ig_s, occ_s = all_results["IG"], all_results["Occlusion"]
        for cls in sorted(idx2label.values()):
            ig_pos,  _ = top_tokens(ig_s.get(cls,  {}), idx2token, 10)
            occ_pos, _ = top_tokens(occ_s.get(cls, {}), idx2token, 10)
            overlap = {t for t, _ in ig_pos} & {t for t, _ in occ_pos}
            print(f"  {cls:<20} overlap={len(overlap)}/10  "
                  f"wspólne: {', '.join(sorted(overlap)) or '—'}")

    print(f"\nWyniki zapisane w: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()