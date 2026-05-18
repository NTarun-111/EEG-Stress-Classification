"""
phase5_utils.py

Shared utilities for Phase 5 deep learning experiments on the SAM 40 dataset.

Everything that gets re-used across Phase 5.1, 5.2, 5.3, 5.4 lives here:
    - Data loading from Phase 2 outputs (the (32, 640) z-scored segments)
    - Subject-grouped train/val/test split (no subject leakage)
    - PyTorch Dataset + DataLoader builders
    - Generic training loop with early stopping
    - Evaluation utilities

Import in every phase5_x script:
    from phase5_utils import (
        load_phase2_data, make_subject_split, make_loaders,
        EEGDataset, train_model, evaluate_model,
        count_parameters, PHASE5_DIR, SEED
    )
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# Paths and constants
# =============================================================================

# Absolute paths because notebook working directory is unreliable in VS Code.
BASE_DIR = r"C:\Users\hibro\OneDrive\Desktop\Desktop_Files\Projects\Python\ML_Models\Cognitive_Stress_Classification\EEG-Stress-Classification"
PHASE2_DIR = os.path.join(BASE_DIR, "Results", "phase2")
PHASE5_DIR = os.path.join(BASE_DIR, "Results", "phase5")

# Single random seed used everywhere. Keeps splits and DL training reproducible.
SEED = 42



# Data loading
# =============================================================================

def load_phase2_data(verbose=True):
    """
    Load the preprocessed segments and labels produced by Phase 2.

    Returns
    -------
    X : np.ndarray of shape (n_segments, 32, 640), dtype float32
        Per-subject z-score normalized EEG segments.
    y_binary : np.ndarray of shape (n_segments,), dtype int64
        Binary labels: 0 = Relaxed, 1 = Stress (Low + High combined).
    subjects : np.ndarray of shape (n_segments,), dtype int
        Subject ID (1..40) for each segment. Used for grouped splits.
    """
    npz_path = os.path.join(PHASE2_DIR, "preprocessed_segments.npz")
    csv_path = os.path.join(PHASE2_DIR, "segment_labels.csv")

    # ----- Load segments (the (2400, 32, 640) array) -----
    npz = np.load(npz_path)
    if verbose:
        print(f"  NPZ keys: {list(npz.keys())}")
    # Phase 2 saved a single array; pick its first key whatever it's named.
    seg_key = list(npz.keys())[0]
    X = npz[seg_key].astype(np.float32)

    # Load labels CSV
    df = pd.read_csv(csv_path)
    if verbose:
        print(f"  CSV columns: {list(df.columns)}")
        print(f"  CSV head:\n{df.head().to_string(index=False)}")

    # Find the relevant columns by keyword match. Adjust here if your column
    # names use different conventions.
    subject_col = next(c for c in df.columns if "subject" in c.lower())
    label_col = next(
        c for c in df.columns
        if "stress" in c.lower() and ("class" in c.lower() or "label" in c.lower() or "level" in c.lower())
    )

    subjects = df[subject_col].values.astype(int)
    labels_raw = df[label_col].values

    # ----- Convert 3-class labels to binary (Relaxed=0, Stress=1) -----
    if labels_raw.dtype.kind in ("U", "O"):  # string labels
        # Anything starting with 'relax' is class 0; everything else is stress.
        y_binary = np.array(
            [0 if str(v).lower().startswith("relax") else 1 for v in labels_raw],
            dtype=np.int64,
        )
    else:  # integer labels — assume 0 = Relaxed, 1/2 = Stress
        y_binary = (labels_raw != 0).astype(np.int64)

    if verbose:
        print(f"\n  X shape: {X.shape}  dtype: {X.dtype}")
        uniq, counts = np.unique(y_binary, return_counts=True)
        print(f"  y_binary classes: {dict(zip(uniq.tolist(), counts.tolist()))}")
        print(f"  unique subjects: {len(np.unique(subjects))}")

    return X, y_binary, subjects



# Subject-grouped split
# =============================================================================
def make_subject_split(subjects, n_train=32, n_val=4, n_test=4, seed=SEED):
    """
    Randomly assign whole SUBJECTS to train / val / test.

    No subject is ever in two splits. We're testing
    generalization to unseen subjects (same principle as LOSO, just chunkier).

    Returns
    -------
    train_idx, val_idx, test_idx : np.ndarray of bool
        Boolean masks of length n_segments selecting each split.
    split_info : dict
        {'train': [list of subject IDs], 'val': [...], 'test': [...]}
    """
    unique = np.unique(subjects)
    total = n_train + n_val + n_test
    assert len(unique) == total, (
        f"Expected exactly {total} unique subjects, got {len(unique)}. "
        "Adjust n_train/n_val/n_test if your dataset differs."
    )

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)

    train_subjects = set(int(s) for s in shuffled[:n_train])
    val_subjects   = set(int(s) for s in shuffled[n_train : n_train + n_val])
    test_subjects  = set(int(s) for s in shuffled[n_train + n_val :])

    train_idx = np.isin(subjects, list(train_subjects))
    val_idx   = np.isin(subjects, list(val_subjects))
    test_idx  = np.isin(subjects, list(test_subjects))

    split_info = {
        "train": sorted(train_subjects),
        "val":   sorted(val_subjects),
        "test":  sorted(test_subjects),
    }
    return train_idx, val_idx, test_idx, split_info



# PyTorch Dataset
# =============================================================================
class EEGDataset(Dataset):
    """
    Wraps EEG segments + labels into PyTorch tensors.

    Each item has shape (1, 32, 640):
      - The leading "1" is the 'IMAGE-CHANNEL' dim that Conv2d expects.
        It is NOT the EEG channels. We treat each segment as a single-channel
        "image" — like a grayscale photo of size 32 x 640.
      - 32 = number of EEG channels (becomes "height" of the image).
      - 640 = number of time samples (becomes "width" of the image).

    A temporal Conv2d kernel of shape (1, K) slides only along time.
    A spatial Conv2d kernel of shape (32, 1) mixes all EEG channels at once.
    Everything in Phase 5 builds on this convention.
    """

    def __init__(self, X, y):
        # unsqueeze(1) inserts the channel axis: (n, 32, 640) -> (n, 1, 32, 640)
        self.X = torch.from_numpy(X).float().unsqueeze(1)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loaders(X, y, train_idx, val_idx, test_idx, batch_size=64):
    """
    Build train/val/test DataLoaders using the boolean split masks.
    Train loader is shuffled; val/test are not (so eval is deterministic).
    """
    train_ds = EEGDataset(X[train_idx], y[train_idx])
    val_ds   = EEGDataset(X[val_idx],   y[val_idx])
    test_ds  = EEGDataset(X[test_idx],  y[test_idx])

    # Generator with fixed seed so shuffle order is reproducible
    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



# Generic training loop
# =============================================================================
def train_model(
    model,
    train_loader,
    val_loader,
    n_epochs=30,
    lr=1e-3,
    weight_decay=1e-4,
    patience=7,
    device="cpu",
    verbose=True,
):
    """
    Train a PyTorch classifier with early stopping on validation loss.

    Parameters
    ----------
    model : nn.Module
        The model to train (already moved to `device`).
    train_loader, val_loader : DataLoader
        Yield (X, y) batches.
    n_epochs : int
        Maximum number of training epochs.
    lr : float
        Adam learning rate.
    weight_decay : float
        L2 regularization strength (helps prevent overfitting on small data).
    patience : int
        Stop if val loss doesn't improve for this many epochs in a row.
    device : 'cpu' or 'cuda'
        Where to run training. CPU only on this machine.
    verbose : bool
        Print per-epoch progress.

    Returns
    -------
    best_state : dict
        Model state_dict at the epoch with the lowest val loss.
    history : dict
        Per-epoch lists: 'train_loss', 'val_loss', 'val_acc'.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, n_epochs + 1):
        # ---------- Training pass ----------
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)               # forward pass
            loss = criterion(logits, y_batch)     # cross-entropy
            loss.backward()                       # backprop gradients
            optimizer.step()                      # update weights
            train_loss_sum += loss.item() * X_batch.size(0)
            train_n += X_batch.size(0)
        train_loss = train_loss_sum / train_n

        # ---------- Validation pass (no gradients) ----------
        model.eval()
        val_loss_sum, val_correct, val_n = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss_sum += loss.item() * X_batch.size(0)
                val_correct  += (logits.argmax(dim=1) == y_batch).sum().item()
                val_n        += X_batch.size(0)
        val_loss = val_loss_sum / val_n
        val_acc  = val_correct / val_n

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            print(
                f"  Epoch {epoch:3d} | "
                f"train_loss {train_loss:.4f} | "
                f"val_loss {val_loss:.4f} | "
                f"val_acc {val_acc:.4f}"
            )

        # ---------- Early stopping logic ----------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # detach + clone so the saved tensors aren't tied to live params
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(no improvement for {patience} epochs).")
                break

    return best_state, history



# Evaluation
# =============================================================================
def evaluate_model(model, test_loader, device="cpu"):
    """
    Run a trained model over a test loader and return metrics + raw predictions.

    Returns
    -------
    results : dict
        accuracy         : float
        f1               : float (binary, positive class = 1)
        f1_macro         : float (macro avg)
        confusion_matrix : np.ndarray of shape (2, 2)
        preds            : np.ndarray of predicted class IDs
        labels           : np.ndarray of true labels
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y_batch.numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    return {
        "accuracy":         accuracy_score(labels, preds),
        "f1":               f1_score(labels, preds, average="binary"),
        "f1_macro":         f1_score(labels, preds, average="macro"),
        "confusion_matrix": confusion_matrix(labels, preds),
        "preds":            preds,
        "labels":           labels,
    }


def count_parameters(model):
    """Return the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
