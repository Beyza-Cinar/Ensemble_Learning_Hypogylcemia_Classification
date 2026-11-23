# ============================================================
# STANDARD LIBRARY
# ============================================================

import os
import re
import csv
import glob
import time
import copy
import random
import datetime

from pathlib import Path
from datetime import timedelta
from collections import Counter

# ============================================================
# NUMERICAL & DATA HANDLING
# ============================================================

import numpy as np
import pandas as pd
import polars as pl

from numpy.random import default_rng
from joblib import load, dump
from tqdm.auto import tqdm


# ============================================================
# SCIKIT-LEARN
# ============================================================

from sklearn.model_selection import (
    train_test_split,
    KFold,
    GroupShuffleSplit
)


from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    auc,
    precision_recall_curve,
    average_precision_score
)


from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# PYTORCH
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset 

# ============================================================
# PYTORCH GEOMETRIC
# ============================================================

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torch_geometric.nn import (
    GATv2Conv,
    global_max_pool
)

from torch_geometric.nn.aggr import MinAggregation

# ============================================================
# LOSS FUNCTIONS
# ============================================================

from torch_focalloss import (
    BinaryFocalLoss,
    MultiClassFocalLoss
)

# ============================================================
# DEVICE CONFIGURATION
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# SETTINGS
# ============================================================

seed_value = 42
epochs = 100
batch_size = 2048
n_classes = 4



# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(seed_value)


# ============================================================
# GPU INITIALIZATION
# ============================================================

REQUIRE_GPU = True

if REQUIRE_GPU and not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU not available. "
        "Check your SLURM GPU allocation and "
        "CUDA-enabled PyTorch installation."
    )

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print("=" * 50)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Selected device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print("Visible GPUs:", torch.cuda.device_count())
    print("CUDA version:", torch.version.cuda)

print("=" * 50)


#---------------- Functions ----------------#
def flatten(subject):
    array_data = [np.array(x) for x in subject]  
    flattened_data = np.concatenate(array_data, axis=0)
    return flattened_data


class NVGDataset(Dataset):

    def __init__(self, X, A, y):

        self.X = np.asarray(X, dtype=np.float32)
        self.A = np.asarray(A, dtype=np.uint8)
        self.y = np.asarray(y, dtype=np.float32)

        assert self.X.shape[1:] == (25, 1)
        assert self.A.shape[1:] == (79,)
        assert len(self.X) == len(self.A) == len(self.y)

        self.T = self.X.shape[1]

        self.time = np.linspace(
            0, 1, self.T, dtype=np.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        values = self.X[idx, :, 0]
        A = np.unpackbits(
            self.A[idx],
            count=625
        ).reshape(25, 25)

        A_with_self = A.copy()
        np.fill_diagonal(A_with_self, 1)

        degree = (
            (A_with_self > 0).sum(axis=1)
            / self.T
        ).astype(np.float32)

        node_features = np.stack(
            [
                values,
                self.time,
                degree
            ],
            axis=1
        ).astype(np.float32)


        A_edges = A.copy()
        np.fill_diagonal(A_edges, 0)

        src, dst = np.nonzero(A_edges)

        edge_index = torch.tensor(
            np.stack([src, dst]),
            dtype=torch.long
        )


        dy = values[src] - values[dst]
        dt = self.time[src] - self.time[dst]

        distances = np.sqrt(
            dy ** 2 + dt ** 2
        ).astype(np.float32)

        edge_attr = torch.from_numpy(
            distances[:, None]
        )


        label = int(np.argmax(self.y[idx]))

        return Data(
            x=torch.from_numpy(node_features),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(label, dtype=torch.long)
        )



def create_nvg_dataset(
    X,
    A,
    y,
    batch_size=2048,
    shuffle=True,
    rand_seed = 42
):
    y_cat = np.eye(4, dtype=np.float32)[y]

    dataset = NVGDataset(X=X, A=A, y=y_cat)

    generator = torch.Generator()
    generator.manual_seed(rand_seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        generator=generator
    )


    return loader

class NVG_GATv2(nn.Module):

    def __init__(self, num_classes=4):
        super().__init__()

        self.node_dense = nn.Linear(3, 64)
        self.edge_dense = nn.Linear(1, 8)

        self.gatv2conv1 = GATv2Conv(
            in_channels=64,
            out_channels=8,
            heads=6,
            concat=True,
            edge_dim=8,
            add_self_loops=True,
            fill_value=0.0
        )

        self.bn1 = nn.BatchNorm1d(48)

        self.gatv2conv2 = GATv2Conv(
            in_channels=48,
            out_channels=8,
            heads=6,
            concat=True,
            edge_dim=8,
            add_self_loops=True,
            fill_value=0.0
        )

        self.bn2 = nn.BatchNorm1d(48)

        self.classifier = nn.Linear(48, num_classes)

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        # Node and edge projections
        x = F.relu(self.node_dense(x))
        edge_attr = F.relu(self.edge_dense(edge_attr))

        # GATv2 block 1
        x = self.gatv2conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.elu(x)

        # GATv2 block 2
        x = self.gatv2conv2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = F.elu(x)

        # Max graph pooling
        x = global_max_pool(x, batch)

        # 4-class logits
        return self.classifier(x)


def GATv2_initialization(Y_train_weights, seed=42):

    set_seed(seed)

    model = NVG_GATv2(
        num_classes=4
    ).to(device)


    if isinstance(Y_train_weights, torch.Tensor):
        y_weights = Y_train_weights.detach().cpu().numpy()
    else:
        y_weights = np.asarray(Y_train_weights)

    class_counts = np.bincount(
        y_weights.astype(np.int64).ravel(),
        minlength=4
    ).astype(np.float32)

    inv_freq = 1.0 / np.maximum(
        class_counts, 1
    )

    alpha = inv_freq / np.mean(inv_freq)

    print("Alpha weights:", alpha)


    criterion = MultiClassFocalLoss(
        gamma=1.5,
        alpha=torch.tensor(
            alpha,
            dtype=torch.float32
        ),
        reduction="none"
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    print("Class weights:", alpha)

    return model, criterion, optimizer



def train_gatv2(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs=100,
    checkpoint_path="best_gatv2.pt"
):

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.2,
        patience=5,
        min_lr=1e-5
    )

    patience = 10
    wait = 0

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": []
    }

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    start_time = time.time()

    for epoch in range(epochs):

        # ====================================
        # TRAINING
        # ====================================

        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:

            batch = batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            logits = model(batch)
            loss = criterion(logits, batch.y.long()).mean()

            loss.backward()
            optimizer.step()

            n = batch.num_graphs

            train_loss_sum += loss.item() * n
            train_correct += (
                logits.argmax(dim=1) == batch.y
            ).sum().item()

            train_total += n

        train_loss = train_loss_sum / train_total
        train_accuracy = train_correct / train_total

        # ====================================
        # VALIDATION 
        # ====================================

        model.eval()

        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.inference_mode():

            for batch in val_loader:

                batch = batch.to(device, non_blocking=True)

                logits = model(batch)
                loss = criterion(logits, batch.y.long()).mean()

                n = batch.num_graphs

                val_loss_sum += loss.item() * n

                val_correct += (
                    logits.argmax(dim=1) == batch.y
                ).sum().item()

                val_total += n

        val_loss = val_loss_sum / val_total
        val_accuracy = val_correct / val_total

        # ====================================
        # CHECKPOINT: BEST VAL LOSS
        # ====================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch + 1
            wait = 0

            best_state = copy.deepcopy(
                model.state_dict()
            )

            torch.save(
                best_state,
                checkpoint_path
            )

            print(
                f"Best model saved | "
                f"Epoch {best_epoch} | "
                f"Val Loss {best_val_loss:.4f}"
            )

        else:
            wait += 1

        # ====================================
        # REDUCE LEARNING RATE
        # ====================================

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        # ====================================
        # STORE METRICS
        # ====================================

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["learning_rate"].append(current_lr)

        print(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # ====================================
        # EARLY STOPPING
        # ====================================

        if wait >= patience:

            print(
                f"\nEarly stopping at epoch {epoch+1}"
            )

            break

    # ========================================
    # RESTORE BEST WEIGHTS
    # ========================================

    if best_state is None:
        raise RuntimeError(
            "No valid checkpoint was produced."
        )

    model.load_state_dict(best_state)

    training_minutes = (
        time.time() - start_time
    ) / 60

    print(
        f"\nBest epoch: {best_epoch}"
        f"\nBest validation loss: {best_val_loss:.4f}"
        f"\nTotal training time: {training_minutes:.2f} min"
    )

    return history, training_minutes, best_epoch

def predict_gatv2(model, test_loader):

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.inference_mode():

        for batch in test_loader:

            batch = batch.to(device, non_blocking=True)

            logits = model(batch)

            probabilities = torch.softmax(
                logits, dim=1
            )

            predictions = probabilities.argmax(dim=1)

            all_labels.append(
                batch.y.cpu().numpy()
            )

            all_predictions.append(
                predictions.cpu().numpy()
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_predictions)
    y_prob = np.concatenate(all_probabilities)

    return y_true, y_pred, y_prob


def calculate_metrics(y_true, y_pred, y_prob):

    labels = np.arange(4)

    results = {}

    results["Accuracy"] = accuracy_score(
        y_true, y_pred
    )

    results["F1_macro"] = f1_score(
        y_true,
        y_pred,
        average="macro",
        labels=labels,
        zero_division=0
    )

    results["Rec_macro"] = recall_score(
        y_true,
        y_pred,
        average="macro",
        labels=labels,
        zero_division=0
    )

    results["Pre_macro"] = precision_score(
        y_true,
        y_pred,
        average="macro",
        labels=labels,
        zero_division=0
    )


    f1_class = f1_score(
        y_true,
        y_pred,
        average=None,
        labels=labels,
        zero_division=0
    )

    rec_class = recall_score(
        y_true,
        y_pred,
        average=None,
        labels=labels,
        zero_division=0
    )

    pre_class = precision_score(
        y_true,
        y_pred,
        average=None,
        labels=labels,
        zero_division=0
    )

    pr_auc_class = []
    pr_avg_class = []

    for cl in labels:

        y_binary = (y_true == cl).astype(int)

        if y_binary.sum() == 0:
            pr_auc = np.nan
            avg_precision = np.nan

        else:
            precision, recall, _ = precision_recall_curve(
                y_binary,
                y_prob[:, cl]
            )

            pr_auc = auc(
                recall,
                precision
            )

            avg_precision = average_precision_score(
                y_binary,
                y_prob[:, cl]
            )

        pr_auc_class.append(pr_auc)
        pr_avg_class.append(avg_precision)


    for cl in labels:

        results[f"F1_class_{cl}"] = f1_class[cl]
        results[f"Rec_class_{cl}"] = rec_class[cl]
        results[f"Pre_class_{cl}"] = pre_class[cl]

        results[f"PR_AUC_class_{cl}"] = pr_auc_class[cl]
        results[f"PR_AVG_class_{cl}"] = pr_avg_class[cl]


    results["PR_AUC_Mac"] = np.nanmean(
        pr_auc_class
    )

    results["PR_AVG_Mac"] = np.nanmean(
        pr_avg_class
    )

    return results


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_class_labels(y):
    y = np.asarray(y)

    if y.ndim == 2 and y.shape[1] == 4:
        return np.argmax(y, axis=1).astype(np.int64)

    return y.reshape(-1).astype(np.int64)


def prepare_gnn_input(X):
    X = np.asarray(X, dtype=np.float32)

    assert X.ndim == 3
    assert X.shape[1] == 25

    # Assumes amplitude is in column 0.
    return X[:, :, :1]


output_dir = Path("Result/GATv2_3Fold")
output_dir.mkdir(parents=True, exist_ok=True)


#------------------- MODELS --------------------#

def main():

    set_seed(seed_value)

    # empty list for the predicted and actual values
    y_true_labels = []
    predicted_labels = []
    predicted_prob_labels = []

    # Results across the three folds
    results = []

    
    count = int(os.environ["FOLD"])

    print(f"\n========== FOLD {count} ==========")

    X_train_fold = np.load(f"/GAT/data/X_train_fold_{count}.npy", mmap_mode="r")
    X_val_fold = np.load(f"/GAT/data/X_val_fold_{count}.npy", mmap_mode="r")
    X_test_fold = np.load(f"/GAT/data/X_test_fold_{count}.npy", mmap_mode="r")

    A_train_fold = np.load(f"/GAT/data/A_nvg_train_fold_{count}.npy", mmap_mode="r")
    A_val_fold = np.load(f"/GAT/data/A_nvg_val_fold_{count}.npy", mmap_mode="r")
    A_test_fold = np.load(f"/GAT/data/A_nvg_test_fold_{count}.npy", mmap_mode="r")

    Y_train_fold = np.load(f"/GAT/data/Y_train_fold_{count}.npy", mmap_mode="r")
    Y_val_fold = np.load(f"/GAT/data/Y_val_fold_{count}.npy", mmap_mode="r")
    Y_test_fold = np.load(f"/GAT/data/Y_test_fold_{count}.npy", mmap_mode="r")

    Y_train_fold= Y_train_fold.reshape(-1).astype('int32')
    Y_val_fold= Y_val_fold.reshape(-1).astype('int32')
    Y_test_fold= Y_test_fold.reshape(-1).astype('int32')

    Train_loader = create_nvg_dataset(
        X_train_fold,A_train_fold,Y_train_fold,
        batch_size=2048,
        shuffle=True,
        rand_seed = 42)

    Val_loader = create_nvg_dataset(
        X_val_fold,A_val_fold,Y_val_fold,
        batch_size=2048,
        shuffle=False,
        rand_seed = 42)

    Test_loader = create_nvg_dataset(
        X_test_fold,A_test_fold,Y_test_fold,
        batch_size=2048,
        shuffle=False,
        rand_seed = 42)

    # -------------------------------------
    # FRESH MODEL FOR EACH FOLD
    # -------------------------------------

    model, criterion, optimizer = GATv2_initialization(
        Y_train_fold, 
        seed=seed_value
    )

    # -------------------------------------
    # TRAINING
    # -------------------------------------

    history, training_minutes, best_epoch = train_gatv2(
        model=model,
        train_loader=Train_loader,
        val_loader=Val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=100,
        checkpoint_path=output_dir / f"GATv2_fold_{count}.pt"
    )

    # -------------------------------------
    # TEST EVALUATION AFTER TRAINING
    # -------------------------------------

    y_true, y_pred, y_prob = predict_gatv2(
        model,
        Test_loader
    )

    assert np.array_equal(
        y_true,
        Y_test_fold
    ), "Test labels or sample order do not match"

    # -------------------------------------
    # METRICS
    # -------------------------------------
    
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    metrics["fold"] = count
    metrics["best_epoch"] = best_epoch
    metrics["training_minutes"] = training_minutes
    
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2, 3],
            digits=4,
            zero_division=0
        )
    )
    
    # SAVE MODEL
    torch.save(
        model.state_dict(),
        output_dir / f"GATv2_fold_{count}.pt"
    )
    
    # SAVE TRAINING HISTORY
    pd.DataFrame(history).to_csv(
        output_dir / f"training_fold_{count}.csv",
        index=False
    )
    
    # SAVE PREDICTIONS
    dump(
        y_true,
        output_dir / f"y_true_GATv2_fold_{count}.joblib",
        compress=("zlib", 1)
    )
    
    dump(
        y_pred,
        output_dir / f"y_pred_GATv2_fold_{count}.joblib",
        compress=("zlib", 1)
    )
    
    dump(
        y_prob,
        output_dir / f"y_pred_prob_GATv2_fold_{count}.joblib",
        compress=("zlib", 1)
    )
    
    # METRIC DEFINITIONS
    metric_names = [
        ("Macro F1 Measure", "F1_macro"),
        ("Macro Recall", "Rec_macro"),
        ("Macro Precision", "Pre_macro"),
        ("Macro PR AUC", "PR_AUC_Mac"),
        ("Macro AVG PR", "PR_AVG_Mac")
    ]
    
    class_metrics = [
        ("F1 Measure per Class", "F1_class"),
        ("Recall per Class", "Rec_class"),
        ("Precision per Class", "Pre_class"),
        ("PR AUC per Class", "PR_AUC_class")
    ]
    
    # SAVE METRICS FOR CURRENT FOLD
    with open(output_dir / f"GATv2_fold_{count}.txt", "w") as f:
    
        for title, key in metric_names:
            f.write(f"{title}: {metrics[key]:.4f}\n\n")
    
        for title, key in class_metrics:
            scores = [
                metrics[f"{key}_{cl}"]
                for cl in range(4)
            ]
            f.write(f"{title}: {scores}\n\n")
    
        f.write("Training information:\n")
        f.write(
            f"Fold {count}: "
            f"Best epoch: {best_epoch}, "
            f"Training time (min): {training_minutes:.2f}\n"
        )
    
    print(f"Fold {count} completed.")
    
if __name__ == "__main__":
    main()
