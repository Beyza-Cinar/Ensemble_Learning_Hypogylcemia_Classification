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
epochs = 75
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


output_dir = Path("Result/GATv2_3Fold/OS")
output_dir.mkdir(parents=True, exist_ok=True)

def train_gatv2_one_shot(
    model,
    train_loader,
    criterion,
    optimizer,
    epochs=75
):

    start_time = time.time()

    for epoch in range(epochs):

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

        print(
            f"Epoch {epoch+1:03d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f}"
        )

    training_minutes = (
        time.time() - start_time
    ) / 60

    print(
        f"\nTotal training time: {training_minutes:.2f} min"
    )

    return training_minutes

#------------------- MODELS --------------------#

def main():

    set_seed(seed_value)

    X_data, Y_data = load("Data/3Fold_Train_Data.joblib")
    A_data = np.load(
    "GAT/data/A_nvg_Full_Train.npy",
    mmap_mode="r"
    )

    X_data = flatten(X_data)
    Y_data = flatten(Y_data).reshape(-1).astype('int32')

    Train_loader = create_nvg_dataset(
        X_data,A_data,Y_data,
        batch_size=2048,
        shuffle=True,
        rand_seed = 42)


    model, criterion, optimizer = GATv2_initialization(
        Y_data, 
        seed=seed_value
    )
    print("Initialized successfully")
    print("Started training")

    # TRAINING
    training_minutes = train_gatv2_one_shot(
        model=model,
        train_loader=Train_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=62
    )
    
    # SAVE MODEL
    torch.save(
        model.state_dict(),
        output_dir / f"GATv2_OS.pt"
    )
    
    print("Model saved")
    print("Training completed.")
    
if __name__ == "__main__":
    main()
