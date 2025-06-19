#!/usr/bin/env python
"""
crossval_resnet.py
------------------
5-fold stratified cross-validation for the ResNet-18 log-Mel classifier.

* keeps every genuine (label 0) and tampered (label 2) file
* samples 30 % of the huge spoof class (label 1)
* trains 3 epochs (early-stop) per fold on CPU or GPU
* reports accuracy and EER per class, plus macro averages
"""

import pathlib, random, time, warnings, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ────────── config ─────────────────────────────────────────────────────────
SEED, BATCH, EPOCHS, MAXF = 42, 8, 3, 400
FOLDS = 5
PROJ   = pathlib.Path(__file__).resolve().parents[1]
FEAT_D = PROJ / "data_feats"
META_C = PROJ / "meta.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

# ────────── augment / dataset / collate ────────────────────────────────────
def specaugment(x: torch.Tensor) -> torch.Tensor:
    _, m, n = x.shape
    t, tw = random.randint(0, n-1), random.randint(10, 30)
    x[:, :, t:t+tw] = 0
    f, fw = random.randint(0, m-1), random.randint(8, 16)
    x[:, f:f+fw, :] = 0
    return x

class LogMelDS(Dataset):
    def __init__(self, rows, aug=False):
        self.rows, self.aug = rows, aug
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        rel, lbl = self.rows[idx]
        x = torch.load(FEAT_D / rel.with_suffix(".pt"))["feat"].float()
        if self.aug and lbl == 0: x = specaugment(x.clone())
        return x, int(lbl)

def collate(batch):
    xs, ys = zip(*batch)
    xs = [x if x.shape[-1] <= MAXF else x[:, :, -MAXF:] for x in xs]
    T = max(x.shape[-1] for x in xs)
    pad = torch.zeros(len(xs), 1, 128, T)
    for i, x in enumerate(xs): pad[i, :, :, :x.shape[-1]] = x
    return pad, torch.tensor(ys)

# ────────── metric helpers ─────────────────────────────────────────────────
def class_eer(y_true_bin, y_prob):
    eers = []
    for i in range(3):
        if 0 < y_true_bin[:, i].sum() < len(y_true_bin):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            fnr = 1 - tpr
            idx = np.nanargmin(np.abs(fnr - fpr))
            eers.append((fpr[idx] + fnr[idx]) / 2)
        else:
            eers.append(np.nan)
    return np.array(eers)

def evaluate(model, loader):
    model.eval(); ALLy, ALLp = [], []
    with torch.no_grad():
        for x, y in loader:
            ALLp.append(model(x.to(DEVICE)).softmax(1).cpu()); ALLy.append(y)
    y = torch.cat(ALLy); p = torch.cat(ALLp)
    acc = (p.argmax(1) == y).float().mean().item()
    eer = class_eer(label_binarize(y, classes=[0,1,2]), p.numpy())
    cm  = confusion_matrix(y, p.argmax(1), labels=[0,1,2])
    return acc, eer, cm

# ────────── load and sample dataset ────────────────────────────────────────
df = pd.read_csv(META_C)
df["path"] = df["path"].apply(pathlib.Path)

def sample_fn(g):
    if g.name in (0,2):               # keep all genuine & tampered
        return g
    return g.sample(frac=0.3, random_state=SEED)  # 30 % of spoof

df = (df.groupby("label", group_keys=False, sort=False)
        .apply(sample_fn)
        .reset_index(drop=True)
        .sample(frac=1, random_state=SEED))

rows   = df[["path", "label"]].values.tolist()
labels = df["label"].to_numpy()

print(f"Total after sampling → {len(rows)} clips")
print("Class counts:", np.bincount(labels, minlength=3))

# ────────── cross-validation ───────────────────────────────────────────────
fold_metrics = []

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, test_idx) in enumerate(skf.split(rows, labels), 1):
    print(f"\n── Fold {fold}/{FOLDS} ──")
    rows_tr  = [rows[i] for i in train_idx]
    rows_tst = [rows[i] for i in test_idx]

    tr_dl  = DataLoader(LogMelDS(rows_tr,  aug=True),  batch_size=BATCH,
                        shuffle=True,  num_workers=0, collate_fn=collate)
    tst_dl = DataLoader(LogMelDS(rows_tst),           batch_size=BATCH,
                        shuffle=False, num_workers=0, collate_fn=collate)

    # model
    model = resnet18(num_classes=3); model.conv1 = nn.Conv2d(1,64,7,2,3,bias=False)
    model.to(DEVICE)
    opt  = torch.optim.AdamW(model.parameters(), 1e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    best_eer, stale = 1.0, 0

    for ep in range(1, EPOCHS+1):
        model.train(); run = 0
        for x, y in tr_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item() * len(y)

        acc, eer, _ = evaluate(model, tst_dl)
        if eer.mean() < best_eer:
            best_eer, stale = eer.mean(), 0
            best_state = model.state_dict()
        else:
            stale += 1
            if stale >= 1: break      # early stop after 1 bad epoch

    model.load_state_dict(best_state)
    acc, eer, cm = evaluate(model, tst_dl)
    fold_metrics.append((acc, eer, cm))
    print(f"Fold {fold}  acc {acc:.3%} | "
          f"EER 0={eer[0]:.3%} 1={eer[1]:.3%} 2={eer[2]:.3%}")

# ────────── aggregate results ─────────────────────────────────────────────
accs, eers, cms = zip(*fold_metrics)
print("\n=== Cross-validation summary ===")
print(f"ACC mean ± std : {np.mean(accs):.3%} ± {np.std(accs):.3%}")
print("EER mean (%)   :", np.nanmean(eers, axis=0) * 100)
print("Confusion matrix (sum over folds):\n", sum(cms))
