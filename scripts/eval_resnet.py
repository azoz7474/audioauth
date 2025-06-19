#!/usr/bin/env python
"""
Evaluate best_resnet18.pt on the held-out TEST fold.
• Re-creates the exact 10 % subset (SEED=42) and 80/10/10 stratified split.
• Prints TEST accuracy, EER, and confusion matrix.
"""

import pathlib, random, warnings, numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── constants (keep in sync with training) ────────────────────────────────
SEED       = 42
BATCH      = 8
MAXF       = 400                       # crop length
PROJ       = pathlib.Path(__file__).resolve().parents[1]
FEAT_DIR   = PROJ / "data_feats"
META_CSV   = PROJ / "meta.csv"
CKPT       = PROJ / "best_resnet18.pt"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

# ── dataset & collate ─────────────────────────────────────────────────────
class LogMelDS(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        rel, lbl = self.rows[idx]
        x = torch.load(FEAT_DIR / rel.with_suffix(".pt"))["feat"].float()
        return x, int(lbl)

def collate(batch):
    xs, ys = zip(*batch)
    xs = [x if x.shape[-1] <= MAXF else x[:, :, -MAXF:] for x in xs]
    T  = max(x.shape[-1] for x in xs)
    pad = torch.zeros(len(xs), 1, 128, T)
    for i, x in enumerate(xs): pad[i, :, :, :x.shape[-1]] = x
    return pad, torch.tensor(ys)

# ── helper: micro-average EER ─────────────────────────────────────────────
def acc_eer_cm(model, loader):
    model.eval(); ALLy, ALLp = [], []
    with torch.no_grad():
        for x, y in loader:
            ALLp.append(model(x.to(DEVICE)).softmax(1).cpu()); ALLy.append(y)
    y = torch.cat(ALLy); p = torch.cat(ALLp)
    acc = (p.argmax(1) == y).float().mean().item()
    yb  = label_binarize(y, classes=[0,1,2]); eers = []
    for i in range(3):
        if 0 < yb[:, i].sum() < len(y):
            fpr, tpr, _ = roc_curve(yb[:, i], p[:, i]); fnr = 1 - tpr
            idx = np.nanargmin(np.abs(fnr - fpr))
            eers.append((fpr[idx] + fnr[idx]) / 2)
    eer = float(np.mean(eers)) if eers else float("nan")
    cm  = confusion_matrix(y, p.argmax(1), labels=[0,1,2])
    return acc, eer, cm

# ── main evaluation routine ───────────────────────────────────────────────
def main():
    # recreate 10 % subset
    df = pd.read_csv(META_CSV)
    df["path"] = df["path"].apply(pathlib.Path)
    df = (df.groupby("label", group_keys=False, sort=False)
            .apply(lambda g: g.sample(frac=0.1, random_state=SEED))
            .reset_index(drop=True)
            .sample(frac=1, random_state=SEED))
    rows   = df[["path", "label"]].values.tolist()
    labels = df["label"].to_numpy()

    # stratified 80/10/10 split → TEST rows
    sss1 = StratifiedShuffleSplit(1, test_size=.2, random_state=SEED)
    _, tmp = next(sss1.split(rows, labels))
    sss2 = StratifiedShuffleSplit(1, test_size=.5, random_state=SEED)
    _, tst_i = next(sss2.split([rows[i] for i in tmp], labels[tmp]))
    rows_tst = [rows[tmp[i]] for i in tst_i]

    tst_dl = DataLoader(LogMelDS(rows_tst), batch_size=BATCH,
                        shuffle=False, num_workers=0, collate_fn=collate)

    # load model
    model = resnet18(num_classes=3)
    model.conv1 = nn.Conv2d(1,64,7,2,3,bias=False)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.to(DEVICE)

    acc, eer, cm = acc_eer_cm(model, tst_dl)
    print(f"TEST accuracy : {acc:.3%}")
    print(f"TEST EER      : {eer:.3%}")
    print("Confusion matrix (rows=true, cols=pred):\n", cm)

# ── Windows-safe entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
