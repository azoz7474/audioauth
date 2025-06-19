#!/usr/bin/env python
"""
calibrate_threshold.py
----------------------
Compute an equal-error-rate threshold θ for the tamper class (label 2)
using the SAME validation split defined in train_resnet.py.
Outputs tamper_threshold.json  = {"theta": 0.1234}
"""

import json, pathlib, random, warnings, numpy as np, pandas as pd, torch
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_curve
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ─── constants (keep in sync with training script) ─────────────────────────
SEED       = 42
BATCH      = 8
MAXF       = 400
PROJ       = pathlib.Path(__file__).resolve().parents[1]
FEAT_DIR   = PROJ / "data_feats"
META_CSV   = PROJ / "meta.csv"
CKPT       = PROJ / "best_resnet18.pt"
OUT_JSON   = PROJ / "tamper_threshold.json"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

# ─── dataset & collate (same as training) ─────────────────────────────────
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

# ─── rebuild the SAME 30 % sample & split ─────────────────────────────────
df = pd.read_csv(META_CSV)
df["path"] = df["path"].apply(pathlib.Path)

def sample_30(g):
    if g.name in (0, 2):         # keep all genuine & tampered
        return g
    return g.sample(frac=0.30, random_state=SEED)

df = (df.groupby("label", group_keys=False, sort=False)
        .apply(sample_30)
        .reset_index(drop=True)
        .sample(frac=1, random_state=SEED))

rows   = df[["path", "label"]].values.tolist()
labels = df["label"].to_numpy()

# 80 / 10 / 10 stratified split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, tmp_idx = next(sss.split(rows, labels))

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
val_idx, _ = next(sss2.split([rows[i] for i in tmp_idx], labels[tmp_idx]))

rows_val = [rows[tmp_idx[i]] for i in val_idx]
val_dl   = DataLoader(LogMelDS(rows_val), batch_size=BATCH,
                      shuffle=False, num_workers=0, collate_fn=collate)

print(f"Validation size: {len(rows_val)} clips "
      f"(labels: {np.bincount([lbl for _,lbl in rows_val], minlength=3)})")

# ─── load model ───────────────────────────────────────────────────────────
model = resnet18(num_classes=3)
model.conv1 = torch.nn.Conv2d(1, 64, 7, 2, 3, bias=False)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.to(DEVICE).eval()

# ─── collect tamper probabilities ─────────────────────────────────────────
tamper_scores, y_true = [], []
with torch.no_grad():
    for x, y in tqdm(val_dl, desc="Scoring"):
        probs = model(x.to(DEVICE)).softmax(1)[:, 2]   # class-2 prob
        tamper_scores.append(probs.cpu())
        y_true.append((y == 2).int())                  # 1 for tampered

tamper_scores = torch.cat(tamper_scores).numpy()
y_true        = torch.cat(y_true).numpy()

# ─── find EER threshold θ ─────────────────────────────────────────────────
fpr, tpr, thr = roc_curve(y_true, tamper_scores)
fnr = 1 - tpr
idx = np.nanargmin(np.abs(fnr - fpr))
theta = float(thr[idx])

print(f"\nEqual-Error point: FPR = FNR = {fpr[idx]:.4%} at θ = {theta:.4f}")

# ─── save to JSON ─────────────────────────────────────────────────────────
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump({"theta": theta}, f, indent=2, ensure_ascii=False)

print(f"Threshold saved to {OUT_JSON}")
