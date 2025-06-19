#!/usr/bin/env python
# Train on 100 % data – GPU – Windows-safe

import pathlib, random, time, warnings, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ────────── hyper-params ──────────
SEED, BATCH, EPOCHS = 42, 32, 10
MAXF   = 400                       # 400 frames ≈ 6.4 s
PROJ   = pathlib.Path(__file__).resolve().parents[1]
FEAT_D = PROJ / "data_feats"
META_C = PROJ / "meta.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

# ────────── data utilities ─────────
def specaugment(x):
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
    T  = max(x.shape[-1] for x in xs)
    pad = torch.zeros(len(xs), 1, 128, T)
    for i, x in enumerate(xs): pad[i, :, :, :x.shape[-1]] = x
    return pad, torch.tensor(ys)

def evaluate(model, loader):
    model.eval(); Y, P = [], []
    with torch.no_grad():
        for x, y in loader:
            P.append(model(x.to(DEVICE)).softmax(1).cpu()); Y.append(y)
    y = torch.cat(Y); p = torch.cat(P)
    acc = (p.argmax(1) == y).float().mean().item()
    eers = []
    yb = label_binarize(y, classes=[0,1,2])
    for i in range(3):
        if 0 < yb[:, i].sum() < len(yb):
            fpr, tpr, _ = roc_curve(yb[:, i], p[:, i])
            fnr = 1 - tpr
            idx = np.nanargmin(np.abs(fnr-fpr))
            eers.append((fpr[idx]+fnr[idx])/2)
    eer = float(np.mean(eers)) if eers else float("nan")
    cm  = confusion_matrix(y, p.argmax(1), labels=[0,1,2])
    return acc, eer, cm

# ────────── main routine ───────────
def main():
    df = pd.read_csv(META_C); df["path"] = df["path"].apply(pathlib.Path)
    rows, labels = df[["path","label"]].values.tolist(), df["label"].to_numpy()

    sss = StratifiedShuffleSplit(1, test_size=0.2, random_state=SEED)
    tr_idx, tmp_idx = next(sss.split(rows, labels))
    sss2 = StratifiedShuffleSplit(1, test_size=0.5, random_state=SEED)
    val_i, tst_i = next(sss2.split([rows[i] for i in tmp_idx], labels[tmp_idx]))

    rows_tr  = [rows[i] for i in tr_idx]
    rows_val = [rows[tmp_idx[i]] for i in val_i]
    rows_tst = [rows[tmp_idx[i]] for i in tst_i]

    tr_dl  = DataLoader(LogMelDS(rows_tr,  aug=True),
                        batch_size=BATCH, shuffle=True,
                        num_workers=8, pin_memory=True, collate_fn=collate)
    val_dl = DataLoader(LogMelDS(rows_val),
                        batch_size=BATCH, shuffle=False,
                        num_workers=4, pin_memory=True, collate_fn=collate)
    tst_dl = DataLoader(LogMelDS(rows_tst),
                        batch_size=BATCH, shuffle=False,
                        num_workers=4, pin_memory=True, collate_fn=collate)

    model = resnet18(num_classes=3)
    model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    model.to(DEVICE)
    opt  = torch.optim.AdamW(model.parameters(), 1e-4, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_eer, stale, patience = 1.0, 0, 2

    for ep in range(1, EPOCHS+1):
        model.train(); run = 0; t0 = time.time()
        for x, y in tqdm(tr_dl, desc=f"Epoch {ep:02d}", ncols=80):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item() * len(y)
        v_acc, v_eer, _ = evaluate(model, val_dl)
        print(f"E{ep:02d} train {run/len(rows_tr):.4f} | val acc {v_acc:.3%} | "
              f"val EER {v_eer:.3%} | {int(time.time()-t0)} s")
        if v_eer < best_eer:
            best_eer, stale = v_eer, 0
            torch.save(model.state_dict(), PROJ/"best_resnet18.pt")
        else:
            stale += 1
            if stale >= patience:
                print("Early stop (no val-EER improvement)."); break

    model.load_state_dict(torch.load(PROJ/"best_resnet18.pt"))
    t_acc, t_eer, cm = evaluate(model, tst_dl)
    print("\nTEST acc", f"{t_acc:.3%}", "| EER", f"{t_eer:.3%}")
    print("Confusion matrix\n", cm)

# ────────── Windows-safe entry ─────
if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
