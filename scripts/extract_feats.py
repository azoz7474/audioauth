#!/usr/bin/env python
# scripts/extract_feats.py
# GPU-aware feature extractor: creates .pt files with log-Mel tensors

import argparse, csv, pathlib, warnings, torch, torchaudio, tqdm

# ── configuration ────────────────────────────────────────────────
SR        = 16_000
N_MELS    = 128
N_FFT     = 1024
HOP       = 256
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
WORKERS   = 0            # keep 0 on Windows/GPU to avoid spawn issues
TORCH_CPU_THREADS = 2    # throttle CPU FFT threads

torch.set_num_threads(TORCH_CPU_THREADS)
warnings.filterwarnings("ignore", category=UserWarning)

# ── CLI ──────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(description="Cache log-Mel features (.pt)")
ap.add_argument("--meta",  default="C:/audioauth/meta.csv",
                help="CSV with columns path,label (relative to project root)")
ap.add_argument("--out",   default="C:/audioauth/data_feats",
                help="Output directory for .pt files")
args = ap.parse_args()

ROOT     = pathlib.Path("C:/audioauth")
FEAT_DIR = pathlib.Path(args.out); FEAT_DIR.mkdir(exist_ok=True)

# ── transforms (on GPU if available) ────────────────────────────
mel   = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP,
            win_length=N_FFT, n_mels=N_MELS).to(DEVICE)
amp2db = torchaudio.transforms.AmplitudeToDB(stype="power").to(DEVICE)

def wav2logmel(path: pathlib.Path):
    wav, sr = torchaudio.load(path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    wav = wav.mean(0, keepdim=True).to(DEVICE)             # mono → (1,T)
    with torch.no_grad():
        feat = amp2db(mel(wav))         # (1,128,T)
        feat = (feat + 80.0) / 80.0     # normalise 0-1
    return feat.cpu()                   # move back to CPU for saving

def process_row(row):
    rel = pathlib.Path(row["path"])
    dst = FEAT_DIR / rel.with_suffix(".pt")
    if dst.exists(): return False       # already done
    try:
        feat = wav2logmel(ROOT / rel)
        torch.save({"feat": feat, "label": int(row["label"])}, dst)
        return True
    except Exception as e:
        warnings.warn(f"[SKIP] {rel}: {e}")
        return False

# ── main loop ───────────────────────────────────────────────────
with open(args.meta, newline='', encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

done = 0
for row in tqdm.tqdm(rows, desc="Extracting", ncols=80):
    done += process_row(row)

print(f"✓ {done} new feature files written to {FEAT_DIR}")