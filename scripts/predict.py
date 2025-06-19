#!/usr/bin/env python
# scripts/predict.py
# Single-file or batch inference – GPU safe (no device mismatch)

import argparse, json, pathlib, torch, torchaudio
from torchvision.models import resnet18

# ─── paths ────────────────────────────────────────────────────────────────
PROJ   = pathlib.Path(__file__).resolve().parents[1]
CKPT   = PROJ / "best_resnet18.pt"
THR_JS = PROJ / "tamper_threshold.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── load threshold ───────────────────────────────────────────────────────
theta = json.load(open(THR_JS, encoding="utf-8"))["theta"]

# ─── load model ───────────────────────────────────────────────────────────
model = resnet18(num_classes=3)
model.conv1 = torch.nn.Conv2d(1, 64, 7, 2, 3, bias=False)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.to(DEVICE).eval()

# ─── GPU transforms (window lives on same device) ─────────────────────────
mel_gpu = torchaudio.transforms.MelSpectrogram(
            sample_rate=16_000,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=128).to(DEVICE)

amp2db_gpu = torchaudio.transforms.AmplitudeToDB(stype="power").to(DEVICE)

# ─── helper: log-Mel (400 frames) on GPU ──────────────────────────────────
def logmel_400(wav, sr):
    if sr != 16_000:
        wav = torchaudio.functional.resample(wav, sr, 16_000)
    wav = wav.mean(0, keepdim=True).to(DEVICE)             # mono → GPU
    with torch.no_grad():
        feat = amp2db_gpu(mel_gpu(wav))                    # GPU → GPU
        feat = (feat + 80.0) / 80.0
    # crop/pad to last 400 frames
    if feat.shape[-1] >= 400:
        feat = feat[:, :, -400:]
    else:
        feat = torch.nn.functional.pad(feat,
                                        (0, 400 - feat.shape[-1]))
    return feat                                            # GPU tensor

# ─── prediction routine ───────────────────────────────────────────────────
def predict_one(fp: pathlib.Path):
    wav, sr = torchaudio.load(fp)
    x = logmel_400(wav, sr).unsqueeze(0)                   # [1,1,128,400] GPU
    with torch.no_grad():
        prob = model(x).softmax(1)[0]
    p_tamper = prob[2].item()
    verdict = "tampered" if p_tamper >= theta else "not_tampered"
    print(f"{str(fp):60s}  →  {verdict:<13s}  (p_tamper={p_tamper:.3f}, θ={theta:.3f})")
# ─── CLI ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tamper detector (GPU safe)")
    ap.add_argument("files", nargs="+", help="Audio files (wav/ogg/opus)")
    args = ap.parse_args()

    for f in args.files:
        predict_one(pathlib.Path(f))
