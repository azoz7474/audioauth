#!/usr/bin/env python
"""
Builds meta.csv for the Audio Authenticity project.

meta.csv structure
------------------
path,label

* path   : relative path from project root to an audio file
* label  : 0 = genuine speech  (YouTube raw)
           1 = spoof / fake    (ASVspoof 2019 LA + PA)
           2 = tampered        (locally-edited YouTube files)

The script walks the ./data/ tree and supports both .wav and .flac.
"""

import csv
import pathlib

# ---------- Configuration ----------------------------------------------------

AUDIO_EXTS = (".wav", ".flac")        # file types to include
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]   # C:\audioauth
DATA_ROOT    = PROJECT_ROOT / "data"
OUT_CSV      = PROJECT_ROOT / "meta.csv"

# ---------- Build table -------------------------------------------------------

rows_written = 0

with OUT_CSV.open("w", newline="", encoding="utf-8") as fp:
    wr = csv.writer(fp)
    wr.writerow(["path", "label"])    # header row

    for audio in DATA_ROOT.rglob("*"):
        if audio.suffix.lower() not in AUDIO_EXTS:
            continue                                   # skip non-audio

        # Decide label from folder name
        if "youtube_edit" in audio.parts:
            label = 2
        elif "youtube_raw" in audio.parts:
            label = 0
        else:
            label = 1                                  # ASVspoof anything

        relpath = audio.relative_to(PROJECT_ROOT)      # store portable path
        wr.writerow([str(relpath), label])
        rows_written += 1

print(f"✓  meta.csv written to {OUT_CSV}")
print(f"   {rows_written} audio files indexed:")
print("   ── label 0 (genuine):  ",
      sum(1 for _ in open(OUT_CSV) if ",0" in _))
print("   ── label 1 (spoof)  :  ",
      sum(1 for _ in open(OUT_CSV) if ",1" in _))
print("   ── label 2 (tamper) :  ",
      sum(1 for _ in open(OUT_CSV) if ",2" in _))
