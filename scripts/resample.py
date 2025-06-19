import torchaudio, glob, os
for path in glob.glob(r'data\youtube_raw\*.wav'):
    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    torchaudio.save(path, wav, 16000)