import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler


class AudioDataset(Dataset):
    def __init__(self, df, background_audio=None):
        self.labels = df["label"].astype(np.int64)
        self.filepaths = df["file"]
        self.label_count = 12
        self.sr = 16000
        if background_audio is not None:
            self.background_audio = [librosa.load(f, sr=self.sr)[0] for f in background_audio["file"].tolist()]
        else:
            self.background_audio = None

    def get_background_audio_sample(self):
        wav = np.random.choice(self.background_audio)
        start = np.random.randint(0, len(wav) - self.sr)
        return wav[start:start + self.sr]

    def mix(self, wavs, ratio=0.25):
        wav = np.zeros(self.sr)
        for w in wavs:
            scale = np.random.random() * ratio
            wav += w * scale
        if wavs:
            return wav / len(wavs)
        else:
            return wav

    def timeshift(self, wav, max_shift=0.2):
        shift = np.random.uniform(-max_shift, max_shift)
        shift = int(len(wav) * shift)
        if shift > 0:
            padded = np.pad(wav, (shift, 0), "constant")
            return padded[:len(wav)]
        else:
            padded = np.pad(wav, (0, -shift), "constant")
            return padded[-len(wav):]

    def generate_silence(self):
        count = np.random.randint(1, 3)
        background_audio = [self.get_background_audio_sample() for _ in range(count)]
        silence = self.mix(background_audio)
        return silence

    def augment(self, wav):
        if np.random.choice([True, False]):
            wav = self.timeshift(wav)
        if np.random.choice([True, False]):
            scale = np.random.uniform(0.75, 1.25)
            noise = self.generate_silence()
            wav = self.pad(wav * scale) + noise
        elif np.random.choice([True, False]):
            scale = np.random.uniform(0.75, 1.25)
            wav = self.pad(wav * scale)
        return wav

    def pad(self, wav):
        return librosa.util.pad_center(wav, self.sr)

    def __getitem__(self, index):
        label = self.labels[index]
        if self.background_audio:
            if self.filepaths[index] == "silence":
                wav = self.generate_silence()
            else:
                wav, _ = librosa.load(self.filepaths[index], sr=self.sr)
                wav = self.augment(wav)
        else:
            wav, _ = librosa.load(self.filepaths[index], sr=self.sr)

        wav = librosa.util.normalize(wav)
        wav = librosa.util.pad_center(wav, self.sr)
        wav = torch.from_numpy(wav).float().unsqueeze(0)
        return {"sound": wav, "label": label}

    def __len__(self):
        return len(self.filepaths)


class AudioPredictionDataset(Dataset):
    def __init__(self, files):
        self.filepaths = files["file"]
        self.sr = 16000

    def __getitem__(self, index):
        wav, _ = librosa.load(self.filepaths[index], sr=self.sr)
        filename = self.filepaths[index]
        wav = librosa.util.normalize(wav)
        wav = librosa.util.pad_center(wav, self.sr)
        wav = torch.from_numpy(wav).float().unsqueeze(0)
        return {"sound": wav, "filename": filename}

    def __len__(self):
        return len(self.filepaths)


def get_sampler(dataset, sample_count=None):
    weights = [0] * dataset.label_count
    for i in dataset.labels:
        weights[i] += 1
    weights = [1/i for i in weights]
    sample_weight = torch.Tensor([weights[i] for i in dataset.labels]).double()
    sample_count = sample_count if sample_count else len(sample_weight)
    return WeightedRandomSampler(sample_weight, sample_count)