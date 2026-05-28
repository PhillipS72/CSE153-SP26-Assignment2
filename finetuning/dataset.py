import os

import torchaudio
from torch.utils.data import Dataset, DataLoader

class TrainMelodyDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.audio_paths, self.texts = self.load_audio_files(cfg)

    def load_audio_files(self, cfg):
        folders = [f"{cfg.dataset.path}/processed"]

        for augmentation in cfg.dataset.augmentations:
            if not os.path.exists(f"{cfg.dataset.path}/augmented/{augmentation}"):
                raise NotImplementedError

            folders.append(f"{cfg.dataset.path}/augmented/{augmentation}")

        audio_paths = []
        texts = []
        for folder in folders:
            for source in cfg.dataset.sources:
                for name in os.listdir(f"{folder}/{source}"):
                    audio_paths.append(f"{folder}/{source}/{name}")

                    if cfg.dataset.conditions == "default":
                        texts.append(cfg.dataset.default_desc)
                    else:
                        raise NotImplementedError

        return audio_paths, texts

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        wav, sr = torchaudio.load(audio_path)

        assert sr == self.cfg.dataset.sample_rate

        return wav, self.texts[idx]

def create_dataloader(cfg):
    dataset = TrainMelodyDataset(cfg)
    dataloader = DataLoader(dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers)

    return dataloader