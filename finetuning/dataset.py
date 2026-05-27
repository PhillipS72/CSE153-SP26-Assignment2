import os

from torch.utils.data import Dataset, DataLoader
from audiocraft.data.audio_dataset import AudioDataset, find_audio_files

def load_audio_files(cfg):
    folders = [f"{cfg.dataset.path}/processed"]

    for augmentation in cfg.dataset.augmentations:
        if not os.path.exists(f"{cfg.dataset.path}/augmented/{augmentation}"):
            raise NotImplementedError

        folders.append(f"{cfg.dataset.path}/augmented/{augmentation}")

    metas = []
    for folder in folders:
        for source in cfg.dataset.sources:
            metas += find_audio_files(f"{folder}/{source}")

    return metas

def create_dataloader(cfg):
    metas = load_audio_files(cfg)
    dataset = AudioDataset(metas, sample_rate=cfg.dataset.sample_rate)
    dataloader = DataLoader(dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        collate_fn=dataset.collater)

    return dataloader