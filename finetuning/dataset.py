import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from torch.utils.data import Dataset, DataLoader
from audiocraft.data.audio_dataset import AudioDataset, find_audio_files

class TrainMelodyDataset(Dataset):
    def __init__(self, cfg, mode):
        self.paths = []

        folders = [f"{cfg.dataset.path}/processed"]
        for augmentation in cfg.dataset.augmentations:
            if not os.path.exists(f"{cfg.dataset.path}/augmented/{augmentation}"):
                raise NotImplementedError

            folders.append(f"{cfg.dataset.path}/augmented/{augmentation}")

        for folder in folders:
            for source in cfg.dataset.sources:
                for name in os.listdir(f"{folder}/{source}"):
                    if name.endswith(".wav"):
                        self.paths.append(f"{folder}/{source}/{name}")

    # def __len__(self):
    #     return len(self.paths)

    # def __getitem__(self, idx):


dataset= AudioDataset.from_path("../data/audio/processed/flat", segment_duration=30, num_samples=1000, channels=1)
print(dataset)