import os

from anticipation.convert import midi_to_events

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class TrainMelodyDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_len = cfg.dataset.max_len
        self.midi_paths, self.texts = self.load_midi_files(cfg)

    def load_midi_files(self, cfg):
        skips = cfg.dataset.get("skips", [])

        midi_paths = []
        texts = []
        for source in cfg.dataset.sources:
            for name in os.listdir(f"{cfg.dataset.path}/{source}"):
                if f"{source}/{name}" in skips:
                    continue

                midi_paths.append(f"{cfg.dataset.path}/{source}/{name}")

                if cfg.dataset.conditions == "default":
                    texts.append(cfg.dataset.default_desc)
                elif cfg.dataset.conditions == "description":
                    text_name = name.replace(".mid", ".txt")
                    with open(f"{cfg.dataset.description}/{source}/{text_name}", "r") as f:
                        text = f.read()
                    texts.append(text)
                else:
                    raise NotImplementedError

        return midi_paths, texts

    def __len__(self):
        return len(self.midi_paths)

    def __getitem__(self, idx):
        midi_path = self.midi_paths[idx]

        events = midi_to_events(midi_path)
        events = F.pad(torch.tensor(events, dtype=torch.long), pad=(0, self.max_len))

        return events, self.texts[idx]

def create_dataloader(cfg):
    dataset = TrainMelodyDataset(cfg)
    dataloader = DataLoader(dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=True)

    return dataloader