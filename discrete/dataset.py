import os
import pickle

from anticipation.convert import midi_to_events

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import T5Tokenizer
from huggingface_hub import hf_hub_download

class TrainMelodyDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_audio_len = cfg.dataset.max_audio_len
        self.max_text_len = cfg.dataset.max_text_len

        self.midi_paths, self.texts = self.load_midi_files(cfg)

        self.text_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

        repo_id = "amaai-lab/text2midi"
        tokenizer_path = hf_hub_download(repo_id=repo_id, filename="vocab_remi.pkl")
        with open(tokenizer_path, "rb") as f:
            self.audio_tokenizer = pickle.load(f)

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
        events = [1] + self.audio_tokenizer.encode(midi_path).ids
        events = torch.tensor(events, dtype=torch.long)

        tgt_mask = torch.ones(events.size(0) - 1)
        tgt = F.pad(events, pad=(0, self.max_audio_len))
        tgt_mask = F.pad(tgt_mask, pad=(0, self.max_audio_len))

        text = self.texts[idx]
        text = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        src = F.pad(text.input_ids, pad=(0, self.max_text_len))
        src_mask = F.pad(text.attention_mask, pad=(0, self.max_text_len))

        print(tgt.tolist())
        return tgt, tgt_mask, src, src_mask

def create_dataloader(cfg):
    dataset = TrainMelodyDataset(cfg)
    dataloader = DataLoader(dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=True)

    return dataloader