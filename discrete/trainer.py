# Credit:
# - https://github.com/junhaopjlab/Musicgen_finetune.git
# - https://github.com/Beinabih/Unconditional-MusicGen-Trainer.git

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from text2midi.model.transformer_model import Transformer

import tqdm

class Text2MidiTrainer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.training.device)

        self.model, self.audio_tokenizer = self.load_model(cfg)
        self.optimizer = self.load_optimizer(cfg)
        self.criterion = nn.CrossEntropyLoss()

    def load_model(self, cfg):
        if cfg.model.name == "text2midi":
            repo_id = "amaai-lab/text2midi"
            model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
            tokenizer_path = hf_hub_download(repo_id=repo_id, filename="vocab_remi.pkl")

            with open(tokenizer_path, "rb") as f:
                audio_tokenizer = pickle.load(f)

            vocab_size = len(audio_tokenizer)
            model = Transformer(vocab_size, 768, 8, 2048, 18, 1024, False, 8, device=self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))

        else:
            raise NotImplementedError

        return model, audio_tokenizer

    def load_optimizer(self, cfg):
        if cfg.training.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg.training.lr, betas=cfg.training.betas, weight_decay=cfg.training.weight_decay)
        else:
            raise NotImplementedError

        return optimizer

    def train(self, dataloader):

        print("Start finetuning a model")
        for epoch in range(self.cfg.training.num_epochs):
            running_loss = 0.0
            total = 0
            for (events, tokens, masks) in tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                self.optimizer.zero_grad()

                outputs = self.model(src=tokens, src_mask=masks, tgt=events)
                print(outputs.size())
                loss = self.criterion(masked_logits, masked_codes)

                running_loss += loss.item() * wav.size(0)
                total += wav.size(0)

                loss.backward()
                self.optimizer.step()

            epoch_loss = running_loss / total

            print(f"[Epoch {epoch+1:2d}] Finished Epoch (loss={epoch_loss:.4f})")
            print(f"[Epoch {epoch+1:2d}] Runninng Inferences")
            self.inference(epoch+1)

        os.makedirs(self.cfg.model.path, exist_ok=True)
        torch.save(self.model.lm.state_dict(), f"{self.cfg.model.path}/{self.cfg.name}.pth")

    def inference(self, epoch):
        self.model.lm.eval()

        num_samples = self.cfg.inference.num_samples
        for duration in self.cfg.inference.durations:
            self.model.set_generation_params(duration=duration)

            if self.cfg.dataset.conditions == "default":
                texts = [self.cfg.dataset.default_desc for _ in range(num_samples)]
            elif self.cfg.dataset.conditions == "description":
                texts = []
                for file_name in os.listdir(self.cfg.dataset.inference):
                    with open(f"{self.cfg.dataset.inference}/{file_name}", "r") as f:
                        texts.append(f.read())
            else:
                raise NotImplementedError

            with torch.no_grad():
                wavs = self.model.generate(texts)

            os.makedirs(f"{self.cfg.inference.path}/{self.cfg.name}/{epoch}/{duration}", exist_ok=True)
            for idx, wav in enumerate(wavs):
                audio_write(
                    f"{self.cfg.inference.path}/{self.cfg.name}/{epoch}/{duration}/{idx+1:03d}",
                    wav.cpu(), self.model.sample_rate, strategy="loudness", loudness_compressor=True)

        self.model.lm.train()
