# Credit:
# - https://github.com/junhaopjlab/Musicgen_finetune.git
# - https://github.com/Beinabih/Unconditional-MusicGen-Trainer.git

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

from transformers import get_scheduler
import tqdm

class MusicGenTrainer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.training.device)

        self.model = self.load_model(cfg)
        self.optimizer = self.load_optimizer(cfg)
        self.criterion = nn.CrossEntropyLoss()

    def load_model(self, cfg):
        if cfg.model.name == "musicgen":
            model = MusicGen.get_pretrained(f"facebook/musicgen-{cfg.model.size}", device=self.device)
            model.lm.train()
            model.lm = model.lm.float()

        else:
            raise NotImplementedError

        return model

    def load_optimizer(self, cfg):
        if cfg.training.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                self.model.lm.parameters(),
                lr=cfg.training.lr, betas=cfg.training.betas, weight_decay=cfg.training.weight_decay)
        else:
            raise NotImplementedError

        return optimizer

    def train(self, dataloader):
        self.inference(0)

        print("Start finetuning a model")
        for epoch in range(self.cfg.training.num_epochs):
            running_loss = 0.0
            total = 0
            for wav, texts in tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                self.optimizer.zero_grad()

                wav = wav.to(self.device)
                with torch.no_grad():
                    codes, scale = self.model.compression_model.encode(wav)

                attributes, _ = self.model._prepare_tokens_and_attributes(texts, prompt=None)
                tokenized = self.model.lm.condition_provider.tokenize(attributes)
                conditions = self.model.lm.condition_provider(tokenized)

                lm_output = self.model.lm.compute_predictions(
                    codes=codes, conditions=[], condition_tensors=conditions)


                logits = lm_output.logits[0]
                mask = lm_output.mask[0].view(-1)

                codes = F.one_hot(codes[0], 2048).float()

                masked_logits = logits.view(-1, 2048)[mask]
                masked_codes = codes.view(-1, 2048)[mask]

                loss = self.criterion(masked_logits, masked_codes)

                running_loss += loss.item() * wav.size(0)
                total += wav.size(0)

                loss.backward()
                self.optimizer.step()

            epoch_loss = running_loss / total

            print(f"[Epoch {epoch+1:2d}] Finished Epoch (loss={epoch_loss:.4f})")
            print(f"[Epoch {epoch+1:2d}] Runninng Inferences")
            self.inference(epoch+1)

            os.makedirs(f"{self.cfg.model.path}/{self.cfg.name}", exist_ok=True)
            torch.save(self.model.lm.state_dict(), f"{self.cfg.model.path}/{self.cfg.name}/{epoch+1}.pth")

        os.makedirs(self.cfg.model.path, exist_ok=True)

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
