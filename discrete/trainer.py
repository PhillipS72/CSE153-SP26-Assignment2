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
from dataset import load_test_data

import tqdm

class Text2MidiTrainer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.training.device)

        self.model, self.audio_tokenizer = self.load_model(cfg)
        self.optimizer = self.load_optimizer(cfg)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

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
        self.inference(0)

        print("Start finetuning a model")
        for epoch in range(self.cfg.training.num_epochs):
            running_loss = 0.0
            total = 0
            for (src_inputs, src_masks, tgt_inputs, tgt_outputs) in tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                self.optimizer.zero_grad()

                src_inputs = src_inputs.to(self.device)
                src_masks = src_masks.to(self.device)
                tgt_inputs = tgt_inputs.to(self.device)
                tgt_outputs = tgt_outputs.to(self.device)

                outputs = self.model(
                    src=src_inputs, src_mask=src_masks,
                    tgt=tgt_inputs, tgt_is_causal=True)

                vocab_size = outputs.size(-1)
                outputs = outputs.reshape(-1, vocab_size)
                tgt_outputs = tgt_outputs.reshape(-1)

                loss = self.criterion(outputs, tgt_outputs)

                running_loss += loss.item()
                total += src_inputs.size(0)

                loss.backward()
                self.optimizer.step()

            epoch_loss = running_loss / (total * self.cfg.dataset.max_audio_len)

            print(f"[Epoch {epoch+1:2d}] Finished Epoch (loss={epoch_loss:.4f})")
            print(f"[Epoch {epoch+1:2d}] Runninng Inferences")

            if epoch + 1 % 10 == 0:
                self.inference(epoch+1)

        os.makedirs(self.cfg.model.path, exist_ok=True)
        torch.save(self.model.state_dict(), f"{self.cfg.model.path}/{self.cfg.name}.pth")

    def inference(self, epoch):
        self.model.eval()

        output_path = f"{self.cfg.inference.path}/{self.cfg.name}/{epoch}"
        os.makedirs(output_path, exist_ok=True)

        src_inputs, src_masks = load_test_data(self.cfg)
        src_inputs = src_inputs.to(self.device)
        src_masks = src_masks.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(src_inputs, src_masks, max_len=self.cfg.inference.output_len)

        for idx in range(outputs.size(0)):
            output_list = outputs[idx].tolist()
            generated_midi = self.audio_tokenizer.decode(output_list)
            generated_midi.dump_midi(f"{output_path}/{idx+1:03d}.mid")

        self.model.train()
