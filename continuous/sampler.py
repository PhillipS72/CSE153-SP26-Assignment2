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

class MusicGenSampler():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.model.device)
        self.conditional_model, self.unconditional_model, self.pretrained_model = self.load_model(cfg)

    def load_model(self, cfg):
        if cfg.model.name == "musicgen":
            conditional_model = MusicGen.get_pretrained(f"facebook/musicgen-{cfg.model.size}")
            state_dict = torch.load(f"{cfg.model.conditional}/{cfg.model.best_epoch}.pth")
            conditional_model.lm.load_state_dict(state_dict)
            conditional_model.lm.eval()

            unconditional_model = MusicGen.get_pretrained(f"facebook/musicgen-{cfg.model.size}")
            state_dict = torch.load(f"{cfg.model.unconditional}/{cfg.model.best_epoch}.pth")
            unconditional_model.lm.load_state_dict(state_dict)
            unconditional_model.lm.eval()

            pretraiend_model = MusicGen.get_pretrained(f"facebook/musicgen-{cfg.model.size}")
            pretraiend_model.lm.eval()

        else:
            raise NotImplementedError

        return conditional_model, unconditional_model, pretraiend_model

    def sample(self):
        texts = []
        for file_name in os.listdir(f"{self.cfg.evaluation.path}/descriptions"):
            with open(f"{self.cfg.evaluation.path}/descriptions/{file_name}", "r") as f:
                texts.append(f.read())

        num_samples = self.cfg.evaluation.num_samples

        models = [
            (self.pretrained_model, "pretrained"),
            (self.conditional_model, "conditional"),
            (self.unconditional_model, "unconditional")]

        for model, model_name in models:
            print(f"Running inference with {model_name}")
            if model_name == "unconditional":
                texts = ["" for _ in texts]

            self.inference(model, model_name, texts, num_samples)

    def inference(self, model, model_name, texts, num_samples):
        output_path = f"{self.cfg.evaluation.path}/{model_name}"
        os.makedirs(output_path, exist_ok=True)

        if len(os.listdir(output_path)) == num_samples:
            return

        model.lm = model.lm.to(self.device)
        model.set_generation_params(duration=self.cfg.evaluation.duration)

        with torch.no_grad():
            wavs = model.generate(texts)

        for idx, wav in enumerate(wavs):
            audio_write(f"{output_path}/{idx+1:03d}", wav.cpu(), model.sample_rate,
                strategy="loudness", loudness_compressor=True)

        model.lm.cpu()
