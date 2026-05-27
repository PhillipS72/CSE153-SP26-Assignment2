# Credit: https://github.com/junhaopjlab/Musicgen_finetune.git

import torch
import torch.nn.functional as F
from audiocraft.models import MusicGen
from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout
import tqdm

from model import load_model

class Trainer():
    def __init__(self, cfg):
        self.cfg = cfg

        self.model = self.load_model(cfg)
        self.optimizer = self.load_optimizer(cfg)
        self.device = torch.device(cfg.training.device)

    def load_model(self, cfg):
        if cfg.model.name == "musicgen":
            model = MusicGen.get_pretrained(f"facebook/musicgen-{cfg.model.size}")
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

    def get_condition_tensor(self, attributes):
        null_conditions = ClassifierFreeGuidanceDropout(p=0.0)(attributes)
        conditions = attributes + null_conditions
        tokenized = self.model.lm.condition_provider.tokenize(conditions)
        cfg_conditions = self.model.lm.condition_provider(tokenized)
        return cfg_conditions

    def train(self, dataloader):
        losses = []
        for i in tqdm.tqdm(range(cfg.training.num_epochs)):
            running_loss = 0.0
            total = 0
            for batch in dataloaderr:
                self.optimizer.zero_grad()

                batch = batch.to(self.device)

                with torch.no_grad():
                    codes, _ = self.model.compression_model.encode()

                if self.cfg.dataset.conditions == "default":
                    descs = [cfg.dataset.default_desc for _ in range(codes.shape[0] // 2)]
                else:
                    raise NotImplementedError

                attributes, _ = self.model._prepare_tokens_and_attributes(descs, None)
                condition_tensors = self.get_condition_tensor(attributes)
                lm_output = self.model.lm.compute_predictions(
                    codes=codes, conditions=[], condition_tensors=condition_tensors)

                logits, mask = lm_ouptut.logits[0], lm_output.mask[0].view(-1)
                codes = F.one_hot(codes[0], 2048).float()

                masked_logits = logits.view(-1, 2048)[mask]
                masked_codes = codes.view(-1, 2048)[mask]

                loss = F.cross_entropy(masked_logits, masked_codes)

                running_loss += loss.item() * batch.size(0)
                total += batch.size(0)

            epoch_loss = running_loss / total
            losses.append(epoch_loss)

            print(f"[Epoch {i+1:2d}] loss={epoch_loss:.4f}")

        return losses