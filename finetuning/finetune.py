import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import create_dataloader
from trainer import MusicGenTrainer

@hydra.main(version_base=None, config_path="configs", config_name="")
def main(cfg):
    dataloader = create_dataloader(cfg)

    if cfg.model.name == "musicgen":
        trainer = MusicGenTrainer(cfg)
    else:
        raise NotImplementedError

    trainer.train(dataloader)


if __name__ == "__main__":
    main()