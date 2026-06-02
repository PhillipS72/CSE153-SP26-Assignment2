import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import create_dataloader
from trainer import Text2MidiTrainer

@hydra.main(version_base=None, config_path="configs", config_name="")
def main(cfg):
    dataloader = create_dataloader(cfg)

    if cfg.model.name == "text2midi":
        trainer = Text2MidiTrainer(cfg)
    else:
        raise NotImplementedError

    trainer.train(dataloader)


if __name__ == "__main__":
    main()