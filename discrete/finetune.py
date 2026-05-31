import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import create_dataloader, TrainMelodyDataset
from trainer import MusicGenTrainer

@hydra.main(version_base=None, config_path="configs", config_name="")
def main(cfg):
    dataloader = create_dataloader(cfg)

    if cfg.model.name == "midillm":
        trainer = MusicLLMTrainer(cfg)
    else:
        raise NotImplementedError

    # trainer.train(dataloader)


if __name__ == "__main__":
    main()