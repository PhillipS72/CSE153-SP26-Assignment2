import hydra
from omegaconf import DictConfig, OmegaConf

from model import load_model
from dataset import create_dataloader

@hydra.main(version_base=None, config_path="configs", config_name="")
def main(cfg):
    dataloader = create_dataloader(cfg)



if __name__ == "__main__":
    main()