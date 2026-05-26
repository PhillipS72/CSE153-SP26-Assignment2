import hydra
from omegaconf import DictConfig, OmegaConf

from model import load_model

@hydra.main(version_base=None, config_path="configs", config_name="")
def main(cfg):


if __name__ == "__main__":
    main()