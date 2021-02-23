import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_name="config_hydra")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__=="__main__":
    my_app()