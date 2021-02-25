import hydra
from omegaconf import DictConfig, OmegaConf

from testfct import testfct

@hydra.main(config_path="divae_configs", config_name="config")
def my_app(cfg : DictConfig) -> None:

    # assert cfg.db1.loompa == 10          # attribute style access
    # assert cfg["db1"]["loompa"] == 10    # dictionary style access
    # assert cfg.db1.zippity == 10         # Value interpolation
    # assert isinstance(cfg.db1.zippity, int)  # Value interpolation type
    # assert cfg.db1.do == "oompa 10"      # string interpolation

    # # cfg.db1.waldo                        # raises an exception
    # print(OmegaConf.to_yaml(cfg))
    testfct(cfg)
    
if __name__=="__main__":
    my_app()