from omegaconf import DictConfig, OmegaConf

def testfct(cfg):
    print(OmegaConf.to_yaml(cfg))
