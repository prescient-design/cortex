import s3fs
import yaml
from omegaconf import DictConfig, OmegaConf


def load_hydra_config(cfg_fpath: str) -> DictConfig:
    """
    Load saved OmegaConf object from YAML file.
    Args:
        cfg_fpath: path to YAML file. Must point to local file or S3 object.
    Returns:
        OmegaConf object
    """
    if "s3://" in cfg_fpath:
        s3 = s3fs.S3FileSystem()
        with s3.open(cfg_fpath, "rb") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    else:
        with open(cfg_fpath, "rb") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    return OmegaConf.create(cfg)
