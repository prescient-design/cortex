import wandb
from omegaconf import DictConfig, OmegaConf

import cortex
from cortex.utils import flatten_config


def wandb_setup(cfg: DictConfig):
    wandb.login(host=cfg.wandb_host)
    wandb.init(
        project=cfg.project_name,
        mode=cfg.wandb_mode,
        group=cfg.exp_name,
    )
    cfg["job_name"] = wandb.run.name
    cfg["__version__"] = cortex.__version__
    log_cfg = flatten_config(OmegaConf.to_container(cfg, resolve=True), sep="/")
    wandb.config.update(log_cfg)
