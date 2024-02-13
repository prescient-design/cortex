from typing import MutableMapping

import wandb
from omegaconf import DictConfig, OmegaConf

import cortex


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


def flatten_config(d: DictConfig, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
