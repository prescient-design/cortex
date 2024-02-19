from typing import MutableMapping

import wandb
from omegaconf import DictConfig, OmegaConf

import cortex


def wandb_setup(cfg: DictConfig):
    if not hasattr(cfg, "wandb_host"):
        cfg["wandb_host"] = "https://api.wandb.ai"

    if not hasattr(cfg, "wandb_mode"):
        cfg["wandb_mode"] = "online"

    if not hasattr(cfg, "project_name"):
        cfg["project_name"] = "cortex"

    if not hasattr(cfg, "exp_name"):
        cfg["exp_name"] = "default_group"

    wandb.login(host=cfg.wandb_host)

    wandb.init(
        project=cfg.project_name,
        mode=cfg.wandb_mode,
        group=cfg.exp_name,
    )
    cfg["job_name"] = wandb.run.name
    cfg["__version__"] = cortex.__version__
    log_cfg = flatten_config(OmegaConf.to_container(cfg, resolve=True))
    wandb.config.update(log_cfg)


def flatten_config(d: DictConfig, parent_key="", sep="/"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
