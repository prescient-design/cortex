import logging
import os
import random
import warnings

import hydra
import lightning as L
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.trainer.supporters import CombinedLoader

from cortex.logging import wandb_setup


@hydra.main(config_path="../config/hydra", config_name="train_cortex_model", version_base=None)
def main(cfg):
    """
    general setup
    """
    random.seed(
        None
    )  # make sure random seed resets between Hydra multirun jobs for random job-name generation

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(cfg.warnings_filter)
            ret_val = execute(cfg)
    except Exception as err:
        ret_val = float("NaN")
        logging.exception(err)

    wandb.finish()  # necessary to log Hydra multirun output to different jobs
    return ret_val


def execute(cfg):
    """
    instantiate and train a multitask neural tree
    """

    trainer = hydra.utils.instantiate(cfg.trainer)

    # seeding
    if cfg.seed is None:
        cfg.seed = random.randint(0, 1000)

    print(trainer.global_rank, trainer.local_rank, trainer.node_rank)
    total_rank = trainer.global_rank + trainer.local_rank + trainer.node_rank
    cfg.seed = cfg.seed + total_rank
    L.seed_everything(seed=cfg.seed, workers=True)

    # logging
    if trainer.global_rank == 0:
        wandb_setup(cfg)  # wandb init, set cfg job name to wandb job name
    cfg = OmegaConf.to_container(cfg, resolve=True)  # Resolve config interpolations
    cfg = DictConfig(cfg)
    if trainer.global_rank == 0:
        print(OmegaConf.to_yaml(cfg))
    trainer.logger = L.pytorch.loggers.WandbLogger()

    # checkpointing
    try:
        ckpt_file = os.path.join(cfg.data_dir, cfg.ckpt_file)
        ckpt_cfg = os.path.join(cfg.data_dir, cfg.ckpt_cfg)
    except TypeError:
        ckpt_file = None
        ckpt_cfg = None

    if os.path.exists(ckpt_file) and cfg.save_ckpt:
        msg = f"checkpoint already exists at {ckpt_file} and will be overwritten!"
        warnings.warn(msg, UserWarning)

    # instantiate model
    model = hydra.utils.instantiate(cfg.tree)
    model.build_tree(cfg, skip_task_setup=False)

    # set up dataloaders
    leaf_train_loaders = {}
    task_test_loaders = {}
    for l_key in model.leaf_nodes:
        task_key, _ = l_key.rsplit("_", 1)
        leaf_train_loaders[l_key] = model.task_dict[task_key].data_module.train_dataloader()
        if task_key not in task_test_loaders:
            task_test_loaders[task_key] = model.task_dict[task_key].data_module.test_dataloader()

    trainer.fit(
        model,
        train_dataloaders=CombinedLoader(leaf_train_loaders, mode="min_size"),
        val_dataloaders=CombinedLoader(task_test_loaders, mode="max_size_cycle"),  # change to max_size when lightning upgraded to >1.9.5
    )

    # save model
    model.cpu()
    if isinstance(ckpt_file, str) and trainer.global_rank == 0:
        torch.save(model.state_dict(), f"{ckpt_file}")
        OmegaConf.save(config=cfg, f=ckpt_cfg)

    return float("NaN")


if __name__ == "__main__":
    main()
