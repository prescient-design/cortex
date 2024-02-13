import importlib.resources
import io

import hydra
import s3fs
import torch
from omegaconf import DictConfig


def get_state_dict(weight_fpath: str, device: torch.device):
    if "s3://" in weight_fpath:
        # load model weights from S3
        s3 = s3fs.S3FileSystem()
        with s3.open(weight_fpath, "rb") as f:
            buffer = io.BytesIO(f.read())
            state_dict = torch.load(buffer, map_location=device)
    else:
        # assume local filepath in cortex/checkpoints otherwise
        try:
            state_dict = torch.load(
                f=(importlib.resources.files("cortex") / "checkpoints" / weight_fpath).as_posix(),
                map_location=device,
            )
        except Exception:
            state_dict = torch.load(f=weight_fpath, map_location=device)
    return state_dict


def load_model_checkpoint(
    cfg: DictConfig,
    weight_fpath: str,
    device: torch.device,
    dtype: torch.dtype,
    skip_task_setup: bool = True,
):
    """
    Load cortex neural tree checkpoint from S3 or local file.
    Args:
        cfg: DictConfig object containing model config
        weight_fpath: path to model checkpoint. Must point to local file or S3 object.
        device: torch.device
        dtype: torch.dtype
    Returns:
        surrogate_model: cortex.tree.NeuralTree object
        task_dict: dict mapping task keys to cortex.task.BaseTask objects
    """
    # instantiate model object with Hydra config
    surrogate_model = hydra.utils.instantiate(cfg.tree)
    task_dict = surrogate_model.build_tree(cfg, skip_task_setup=skip_task_setup)
    surrogate_model = surrogate_model.to(device=device, dtype=dtype)

    state_dict = get_state_dict(weight_fpath, device)
    try:
        surrogate_model.load_state_dict(state_dict)
    except RuntimeError:
        # hotfix for Botorch v0.8.6
        for leaf_key, leaf_node in surrogate_model.leaf_nodes.items():
            if hasattr(leaf_node, "outcome_transform"):
                k = f"leaf_nodes.{leaf_key}.outcome_transform._is_trained"
                state_dict[k] = torch.tensor(True)
        surrogate_model.load_state_dict(state_dict)

    print(f"Loaded cortex checkpoint {weight_fpath}")

    return surrogate_model, task_dict
