from typing import Optional

import numpy as np
import pandas as pd
import torch

from cortex.io import load_hydra_config, load_model_checkpoint


def infer_with_model(
    data: pd.DataFrame,
    model: Optional[torch.nn.Module] = None,
    cfg_fpath: Optional[str] = None,
    weight_fpath: Optional[str] = None,
    batch_limit: int = 32,
    cpu_offload: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> dict[str, np.ndarray]:
    """
    A functional interface for inference with a cortex model.

    Usage:

    ```python title="Example of inference with a cortex model checkpoint."
    from cortex.model import infer_with_model

    ckpt_dir = <TODO>
    ckpt_name = <TODO>
    predictions = infer_with_model(
        data=df,
        cfg_fpath=f"{ckpt_dir}/{ckpt_name}.yaml",
        weight_fpath=f"{ckpt_dir}/{ckpt_name}.pt",
    )
    ```

    Args:
        data (pd.DataFrame): A dataframe containing the sequences to predict on.
        cfg_fpath (str): The path to the Hydra config file on S3.
        weight_fpath (str): The path to the PyTorch model weights on S3.
        batch_limit (int, optional): The maximum number of sequences to predict on at once. Defaults to 32.
        cpu_offload (bool, optional): Whether to use cpu offload.
            If true, will run prediction with cpu offload. Defaults to True
        device (torch.device, optional): The device to run the model on. Defaults to None.
        dtype (torch.dtype, optional): The dtype to run the model on. Defaults to None.

    Returns:
        dict[str, np.ndarray]: A dict of NumPy arrays of the predictions.
    """
    # set default device and dtype
    if device is None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if dtype is None and torch.cuda.is_available():
        dtype = torch.float32
    elif dtype is None:
        dtype = torch.float64

    if model is None and (cfg_fpath is None or weight_fpath is None):
        raise ValueError("Either model or cfg_fpath and weight_fpath must be provided")

    if model is not None and (cfg_fpath is not None or weight_fpath is not None):
        raise ValueError("Only one of model or cfg_fpath and weight_fpath must be provided")

    if model is None:
        # load Hydra config from s3 or locally
        cfg = load_hydra_config(cfg_fpath)
        # load model checkpoint from s3 or locally
        model, _ = load_model_checkpoint(cfg, weight_fpath, device=device, dtype=dtype)

    # model forward pass
    with torch.inference_mode():
        return model.predict(
            data=data,
            batch_limit=batch_limit,
            predict_tasks=None,
            cpu_offload=cpu_offload,
        )
