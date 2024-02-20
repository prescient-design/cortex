import copy
import math
from typing import Dict, Optional

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
from botorch.models.transforms.outcome import OutcomeTransform
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch import nn

from cortex.model import online_weight_update_
from cortex.model.leaf import (
    ClassifierLeaf,
    DenoisingLanguageModelLeaf,
    RegressorLeaf,
    format_classifier_ensemble_output,
    format_denoising_lm_ensemble_output,
    format_regressor_ensemble_output,
)
from cortex.model.tree import NeuralTree


class SequenceModelTree(NeuralTree, L.LightningModule):
    def __init__(
        self,
        root_nodes: Optional[nn.ModuleDict] = None,
        trunk_node: Optional[nn.Module] = None,
        branch_nodes: Optional[nn.ModuleDict] = None,
        leaf_nodes: Optional[nn.ModuleDict] = None,
        fit_cfg: Optional[DictConfig] = None,  # hyperparameters for training
    ):
        root_nodes = root_nodes or nn.ModuleDict()
        branch_nodes = branch_nodes or nn.ModuleDict()
        leaf_nodes = leaf_nodes or nn.ModuleDict()
        super().__init__(
            root_nodes=root_nodes,
            trunk_node=trunk_node,
            branch_nodes=branch_nodes,
            leaf_nodes=leaf_nodes,
        )
        # self.task_dict = self.build_tree(model_cfg, skip_task_setup=False)

        # used for weight averaging
        self._train_state_dict = None
        self._eval_state_dict = None
        self._w_avg_step_count = 1

        # decoupled multi-task training requires manual optimization
        self.automatic_optimization = False
        self.save_hyperparameters(
            ignore=[
                "root_nodes",
                "trunk_node",
                "branch_nodes",
                "leaf_nodes",
            ]
        )

    def train(self, *args, **kwargs):
        if not self.training and self._eval_state_dict is not None:
            self.load_state_dict(self._train_state_dict)
        return super().train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        if self.training and self._eval_state_dict is not None:
            self._train_state_dict = copy.deepcopy(self.state_dict())
            self.load_state_dict(self._eval_state_dict)
        return super().eval(*args, **kwargs)

    def get_dataloader(self, split="train"):
        loaders = {}
        for l_key in self.leaf_nodes:
            task_key, _ = l_key.rsplit("_", 1)
            if split == "train":
                loaders[l_key] = self.task_dict[task_key].data_module.train_dataloader()
            elif split == "val" and task_key not in loaders:
                loaders[task_key] = self.task_dict[task_key].data_module.test_dataloader()
            elif split == "val":
                pass
            else:
                raise ValueError(f"Invalid split {split}")

        # change val to max_size when lightning upgraded to >1.9.5
        mode = "min_size" if split == "train" else "max_size_cycle"
        return CombinedLoader(loaders, mode=mode)

    def training_step(self, batch: dict, batch_idx: int, dataloader_idx: Optional[int] = None):
        leaf_keys = list(batch.keys())
        rng = np.random.default_rng()
        rng.shuffle(leaf_keys)

        opt = self.optimizers()

        self.train()
        self.requires_grad_(True)
        if self.hparams.fit_cfg.linear_probing:
            self.freeze_roots()
            self.freeze_trunk()

        # fit outcome normalization transforms (if any)
        step_metrics = {}
        batch_size = {}
        for l_key, l_node in self.leaf_nodes.items():
            task_key, _ = l_key.rsplit("_", 1)
            task = self.task_dict[task_key]
            l_batch = task.format_batch(batch[l_key])
            root_inputs = l_batch["root_inputs"]
            leaf_targets = l_batch["leaf_targets"].get(task_key, {})
            tree_outputs = self(root_inputs, leaf_keys=[l_key])
            l_node = self.leaf_nodes[l_key]
            # link leaf output to root input if needed (e.g. for masked language model leaves)
            r_key = l_node.root_key if hasattr(l_node, "root_key") else None
            root_outputs = tree_outputs.root_outputs[r_key] if r_key in tree_outputs.root_outputs else None
            leaf_outputs = tree_outputs.leaf_outputs[l_key]

            # get leaf loss and update weights
            opt.zero_grad()
            loss = l_node.loss(
                leaf_outputs=leaf_outputs,
                root_outputs=root_outputs,
                **leaf_targets,
            )
            self.manual_backward(loss)
            opt.step()
            step_metrics.setdefault(task_key, []).append(loss.item())
            batch_size.setdefault(task_key, []).append(batch[l_key]["batch_size"])

        # log average task losses
        step_metrics = {f"{task_key}/train_loss": np.mean(losses) for task_key, losses in step_metrics.items()}
        step_metrics.update(
            {f"{task_key}/train_batch_size": np.mean(batch_sizes) for task_key, batch_sizes in batch_size.items()}
        )
        return step_metrics

    def training_step_end(self, step_metrics):
        # weight averaging
        w_avg_enabled = self.hparams.fit_cfg.weight_averaging is not None
        if w_avg_enabled:
            self._w_avg_step_count = self._weight_average_update(self._w_avg_step_count)

        # step_metrics = pd.DataFrame.from_records(step_metrics)
        # step_metrics = step_metrics.mean().to_dict()

        # task_keys = set()
        # for key in step_metrics.keys():
        #     task_key = key.split("/")[0]
        #     task_keys.add(task_key)

        # for t_key in task_keys:
        #     task_metrics = {key: val for key, val in step_metrics.items() if key.startswith(t_key)}
        #     batch_size = task_metrics[f"{t_key}/train_batch_size"]
        #     del task_metrics[f"{t_key}/train_batch_size"]
        #     self.log_dict(task_metrics, logger=True, prog_bar=True, batch_size=batch_size)

        return step_metrics

        # log metrics
        # step_metrics = {key: sum(val) / len(val) for key, val in step_metrics.items()}

        # self.log_dict(step_metrics, prog_bar=True)

    def training_epoch_end(self, step_metrics):
        step_metrics = pd.DataFrame.from_records(step_metrics)
        step_metrics = step_metrics.mean().to_dict()

        task_keys = set()
        for key in step_metrics.keys():
            task_key = key.split("/")[0]
            task_keys.add(task_key)

        for t_key in task_keys:
            task_metrics = {key: val for key, val in step_metrics.items() if key.startswith(t_key)}
            batch_size = task_metrics[f"{t_key}/train_batch_size"]
            del task_metrics[f"{t_key}/train_batch_size"]
            self.log_dict(task_metrics, prog_bar=True, batch_size=batch_size)

    def _weight_average_update(
        self,
        w_avg_step_count: int,
    ) -> int:
        w_avg_start_epoch = int(self.hparams.fit_cfg.weight_averaging.start_epoch)
        if self.current_epoch < w_avg_start_epoch:
            return w_avg_step_count

        self.requires_grad_(False)
        w_avg_method = self.hparams.fit_cfg.weight_averaging.method
        self._eval_state_dict = self._eval_state_dict or copy.deepcopy(self.state_dict())
        if w_avg_method == "sma":
            decay_rate = w_avg_step_count / (w_avg_step_count + 1)
        elif w_avg_method == "ema":
            decay_rate = float(self.hparams.fit_cfg.weight_averaging.decay_rate)
        else:
            raise ValueError("Invalid weight averaging method")
        online_weight_update_(self.state_dict(), self._eval_state_dict, decay_rate, param_prefixes=None)
        return w_avg_step_count + 1

    def on_train_epoch_start(self):
        self.lr_schedulers().step()

    def on_fit_start(self) -> None:
        # Initialize root node weights with pretrained weights or random initialization if training from scratch
        if self.hparams.fit_cfg.reinitialize_roots:
            for root_node in self.root_nodes.values():
                root_node.initialize_weights()

        for l_key, l_node in self.leaf_nodes.items():
            task_key, _ = l_key.rsplit("_", 1)
            if task_key not in self.task_dict:
                continue
            if hasattr(l_node, "outcome_transform"):
                transform = l_node.outcome_transform
                if isinstance(transform, OutcomeTransform):
                    task = self.task_dict[task_key]
                    task.fit_transform(transform, self.device, self.dtype)

    def configure_optimizers(self):
        fit_cfg = self.hparams.fit_cfg
        self.requires_grad_(True)
        if fit_cfg.linear_probing:
            self.freeze_roots()
            self.freeze_trunk()
        params = self.get_trainable_params()
        optimizer = hydra.utils.instantiate(fit_cfg.optimizer, params=params)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": hydra.utils.call(fit_cfg.lr_scheduler, optimizer=optimizer)},
        }

    def evaluate(self, *args, **kwargs):
        pass

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: Optional[int] = None):
        step_metrics = {}
        for task_key, task_batch in batch.items():
            if task_batch is None or len(task_batch) == 0:
                continue
            task = self.task_dict[task_key]

            predict_targets = {}
            if hasattr(task, "format_targets"):
                predict_targets[task_key] = task.format_targets(task_batch)[task_key]
            else:
                predict_targets[task_key] = None

            predict_out = self.predict(data=task_batch, batch_limit=len(task_batch), predict_tasks=[task_key])

            step_metrics.update(
                self.prediction_metrics(
                    task_dict=self.task_dict,
                    predict_out=predict_out,
                    predict_targets=predict_targets,
                    log_prefix="val",
                )
            )
            step_metrics[f"{task_key}/val_batch_size"] = len(task_batch)

        return step_metrics

    def validation_epoch_end(self, step_metrics):
        step_metrics = pd.DataFrame.from_records(step_metrics)
        step_metrics = step_metrics.mean().to_dict()

        task_keys = set()
        for key in step_metrics.keys():
            task_key = key.split("/")[0]
            task_keys.add(task_key)

        for t_key in task_keys:
            task_metrics = {key: val for key, val in step_metrics.items() if key.startswith(t_key)}
            batch_size = task_metrics[f"{t_key}/val_batch_size"]
            del task_metrics[f"{t_key}/val_batch_size"]
            self.log_dict(task_metrics, prog_bar=False, batch_size=batch_size)

        # step_metrics = {key: sum(val) / len(val) for key, val in step_metrics.items()}
        # self.log_dict(step_metrics, prog_bar=True, batch_size=len(task_batch))
        # self.log_dict(step_metrics, prog_bar=True)

    def finetune(
        self,
        cfg: DictConfig,
        task_dict: dict,
        log_prefix: Optional[str] = "",
        ckpt_file: Optional[str] = None,
        ckpt_cfg: Optional[str] = None,
        tune_branches: bool = True,
        reinitialize_leaves: bool = False,
        linear_probing: bool = False,
        **kwargs,
    ) -> list[Dict]:
        if not tune_branches:
            self.freeze_branches()
        if reinitialize_leaves:
            self.initialize_leaves(task_dict)
        return self.fit(
            cfg,
            task_dict,
            log_prefix=log_prefix,
            ckpt_file=ckpt_file,
            ckpt_cfg=ckpt_cfg,
            linear_probing=linear_probing,
            **kwargs,
        )

    # TODO test this feature
    def add_task_nodes(
        self,
        branch_key: str,
        task_key: str,
        task_cfg: OmegaConf,
        branch_cfg: Optional[OmegaConf] = None,
        data_dir: str = "./data",
    ) -> None:
        task = hydra.utils.instantiate(task_cfg, leaf_key=task_key, data_dir=data_dir)
        self.task_dict[task_key] = task
        for ens_idx in range(task_cfg.ensemble_size):
            # add branch node
            b_key = "_".join((branch_key, str(ens_idx)))
            if b_key not in self.branch_nodes and branch_cfg is None:
                msg = f"Branch node {b_key} not found, you must provide a branch config to add new branches."
                raise ValueError(msg)
            self.add_branch(branch_cfg, b_key)

            # add leaf node
            leaf_in_dim = branch_cfg.out_dim
            l_key = "_".join((task_key, str(ens_idx)))
            self.add_leaf(task.create_leaf((leaf_in_dim, b_key), l_key))

    def initialize_leaves(self, task_dict) -> None:
        for l_key, l_node in self.leaf_nodes.items():
            task_key, _ = l_key.rsplit("_", 1)
            if task_key in task_dict:
                l_node.initialize()

    def build_tree(self, cfg: OmegaConf, skip_task_setup: bool = False) -> dict:
        # create root nodes
        for root_key, root_cfg in cfg.roots.items():
            self.root_nodes[root_key] = hydra.utils.instantiate(root_cfg, device=self.device, dtype=self.dtype)

        # create trunk node
        root_out_dims = [r_node.out_dim for r_node in self.root_nodes.values()]
        if not hasattr(cfg.trunk, "out_dim"):
            cfg.trunk["out_dim"] = max(root_out_dims)
        self.trunk_node = hydra.utils.instantiate(
            cfg.trunk,
            in_dims=root_out_dims,
        ).to(self.device, self.dtype)

        # instantiate tasks, branch and leaf nodes
        task_dict = {}
        for branch_key, branch_tasks in cfg.tasks.items():
            for task_key, task_cfg in branch_tasks.items():
                # TODO revisit leaf_key argument usage
                task_cfg.data_module["skip_task_setup"] = skip_task_setup
                task = hydra.utils.instantiate(
                    task_cfg,
                    leaf_key=task_key,
                    data_dir=cfg.data_dir,
                )
                task_dict[task_key] = task
                for idx in range(task_cfg.ensemble_size):
                    b_key = "_".join((branch_key, str(idx)))
                    l_key = "_".join((task_key, str(idx)))
                    cfg.branches[branch_key]["in_dim"] = self.trunk_node.out_dim
                    self.add_branch(cfg.branches[branch_key], b_key)
                    leaf_in_dim = cfg.branches[branch_key].out_dim
                    self.add_leaf(task.create_leaf(leaf_in_dim, b_key), l_key)
        self.branch_nodes = self.branch_nodes.to(self.device, self.dtype)
        self.leaf_nodes = self.leaf_nodes.to(self.device, self.dtype)

        self.task_dict = task_dict

        return task_dict

    def predict(
        self,
        data: pd.DataFrame,
        batch_limit: Optional[int] = None,
        predict_tasks: Optional[list[str]] = None,
        format_outputs: bool = True,
        cpu_offload: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            predict_inputs: array of inputs (use `AbModel.df_to_pred_inputs` to convert a dataframe)
        Returns:
            predict_out: dict[str, torch.Tensor] of task prediction outputs for `predict_inputs`.
        """
        self.eval()
        batch_limit = len(data) if batch_limit is None else batch_limit
        num_chunks = math.ceil(len(data) / batch_limit)
        if num_chunks > 1:
            batches = np.array_split(data, num_chunks)
        else:
            batches = [data]

        def data_mapper(x):
            if cpu_offload:
                return x.cpu()
            else:
                return x

        batched_out = [
            {
                k: data_mapper(v)
                for k, v in self._predict_batch(data=b, task_keys=predict_tasks, format_outputs=format_outputs).items()
            }
            for b in batches
        ]
        predict_out = {key: [b_out[key] for b_out in batched_out] for key in batched_out[0]}
        predict_out = {key: torch.cat(vals, dim=1) for key, vals in predict_out.items()}
        return predict_out

    def prediction_metrics(self, task_dict, predict_out, predict_targets, log_prefix=""):
        metrics = {}
        for task_key, task in task_dict.items():
            if not any([task_key in out_key for out_key in predict_out]):
                continue

            targets = predict_targets[task_key]
            task_metrics = task.compute_eval_metrics(predict_out, targets, task_key)

            metrics.update({f"{task_key}/{log_prefix}_{m_key}": val for m_key, val in task_metrics.items()})
        return metrics

    def _predict_batch(
        self,
        data: pd.DataFrame,
        format_outputs: bool = True,
        task_keys: Optional[list[str]] = None,
    ) -> dict:
        """
        Args:
            aa_seqs: [(AbHC, AbLC, Ag), ...]
            predict_tasks: list of task keys indicating which task predictions are required.
                Defaults to `None`, which indicates all tasks are required without enumeration.
        """
        task_out = {}
        task_keys = list(self.task_dict.keys()) if task_keys is None else task_keys

        group_leaves = []
        group_tasks = []
        for l_key in self.leaf_nodes:
            # strip off the ensemble index {task_key}_{idx}
            task_key, _ = l_key.rsplit("_", 1)
            if not self.task_dict[task_key].corrupt_inference_inputs:
                group_leaves.append(l_key)
                group_tasks.append(task_key)

        if len(group_leaves) > 0:
            group_inputs = {}
            for task_key in group_tasks:
                group_inputs.update(self.task_dict[task_key].format_inputs(data, corrupt_frac=0.0))
            with torch.no_grad():
                group_outputs = self(group_inputs, leaf_keys=group_leaves)
            task_out.update(group_outputs.branch_outputs)
            task_out.update(group_outputs.leaf_outputs)

        group_leaves = []
        group_tasks = []
        for l_key in self.leaf_nodes:
            # strip off the ensemble index {task_key}_{idx}
            task_key, _ = l_key.rsplit("_", 1)
            if self.task_dict[task_key].corrupt_inference_inputs:
                group_leaves.append(l_key)
                group_tasks.append(task_key)
        if len(group_leaves) > 0:
            group_inputs = {}
            for task_key in group_tasks:
                group_inputs.update(self.task_dict[task_key].format_inputs(data, corrupt_frac=0.5))
            with torch.no_grad():
                group_outputs = self(group_inputs, leaf_keys=group_leaves)
            task_out.update(group_outputs.root_outputs)
            task_out.update(group_outputs.leaf_outputs)

        task_leaves = {}
        for task_key in self.task_dict:
            task_leaves[task_key] = [l_key for l_key in self.leaf_nodes.keys() if l_key.startswith(task_key)]

        if format_outputs:
            predict_out = self.format_task_outputs(
                task_out=task_out,
                task_keys=task_keys,
                task_leaves=task_leaves,
            )
        else:
            predict_out = task_out

        return predict_out

    def format_task_outputs(self, task_out, task_keys, task_leaves):
        # format leaf output to more useful form
        predict_out = {}
        for t_key in task_keys:
            # classifier leaves
            values = [
                l_out
                for l_key, l_out in task_out.items()
                if l_key in task_leaves[t_key] and type(self.leaf_nodes[l_key]) is ClassifierLeaf
            ]
            if len(values) > 0:
                predict_out.update(format_classifier_ensemble_output(leaf_outputs=values, task_key=t_key))

            # regressor leaves
            values = [
                l_out
                for l_key, l_out in task_out.items()
                if l_key in task_leaves[t_key] and isinstance(self.leaf_nodes[l_key], RegressorLeaf)
            ]
            if len(values) > 0:
                predict_out.update(format_regressor_ensemble_output(leaf_outputs=values, task_key=t_key))

            # denoising language model leaves
            values = [
                l_out
                for l_key, l_out in task_out.items()
                if l_key in task_leaves[t_key] and isinstance(self.leaf_nodes[l_key], DenoisingLanguageModelLeaf)
            ]
            if len(values) > 0:
                root_keys = [self.leaf_nodes[l_key].root_key for l_key in task_leaves[t_key]]
                root_outputs = [task_out[r_key] for r_key in root_keys]
                predict_out.update(
                    format_denoising_lm_ensemble_output(leaf_outputs=values, root_outputs=root_outputs, task_key=t_key)
                )

        return predict_out

    def call_from_str_array(
        self, str_array, root_key: Optional[str] = None, leaf_keys: Optional[list[str]] = None, **kwargs
    ):
        if root_key is None:
            root_key = _infer_root_key(self.root_nodes)
        root_inputs = {root_key: {"seq_array": str_array, **kwargs}}
        return self(root_inputs=root_inputs, leaf_keys=leaf_keys)

    def call_from_tok_idxs(
        self,
        tok_idxs: torch.LongTensor,
        root_key: Optional[str] = None,
        leaf_keys: Optional[list[str]] = None,
        **kwargs,
    ):
        if root_key is None:
            root_key = _infer_root_key(self.root_nodes)
        root_inputs = {root_key: {"tgt_tok_idxs": tok_idxs, **kwargs}}
        return self(root_inputs=root_inputs, leaf_keys=leaf_keys)

    def call_from_tok_embs(
        self,
        tok_embs: torch.FloatTensor,
        root_key: Optional[str] = None,
        leaf_keys: Optional[list[str]] = None,
        **kwargs,
    ):
        if root_key is None:
            root_key = _infer_root_key(self.root_nodes)
        root_inputs = {root_key: {"src_tok_embs": tok_embs, **kwargs}}
        return self(root_inputs=root_inputs, leaf_keys=leaf_keys)

    def get_tokenizer(self, root_key: Optional[str] = None):
        if root_key is None:
            root_key = _infer_root_key(self.root_nodes)
        return self.root_nodes[root_key].tokenizer


def _infer_root_key(root_nodes):
    if len(root_nodes) == 1:
        return list(root_nodes.keys())[0]
    else:
        raise ValueError("root_key must be provided when there are multiple root nodes")


def get_param_prefixes(tree_outputs):
    param_prefixes = []
    for root_key in tree_outputs.root_outputs:
        param_prefixes.append(f"root_nodes.{root_key}")

    # for trunk_key in tree_output.trunk_output:
    #     param_prefixes.append(f"trunk_node.{trunk_key}")

    for branch_key in tree_outputs.branch_outputs:
        param_prefixes.append(f"branch_nodes.{branch_key}")

    for leaf_key in tree_outputs.leaf_outputs:
        param_prefixes.append(f"leaf_nodes.{leaf_key}")

    return param_prefixes
