import math
import pprint
import warnings
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.distributions.kl import kl_divergence

from cortex.attribution import approximate_occlusion
from cortex.corruption import GaussianCorruptionProcess, MaskCorruptionProcess
from cortex.model.leaf import mlm_pseudo_log_likelihood


class LaMBO(object):
    """
    This class implements the LaMBO-2 algorithm for optimization of discrete sequences.

    https://arxiv.org/abs/2305.20009
    """

    def __init__(
        self,
        params: torch.LongTensor,
        is_mutable: torch.BoolTensor,
        model,
        objective,
        max_num_solutions: int,
        num_mutations_per_step: Optional[int] = 2,
        max_guidance_updates: int = 16,
        guidance_step_size: float = 1.0,
        guidance_layer: str = "trunk",
        kl_weight: float = 0.5,
        feature_attr_temp: float = 0.125,
        constraint_fn: Optional[Callable[[str], bool]] = None,
        domain_name: Optional[str] = None,
        exclude_initial_solution: bool = False,
        resample_edit_positions: bool = False,
    ) -> None:
        self.model = model
        self.objective = objective
        self.max_num_solutions = max_num_solutions
        self.num_mutations_per_step = num_mutations_per_step
        self.max_guidance_updates = max_guidance_updates
        self.guidance_step_size = guidance_step_size
        self.guidance_layer = guidance_layer
        self.kl_weight = kl_weight
        self.feature_attr_temp = feature_attr_temp
        self.constraint_fn = constraint_fn
        self.domain_name = "iterate" if domain_name is None else domain_name
        self.resample_edit_positions = resample_edit_positions

        self.initial_solution = np.array([self.tokenizer.decode(t_idxs) for t_idxs in params])

        initial_obj_val, _ = self.score_sequences(self.initial_solution)
        print(initial_obj_val)
        self.initial_obj_val = initial_obj_val
        self.is_mutable = is_mutable

        self._step_count = 0
        self._buffer = pd.DataFrame(
            {
                "iteration": self._step_count,
                domain_name: self.initial_solution,
                "obj_val": self.initial_obj_val.cpu().numpy(),
            },
            index=list(range(initial_obj_val.size(0))),
        )
        self.exclude_initial_solution = exclude_initial_solution

    @property
    def tokenizer(self):
        return self.model.root_nodes[self.domain_name].tokenizer

    @property
    def tokens_to_long_tensor(self):
        return self.model.root_nodes[self.domain_name].eval_transform

    def step(self) -> None:
        """
        Each call of LaMBO.step() corresponds to one guided diffusion step
        """
        self.model.eval()
        self.model.requires_grad_(False)

        print(f"==== LaMBO-2 Step {self._step_count + 1} ====")
        prev_frame = self._buffer.iloc[-self.initial_solution.shape[0] :]
        base_solutions = prev_frame[self.domain_name].values

        # TODO figure out cleaner structure
        base_tok_idxs = self.tokens_to_long_tensor(base_solutions)
        base_output = self.model.call_from_tok_idxs(base_tok_idxs, corrupt_frac=0.0)
        base_obj_vals = self.objective(base_output)
        base_tok_embs = base_output.root_outputs[self.domain_name].src_tok_embs
        base_padding_mask = base_output.root_outputs[self.domain_name].padding_mask

        # tensor of token indices that are not viable sample output
        non_viable_idxs = torch.tensor(
            [self.tokenizer.vocab[token] for token in self.tokenizer.sampling_vocab_excluded]
        )

        # check feasibility constraints
        base_is_feasible = self.check_constraints(base_tok_idxs).to(base_tok_idxs.device)
        if not base_is_feasible.any():
            msg = "No feasible starting points, search may fail to find feasible solutions. Try changing the initial solution or the constraint function."
            warnings.warn(msg, stacklevel=2)

        print(f"Optimizing {base_obj_vals.size(0)} solutions")

        # create mutable tensors for optimization
        tgt_tok_idxs = base_tok_idxs.clone()
        tgt_tok_embs = base_tok_embs.clone()
        tgt_padding_mask = base_padding_mask.clone()

        # choose edit positions
        is_corruptible = self.tokenizer.get_corruptible_mask(tgt_tok_idxs)
        self._coordinate_selection(tgt_tok_idxs, tgt_tok_embs, tgt_padding_mask, is_corruptible)

        # score current solutions
        if self._step_count == 0 and self.exclude_initial_solution:
            tgt_obj_vals = torch.full_like(base_obj_vals, float("-inf"))
        else:
            tgt_obj_vals = base_obj_vals
        if self.kl_weight > 0.0:
            tgt_obj_vals *= 1 - self.kl_weight
            tgt_obj_vals += self.kl_weight * mlm_pseudo_log_likelihood(
                tgt_tok_idxs,
                null_value=self.tokenizer.masking_idx,
                model=self.model.call_from_tok_idxs,
                root_key=self.domain_name,
                is_excluded=~self.tokenizer.get_corruptible_mask(tgt_tok_idxs),
            )

        # set up forwards pass inputs
        generation_inputs = self._set_up_root_inputs(tgt_tok_idxs, tgt_tok_embs, tgt_padding_mask)
        is_corrupted = generation_inputs[self.domain_name]["is_corrupted"]

        # get latent features, make them leaf variables
        activations, trunk_outputs = self._get_latent_variables(generation_inputs)

        delta = torch.nn.Parameter(torch.zeros_like(activations))
        optimizer = torch.optim.Adam([delta], lr=self.guidance_step_size, betas=(0.09, 0.0999))
        metrics = {"step": self._step_count}

        # get initial solution before guidance
        tgt_tok_idxs, tgt_obj_vals = self._update_solution(
            trunk_outputs,
            activations,
            delta,
            tgt_tok_idxs,
            tgt_obj_vals,
            is_corrupted,
            self.tokenizer,
            non_viable_idxs,
        )

        print("\n")
        for lang_step in range(self.max_guidance_updates):
            delta.grad = None

            # forward pass from modified activations
            if self.guidance_layer == "root":
                masked_delta = torch.where(is_corrupted[..., None], delta, 0.0)
                generation_inputs[self.domain_name]["src_tok_embs"] = activations + masked_delta
                tree_output = self.model(root_inputs=generation_inputs)
                trunk_outputs = tree_output.trunk_outputs
            elif self.guidance_layer == "trunk":
                masked_delta = torch.where(is_corrupted[..., None], delta, 0.0)
                trunk_outputs.trunk_features = activations + masked_delta
                tree_output = self.model.call_from_trunk_output(trunk_outputs)

            # guided token distribution logits
            token_logits = tree_output.leaf_outputs[f"{self.domain_name}_0"].logits

            # fix unguided reference token distribution
            if lang_step == 0:
                adj_logits = token_logits.scatter(
                    dim=-1,
                    index=non_viable_idxs.expand(*token_logits.shape[:-1], -1).to(token_logits.device),
                    value=float("-inf"),
                )
                base_probs = adj_logits.detach().clone().softmax(-1).clamp_min(1e-6)
                base_dist = torch.distributions.Categorical(probs=base_probs)

            # compute guidance loss
            guided_dist = torch.distributions.Categorical(logits=token_logits)
            entropy = torch.masked_select(guided_dist.entropy(), is_corrupted).mean()
            kl_div = torch.masked_select(kl_divergence(guided_dist, base_dist), is_corrupted).mean()
            obj_loss = -1.0 * self.objective(tree_output).mean()
            design_loss = self.kl_weight * kl_div + (1.0 - self.kl_weight) * obj_loss
            design_loss.backward()
            feature_grad = delta.grad.detach().clone()
            optimizer.step()

            # update solution
            tgt_tok_idxs, tgt_obj_vals = self._update_solution(
                trunk_outputs,
                activations,
                delta,
                tgt_tok_idxs,
                tgt_obj_vals,
                is_corrupted,
                self.tokenizer,
                non_viable_idxs,
            )

            grad_norm = feature_grad.norm(dim=(-2, -1), keepdim=True)
            metrics.update(
                {
                    "act_obj_val": tgt_obj_vals.mean().item(),
                    "masked_design_loss": design_loss.item(),
                    "masked_design_loss_grad_norm": grad_norm.mean().item(),
                    "masked_token_loss": kl_div.item(),
                    "masked_obj_loss": obj_loss.item(),
                    "token_entropy": entropy.item(),
                }
            )
            pprint.pp(metrics)

        self._step_count += 1

        tgt_str_array = [self.tokenizer.decode(t_idxs) for t_idxs in tgt_tok_idxs]
        df = pd.DataFrame({self.domain_name: tgt_str_array})
        df.loc[:, "obj_val"] = tgt_obj_vals.cpu().numpy()
        df.loc[:, "iteration"] = self._step_count

        self._buffer = pd.concat([self._buffer, df], ignore_index=True)

        return metrics

    def _coordinate_selection(
        self,
        tok_idxs: torch.LongTensor,
        tok_embeddings: torch.FloatTensor,
        padding_mask: torch.BoolTensor,
        is_corruptible: torch.BoolTensor,
    ):
        """
        Choose edit positions (i.e. the infilling region) for diffusion.
        If `resample_edit_positions` is True, the infilling region can change between steps.
        """

        def coord_score(tok_embeddings):
            tree_output = self.model.call_from_tok_embs(
                tok_embeddings, root_key=self.domain_name, corrupt_frac=0.0, padding_mask=padding_mask
            )
            return self.objective(tree_output)

        null_embedding = self.model.root_nodes[self.domain_name].get_token_embedding(self.tokenizer.masking_idx)

        # edit_idxs are all corruptible and mutable positions
        pos_is_feasible = is_corruptible * self.is_mutable
        if self.num_mutations_per_step is None:
            self._corruption_allowed = pos_is_feasible
        elif self._step_count == 0 or self.resample_edit_positions:
            position_scores = approximate_occlusion(
                coord_score,
                tok_embeddings,
                null_embedding,
                is_excluded=~pos_is_feasible,
            )
            denom = torch.where(position_scores > float("-inf"), position_scores, 0.0).abs().sum(-1, keepdim=True)
            position_scores = position_scores / (denom + 1e-6)

            position_probs = (position_scores / self.feature_attr_temp).softmax(-1)
            hand_tuned_entropy = torch.distributions.Categorical(probs=position_probs).entropy().median()
            print(f"[INFO][LaMBO-2]: Hand-tuned entropy = {hand_tuned_entropy}")

            edit_idxs = torch.multinomial(position_probs, self.num_mutations_per_step, replacement=False)
            edit_idxs = edit_idxs.sort(dim=-1).values

            self._corruption_allowed = torch.zeros_like(tok_idxs)
            self._corruption_allowed = self._corruption_allowed.scatter(dim=-1, index=edit_idxs, value=1).bool()
            print(f"Selected edit positions: {edit_idxs}")

    def _get_latent_variables(
        self,
        generation_inputs: dict,
    ):
        with torch.no_grad():
            tree_outputs = self.model(generation_inputs, leaf_keys=[f"{self.domain_name}_0"])
        if self.guidance_layer == "root":
            activations = tree_outputs.root_outputs[self.domain_name].src_tok_embs

        elif self.guidance_layer == "trunk":
            activations = tree_outputs.trunk_outputs.trunk_features

        trunk_outputs = tree_outputs.trunk_outputs

        return activations, trunk_outputs

    def _update_solution(
        self,
        trunk_outputs,
        activations,
        delta,
        tgt_tok_idxs,
        tgt_obj_vals,
        is_corrupted,
        tokenizer,
        non_viable_idxs,
    ):
        """
        Update the guided activations, decode out to sequence and check for improvement.
        """
        # update latent features only at masked locations
        with torch.no_grad():
            new_activations = torch.where(is_corrupted[..., None], activations + delta, activations)
            # compute token logits from updated features
            trunk_outputs.trunk_features = new_activations
            sample_tok_idxs = self.decode(trunk_outputs, non_viable_idxs)
            sample_tok_idxs = torch.where(is_corrupted, sample_tok_idxs, tgt_tok_idxs)

            sample_obj_vals, sample_tok_embs = self.score_sequences(sample_tok_idxs)
            sample_obj_vals *= 1 - self.kl_weight
            if self.kl_weight > 0.0:
                sample_obj_vals += self.kl_weight * mlm_pseudo_log_likelihood(
                    sample_tok_idxs,
                    null_value=tokenizer.masking_idx,
                    model=self.model.call_from_tok_idxs,
                    root_key=self.domain_name,
                    is_excluded=~tokenizer.get_corruptible_mask(tgt_tok_idxs),
                )

            if tgt_obj_vals is None:
                tgt_obj_vals = sample_obj_vals
                sample_is_improved = torch.ones_like(sample_obj_vals).bool()
            else:
                sample_is_improved = sample_obj_vals >= tgt_obj_vals

            sample_is_feasible = self.check_constraints(sample_tok_idxs)
            print(f"Feasible samples: {sample_is_feasible.sum()}/{sample_is_feasible.size(0)}")

            # keep improved feasible sequences
            replace_mask = sample_is_improved * sample_is_feasible.to(sample_is_improved)
            # tgt_tok_embs = torch.where(replace_mask[..., None, None], sample_tok_embs, tgt_tok_embs)
            tgt_tok_idxs = torch.where(replace_mask[..., None], sample_tok_idxs, tgt_tok_idxs)
            tgt_obj_vals = torch.where(replace_mask, sample_obj_vals, tgt_obj_vals)

        return tgt_tok_idxs, tgt_obj_vals

    def _set_up_root_inputs(
        self,
        tgt_tok_idxs,
        tgt_tok_embs,
        tgt_padding_mask,
    ):
        """
        Set up inputs for the forward pass of each guidance update.
        Corrupt a random subset of the positions selected previously as the infilling region.
        """
        corrupt_frac = 1.0 / math.sqrt(1 + self._step_count)

        root_inputs = {self.domain_name: {}}
        corrupt_kwargs = {
            "corrupt_frac": corrupt_frac,
        }
        corruption_process = self.model.root_nodes[self.domain_name].corruption_process

        # corrupt random subset of positions where self._corruption_allowed is True
        if isinstance(corruption_process, MaskCorruptionProcess):
            corrupt_kwargs["x_start"] = tgt_tok_idxs
            corrupt_kwargs["corruption_allowed"] = self._corruption_allowed
            corrupt_kwargs["mask_val"] = self.tokenizer.masking_idx
            src_tok_idxs, is_corrupted = corruption_process(**corrupt_kwargs)
            root_inputs[self.domain_name]["tgt_tok_idxs"] = src_tok_idxs
            root_inputs[self.domain_name]["is_corrupted"] = is_corrupted

        # corrupt all positions where self._corruption_allowed is True
        elif isinstance(corruption_process, GaussianCorruptionProcess):
            corrupt_kwargs["x_start"] = tgt_tok_embs
            corrupt_kwargs["corruption_allowed"] = self._corruption_allowed[..., None]
            src_tok_embs, is_corrupted = corruption_process(**corrupt_kwargs)
            is_corrupted = is_corrupted.sum(-1).bool()
            root_inputs[self.domain_name]["src_tok_embs"] = src_tok_embs
            root_inputs[self.domain_name]["is_corrupted"] = is_corrupted
            root_inputs[self.domain_name]["padding_mask"] = tgt_padding_mask
        else:
            raise NotImplementedError

        return root_inputs

    def check_constraints(self, sample_tok_idxs):
        tokenizer = self.model.root_nodes[self.domain_name].tokenizer
        if self.constraint_fn is not None:
            sample_seqs = [tokenizer.decode(t_idxs) for t_idxs in sample_tok_idxs]
            sample_seqs = np.array(sample_seqs)
            sample_is_feasible = self.constraint_fn(sample_seqs)
        else:
            sample_is_feasible = np.array([True for _ in sample_tok_idxs])
        return torch.from_numpy(sample_is_feasible)

    def score_sequences(self, sequences):
        with torch.inference_mode():
            if isinstance(sequences, np.ndarray):
                tree_output = self.model.call_from_str_array(sequences, corrupt_frac=0.0)
            elif isinstance(sequences, torch.Tensor):
                tree_output = self.model.call_from_tok_idxs(sequences, corrupt_frac=0.0)
            else:
                raise ValueError("Invalid sequences type")
        sample_tok_embs = tree_output.root_outputs[self.domain_name].src_tok_embs
        sample_obj_vals = self.objective(tree_output)

        return sample_obj_vals, sample_tok_embs

    def get_best_solutions(self) -> pd.DataFrame:
        res = self._buffer.iloc[-self.initial_solution.shape[0] :].copy()
        res["obj_val_init"] = self.initial_obj_val.cpu().numpy()
        return res

    def decode(self, trunk_outputs, non_viable_idxs):
        leaf_key = f"{self.domain_name}_0"
        with torch.no_grad():
            tree_output = self.model(root_inputs=None, trunk_outputs=trunk_outputs, leaf_keys=[leaf_key])
        logits = tree_output.leaf_outputs[leaf_key].logits

        # adjust logits to prevent sampling utility tokens
        adj_logits = logits.scatter(
            dim=-1,
            index=non_viable_idxs.expand(*logits.shape[:-1], -1).to(logits.device),
            value=float("-inf"),
        )

        # sample new tokens at masked locations
        sample_dist = torch.distributions.Categorical(logits=adj_logits)
        return sample_dist.sample()
