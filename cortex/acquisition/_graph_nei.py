from typing import Optional

import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.acquisition.objective import IdentityMCObjective
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import infer_reference_point
from botorch.utils.multi_objective.pareto import is_non_dominated
from torch import Tensor

from cortex.model.tree import NeuralTree, NeuralTreeOutput, fetch_task_outputs

GRAPH_OBJECTIVES = ["stability", "log_fluorescence"]
GRAPH_CONSTRAINTS = {}
# rescale stability and log_fluorescence to [0, 1]
GRAPH_OBJ_TRANSFORM = {
    "stability": {"scale": 1 / 2.0, "shift": 2.0},
    "log_fluorescence": {"scale": 1 / 7.0, "shift": -4.0},
}


def get_joint_objective_values(
    inputs: dict[str, Tensor],
    objectives: list[str],
    constraints: Optional[dict[str, list[str]]] = None,
    scaling: Optional[dict[str, dict[str, float]]] = None,
) -> Tensor:
    """Get joint objective values from predicted properties based on objectives and constraints.

    Parameters
    ----------
    inputs : dict[str, Tensor]
        dictionary of predicted properties. Each key is a property name and each value is a tensor of shape (ensemble size, batch_size)
    objectives : list[str]
        list of objective names. Each objective name must be a key in inputs.
    constraints : Optional[dict[str, list[str]]], optional
        dictionary of constraints. Each key is a constraint name and each value is a list of objective names that are constrained by the constraint.
    scaling : Optional[dict[str, dict[str, float]]], optional
        dictionary of scaling parameters. Each key is a property name and each value is a dictionary with keys "scale" and "shift".

    Returns
    -------
    Tensor
        Joint objective values of shape (ensemble size, batch_size, num_objectives)

    """

    if not all([obj in inputs for obj in objectives]):
        raise ValueError(f"Not all objectives {objectives} in predicted_properties {inputs.keys()}")

    objective_values: list[Tensor] = []

    for obj in objectives:
        pred_means = inputs[obj]

        if scaling is not None and obj in scaling:
            pred_means = scale_value(pred_means, shift=scaling[obj]["shift"], scale=scaling[obj]["scale"])

        objective_values.append(pred_means)

    objective_values = torch.stack(objective_values, dim=-1)

    if constraints is None:
        return objective_values

    constraint_values: list[Tensor] = []

    for obj in objectives:
        if obj in constraints:
            constraint_list = constraints[obj]
            _current = [inputs[const] for const in constraint_list]
            constraint_values.append(torch.stack(_current, dim=-1).prod(-1))

    constraint_values = torch.stack(constraint_values, dim=-1)

    objective_values = objective_values * constraint_values

    return objective_values


def scale_value(value: Tensor, *, shift: float, scale: float) -> Tensor:
    return (value + shift) * scale


def tree_output_to_dict(
    tree_output: NeuralTreeOutput,
    objectives: list[str],
    constraints: Optional[dict[str, list[str]]] = None,
    scaling: Optional[dict[str, dict[str, float]]] = None,
) -> dict[str, Tensor]:
    """Convert tree output to dictionary of tensors.

    Parameters
    ----------
    tree_output : NeuralTreeOutput
        Tree output
    objectives : list[str]
        list of objective names. Each objective adds a key to the output dictionary.
    constraints : Optional[dict[str, list[str]]], optional
        Optional dictionary of constraints. Each key is added to the output dictionary.
    scaling : Optional[dict[str, dict[str, float]]], optional
        Optional dictionary of scaling parameters. Must be a subset of objectives and each value is a dictionary with keys "scale" and "shift".

    Returns
    -------
    dict[str, Tensor]
        dictionary of tensors with keys corresponding to objectives and constraints.
    """

    result: dict[str, Tensor] = {}

    for objective in objectives:
        result[objective] = fetch_task_outputs(tree_output, objective)["loc"].squeeze(-1)

        if scaling is not None and objective in scaling:
            result[f"{objective}_scaled"] = scale_value(
                value=result[objective],
                shift=scaling[objective]["shift"],
                scale=scaling[objective]["scale"],
            )

    if constraints is not None:
        for constraint in constraints:
            constraint_values = fetch_task_outputs(tree_output, constraint)["logits"]
            constraint_values = constraint_values.softmax(dim=-1)[..., 1]

            result[constraint] = constraint_values

    return result


def get_graph_nei_runtime_kwargs(
    model: NeuralTree,
    candidate_points: np.ndarray,
    objectives: list[str] = GRAPH_OBJECTIVES,
    constraints: dict[str, list[str]] = GRAPH_CONSTRAINTS,
    scaling: dict[str, dict[str, float]] = GRAPH_OBJ_TRANSFORM,
):
    print("==== predicting baseline point objective values ====")
    with torch.inference_mode():
        tree_output = model.call_from_str_array(candidate_points, corrupt_frac=0.0)

    tree_output_dict = tree_output_to_dict(tree_output, objectives=objectives, constraints=constraints, scaling=scaling)
    f_baseline = get_joint_objective_values(
        inputs=tree_output_dict,
        objectives=objectives,
        constraints=constraints,
        scaling=scaling,
    )  # (num_samples, num_baseline, num_objectives)

    f_baseline_flat = f_baseline.reshape(-1, len(objectives))
    f_baseline_non_dom = f_baseline_flat[is_non_dominated(f_baseline_flat)]
    print(f_baseline_non_dom)
    f_ref = infer_reference_point(f_baseline_non_dom)
    print(f"reference point: {f_ref}")
    res = {
        "f_ref": f_ref,
        "f_baseline": f_baseline,
    }
    return res


class GraphNEI(object):
    def __init__(
        self,
        objectives: list[str],
        constraints: dict[str, list[str]],
        scaling: dict[str, dict[str, float]],
        f_ref: torch.Tensor,  # (num_objectives,)
        f_baseline: torch.Tensor,  # (num_samples, num_baseline, num_objectives)
    ) -> None:
        """
        Very simple implementation of PropertyDAG + NEHVI
        """
        self.objectives = objectives
        self.constraints = constraints
        self.scaling = scaling

        f_non_dom = []
        for f in f_baseline:
            f_non_dom.append(f[is_non_dominated(f)])

        self._obj_dim = len(objectives)
        if self._obj_dim == 1:
            f_best = f_baseline.max(dim=-2).values.squeeze(-1)
            self.acq_functions = [
                qLogExpectedImprovement(
                    model=None,
                    best_f=f,
                    objective=IdentityMCObjective(),
                )
                for f in f_best
            ]
        else:
            self.acq_functions = [
                qLogExpectedHypervolumeImprovement(
                    model=None,
                    ref_point=f_ref,
                    partitioning=FastNondominatedPartitioning(f_ref, f),
                )
                for f in f_non_dom
            ]
        self.has_pointwise_reference = False

    def get_objective_vals(self, tree_output: NeuralTreeOutput):
        if isinstance(tree_output, NeuralTreeOutput):
            tree_output_dict = tree_output_to_dict(tree_output, self.objectives, self.constraints, self.scaling)
        return get_joint_objective_values(
            tree_output_dict,
            self.objectives,
            self.constraints,
            self.scaling,
        )

    def __call__(self, input: NeuralTreeOutput | torch.Tensor, pointwise=True):
        if isinstance(input, NeuralTreeOutput):
            obj_val_samples = self.get_objective_vals(input)

        else:
            obj_val_samples = input

        if pointwise:
            obj_val_samples = obj_val_samples.unsqueeze(-2)  # (num_samples, num_designs, 1, num_objectives)

        # assumes the first dimension of obj_vals corresponds to the qEHVI partitions
        if self._obj_dim == 1:
            acq_vals = torch.stack(
                [fn._sample_forward(vals) for fn, vals in zip(self.acq_functions, obj_val_samples.squeeze(-1))]
            ).squeeze(-1)
        else:
            acq_vals = torch.stack(
                [fn._compute_log_qehvi(vals.unsqueeze(0)) for fn, vals in zip(self.acq_functions, obj_val_samples)]
            )

        return acq_vals.mean(0)
