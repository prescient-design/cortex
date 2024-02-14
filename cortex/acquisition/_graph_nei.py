import numpy as np
import torch
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.acquisition.objective import IdentityMCObjective
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning
from botorch.utils.multi_objective.hypervolume import infer_reference_point
from botorch.utils.multi_objective.pareto import is_non_dominated

from cortex.model.tree import NeuralTree, NeuralTreeOutput, fetch_task_outputs

GRAPH_OBJECTIVES = ["stability", "log_fluorescence"]
GRAPH_CONSTRAINTS = {}
# rescale stability and log_fluorescence to [0, 1]
GRAPH_OBJ_TRANSFORM = {
    "stability": {"scale": 1 / 2.0, "shift": 2.0},
    "log_fluorescence": {"scale": 1 / 7.0, "shift": -4.0},
}


def get_obj_vals(
    tree_output: NeuralTreeOutput,
    objectives: list[str],
    constraints: dict[str, list[str]],
    scaling: dict[str, dict[str, float]],
):
    res = {}
    obj_vals = []
    for obj in objectives:
        obj_outputs = fetch_task_outputs(tree_output, obj)
        pred_means = obj_outputs["loc"].squeeze(-1)
        scaled_pred_means = (pred_means + scaling[obj]["shift"]) * scaling[obj]["scale"]
        obj_vals.append(scaled_pred_means)
        res[obj] = pred_means
        res[f"{obj}_scaled"] = scaled_pred_means
    obj_vals = torch.stack(obj_vals, dim=-1)  # (num_samples, num_objectives)

    if len(constraints) == 0:
        constraint_vals = torch.ones_like(obj_vals)
    else:
        constraint_vals = []
        for obj in constraints:
            _current = []
            for const in constraints[obj]:
                const_outputs = fetch_task_outputs(tree_output, const)
                satisfied_prob = const_outputs["logits"].softmax(-1)[..., 1]
                _current.append(satisfied_prob)
                res[const] = satisfied_prob
            constraint_vals.append(torch.stack(_current, dim=-1).prod(-1))
        constraint_vals = torch.stack(constraint_vals, dim=-1)  # (num_samples, num_objectives)

    res["joint"] = obj_vals * constraint_vals

    return res


def get_graph_nei_runtime_kwargs(
    model: NeuralTree,
    candidate_points: np.ndarray,
    objectives: list[str] = GRAPH_OBJECTIVES,
    constraints: dict[str, list[str]] = GRAPH_CONSTRAINTS,
    scaling: dict[str, dict[str, float]] = GRAPH_OBJ_TRANSFORM,
):
    with torch.inference_mode():
        tree_output = model.call_from_str_array(candidate_points, corrupt_frac=0.0)
    seed_preds = get_obj_vals(
        tree_output,
        objectives=objectives,
        constraints=constraints,
        scaling=scaling,
    )

    print("==== constructing acquisition function ====")
    f_baseline = seed_preds["joint"]  # (num_samples, num_baseline, num_objectives)
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
        objectives: list,
        constraints: dict,
        scaling: dict,
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

    def get_objective_vals(self, tree_output):
        return get_obj_vals(tree_output, self.objectives, self.constraints, self.scaling)

    def __call__(self, tree_output, pointwise=True):
        obj_vals = self.get_objective_vals(tree_output)["joint"]

        if pointwise:
            obj_vals = obj_vals.unsqueeze(-2)  # (num_samples, num_designs, 1, num_objectives)

        # assumes the first dimension of obj_vals corresponds to the qEHVI partitions
        if self._obj_dim == 1:
            acq_vals = torch.stack(
                [fn._sample_forward(vals) for fn, vals in zip(self.acq_functions, obj_vals.squeeze(-1))]
            ).squeeze(-1)
        else:
            acq_vals = torch.stack(
                [fn._compute_log_qehvi(vals.unsqueeze(0)) for fn, vals in zip(self.acq_functions, obj_vals)]
            )

        return acq_vals.mean(0)
