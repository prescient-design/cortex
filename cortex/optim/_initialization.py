import pandas as pd
from botorch.utils.multi_objective.pareto import is_non_dominated

from cortex.acquisition import get_joint_objective_values
from cortex.model import infer_with_model
from cortex.model.tree import NeuralTree


def select_initial_sequences(
    data: pd.DataFrame,
    model: NeuralTree,
    graph_objectives: list[str],
    graph_constraints: dict[str, list[str]],
    graph_obj_transform: dict[str, dict[str, float]],
):
    predictions = infer_with_model(
        data=data,
        model=model,
    )

    obj_vals = get_joint_objective_values(
        input=predictions,
        objectives=graph_objectives,
        constraints=graph_constraints,
        scaling=graph_obj_transform,
    )

    non_dom_seeds = []
    for obj_val_sample in obj_vals:
        is_non_dom = is_non_dominated(obj_val_sample)
        non_dom_seeds.append(data.loc[is_non_dom, :])

    return pd.concat(non_dom_seeds, ignore_index=True)
