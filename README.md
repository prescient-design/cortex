# Cortex
## A Modular Architecture for Deep Learning Systems

<p align="center">
<img src="docs/assets/neural_tree.png" width=400px>
</p>

## What is `cortex`?

The `cortex` library provides a modular framework for neural network model composition for advanced use cases like multitask models, guided generative models, and multi-modal models.
Rather than tack on auxiliary abstractions to a single input --> single task model, `cortex` is built from the ground up to make adding new inputs or tasks to the model as seamless as possible.

In addition to streamlining advanced model construction, `cortex` provides meticulous implementations of classification and regression, addressing common challenges such as
- imbalanced classes
- disjoint multitask datasets
- noisy observations
- variable length inputs

Currently `cortex` supports SMILES-encoded small molecule and protein sequence inputs.
`cortex` can also provide epistemic and aleatoric uncertainty estimates (i.e. model uncertainty and measurement uncertainty) for use in decision-making problems such as active learning or black-box optimization.

## Why should I use `cortex`?

Deep learning is easy to learn and difficult to master. Seemingly insignificant bugs or design choices can dramatically affect performance. Unfortunately highly optimized model implementations are often very difficult to extend to new problems because the code is overly prescriptive.
`cortex` is designed to abstract away as many of these details as possible, while providing maximum flexibility to the user.


## Installation

1.  Create a new conda environment.

    ```bash
    conda create --name cortex-env python=3.10 -y && conda activate cortex-env
    ```

2.  (optional) If desired install dependencies from frozen requirements files.

    `pip install -r requirements.txt -r requirements-dev.txt`

    These files fix the exact version of all dependencies and therefore should create a known good environment.
    However, this is likely more stringent than strictly necessary and can make it difficult to work in environments with multiple projects installed.
    If you skip this step, all dependencies will be fetched during package installation based on `requirements.in` which attempts to be as loose as possible in specifying compatible package versions.

    To update the frozen dependencies run

    `pip-compile --resolver=backtracking requirements.in > requirements.txt`.

3.  Install cortex.

    `pip install -e .[dev]`

## Running
  
Use `cortex_train_model --config-name <CONFIG_NAME>` to train, e.g.:
```
cortex_train_model --config-name train_ab_seqcnn wandb_mode=offline fit=smoke_test
```

Supported configs are

- `train_ab_seqcnn` to train a SeqCNN from scratch.


## How to launch a WANDB sweep on a cluster

1. Configure the sweep `.yaml`, e.g. `./wandb_config/ab_model_sweep.yaml`
2. Run `wandb sweep wandb_config/ab_model_sweep.yaml`
3. Copy the sweep id to `scripts/wandb_agent_array.bsub`
4. Run `bsub < scripts/wandb_agent_array.bsub`

## Contributing

Contributions are welcome, especially tutorials and documentation.

### Testing

`pytest -v --cov-report term-missing --cov=./cortex ./tests`


### Maintainers

- Samuel Stanton


## What's in the name?

The cerebral cortex is the outer layer of the brain (i.e. the grey matter). Different parts of the cortex are responsible for different functions, including sensation, motor function, and reasoning. The cortex is supported and interconnected by subcortical structures (i.e. the white matter). Similarly the `cortex` package provides modular neural network components for different functions, such as extracting features from different types of inputs, or making predictions for different tasks.
`cortex` also provides a flexible superstructure to compose those modules into advanced architectures and route signals between them.
