<p align="center">
<img src="docs/assets/cortex_logo_concept_v1.png" width=300px>
</p>

# A Modular Architecture for Deep Learning Systems

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

    ```bash
    conda create --name cortex-env python=3.10 -y && conda activate cortex-env
    python -m pip install -r requirements.in
    pip install -e .
    ```


If you have a package version issue we provide pinned versions of all dependencies in `requirements.txt`.
To update the frozen dependencies run

```bash
pip-compile --resolver=backtracking requirements.in
```


## Running

Use `cortex_train_model --config-name <CONFIG_NAME>` to train, e.g.:
```
cortex_train_model --config-name train_protein_model wandb_mode=offline
```


## How to launch a WANDB sweep

1. Configure the sweep `.yaml`, e.g. `./wandb_config/ab_model_sweep.yaml`
2. Run `wandb sweep wandb_config/ab_model_sweep.yaml`
3. Launch the wandb agents using a scheduler of your choice, e.g. SLURM or LSF


## Contributing

Contributions are welcome!

### Install dev requirements and pre-commit hooks

```bash
python -m pip install -r requirements-dev.in
pre-commit install
```

### Testing

```bash
pytest -v --cov-report term-missing --cov=./cortex ./tests
```

### Build and browse docs locally

```bash
make -C docs html
cd docs/build/html
python -m http.server
```

Then open `http://localhost:8000` in your browser.
```


### Maintainers

- Samuel Stanton


## What's in the name?

The cerebral cortex is the outer layer of the brain (i.e. the grey matter). Different parts of the cortex are responsible for different functions, including sensation, motor function, and reasoning. The cortex is supported and interconnected by subcortical structures (i.e. the white matter). Similarly the `cortex` package provides modular neural network components for different functions, such as extracting features from different types of inputs, or making predictions for different tasks.
`cortex` also provides a flexible superstructure to compose those modules into advanced architectures and route signals between them.

<p align="center">
<img src="docs/assets/neural_tree_banner.png" width=1200px>
</p>
