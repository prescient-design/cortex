# Cortex Architecture Refactor: HuggingFace Native Redesign

## Executive Summary

After 2.5 years of development, cortex has proven its core algorithmic innovations but suffers from infrastructure limitations that prevent performance optimization and broader adoption. The solution is to preserve cortex's novel ML contributions while migrating to HuggingFace/Lightning native architecture for modern infrastructure benefits.

## Current State Analysis

### What Cortex Got Right ✅

1. **NeuralTree Architecture**: The root/trunk/branch/leaf abstraction is genuinely innovative and enables clean multi-task model composition
2. **Sophisticated ML Algorithms**:
   - Regression parameterization with natural parameters and label smoothing
   - Round-robin minority upsampling for balanced training
   - Discriminative input corruption for robust learning
   - Guided discrete diffusion (LaMBO) for sequence optimization
3. **Clean Task Abstraction**: The `model ↔ task ↔ data` boundary provides good separation of concerns
4. **Hydra Configuration**: Composable config system enables flexible model architecture specification

### Core Performance Problems ❌

1. **GPU Underutilization**: Transforms in forward pass prevent dataloader parallelism
2. **torch.compile Incompatibility**: Dynamic control flow and isinstance checks break compilation
3. **Transform Ownership vs. Execution**: Tokenizers logically belong to root nodes but executing them there kills performance
4. **Multi-task Transform Complexity**: Different tasks need different tokenizers but current architecture makes this awkward

### Infrastructure Gaps ❌

1. **No HuggingFace Integration**: Can't leverage pretrained models or standard processors
2. **Awkward Lightning Integration**: Manual optimization and multi-task training don't fit Lightning's assumptions
3. **Limited Ecosystem Compatibility**: Custom implementations instead of standard interfaces

## Root Cause: Architectural Coupling

The fundamental issue is **necessary algorithmic coupling** (corruption processes need model state for guided generation) got mixed with **unnecessary infrastructure coupling** (tokenization happening in forward pass). This created performance bottlenecks and prevented modern optimization techniques.

### Specific Coupling Issues

**Transform Location**:
- Problem: `TransformerRoot.forward()` does tokenization → blocks parallelism
- Root Cause: Convenience coupling, not algorithmic necessity

**Dynamic Control Flow**:
```python
# Breaks torch.compile
if isinstance(self.corruption_process, MaskCorruptionProcess):
    # different path
elif isinstance(self.corruption_process, GaussianCorruptionProcess):
    # different path
```

**Multi-task Transform Ownership**:
- Problem: Tasks don't know which tokenizer to use without model
- Current: Circular dependency between task formatting and model transforms

## Refactor Strategy: HuggingFace Native Architecture

### Core Principle
**Preserve algorithmic innovations, modernize infrastructure**

- Keep: Tree architecture, ML algorithms, guided generation, Hydra composition
- Replace: Model base classes, config system, transform execution, training loop

### Phase 1: Infrastructure Migration

#### 1.1 HuggingFace Model Integration
```python
class NeuralTreeModel(PreTrainedModel):
    config_class = NeuralTreeConfig

    def __init__(self, config):
        super().__init__(config)

        # Preserve existing tree composition via Hydra
        self.root_nodes = nn.ModuleDict()
        for name, root_config in config.roots.items():
            if root_config.use_hf_model:
                # Native HF integration
                self.root_nodes[name] = AutoModel.from_config(root_config.hf_config)
            else:
                # Keep custom roots
                self.root_nodes[name] = hydra.utils.instantiate(root_config.cortex_config)

        # Existing trunk/branch/leaf logic unchanged
        self.trunk_node = hydra.utils.instantiate(config.trunk)
        self.branch_nodes = nn.ModuleDict(...)
        self.leaf_nodes = nn.ModuleDict(...)
```

#### 1.2 Config System Redesign
```python
@dataclass
class NeuralTreeConfig(PretrainedConfig):
    model_type = "neural_tree"

    # Preserve Hydra composition
    roots: Dict[str, RootConfig] = field(default_factory=dict)
    trunk: Dict[str, Any] = field(default_factory=dict)
    branches: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tasks: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # New: Transform registry
    processors: Dict[str, str] = field(default_factory=dict)  # root_name -> processor_name

@dataclass
class RootConfig:
    # Dual mode: HF or custom
    use_hf_model: bool = False
    hf_config: Optional[AutoConfig] = None
    cortex_config: Optional[Dict[str, Any]] = None
    processor_name: Optional[str] = None
```

#### 1.3 Transform Execution Separation
```python
class CortexDataset(Dataset):
    def __init__(self, hf_dataset, model_config):
        self.dataset = hf_dataset

        # Build processors from model config
        self.processors = {}
        for root_name, processor_name in model_config.processors.items():
            self.processors[root_name] = AutoProcessor.from_pretrained(processor_name)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Apply static transforms in dataloader (parallel execution)
        processed = {}
        for root_name, processor in self.processors.items():
            if root_name in item:
                processed[root_name] = processor(item[root_name], return_tensors="pt")

        return processed
```

### Phase 2: torch.compile Compatibility

#### 2.1 Corruption Layer Redesign
Apply the "always apply" pattern from modern diffusion models:

```python
class CorruptionLayer(nn.Module):
    """Compilation-friendly corruption that always applies operations."""

    def forward(self, embeddings, corruption_params):
        # Always apply both corruption types, use params to weight them
        mask_result = self.mask_corruption(embeddings, corruption_params.mask_noise)
        gaussian_result = self.gaussian_corruption(embeddings, corruption_params.gaussian_noise)

        # Use corruption_type as binary weights (0.0 or 1.0)
        return (corruption_params.mask_weight * mask_result +
                corruption_params.gaussian_weight * gaussian_result)
```

#### 2.2 Static Forward Pass
```python
def forward(self, inputs, corruption_params=None):
    # All inputs pre-processed, no dynamic transforms
    # Single path through model with tensor operations only

    root_outputs = {}
    for root_name, root_input in inputs.items():
        root_outputs[root_name] = self.root_nodes[root_name](root_input)

    # Apply corruption if specified (always same operations)
    if corruption_params is not None:
        for root_name in root_outputs:
            root_outputs[root_name] = self.corruption_layer(
                root_outputs[root_name],
                corruption_params[root_name]
            )

    # Rest of tree forward pass unchanged
    trunk_outputs = self.trunk_node(*root_outputs.values())
    # ...
```

### Phase 3: Lightning Training Integration

#### 3.1 Clean Multi-task Training
```python
class NeuralTreeModule(LightningModule):
    def __init__(self, model_config, task_configs):
        super().__init__()
        self.model = NeuralTreeModel.from_config(model_config)
        self.tasks = {name: hydra.utils.instantiate(cfg) for name, cfg in task_configs.items()}

    def training_step(self, batch, batch_idx):
        # Clean single-responsibility training step
        total_loss = 0

        for task_name, task_batch in batch.items():
            task = self.tasks[task_name]

            # Model forward pass (compilable)
            outputs = self.model(task_batch)

            # Task-specific loss computation
            task_loss = task.compute_loss(outputs, task_batch)
            total_loss += task_loss

            self.log(f"{task_name}/loss", task_loss)

        return total_loss

    def configure_optimizers(self):
        # Standard Lightning optimizer configuration
        return torch.optim.AdamW(self.parameters(), lr=1e-4)
```

### Phase 4: Guided Generation Modernization

#### 4.1 Clean LaMBO API
```python
class LaMBOOptimizer:
    def __init__(self, model, objective, config):
        self.model = model
        self.objective = objective
        self.corruption_scheduler = CorruptionScheduler(config)

    def step(self, sequences):
        # Clean separation: scheduler provides corruption params
        corruption_params = self.corruption_scheduler.get_params(self.step_count)

        # Model provides clean guided forward interface
        outputs = self.model.guided_forward(
            sequences=sequences,
            corruption_params=corruption_params,
            guidance_layer="trunk"
        )

        # Optimization logic isolated from model internals
        return self.optimize_sequences(outputs)
```

## Implementation Plan

### Milestone 1: HF Model Integration (2-3 weeks)
- [ ] Create `NeuralTreeConfig` class extending `PretrainedConfig`
- [ ] Implement `NeuralTreeModel(PreTrainedModel)` wrapper
- [ ] Migrate one root node to support both HF and custom models
- [ ] Test config serialization/deserialization
- [ ] Verify existing Hydra configs still work

### Milestone 2: Transform Execution Migration (2-3 weeks)
- [ ] Create `CortexDataset` with processor integration
- [ ] Move tokenization from `TransformerRoot.forward()` to dataloader
- [ ] Implement dual-mode operation (training vs inference)
- [ ] Add processor auto-detection from model config
- [ ] Benchmark dataloader parallelism improvements

### Milestone 3: torch.compile Compatibility (2-3 weeks)
- [ ] Redesign corruption as "always apply" pattern
- [ ] Remove all dynamic control flow from forward pass
- [ ] Create compilation-friendly model entry points
- [ ] Add compilation benchmarks and tests
- [ ] Verify guided generation still works correctly

### Milestone 4: Lightning Integration (1-2 weeks)
- [ ] Create `NeuralTreeModule(LightningModule)`
- [ ] Clean up multi-task training loop
- [ ] Remove manual optimization complexity
- [ ] Add standard Lightning features (callbacks, logging)
- [ ] Migration guide for existing training scripts

### Milestone 5: LaMBO Modernization (2-3 weeks)
- [ ] Extract model manipulation into clean interfaces
- [ ] Create `CorruptionScheduler` abstraction
- [ ] Implement `guided_forward()` model method
- [ ] Test algorithmic equivalence with current implementation
- [ ] Performance benchmarks

## Success Metrics

### Performance Targets
- **GPU Utilization**: 2x improvement from dataloader parallelism
- **Training Speed**: 1.5x improvement from torch.compile
- **Memory Efficiency**: Comparable or better than current implementation

### Functionality Preservation
- **Algorithmic Equivalence**: All ML innovations produce identical results
- **Config Compatibility**: Existing Hydra configs work with minimal changes
- **API Stability**: Core user-facing APIs remain similar

### Infrastructure Benefits
- **HF Ecosystem**: Can load/save models to HF Hub
- **Pretrained Models**: Can use any HF transformer as root node
- **Standard Training**: Compatible with HF Trainer and Lightning
- **Modern Optimization**: torch.compile, mixed precision, multi-GPU

## Risk Mitigation

### Backwards Compatibility
- Maintain existing API during transition
- Provide clear migration guides
- Keep old code paths until new ones are proven

### Performance Validation
- Comprehensive benchmarks at each milestone
- A/B testing between old and new implementations
- Memory profiling to catch regressions

### Algorithmic Correctness
- Unit tests for each ML component
- End-to-end integration tests
- Numerical equivalence verification

## Migration Strategy for Existing Users

Since cortex has seen minimal external adoption, focus on **internal migration**:

1. **Parallel Implementation**: Build new architecture alongside existing code
2. **Gradual Migration**: Move one component at a time
3. **Performance Validation**: Benchmark each change
4. **Clean Cutover**: Remove old code once new is proven

## Long-term Vision

Post-refactor, cortex becomes:
- **Best-in-class multi-task learning framework** with HF ecosystem integration
- **Production-ready guided generation** with modern optimization
- **Research platform** that doesn't sacrifice performance for flexibility
- **Genuinely reusable** architecture that others can build upon

The refactor preserves your 2.5 years of ML innovation while providing the infrastructure needed for continued research and potential broader adoption.
