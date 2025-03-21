# Task-Leaf Configuration Issue

## Problem

The current architecture in the Cortex library has a code smell related to how the leaf node parameters are configured. The issue is that leaf nodes are not instantiated directly from configuration but through task objects, which leads to several problems:

1. **Parameter Pass-Through Gap**: When a new parameter is added to a leaf node class (like `corruption_process` and `corruption_rate` in `DenoisingLanguageModelLeaf`), there's no direct way to configure it through Hydra config files. The task class doesn't automatically have parameters that correspond to all possible leaf parameters.

2. **Configuration Duplication**: To support a new leaf parameter, we need to duplicate that parameter in the task class and pass it through in the `create_leaf` method. This creates a maintenance burden as new leaf features require updating multiple classes.

3. **Hidden Parameters**: The parameters available for configuring a leaf node are not explicitly visible in the configuration schema, making it hard for users to discover and use them.

4. **Tight Coupling**: Tasks are tightly coupled to specific leaf implementations, making it difficult to extend leaf functionality without modifying the task code.

## Current Approach

Currently, when adding a new parameter to a leaf node, we need to:

1. Add the parameter to the leaf node class (e.g., `DenoisingLanguageModelLeaf`)
2. Add the same parameter to the corresponding task class (e.g., `DenoisingLanguageModelTask`)
3. Pass the parameter from the task's `create_leaf` method to the leaf constructor

This approach works but leads to code duplication and tight coupling between tasks and leaves.

## Potential Solutions

### Short-Term (Current Approach)

- Continue duplicating parameters between task and leaf classes
- Document the pattern clearly for developers
- Ensure parameter names and defaults are consistent between task and leaf

### Medium-Term (Configuration Dictionary)

Modify the task classes to accept a generic dictionary of leaf arguments that get passed directly to the leaf constructor:

```python
def __init__(
    self,
    # ... existing parameters ...
    leaf_args: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    # ... existing initialization ...
    self.leaf_args = leaf_args or {}

def create_leaf(self, in_dim: int, branch_key: str) -> LeafNode:
    return LeafNodeType(
        in_dim=in_dim,
        # ... other required parameters ...
        **self.leaf_args
    )
```

This would allow configuring leaf-specific parameters without modifying task classes each time:

```yaml
task:
  _target_: cortex.task.DenoisingLanguageModelTask
  # ... other parameters ...
  leaf_args:
    corruption_process:
      _target_: cortex.corruption.SubstitutionCorruptionProcess.from_blosum62
    corruption_rate: 0.05
```

### Long-Term (Architecture Refactoring)

A more complete solution would involve restructuring the architecture to separate the configuration of tasks and leaves:

1. **Decoupled Configuration**: Allow leaf nodes to be configured separately from tasks, with tasks referencing leaf configurations

2. **Configuration Schema Registry**: Create a registry of available configuration schemas that automatically includes all parameters for each component

3. **Factory Pattern**: Implement a proper factory pattern where task creation and leaf creation are separated but coordinated

Challenges with this approach include:
- Some properties of leaves (like output dimensionality) are determined by the task, so completely decoupling isn't trivial
- Backward compatibility would need to be maintained
- Significant refactoring would be required

## Recommendation

The medium-term solution (configuration dictionary) offers the best balance between addressing the issue and minimizing changes to the existing architecture. It allows for flexible leaf configuration without requiring changes to task classes for each new leaf parameter.

This approach could be implemented incrementally, starting with the most complex leaf types (like DenoisingLanguageModelLeaf) that are likely to require more configuration options in the future.
