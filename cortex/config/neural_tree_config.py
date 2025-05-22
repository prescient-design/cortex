"""HuggingFace-compatible configuration for NeuralTree models."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from transformers import PretrainedConfig


@dataclass
class RootConfig:
    """Configuration for a single root node in the neural tree."""

    # Dual mode: HuggingFace or custom
    use_hf_model: bool = False
    hf_config: Optional[Dict[str, Any]] = None
    cortex_config: Optional[Dict[str, Any]] = None
    processor_name: Optional[str] = None

    def __post_init__(self):
        if self.use_hf_model and self.hf_config is None:
            raise ValueError("hf_config must be provided when use_hf_model=True")
        if not self.use_hf_model and self.cortex_config is None:
            raise ValueError("cortex_config must be provided when use_hf_model=False")


class NeuralTreeConfig(PretrainedConfig):
    """
    Configuration class for NeuralTree models that preserves Hydra composition
    while enabling HuggingFace ecosystem integration.

    This configuration supports both traditional cortex components and modern
    HuggingFace pretrained models within the same neural tree architecture.
    """

    model_type = "neural_tree"

    def __init__(
        self,
        roots: Optional[Dict[str, Any]] = None,
        trunk: Optional[Dict[str, Any]] = None,
        branches: Optional[Dict[str, Dict[str, Any]]] = None,
        tasks: Optional[Dict[str, Dict[str, Any]]] = None,
        processors: Optional[Dict[str, str]] = None,
        optimizer_config: Optional[Dict[str, Any]] = None,
        lr_scheduler_config: Optional[Dict[str, Any]] = None,
        ensemble_size: int = 1,
        channel_dim: int = 64,
        dropout_prob: float = 0.0,
        enable_torch_compile: bool = False,
        compile_mode: str = "default",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Core tree architecture (preserved from existing cortex)
        self.roots = roots or {}
        self.trunk = trunk or {}
        self.branches = branches or {}
        self.tasks = tasks or {}

        # New: Transform and processor registry for dataloader execution
        self.processors = processors or {}  # root_name -> processor_name

        # Training configuration (migrated from fit_cfg)
        self.optimizer_config = optimizer_config or {}
        self.lr_scheduler_config = lr_scheduler_config or {}

        # Model global settings
        self.ensemble_size = ensemble_size
        self.channel_dim = channel_dim
        self.dropout_prob = dropout_prob

        # Compilation and performance settings
        self.enable_torch_compile = enable_torch_compile
        self.compile_mode = compile_mode  # "default", "reduce-overhead", "max-autotune"

        # Convert root configs to RootConfig objects if they're dicts
        if self.roots:
            for root_name, root_cfg in self.roots.items():
                if isinstance(root_cfg, dict):
                    self.roots[root_name] = RootConfig(**root_cfg)

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        output = super().to_dict()

        # Convert RootConfig objects to dicts
        if hasattr(self, "roots") and self.roots:
            roots_dict = {}
            for root_name, root_config in self.roots.items():
                if isinstance(root_config, RootConfig):
                    roots_dict[root_name] = {
                        "use_hf_model": root_config.use_hf_model,
                        "hf_config": root_config.hf_config,
                        "cortex_config": root_config.cortex_config,
                        "processor_name": root_config.processor_name,
                    }
                else:
                    roots_dict[root_name] = root_config
            output["roots"] = roots_dict

        return output

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """Create from dictionary (used during deserialization)."""
        # Convert root configs back to RootConfig objects
        if "roots" in config_dict:
            roots = {}
            for root_name, root_data in config_dict["roots"].items():
                if isinstance(root_data, dict) and "use_hf_model" in root_data:
                    roots[root_name] = RootConfig(**root_data)
                else:
                    roots[root_name] = root_data
            config_dict["roots"] = roots

        return super().from_dict(config_dict, **kwargs)

    def add_root(self, name: str, root_config: RootConfig):
        """Add a root node configuration."""
        self.roots[name] = root_config

    def add_hf_root(self, name: str, model_name_or_path: str, processor_name: Optional[str] = None):
        """Convenience method to add a HuggingFace pretrained root."""
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_name_or_path)
        root_config = RootConfig(
            use_hf_model=True, hf_config=hf_config.to_dict(), processor_name=processor_name or model_name_or_path
        )
        self.add_root(name, root_config)

        # Register processor for dataloader execution
        if processor_name:
            self.processors[name] = processor_name

    def add_cortex_root(self, name: str, cortex_config: Dict[str, Any], processor_name: Optional[str] = None):
        """Convenience method to add a traditional cortex root."""
        root_config = RootConfig(use_hf_model=False, cortex_config=cortex_config, processor_name=processor_name)
        self.add_root(name, root_config)

        # Register processor if provided
        if processor_name:
            self.processors[name] = processor_name

    def to_hydra_config(self) -> Dict[str, Any]:
        """
        Convert back to Hydra-style configuration for backwards compatibility.
        This allows existing training scripts to work with minimal changes.
        """
        hydra_config = {
            "roots": {},
            "trunk": self.trunk,
            "branches": self.branches,
            "tasks": self.tasks,
            "ensemble_size": self.ensemble_size,
            "channel_dim": self.channel_dim,
            "dropout_prob": self.dropout_prob,
        }

        # Convert root configs back to Hydra format
        for root_name, root_config in self.roots.items():
            if root_config.use_hf_model:
                # For HF models, we'll need to create a wrapper config
                hydra_config["roots"][root_name] = {
                    "_target_": "cortex.model.root.HuggingFaceRoot",
                    "model_name_or_path": root_config.processor_name,
                    "config": root_config.hf_config,
                }
            else:
                # Use existing cortex config directly
                hydra_config["roots"][root_name] = root_config.cortex_config

        return hydra_config

    @classmethod
    def from_hydra_config(cls, hydra_config: Dict[str, Any]) -> "NeuralTreeConfig":
        """
        Create NeuralTreeConfig from existing Hydra configuration.
        This enables migration from existing configs.
        """
        config = cls()

        # Extract core tree components
        config.trunk = hydra_config.get("trunk", {})
        config.branches = hydra_config.get("branches", {})
        config.tasks = hydra_config.get("tasks", {})

        # Extract global settings
        config.ensemble_size = hydra_config.get("ensemble_size", 1)
        config.channel_dim = hydra_config.get("channel_dim", 64)
        config.dropout_prob = hydra_config.get("dropout_prob", 0.0)

        # Convert root configurations
        for root_name, root_cfg in hydra_config.get("roots", {}).items():
            if isinstance(root_cfg, dict):
                # Detect if this is a HF model based on target or model_name_or_path
                target = root_cfg.get("_target_", "")
                if "HuggingFace" in target or "model_name_or_path" in root_cfg:
                    config.add_hf_root(
                        root_name, root_cfg.get("model_name_or_path", ""), root_cfg.get("processor_name")
                    )
                else:
                    config.add_cortex_root(root_name, root_cfg)

        return config
