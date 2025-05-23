"""
Corruption Layer v2: torch.compile compatible corruption with always-apply pattern.

This module implements a compilation-friendly corruption layer that eliminates
dynamic control flow by always applying all corruption operations and using
weights to control their contribution.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from cortex.corruption._gaussian_corruption import GaussianCorruptionProcess
from cortex.corruption._mask_corruption import MaskCorruptionProcess
from cortex.optim.generative._lambo_v2 import CorruptionParams


class CorruptionLayerV2(nn.Module):
    """
    torch.compile compatible corruption layer using always-apply pattern.

    Instead of dynamic control flow based on corruption type, this layer
    always applies all corruption operations and uses weights to control
    their contribution. This enables torch.compile optimization.
    """

    def __init__(
        self,
        mask_corruption_config: Optional[Dict[str, Any]] = None,
        gaussian_corruption_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        # Initialize both corruption processes
        self.mask_corruption = MaskCorruptionProcess(**(mask_corruption_config or {}))
        self.gaussian_corruption = GaussianCorruptionProcess(**(gaussian_corruption_config or {}))

    def forward(self, embeddings: torch.Tensor, corruption_params: CorruptionParams) -> torch.Tensor:
        """
        Apply corruption using always-apply pattern.

        Args:
            embeddings: Input embeddings to corrupt [batch_size, seq_len, embed_dim]
            corruption_params: Parameters controlling corruption application

        Returns:
            Corrupted embeddings with same shape as input
        """
        # Always apply both corruption types
        mask_result = self._apply_mask_corruption(embeddings, corruption_params)
        gaussian_result = self._apply_gaussian_corruption(embeddings, corruption_params)

        # Use weights to control contribution (0.0 or 1.0 for discrete selection)
        # This is compilation-friendly since it's pure tensor operations
        weighted_mask = corruption_params.mask_weight * mask_result
        weighted_gaussian = corruption_params.gaussian_weight * gaussian_result
        weighted_original = (1.0 - corruption_params.mask_weight - corruption_params.gaussian_weight) * embeddings

        # Sum all contributions
        return weighted_mask + weighted_gaussian + weighted_original

    def _apply_mask_corruption(self, embeddings: torch.Tensor, corruption_params: CorruptionParams) -> torch.Tensor:
        """Apply mask corruption process."""
        if corruption_params.mask_noise is not None:
            # Use provided noise
            return self.mask_corruption.apply_corruption(embeddings, noise=corruption_params.mask_noise)
        else:
            # Generate noise internally
            return self.mask_corruption(embeddings)

    def _apply_gaussian_corruption(self, embeddings: torch.Tensor, corruption_params: CorruptionParams) -> torch.Tensor:
        """Apply Gaussian corruption process."""
        if corruption_params.gaussian_noise is not None:
            # Use provided noise
            return self.gaussian_corruption.apply_corruption(embeddings, noise=corruption_params.gaussian_noise)
        else:
            # Generate noise internally
            return self.gaussian_corruption(embeddings)


class StaticCorruptionMixin:
    """
    Mixin to add static corruption capability to neural tree components.

    This replaces the dynamic isinstance-based corruption selection with
    a static always-apply approach that's compatible with torch.compile.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize v2 corruption layer if corruption configs are provided
        if hasattr(self, "corruption_process"):
            self.corruption_layer = self._create_corruption_layer_v2()

    def _create_corruption_layer_v2(self) -> CorruptionLayerV2:
        """Create v2 corruption layer from existing corruption process."""
        # Extract config from existing corruption process
        mask_config = None
        gaussian_config = None

        # Handle existing corruption process types
        if hasattr(self.corruption_process, "mask_token_id"):
            mask_config = {
                "mask_token_id": self.corruption_process.mask_token_id,
                "corruption_prob": getattr(self.corruption_process, "corruption_prob", 0.15),
            }

        if hasattr(self.corruption_process, "noise_std"):
            gaussian_config = {"noise_std": self.corruption_process.noise_std}

        return CorruptionLayerV2(mask_corruption_config=mask_config, gaussian_corruption_config=gaussian_config)

    def apply_corruption_v2(self, embeddings: torch.Tensor, corruption_params: CorruptionParams) -> torch.Tensor:
        """Apply corruption using v2 static approach."""
        if hasattr(self, "corruption_layer"):
            return self.corruption_layer(embeddings, corruption_params)
        else:
            # Fallback to no corruption
            return embeddings


@dataclass
class CorruptionConfig:
    """Configuration for corruption layer v2."""

    mask_corruption: bool = True
    gaussian_corruption: bool = True
    mask_token_id: int = 103  # [MASK] token ID
    mask_corruption_prob: float = 0.15
    gaussian_noise_std: float = 0.1

    def create_layer(self) -> CorruptionLayerV2:
        """Create corruption layer from config."""
        mask_config = None
        gaussian_config = None

        if self.mask_corruption:
            mask_config = {"mask_token_id": self.mask_token_id, "corruption_prob": self.mask_corruption_prob}

        if self.gaussian_corruption:
            gaussian_config = {"noise_std": self.gaussian_noise_std}

        return CorruptionLayerV2(mask_corruption_config=mask_config, gaussian_corruption_config=gaussian_config)


def convert_corruption_process_to_v2(
    corruption_process: Any, corruption_config: Optional[CorruptionConfig] = None
) -> CorruptionLayerV2:
    """
    Convert existing v1 corruption process to v2 layer.

    Args:
        corruption_process: Existing corruption process
        corruption_config: Optional override configuration

    Returns:
        Equivalent v2 corruption layer
    """
    if corruption_config is None:
        corruption_config = CorruptionConfig()

    # Extract parameters from existing process
    if isinstance(corruption_process, MaskCorruptionProcess):
        corruption_config.mask_token_id = corruption_process.mask_token_id
        corruption_config.mask_corruption_prob = getattr(corruption_process, "corruption_prob", 0.15)
    elif isinstance(corruption_process, GaussianCorruptionProcess):
        corruption_config.gaussian_noise_std = corruption_process.noise_std

    return corruption_config.create_layer()
