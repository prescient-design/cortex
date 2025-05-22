"""HuggingFace pretrained model root node for NeuralTree."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from transformers import AutoModel, AutoConfig, PreTrainedModel

from cortex.model.root import RootNode, RootNodeOutput


@dataclass
class HuggingFaceRootOutput(RootNodeOutput):
    """Extended root output that preserves HF model outputs."""

    # Standard fields from RootNodeOutput
    root_features: torch.Tensor
    corrupt_frac: Optional[torch.Tensor] = None

    # Additional HF-specific fields
    attention_mask: Optional[torch.Tensor] = None
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None
    last_hidden_state: Optional[torch.Tensor] = None
    pooler_output: Optional[torch.Tensor] = None

    # Raw HF model output for advanced use cases
    raw_output: Optional[Any] = None


class HuggingFaceRoot(RootNode):
    """
    Root node that wraps any HuggingFace pretrained model.

    This enables using pretrained transformers (BERT, RoBERTa, T5, etc.)
    as root nodes in the NeuralTree architecture while preserving
    cortex's corruption and transform capabilities.
    """

    def __init__(
        self,
        model_name_or_path: str,
        config: Optional[Union[Dict[str, Any], AutoConfig]] = None,
        trust_remote_code: bool = False,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        feature_extraction_layer: int = -1,  # Which layer to use for root_features
        pooling_strategy: str = "mean",  # "mean", "cls", "max", "pooler"
        freeze_pretrained: bool = False,
        corruption_process: Optional[Any] = None,
        **model_kwargs,
    ):
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.feature_extraction_layer = feature_extraction_layer
        self.pooling_strategy = pooling_strategy
        self.corruption_process = corruption_process

        # Load HuggingFace model
        if config is not None:
            if isinstance(config, dict):
                config = AutoConfig.from_dict(config)
            self.model = AutoModel.from_config(config, **model_kwargs)
        else:
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                **model_kwargs,
            )

        # Freeze pretrained weights if requested
        if freeze_pretrained:
            for param in self.model.parameters():
                param.requires_grad = False

        # Store model config for introspection
        self.config = self.model.config

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> HuggingFaceRootOutput:
        """
        Forward pass through HuggingFace model with cortex-compatible output.

        Args:
            input_ids: Token indices sequence tensor
            attention_mask: Mask to avoid attention on padding tokens
            token_type_ids: Segment token indices (for models like BERT)
            position_ids: Position indices
            inputs_embeds: Direct embedding inputs (alternative to input_ids)
            **kwargs: Additional model-specific arguments

        Returns:
            HuggingFaceRootOutput with extracted root_features and HF outputs
        """
        # Forward through HuggingFace model
        model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        # Extract features for cortex tree
        if hasattr(model_output, "hidden_states") and model_output.hidden_states is not None:
            # Use specified layer from hidden states
            hidden_state = model_output.hidden_states[self.feature_extraction_layer]
        elif hasattr(model_output, "last_hidden_state"):
            # Use last hidden state
            hidden_state = model_output.last_hidden_state
        else:
            # Fallback to main output tensor
            hidden_state = model_output[0] if isinstance(model_output, tuple) else model_output

        # Apply pooling strategy to get root_features
        root_features = self._pool_features(hidden_state, attention_mask)

        # Apply corruption if specified (for guided generation)
        corrupt_frac = None
        if self.corruption_process is not None:
            # This will be modernized in the torch.compile milestone
            corrupted_output = self.corruption_process(
                torch.stack([root_features]), **kwargs.get("corruption_params", {})
            )
            if hasattr(corrupted_output, "root_features"):
                root_features = corrupted_output.root_features[0]
                corrupt_frac = corrupted_output.corrupt_frac

        return HuggingFaceRootOutput(
            root_features=root_features,
            corrupt_frac=corrupt_frac,
            attention_mask=attention_mask,
            hidden_states=getattr(model_output, "hidden_states", None),
            attentions=getattr(model_output, "attentions", None),
            last_hidden_state=getattr(model_output, "last_hidden_state", None),
            pooler_output=getattr(model_output, "pooler_output", None),
            raw_output=model_output,
        )

    def _pool_features(self, hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply pooling strategy to extract root_features from hidden states.

        Args:
            hidden_state: [batch_size, seq_len, hidden_size] tensor
            attention_mask: [batch_size, seq_len] mask tensor

        Returns:
            root_features: [batch_size, hidden_size] tensor
        """
        if self.pooling_strategy == "cls":
            # Use [CLS] token (first token)
            return hidden_state[:, 0, :]

        elif self.pooling_strategy == "mean":
            # Mean pooling over sequence dimension
            if attention_mask is not None:
                # Mask out padding tokens
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
                sum_hidden = torch.sum(hidden_state * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_hidden / sum_mask
            else:
                return torch.mean(hidden_state, dim=1)

        elif self.pooling_strategy == "max":
            # Max pooling over sequence dimension
            if attention_mask is not None:
                # Set padding positions to large negative value
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size())
                hidden_state = hidden_state.masked_fill(~mask_expanded.bool(), -1e9)
            return torch.max(hidden_state, dim=1)[0]

        elif self.pooling_strategy == "pooler":
            # Use model's pooler output if available
            model_output = self.model.get_output_embeddings() if hasattr(self.model, "get_output_embeddings") else None
            if hasattr(self.model, "pooler") and self.model.pooler is not None:
                return self.model.pooler(hidden_state)
            else:
                # Fallback to CLS token
                return hidden_state[:, 0, :]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings (useful for adding special tokens)."""
        return self.model.resize_token_embeddings(new_num_tokens)

    def get_input_embeddings(self):
        """Get input embedding layer."""
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        """Set input embedding layer."""
        self.model.set_input_embeddings(value)

    @property
    def device(self):
        """Get model device."""
        return next(self.model.parameters()).device

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "HuggingFaceRoot":
        """
        Create HuggingFaceRoot from pretrained model.

        This is the primary way to create HF root nodes in practice.
        """
        return cls(model_name_or_path=model_name_or_path, **kwargs)
