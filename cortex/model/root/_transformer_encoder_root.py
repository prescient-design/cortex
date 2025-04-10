import warnings
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig
from torch import LongTensor, nn
from transformers import AutoConfig, AutoModel

from cortex.corruption import CorruptionProcess, MaskCorruptionProcess
from cortex.model.root import RootNode, RootNodeOutput
from cortex.transforms import HuggingFaceTokenizerTransform, PadTransform, ToTensor


@dataclass
class TransformerEncoderRootOutput(RootNodeOutput):
    """Output of TransformerEncoderRoot."""

    root_features: torch.Tensor
    padding_mask: torch.Tensor
    corrupt_frac: Optional[torch.Tensor] = None
    src_tok_idxs: Optional[torch.LongTensor] = None
    tgt_tok_idxs: Optional[torch.LongTensor] = None
    src_tok_embs: Optional[torch.Tensor] = None
    is_corrupted: Optional[torch.Tensor] = None


class TransformerEncoderRoot(RootNode):
    """
    A root node that wraps a Hugging Face transformer encoder model (e.g., BERT, RoBERTa, LBSTER encoders).

    Example Hydra Config:
    ```yaml
    roots:
      protein_encoder:
        _target_: cortex.model.root.TransformerEncoderRoot
        tokenizer_transform: ??? # Needs instantiation elsewhere
        model_name_or_path: "facebook/esm2_t6_8M_UR50D"
        use_pretrained: True
        max_len: 512
        attn_implementation: "sdpa"
        out_dim: 320 # Example, will be inferred
    ```
    """

    def __init__(
        self,
        tokenizer_transform: HuggingFaceTokenizerTransform,
        model_name_or_path: str,
        max_len: int,
        out_dim: int = None,
        use_pretrained: bool = True,
        attn_implementation: Optional[str] = "sdpa",
        config_overrides: Optional[DictConfig] = None,
        corruption_process: Optional[CorruptionProcess] = None,
        train_transforms=None,
        eval_transforms=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer_transform.tokenizer
        self.pad_tok_idx = self.tokenizer.padding_idx
        self.masking_idx = getattr(self.tokenizer, "masking_idx", None)
        self.max_len = max_len

        # Load or create Hugging Face model configuration
        if use_pretrained:
            self.transformer = AutoModel.from_pretrained(model_name_or_path, attn_implementation=attn_implementation)
        else:
            config = AutoConfig.from_pretrained(model_name_or_path)
            # Apply configuration overrides if specified
            if config_overrides is not None:
                for key, value in config_overrides.items():
                    setattr(config, key, value)
            self.transformer = AutoModel.from_config(config)

        # Determine output dimension from model
        self._out_dim = self.transformer.config.hidden_size

        # Validate against provided out_dim if specified
        if out_dim is not None and out_dim != self._out_dim:
            warnings.warn(
                f"Provided out_dim ({out_dim}) does not match model's hidden_size ({self._out_dim}). "
                f"Using model's hidden_size.",
                stacklevel=2,
            )

        # Set up transforms
        shared_transforms = [
            tokenizer_transform,
            ToTensor(padding_value=self.pad_tok_idx),
            PadTransform(max_length=self.max_len, pad_value=self.pad_tok_idx),
        ]
        train_transforms = [] if train_transforms is None else list(train_transforms.values())
        eval_transforms = [] if eval_transforms is None else list(eval_transforms.values())
        self.train_transform = nn.Sequential(*(train_transforms + shared_transforms))
        self.eval_transform = nn.Sequential(*(eval_transforms + shared_transforms))

        self.corruption_process = corruption_process

    @property
    def out_dim(self):
        return self._out_dim

    @property
    def device(self):
        return next(self.transformer.parameters()).device

    def initialize_weights(self, **kwargs):
        # Default random initialization or handled by HF
        pass

    def init_seq(
        self,
        inputs: Optional[Union[np.ndarray, torch.Tensor]] = None,
        seq_array: Optional[np.ndarray] = None,
        tgt_tok_idxs: Optional[LongTensor] = None,
        src_tok_embs: Optional[torch.Tensor] = None,
        corrupt_frac: float = 0.0,
        **kwargs,
    ):
        # infer input type if not specified
        if inputs is not None:
            if isinstance(inputs, np.ndarray):
                seq_array = inputs
            if isinstance(inputs, LongTensor):
                tgt_tok_idxs = inputs
            elif isinstance(inputs, torch.Tensor):
                src_tok_embs = inputs
            msg = "inputs is deprecated, use a specific argument instead"
            warnings.warn(msg, PendingDeprecationWarning, stacklevel=2)

        # Determine batch size from any available input
        batch_size = None
        if seq_array is not None:
            batch_size = seq_array.shape[0]
        elif tgt_tok_idxs is not None:
            batch_size = tgt_tok_idxs.shape[0]
        elif src_tok_embs is not None:
            batch_size = src_tok_embs.shape[0]

        # Fallback to default batch size of 1 if no inputs are provided
        if batch_size is None:
            batch_size = 1

        if "mask_frac" in kwargs:
            corrupt_frac = kwargs["mask_frac"]
            msg = "mask_frac is deprecated, use corrupt_frac instead."
            warnings.warn(msg, PendingDeprecationWarning, stacklevel=2)

        if self.corruption_process is not None and corrupt_frac is None:
            corrupt_frac = self.corruption_process.sample_corrupt_frac(n=batch_size).to(self.device)
        elif isinstance(corrupt_frac, float):
            corrupt_frac = torch.full((batch_size,), corrupt_frac, device=self.device)
        elif isinstance(corrupt_frac, torch.Tensor):
            # Move tensor to the correct device
            corrupt_frac = corrupt_frac.to(self.device)
        else:
            corrupt_frac = torch.full((batch_size,), 0.0, device=self.device)

        return seq_array, tgt_tok_idxs, src_tok_embs, corrupt_frac

    def tokenize_seq(
        self,
        seq_array: Optional[np.ndarray] = None,
        tgt_tok_idxs: Optional[LongTensor] = None,
        src_tok_embs: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        corrupt_frac: Union[float, torch.Tensor] = 0.0,
        is_corrupted: Optional[torch.Tensor] = None,
        corruption_allowed: Optional[torch.Tensor] = None,
    ):
        # begin forward pass from raw sequence
        if seq_array is not None:
            assert tgt_tok_idxs is None
            assert src_tok_embs is None
            if self.training:
                tgt_tok_idxs = self.train_transform(seq_array)
            else:
                tgt_tok_idxs = self.eval_transform(seq_array)
            tgt_tok_idxs = tgt_tok_idxs.to(self.device)

        # truncate token sequence to max context length
        if tgt_tok_idxs is not None:
            assert src_tok_embs is None
            # truncate to max context length, keep final stop token
            if tgt_tok_idxs.size(-1) > self.max_len:
                tmp_tok_idxs = tgt_tok_idxs[..., : self.max_len - 1]
                tgt_tok_idxs = torch.cat([tmp_tok_idxs, tgt_tok_idxs[..., -1:]], dim=-1)

        if corruption_allowed is None and tgt_tok_idxs is not None:
            corruption_allowed = self.tokenizer.get_corruptible_mask(tgt_tok_idxs)

        # begin forward pass from tokenized sequence
        if tgt_tok_idxs is not None:
            # apply masking corruption
            if isinstance(self.corruption_process, MaskCorruptionProcess) and (
                (isinstance(corrupt_frac, float) and corrupt_frac > 0.0)
                or (isinstance(corrupt_frac, torch.Tensor) and torch.any(corrupt_frac > 0.0))
            ):
                src_tok_idxs, is_corrupted = self.corruption_process(
                    x_start=tgt_tok_idxs,
                    mask_val=self.masking_idx or self.tokenizer.mask_token_id,
                    corruption_allowed=corruption_allowed,
                    corrupt_frac=corrupt_frac,
                )
            else:
                src_tok_idxs = tgt_tok_idxs
                is_corrupted = (
                    torch.full_like(src_tok_idxs, False, dtype=torch.bool) if is_corrupted is None else is_corrupted
                )

            padding_mask = src_tok_idxs != self.pad_tok_idx

        if src_tok_embs is not None:
            assert seq_array is None
            assert padding_mask is not None
            src_tok_idxs = None

        return (
            src_tok_idxs,
            tgt_tok_idxs,
            corruption_allowed,
            is_corrupted,
            padding_mask,
        )

    def forward(
        self,
        inputs: Optional[Union[np.ndarray, torch.Tensor]] = None,
        seq_array: Optional[np.ndarray] = None,
        tgt_tok_idxs: Optional[LongTensor] = None,
        src_tok_embs: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        corrupt_frac: Union[float, torch.Tensor] = 0.0,
        is_corrupted: Optional[torch.Tensor] = None,
        corruption_allowed: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> TransformerEncoderRootOutput:
        """
        Args:
            seq_array: (batch_size,) array of discrete sequences (e.g. text strings)
            tgt_tok_idxs: Optional pre-tokenized inputs
            src_tok_embs: Optional pre-embedded inputs
            padding_mask: Optional padding mask for pre-embedded inputs
            corrupt_frac: Fraction of tokens to corrupt
            is_corrupted: Optional pre-computed corruption mask
            corruption_allowed: Optional mask of tokens that can be corrupted

        Returns:
            TransformerEncoderRootOutput containing:
                root_features: Transformer encoder output representations
                padding_mask: Attention mask (1 for keep, 0 for padding)
                src_tok_idxs: Source token indices (possibly corrupted)
                tgt_tok_idxs: Target token indices (original)
                src_tok_embs: Source token embeddings
                is_corrupted: Mask indicating which tokens were corrupted
                corrupt_frac: Fraction of tokens corrupted
        """
        seq_array, tgt_tok_idxs, src_tok_embs, corrupt_frac = self.init_seq(
            inputs, seq_array, tgt_tok_idxs, src_tok_embs, corrupt_frac, **kwargs
        )
        (
            src_tok_idxs,
            tgt_tok_idxs,
            corruption_allowed,
            is_corrupted,
            padding_mask,
        ) = self.tokenize_seq(
            seq_array,
            tgt_tok_idxs,
            src_tok_embs,
            padding_mask,
            corrupt_frac,
            is_corrupted,
            corruption_allowed,
        )

        # Create HF attention mask (1 for keep, 0 for padding) from padding_mask
        attention_mask = padding_mask.long()

        # Process through transformer model
        if src_tok_idxs is not None:
            outputs = self.transformer(input_ids=src_tok_idxs, attention_mask=attention_mask, return_dict=True)
            # Extract last hidden state
            root_features = outputs.last_hidden_state
        else:
            # Handle the case when src_tok_embs is provided (less common for transformers)
            outputs = self.transformer(inputs_embeds=src_tok_embs, attention_mask=attention_mask, return_dict=True)
            root_features = outputs.last_hidden_state

        # Make sure corrupt_frac is on the same device as other tensors
        if isinstance(corrupt_frac, torch.Tensor):
            corrupt_frac = corrupt_frac.to(root_features.device)

        outputs = TransformerEncoderRootOutput(
            root_features=root_features.contiguous(),
            padding_mask=padding_mask,
            src_tok_idxs=src_tok_idxs,
            tgt_tok_idxs=tgt_tok_idxs,
            src_tok_embs=src_tok_embs,
            is_corrupted=is_corrupted,
            corrupt_frac=corrupt_frac,
        )
        return outputs
