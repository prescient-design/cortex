import math
import warnings
from typing import Optional, Union

import numpy as np
import torch
from torch import LongTensor, nn

from cortex.corruption import CorruptionProcess, GaussianCorruptionProcess, MaskCorruptionProcess
from cortex.model.block import TransformerBlock
from cortex.model.elemental import SinePosEncoder
from cortex.model.root._abstract_root import RootNode
from cortex.model.root._transformer_root import TransformerRootOutput
from cortex.transforms import HuggingFaceTokenizerTransform


class TransformerRootV2(RootNode):
    """
    Updated TransformerRoot that accepts pre-tokenized inputs from CortexDataset.

    Moves tokenization to dataloader for parallel execution and improved GPU utilization.
    """

    def __init__(
        self,
        tokenizer_transform: HuggingFaceTokenizerTransform,
        max_len: int,
        out_dim: int = 64,
        embed_dim: int = 64,
        channel_dim: int = 256,
        num_blocks: int = 2,
        num_heads: int = 4,
        is_causal: bool = False,
        dropout_prob: float = 0.0,
        pos_encoding: bool = True,
        corruption_process: Optional[CorruptionProcess] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer_transform.tokenizer
        self.vocab_size = len(self.tokenizer.vocab)
        self.max_len = max_len
        self.pad_tok_idx = self.tokenizer.padding_idx

        if num_blocks >= 1:
            self.tok_encoder = nn.Embedding(self.vocab_size, embed_dim, padding_idx=self.pad_tok_idx)

        # optional positional encoding
        if pos_encoding:
            self.pos_encoder = SinePosEncoder(embed_dim, dropout_prob, max_len, batch_first=True)
        else:
            self.pos_encoder = None

        # create encoder
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        if num_blocks >= 1:
            self.out_dim = out_dim
            encoder_modules = []
            resid_block_kwargs = {
                "num_heads": num_heads,
                "dropout_p": dropout_prob,
                "is_causal": is_causal,
            }
            if num_blocks == 1:
                encoder_modules.append(TransformerBlock(embed_dim, out_dim, **resid_block_kwargs))
            else:
                encoder_modules.append(TransformerBlock(embed_dim, channel_dim, **resid_block_kwargs))

                encoder_modules.extend(
                    [
                        TransformerBlock(
                            channel_dim,
                            channel_dim,
                            **resid_block_kwargs,
                        )
                        for _ in range(num_blocks - 2)
                    ]
                )

                encoder_modules.append(
                    TransformerBlock(
                        channel_dim,
                        out_dim,
                        **resid_block_kwargs,
                    )
                )
            self.encoder = nn.Sequential(*encoder_modules)

        self.corruption_process = corruption_process

    def initialize_weights(self, **kwargs):
        # default random initialization
        pass

    def get_token_embedding(self, tok_idx: int):
        return self.tok_encoder(torch.tensor(tok_idx, device=self.device))

    @property
    def device(self):
        return self.tok_encoder.weight.device

    def init_seq(
        self,
        tgt_tok_idxs: Optional[LongTensor] = None,
        src_tok_embs: Optional[torch.Tensor] = None,
        corrupt_frac: Union[float, torch.Tensor] = 0.0,
        **kwargs,
    ):
        """Initialize sequence processing with pre-tokenized inputs."""

        # Determine batch size from available inputs
        batch_size = None
        if tgt_tok_idxs is not None:
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

        return tgt_tok_idxs, src_tok_embs, corrupt_frac

    def apply_corruption(
        self,
        tgt_tok_idxs: Optional[LongTensor] = None,
        src_tok_embs: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        corrupt_frac: Union[float, torch.Tensor] = 0.0,
        is_corrupted: Optional[torch.Tensor] = None,
        corruption_allowed: Optional[torch.Tensor] = None,
    ):
        """Apply corruption to pre-tokenized sequences."""

        # For pre-tokenized inputs, truncate to max context length
        if tgt_tok_idxs is not None:
            assert src_tok_embs is None
            # truncate to max context length, keep final stop token
            if tgt_tok_idxs.size(-1) > self.max_len:
                tmp_tok_idxs = tgt_tok_idxs[..., : self.max_len - 1]
                tgt_tok_idxs = torch.cat([tmp_tok_idxs, tgt_tok_idxs[..., -1:]], dim=-1)

        if corruption_allowed is None and tgt_tok_idxs is not None:
            corruption_allowed = self.tokenizer.get_corruptible_mask(tgt_tok_idxs)

        # Apply corruption to pre-tokenized sequences
        if tgt_tok_idxs is not None:
            # apply masking corruption
            if isinstance(self.corruption_process, MaskCorruptionProcess) and (
                (isinstance(corrupt_frac, float) and corrupt_frac > 0.0)
                or (isinstance(corrupt_frac, torch.Tensor) and torch.any(corrupt_frac > 0.0))
            ):
                src_tok_idxs, is_corrupted = self.corruption_process(
                    x_start=tgt_tok_idxs,
                    mask_val=self.tokenizer.masking_idx,
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
            assert padding_mask is not None
            src_tok_idxs = None

        return (
            src_tok_idxs,
            tgt_tok_idxs,
            corruption_allowed,
            is_corrupted,
            padding_mask,
        )

    def embed_seq(
        self,
        src_tok_idxs: Optional[LongTensor] = None,
        src_tok_embs: Optional[torch.Tensor] = None,
        corrupt_frac: Union[float, torch.Tensor] = 0.0,
        is_corrupted: Optional[torch.Tensor] = None,
        corruption_allowed: Optional[torch.Tensor] = None,
        normalize_embeds: bool = True,
    ):
        """Embed token sequences."""
        # begin forward pass from token embeddings
        if src_tok_embs is None:
            src_tok_embs = self.tok_encoder(src_tok_idxs)
            if normalize_embeds:
                src_tok_embs = src_tok_embs / src_tok_embs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                src_tok_embs = src_tok_embs * math.sqrt(self.embed_dim)

        # apply gaussian embedding corruption
        if isinstance(self.corruption_process, GaussianCorruptionProcess) and (
            (isinstance(corrupt_frac, float) and corrupt_frac > 0.0)
            or (isinstance(corrupt_frac, torch.Tensor) and torch.any(corrupt_frac > 0.0))
        ):
            assert corruption_allowed is not None
            src_tok_embs, is_corrupted = self.corruption_process(
                x_start=src_tok_embs,
                corruption_allowed=corruption_allowed[..., None],
                corrupt_frac=corrupt_frac,
            )
            is_corrupted = is_corrupted.sum(-1).bool()
        else:
            none_corrupted = torch.zeros(*src_tok_embs.shape[:-1], dtype=torch.bool).to(src_tok_embs.device)
            is_corrupted = none_corrupted if is_corrupted is None else is_corrupted

        return src_tok_embs, is_corrupted

    def process_seq(
        self,
        src_tok_embs: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        """Process embedded sequences through transformer blocks."""
        # apply positional encoding if it exists
        if self.pos_encoder is not None:
            src_features = self.pos_encoder(src_tok_embs)
        else:
            src_features = src_tok_embs

        # main forward pass
        src_features, _ = self.encoder((src_features, padding_mask.to(src_features)))

        return src_features

    def forward(
        self,
        # Pre-tokenized inputs from CortexDataset
        tgt_tok_idxs: Optional[LongTensor] = None,
        src_tok_embs: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        corrupt_frac: Union[float, torch.Tensor] = 0.0,
        is_corrupted: Optional[torch.Tensor] = None,
        corruption_allowed: Optional[torch.Tensor] = None,
        # Backward compatibility (deprecated)
        inputs: Optional[Union[np.ndarray, torch.Tensor]] = None,
        seq_array: Optional[np.ndarray] = None,
        **kwargs,
    ) -> TransformerRootOutput:
        """
        Forward pass with pre-tokenized inputs from CortexDataset.

        Args:
            tgt_tok_idxs: Pre-tokenized and padded sequences from dataloader
            src_tok_embs: Pre-computed embeddings (optional)
            padding_mask: Attention mask from dataloader
            corrupt_frac: Corruption fraction for guided generation

        Returns:
            TransformerRootOutput with processed features
        """

        # Backward compatibility: fallback to old tokenization path
        if inputs is not None or seq_array is not None:
            warnings.warn(
                "Using deprecated seq_array/inputs. Use CortexDataset with pre-tokenized tgt_tok_idxs instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            # Fall back to old tokenization behavior
            from cortex.model.root._transformer_root import TransformerRoot

            legacy_root = TransformerRoot.__new__(TransformerRoot)
            legacy_root.__dict__.update(self.__dict__)
            return legacy_root.forward(inputs=inputs, seq_array=seq_array, **kwargs)

        # Main path: pre-tokenized inputs
        tgt_tok_idxs, src_tok_embs, corrupt_frac = self.init_seq(tgt_tok_idxs, src_tok_embs, corrupt_frac, **kwargs)

        (
            src_tok_idxs,
            tgt_tok_idxs,
            corruption_allowed,
            is_corrupted,
            padding_mask,
        ) = self.apply_corruption(
            tgt_tok_idxs,
            src_tok_embs,
            padding_mask,
            corrupt_frac,
            is_corrupted,
            corruption_allowed,
        )

        src_tok_embs, is_corrupted = self.embed_seq(
            src_tok_idxs, src_tok_embs, corrupt_frac, is_corrupted, corruption_allowed
        )

        src_features = self.process_seq(src_tok_embs, padding_mask)

        # Make sure corrupt_frac is on the same device as other tensors
        if isinstance(corrupt_frac, torch.Tensor):
            corrupt_frac = corrupt_frac.to(src_tok_embs.device)

        outputs = TransformerRootOutput(
            root_features=src_features.contiguous(),
            padding_mask=padding_mask,
            src_tok_embs=src_tok_embs,
            src_tok_idxs=src_tok_idxs,
            tgt_tok_idxs=tgt_tok_idxs,
            is_corrupted=is_corrupted,
            corrupt_frac=corrupt_frac,
        )
        return outputs
