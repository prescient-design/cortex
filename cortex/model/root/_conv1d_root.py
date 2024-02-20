import math
import warnings
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
from torch import LongTensor, nn
from torchtext.transforms import PadTransform, ToTensor

from cortex.corruption import CorruptionProcess, GaussianCorruptionProcess, MaskCorruptionProcess
from cortex.model.block import Conv1dResidBlock
from cortex.model.elemental import Apply, Expression, SinePosEncoder, permute_spatial_channel_dims
from cortex.model.root import RootNode, RootNodeOutput
from cortex.transforms import HuggingFaceTokenizerTransform


@dataclass
class Conv1dRootOutput(RootNodeOutput):
    padding_mask: torch.Tensor
    src_tok_idxs: Optional[torch.LongTensor] = None
    tgt_tok_idxs: Optional[torch.LongTensor] = None
    src_tok_embs: Optional[torch.Tensor] = None
    is_corrupted: Optional[torch.Tensor] = None
    corrupt_frac: Optional[float] = None


class Conv1dRoot(RootNode):
    """
    A root node transforming an array of discrete sequences to an array of continuous sequence embeddings
    """

    def __init__(
        self,
        tokenizer_transform: HuggingFaceTokenizerTransform,
        max_len: int,
        out_dim: int = 64,
        embed_dim: int = 64,
        channel_dim: int = 256,
        num_blocks: int = 2,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout_prob: float = 0.0,
        layernorm: bool = True,
        pos_encoding: bool = True,
        train_transforms=None,
        eval_transforms=None,
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
            encoder_modules = [
                Apply(Expression(permute_spatial_channel_dims)),  # (B,N,C) -> (B,C,N)
            ]
            resid_block_kwargs = {
                "kernel_size": kernel_size,
                "layernorm": layernorm,
            }
            if num_blocks == 1:
                encoder_modules.append(
                    Conv1dResidBlock(embed_dim, out_dim, dropout_p=dropout_prob, **resid_block_kwargs)
                )
            else:
                encoder_modules.append(Conv1dResidBlock(embed_dim, channel_dim, **resid_block_kwargs))

                encoder_modules.extend(
                    [
                        Conv1dResidBlock(
                            channel_dim,
                            channel_dim,
                            dilation=dilation,
                            **resid_block_kwargs,
                        )
                        for _ in range(num_blocks - 2)
                    ]
                )

                encoder_modules.append(
                    Conv1dResidBlock(
                        channel_dim,
                        out_dim,
                        dropout_p=dropout_prob,
                        **resid_block_kwargs,
                    )
                )

            encoder_modules.append(
                Apply(Expression(permute_spatial_channel_dims)),  # (B,C,N) -> (B,N,C)
            )
            self.encoder = nn.Sequential(*encoder_modules)

        shared_transforms = [
            tokenizer_transform,  # convert np.array([str, str, ...]) to list[list[int, int, ...]]
            ToTensor(padding_value=self.pad_tok_idx),  # convert list[list[int, int, ...]] to tensor
            PadTransform(max_length=self.max_len, pad_value=self.pad_tok_idx),  # pad to max_len
        ]
        train_transforms = [] if train_transforms is None else list(train_transforms.values())
        eval_transforms = [] if eval_transforms is None else list(eval_transforms.values())
        self.train_transform = nn.Sequential(*(train_transforms + shared_transforms))
        self.eval_transform = nn.Sequential(*(eval_transforms + shared_transforms))
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
        inputs: Optional[Union[np.ndarray, torch.Tensor]] = None,  # TODO deprecate
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

        if "mask_frac" in kwargs:
            corrupt_frac = kwargs["mask_frac"]
            msg = "mask_frac is deprecated, use corrupt_frac instead."
            warnings.warn(msg, PendingDeprecationWarning, stacklevel=2)

        if self.corruption_process is not None and corrupt_frac is None:
            corrupt_frac = self.corruption_process.sample_corrupt_frac()
        else:
            corrupt_frac = 0.0

        return seq_array, tgt_tok_idxs, src_tok_embs, corrupt_frac

    def tokenize_seq(
        self,
        seq_array: Optional[np.ndarray] = None,
        tgt_tok_idxs: Optional[LongTensor] = None,
        src_tok_embs: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        corrupt_frac: float = 0.0,
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
            if isinstance(self.corruption_process, MaskCorruptionProcess) and corrupt_frac > 0.0:
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

    def embed_seq(
        self,
        src_tok_idxs: Optional[LongTensor] = None,
        src_tok_embs: Optional[torch.Tensor] = None,
        corrupt_frac: float = 0.0,
        is_corrupted: Optional[torch.Tensor] = None,
        corruption_allowed: Optional[torch.Tensor] = None,
        normalize_embeds: bool = True,
    ):
        # begin forward pass from token embeddings
        if src_tok_embs is None:
            src_tok_embs = self.tok_encoder(src_tok_idxs)
            if normalize_embeds:
                src_tok_embs = src_tok_embs / src_tok_embs.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                src_tok_embs = src_tok_embs * math.sqrt(self.embed_dim)

        # apply gaussian embedding corruption
        if isinstance(self.corruption_process, GaussianCorruptionProcess) and corrupt_frac > 0.0:
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
        inputs: Optional[Union[np.ndarray, torch.Tensor]] = None,  # TODO deprecate
        seq_array: Optional[np.ndarray] = None,
        tgt_tok_idxs: Optional[LongTensor] = None,
        src_tok_embs: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        corrupt_frac: float = 0.0,
        is_corrupted: Optional[torch.Tensor] = None,
        corruption_allowed: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Conv1dRootOutput:
        """
        Args:
            seq_array: (batch_size,) array of discrete sequences (e.g. text strings)
        Returns:
            outputs: {'root_features': torch.Tensor, 'padding_mask': torch.Tensor}
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
        src_tok_embs, is_corrupted = self.embed_seq(
            src_tok_idxs, src_tok_embs, corrupt_frac, is_corrupted, corruption_allowed
        )
        src_features = self.process_seq(src_tok_embs, padding_mask)
        corrupt_frac = torch.tensor((corrupt_frac,)).to(src_tok_embs)

        outputs = Conv1dRootOutput(
            root_features=src_features.contiguous(),
            padding_mask=padding_mask,
            src_tok_embs=src_tok_embs,
            src_tok_idxs=src_tok_idxs,
            tgt_tok_idxs=tgt_tok_idxs,
            is_corrupted=is_corrupted,
            corrupt_frac=corrupt_frac,
        )
        return outputs
