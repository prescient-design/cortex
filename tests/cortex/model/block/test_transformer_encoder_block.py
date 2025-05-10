import torch

from cortex.model.block import TransformerEncoderBlock

BATCH_SIZE = 2
NUM_HEADS = 3
EMBED_DIM = 12
SEQ_LEN = 5


def test_transformer_encoder_block():
    module = TransformerEncoderBlock(
        in_channels=EMBED_DIM,
        out_channels=EMBED_DIM,
        num_heads=NUM_HEADS,
    )

    x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    padding_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.bool)
    x_prime, _ = module((x, padding_mask))

    assert x_prime.shape == x.shape
