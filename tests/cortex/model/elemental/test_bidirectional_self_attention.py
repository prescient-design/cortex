import torch

from cortex.model.elemental import BidirectionalSelfAttention

BATCH_SIZE = 2
NUM_HEADS = 3
EMBED_DIM = 12
SEQ_LEN = 5


def test_bidirectional_self_attention():
    module = BidirectionalSelfAttention(num_heads=NUM_HEADS, embed_dim=EMBED_DIM, dropout_p=0.0, bias=False)

    x = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    padding_mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.bool)
    x_prime, _ = module((x, padding_mask))

    assert x_prime.shape == x.shape
