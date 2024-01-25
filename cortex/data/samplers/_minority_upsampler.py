from typing import List

from torch.utils.data import Sampler

from cortex.data.samplers.functional import SizedIterable, round_robin_longest


class MinorityUpsampler(Sampler[int]):
    """Upsamples shorter length lists of indices by cycling through them until
    until the longer ones are exhausted.
    """

    def __init__(self, index_list: List[SizedIterable[int]]):
        self.index_list = index_list

    def __iter__(self):
        yield from round_robin_longest(self.index_list)
