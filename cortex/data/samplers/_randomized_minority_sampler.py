import random

from cortex.data.samplers.functional import round_robin_longest

from ._minority_upsampler import MinorityUpsampler


class RandomizedMinorityUpsampler(MinorityUpsampler):
    """Randomized version of Upsampler."""

    def __iter__(self):
        index_list = [idxlist.copy() for idxlist in self.index_list]
        random.shuffle(index_list)

        for idxlist_copy in index_list:
            random.shuffle(idxlist_copy)
        yield from round_robin_longest(index_list)
