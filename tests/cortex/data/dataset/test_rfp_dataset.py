import os

from cortex.data.dataset import RedFluorescentProteinDataset


def test_rfp_dataset():
    # make temp root dir
    root = "./temp/"
    os.makedirs(root, exist_ok=True)
    _ = RedFluorescentProteinDataset(
        root=root,
        download=True,
    )
