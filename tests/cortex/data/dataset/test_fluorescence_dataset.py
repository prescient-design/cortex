import os

from cortex.data.dataset import TAPEFluorescenceDataset


def test_tape_fluorescence_dataset():
    root = "./temp/"
    os.makedirs(root, exist_ok=True)
    dataset = TAPEFluorescenceDataset(
        root=root,
        download=True,
    )
