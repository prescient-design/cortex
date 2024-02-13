import os

from cortex.data.dataset import GFPFluorescenceDataset


def test_gfp_fluorescence_dataset():
    root = "./temp/"
    os.makedirs(root, exist_ok=True)
    dataset = GFPFluorescenceDataset(
        root=root,
        download=True,
    )
