import os

from cortex.data.dataset import RedFluorescentProteinDataset


def test_rfp_dataset():
    # make temp root dir
    root = "./temp/"
    os.makedirs(root, exist_ok=True)
    dataset = RedFluorescentProteinDataset(
        root=root,
        download=True,
    )

    item = dataset[0]

    assert (
        item["foldx_seq"]
        == "LSKHGLTKDMTMKYRMEGCVDGHKFVITGHGNGSPFEGKQTINLCVVEGGPLPFSEDILSAVFNRVFTDYPQGMVDFFKNSCPAGYTWQRSLLFEDGAVCTASADITVSVEENCFYHESKFHGVNFPADGPVMKKMTINWEPCCEKIIPVPRQGILKGDVAMYLLLKDGGRYRCQFDTVYKAKTDSKKMPEWHFIQHKLTREDRSDAKNQKWQLAEHSVASRSALA"
    )
    assert item["foldx_total_energy"] == -39.8155
    assert item["SASA"] == 11189.00587945787
