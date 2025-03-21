ANTIGEN_COMPLEX_TOKEN = "<ag>"  # Antigen chain start token
AB_AG_COMPLEX_COL = "tokenized_ab_ag_complex"  # Tokenized complex column in dataframes
ANTIGEN_COL = "affinity_antigen_sequence"  # Antigen sequence column in dataframes

# Dayhoff classification of amino acids based on physicochemical properties
# These groups represent amino acids that can often substitute for each other
AMINO_ACID_GROUPS = {
    "sulfur_polymerization": ["C"],  # Cysteine
    "small": ["A", "G", "P", "S", "T"],  # Alanine, Glycine, Proline, Serine, Threonine
    "acid_and_amide": ["D", "E", "N", "Q"],  # Aspartic acid, Glutamic acid, Asparagine, Glutamine
    "basic": ["H", "K", "R"],  # Histidine, Lysine, Arginine
    "hydrophobic": ["I", "L", "M", "V"],  # Isoleucine, Leucine, Methionine, Valine
    "aromatic": ["F", "W", "Y"],  # Phenylalanine, Tryptophan, Tyrosine
}

# Combined list of all canonical amino acids
CANON_AMINO_ACIDS = []
for group in AMINO_ACID_GROUPS.values():
    CANON_AMINO_ACIDS.extend(group)

STANDARD_AA_FREQS = {
    "A": 0.0825,
    "R": 0.0553,
    "N": 0.0406,
    "D": 0.0545,
    "C": 0.0137,
    "Q": 0.0393,
    "E": 0.0675,
    "G": 0.0707,
    "H": 0.0227,
    "I": 0.0595,
    "L": 0.0966,
    "K": 0.0584,
    "M": 0.0242,
    "F": 0.0386,
    "P": 0.0470,
    "S": 0.0657,
    "T": 0.0534,
    "W": 0.0108,
    "Y": 0.0292,
    "V": 0.0687,
}

VARIABLE_HEAVY_CHAIN_TOKEN = "<vh>"  # Variable heavy chain start token
VARIABLE_HEAVY_COL = "fv_heavy_aho"  # Aligned VH sequence column in dataframes
VARIABLE_LIGHT_CHAIN_TOKEN = "<vl>"  # Variable light chain start token
VARIABLE_LIGHT_COL = "fv_light_aho"  # Aligned VL sequence column in dataframes
