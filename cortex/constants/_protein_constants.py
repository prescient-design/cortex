ANTIGEN_COMPLEX_TOKEN = "<ag>"  # Antigen chain start token
AB_AG_COMPLEX_COL = "tokenized_ab_ag_complex"  # Tokenized complex column in dataframes
ANTIGEN_COL = "affinity_antigen_sequence"  # Antigen sequence column in dataframes

CANON_AMINO_ACIDS = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]

VARIABLE_HEAVY_CHAIN_TOKEN = "<vh>"  # Variable heavy chain start token
VARIABLE_HEAVY_COL = "fv_heavy_aho"  # Aligned VH sequence column in dataframes
VARIABLE_LIGHT_CHAIN_TOKEN = "<vl>"  # Variable light chain start token
VARIABLE_LIGHT_COL = "fv_light_aho"  # Aligned VL sequence column in dataframes
