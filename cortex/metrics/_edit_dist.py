import edlib


def edit_dist(x: str, y: str):
    """
    Computes the edit distance between two strings.
    """
    return edlib.align(x, y)["editDistance"]
