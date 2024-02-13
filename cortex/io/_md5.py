import hashlib
import sys


def md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    if sys.version_info >= (3, 9):
        checksum = hashlib.md5(usedforsecurity=False)
    else:
        checksum = hashlib.md5()

    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            checksum.update(chunk)

    return checksum.hexdigest()
