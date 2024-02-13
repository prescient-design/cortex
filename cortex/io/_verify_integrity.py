import os.path
from typing import Optional

from ._verify_checksum import verify_checksum


def verify_integrity(path: str, checksum: Optional[str] = None) -> bool:
    if not os.path.isfile(path):
        return False

    if checksum is None:
        return True

    return verify_checksum(path, checksum)
