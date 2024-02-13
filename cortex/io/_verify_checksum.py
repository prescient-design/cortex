from ._md5 import md5


def verify_checksum(path: str, checksum: str, **kwargs) -> bool:
    return checksum == md5(path, **kwargs)
