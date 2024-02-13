from pathlib import Path
from typing import Union
from urllib.parse import urlparse

from upath import UPath


def parse_s3_path(s3_path: Union[UPath, Path, str]):
    """Parse an S3 path and return the bucket name and key.

    Parameters
    ----------
    s3_path : Union[UPath, Path, str]
        The S3 bucket path of the file to be downloaded.

    Returns
    -------
    bucket_name : str
        The name of the S3 bucket.
    bucket_key : str
        The key of the S3 bucket.
    """
    parsed = urlparse(str(s3_path), allow_fragments=True)
    bucket_name = parsed.netloc
    bucket_key = parsed.path.lstrip("/")

    return bucket_name, bucket_key
