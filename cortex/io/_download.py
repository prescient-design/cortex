# This file is a temporary duplicate of the _download.py file in the prescient package.
import bz2
import contextlib
import gzip
import importlib.metadata
import itertools
import lzma
import os
import os.path
import re
import tarfile
import urllib.parse
import urllib.request
import zipfile
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import IO, Callable, Iterator, Optional, Tuple, Union
from urllib.error import URLError
from urllib.request import Request

import boto3
from botocore.client import BaseClient
from requests import Session
from tqdm import tqdm
from upath import UPath

from ._parse_s3_path import parse_s3_path
from ._verify_checksum import verify_checksum
from ._verify_integrity import verify_integrity

try:
    USER_AGENT = f"cortex/{importlib.metadata.version('cortex')}"
except PackageNotFoundError:
    USER_AGENT = "cortex"

_COMPRESSED_FILE_OPENERS: dict[str, Callable[..., IO]] = {
    ".bz2": bz2.open,
    ".gz": gzip.open,
    ".xz": lzma.open,
}


def extract_archive(
    source: str,
    destination: Optional[str] = None,
    remove_archive: bool = False,
) -> str:
    if destination is None:
        destination = os.path.dirname(source)

    suffix, archive_type, compression_type = None, None, None

    suffixes = Path(source).suffixes

    if not suffixes:
        raise RuntimeError

    suffix_a = suffixes[-1]

    aliases = {
        ".tbz": (".tar", ".bz2"),
        ".tbz2": (".tar", ".bz2"),
        ".tgz": (".tar", ".gz"),
    }

    if suffix_a in aliases:
        suffix, archive_type, compression_type = (
            suffix_a,
            aliases[suffix_a][0],
            aliases[suffix_a][1],
        )
    elif suffix_a in _ARCHIVE_EXTRACTORS:
        suffix, archive_type, compression_type = suffix_a, suffix_a, None
    elif suffix_a in _COMPRESSED_FILE_OPENERS:
        if len(suffixes) > 1:
            suffix_b = suffixes[-2]

            if suffix_b in _ARCHIVE_EXTRACTORS:
                suffix, archive_type, compression_type = (
                    f"{suffix_b}{suffix_a}",
                    suffix_b,
                    suffix_a,
                )
            else:
                suffix, archive_type, compression_type = (
                    suffix_a,
                    None,
                    suffix_a,
                )

        else:
            suffix, archive_type, compression_type = suffix_a, None, suffix_a
    else:
        raise RuntimeError

    if not archive_type:
        destination = os.path.join(
            destination,
            os.path.basename(source).replace(suffix, ""),
        )

        if not compression_type:
            raise RuntimeError

        if destination is None:
            if archive_type is None:
                archive_type = ""

            destination = source.replace(suffix, archive_type)

        compressed_file_opener = _COMPRESSED_FILE_OPENERS[compression_type]

        with (
            compressed_file_opener(source, "rb") as reader,
            open(destination, "wb") as writer,
        ):
            writer.write(reader.read())

        if remove_archive:
            os.remove(source)

        return destination

    extract = _ARCHIVE_EXTRACTORS[archive_type]

    extract(source, destination, compression_type)

    if remove_archive:
        os.remove(source)

    return destination


def _extract_tar(
    source: str,
    destination: str,
    compression: Optional[str],
) -> None:
    if compression is not None:
        mode = f"r:{compression[1:]}"
    else:
        mode = "r"

    with tarfile.open(source, mode) as f:
        f.extractall(destination)


def _extract_zip(
    source: str,
    destination: str,
    compression: Optional[str],
) -> None:
    if compression is not None:
        compression = {
            ".bz2": zipfile.ZIP_BZIP2,
            ".xz": zipfile.ZIP_LZMA,
        }[compression]
    else:
        compression = zipfile.ZIP_STORED

    with zipfile.ZipFile(source, "r", compression=compression) as f:
        f.extractall(destination)


_ARCHIVE_EXTRACTORS: dict[str, Callable[[str, str, Optional[str]], None]] = {
    ".tar": _extract_tar,
    ".zip": _extract_zip,
}


def _get_google_drive_file_id(url: str) -> Optional[str]:
    parts = urllib.parse.urlparse(url)

    if re.match(r"(drive|docs)[.]google[.]com", parts.netloc) is None:
        return None

    match = re.match(r"/file/d/(?P<id>[^/]*)", parts.path)

    if match is None:
        return None

    return match.group("id")


def _get_redirect_url(url: str, maximum_hops: int = 3) -> str:
    headers = {"Method": "HEAD", "User-Agent": USER_AGENT}

    for _ in range(maximum_hops + 1):
        import urllib.request

        # Make the request using the custom SSL context
        with urllib.request.urlopen(Request(url, headers=headers)) as response:
            # with urllib.request.urlopen(Request(url, headers=headers)) as response:
            if response.url == url or response.url is None:
                return url

            url = response.url
    else:
        raise RecursionError


def _google_drive_download(
    source: str,
    destination: str,
    filename: Optional[str] = None,
    checksum: Optional[str] = None,
):
    destination = os.path.expanduser(destination)

    if not filename:
        filename = source

    path = os.path.join(destination, filename)

    os.makedirs(destination, exist_ok=True)

    if verify_integrity(path, checksum):
        return

    params = {"id": source, "export": "download"}

    with Session() as session:
        response = session.get(
            "https://drive.google.com/uc",
            params=params,
            stream=True,
        )

        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                token = value

                break
        else:
            response, content = _parse_google_drive_response(response)

            if response == "Virus scan warning":
                token = "t"
            else:
                token = None

        if token is not None:
            response, content = _parse_google_drive_response(
                session.get(
                    "https://drive.google.com/uc",
                    params=dict(params, confirm=token),
                    stream=True,
                ),
            )

        if response == "Quota exceeded":
            raise RuntimeError

        _save_response_content(content, path)

    if os.stat(path).st_size < 10 * 1024:
        with contextlib.suppress(UnicodeDecodeError), open(path) as file:
            text = file.read()

            if re.search(
                r"</?\s*[a-z-][^>]*\s*>|(&(?:[\w\d]+|#\d+|#x[a-f\d]+);)",
                text,
            ):
                raise ValueError

    if checksum and not verify_checksum(path, checksum):
        raise RuntimeError


def _parse_google_drive_response(
    response,
    chunk_size: int = 32 * 1024,
) -> Tuple[bytes, Iterator[bytes]]:
    content = response.iter_content(chunk_size)

    first_chunk = None

    while not first_chunk:
        first_chunk = next(content)

    content = itertools.chain([first_chunk], content)

    try:
        matches = re.search(
            "<title>Google Drive - (?P<response>.+?)</title>",
            first_chunk.decode(),
        )

        if matches is not None:
            response = matches["response"]
        else:
            response = None
    except UnicodeDecodeError:
        response = None

    return response, content


def _save_response_content(
    chunks: Iterator[bytes],
    destination: str,
    length: Optional[int] = None,
):
    with open(destination, "wb") as file, tqdm(total=length) as progress_bar:
        for chunk in chunks:
            if not chunk:
                continue

            file.write(chunk)

            progress_bar.update(len(chunk))


def _urlretrieve(url: str, filename: str, chunk_size: int = 1024 * 32):
    headers = {"User-Agent": USER_AGENT}

    import urllib.request

    # Make the request using the custom SSL context
    with urllib.request.urlopen(Request(url, headers=headers)) as response:
        # with urllib.request.urlopen(Request(url, headers=headers)) as response:
        _save_response_content(
            iter(lambda: response.read(chunk_size), b""),
            filename,
            length=response.length,
        )


def _s3_download(
    s3_path: Union[UPath, Path, str],
    local_path: Union[UPath, Path, str],
    boto3_s3_client: Optional[BaseClient] = None,
):
    """
    Download a file from an S3 bucket and save it to a local path.

    Parameters
    ----------
    s3_path : Union[UPath, Path, str]
        The S3 bucket path of the file to be downloaded.
    local_path : Union[UPath, Path, str]
        The local file path where the downloaded file will be saved.
    boto3_s3_client : Optional[BaseClient], optional
        The boto S3 client to use, by default None.

    """
    bucket_name, bucket_key = parse_s3_path(s3_path=s3_path)

    if boto3_s3_client is None:
        boto3_s3_client = boto3.client("s3")

    boto3_s3_client.download_file(bucket_name, bucket_key, str(local_path))


def download(
    source: Union[str, Path],
    destination: Union[str, Path],
    filename: Union[str, None] = None,
    checksum: Union[str, None] = None,
    maximum_redirect_url_hops: int = 3,
    boto3_s3_client: Union[BaseClient, None] = None,
):
    """
    Download a file from a URL and save it to a local path.
    Supports standard URLs, Google Drive, and S3.

    Parameters
    ----------
    source : str | Path
        The URL of the file to be downloaded.
    destination : str | Path
        The local directory where the downloaded file will be saved.
    filename : str | None, optional
        The name of the file to be saved, by default None.
    checksum : str | None, optional
        The checksum of the file to be downloaded, by default None.
    maximum_redirect_url_hops : int, optional
        The maximum number of hops to follow when resolving a URL redirect, by default 3.
    boto3_s3_client : BaseClient | None, optional
        The boto S3 client to use, by default None.

    """
    destination = os.path.expanduser(destination)

    if not filename:
        filename = os.path.basename(source)

    path = os.path.join(destination, filename)

    os.makedirs(destination, exist_ok=True)

    if verify_integrity(path, checksum):
        return

    if urllib.parse.urlparse(source).scheme == "s3":
        return _s3_download(source, path, boto3_s3_client)

    source = _get_redirect_url(
        source,
        maximum_hops=maximum_redirect_url_hops,
    )

    # TEST IF FILE IS ON GOOGLE DRIVE:
    google_drive_file_id = _get_google_drive_file_id(source)

    if google_drive_file_id is not None:
        return _google_drive_download(
            google_drive_file_id,
            destination,
            filename,
            checksum,
        )

    try:
        _urlretrieve(source, path)
    except (URLError, OSError) as error:
        if source[:5] == "https":
            source = source.replace("https:", "http:")

            # TODO: Add an insecure connection warning?

            _urlretrieve(source, path)
        else:
            raise error

    if not verify_integrity(path, checksum):
        raise RuntimeError


def download_and_extract_archive(
    resource: Union[str, Path],
    source: Union[str, Path],
    destination: Union[str, Path, None] = None,
    name: Union[str, None] = None,
    checksum: Union[str, None] = None,
    remove_archive: bool = False,
) -> None:
    """Download and extract an archive file.

    Parameters
    ----------
    resource : str
        The URL of the resource to download.
    source : str
        The directory where the archive file will be downloaded.
    destination : str, optional
        The directory where the archive file will be extracted, by default None.
    name : str, optional
        The name of the archive file, by default None.
    checksum : str, optional
        The checksum of the archive file, by default None.
    remove_archive : bool, optional
        Whether to remove the archive file after extraction, by default False.
    """
    source = os.path.expanduser(source)

    if destination is None:
        destination = source

    if not name:
        name = os.path.basename(resource)

    download(resource, source, name, checksum)

    archive = os.path.join(source, name)

    extract_archive(archive, destination, remove_archive)
