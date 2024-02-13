from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pytorch-cortex")
except PackageNotFoundError:
    __version__ = "unknown version"
