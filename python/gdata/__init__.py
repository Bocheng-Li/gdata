"""Python namespace for the native :mod:`gdata` extension."""

from . import gdata as _gdata
from .gdata import *

__doc__ = _gdata.__doc__
__version__ = _gdata.__version__

# The adapter keeps PyTorch as a lazy runtime dependency.  Importing gdata
# therefore remains lightweight, while the adapter is still part of the
# public gdata namespace.
from .native_dataloader import (
    NativeGDataDataLoader,
    _codes_to_one_hot,
    make_native_gdata_dataloader,
)

__all__ = [
    name for name in dir(_gdata) if not name.startswith("_")
]
__all__ += [
    "NativeGDataDataLoader",
    "make_native_gdata_dataloader",
    "_codes_to_one_hot",
]
