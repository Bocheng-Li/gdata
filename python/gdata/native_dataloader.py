"""Optional PyTorch batching adapter for the native :mod:`gdata` iterators.

This module is part of ``genomics-dataloader`` itself.  The Rust extension
owns genomic-record reading and native batch concatenation; this small adapter
only converts the returned NumPy/DLPack objects to PyTorch tensors and applies
a DNA tokenizer.

PyTorch and NumPy are imported lazily so that ``import gdata`` remains usable
for callers that only need the Rust/NumPy APIs.  Calling
``NativeGDataDataLoader`` requires a PyTorch installation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _torch_modules():
    """Import the optional PyTorch dependencies with a useful error message."""

    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "gdata.NativeGDataDataLoader requires PyTorch and NumPy. "
            "Install the optional PyTorch dependencies or use the raw "
            "gdata iterator/next_batch API."
        ) from exc
    return np, torch, F


def _codes_to_one_hot(codes: Any):
    """Convert gdata A/C/G/T/N codes to a float32 one-hot tensor."""

    _np, torch, F = _torch_modules()
    codes = codes.to(torch.long)
    if torch.any((codes < 0) | (codes > 4)):
        raise ValueError("gdata DNA codes must be in [0, 4]")
    return F.one_hot(codes, num_classes=5)[..., :4].to(torch.float32)


class NativeGDataDataLoader:
    """Batch a native center/split iterator for PyTorch.

    The raw Rust iterator yields all segments from one parent in one item.
    ``next_batch`` concatenates complete parents in Rust, reducing Python
    boundary crossings for ordinary batch sizes.  If ``batch_size`` is not a
    multiple of the iterator's ``segments_length``, a Python leftover-buffer
    fallback is used so that arbitrary batch sizes remain supported.

    Parameters
    ----------
    iterator:
        A native ``GenomeDataLoader`` or ``GenomeDataLoaderMap``
        ``iter_bfloat16_dlpack_center_split`` iterator.
    batch_size:
        Number of output segments in each batch.
    tokenizer:
        Callable receiving a CPU integer tensor of shape ``(B, L)`` and
        returning ``(B, L, D)``.  The default is :func:`_codes_to_one_hot`.
    drop_last:
        Drop a final incomplete batch.
    """

    def __init__(
        self,
        iterator: Any,
        *,
        batch_size: int,
        tokenizer: Any | None = None,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        next_batch = getattr(iterator, "next_batch", None)
        if next_batch is None or not callable(next_batch):
            raise TypeError(
                "iterator must be a native center/split iterator from the "
                "rebuilt gdata extension (missing next_batch)"
            )
        if tokenizer is not None and not callable(tokenizer):
            raise TypeError("tokenizer must be callable")

        self.iterator = iterator
        self.batch_size = int(batch_size)
        self.tokenizer = _codes_to_one_hot if tokenizer is None else tokenizer
        self.drop_last = bool(drop_last)
        self._use_rust_batch = True
        self._pending_sequence = None
        self._pending_tracks = None
        self._pending_metadata: list[Any] = []

    def __iter__(self) -> "NativeGDataDataLoader":
        return self

    @staticmethod
    def _convert_raw_item(item: Any):
        np, torch, _F = _torch_modules()
        sequence_np, values_capsules, metadata = item
        sequence_codes = torch.from_numpy(np.asarray(sequence_np))
        if isinstance(values_capsules, Mapping):
            tracks = {
                head: torch.from_dlpack(capsule)
                for head, capsule in values_capsules.items()
            }
        else:
            tracks = torch.from_dlpack(values_capsules)
        return sequence_codes, tracks, list(metadata)

    @staticmethod
    def _cat_tracks(left: Any, right: Any) -> Any:
        _np, torch, _F = _torch_modules()
        if isinstance(left, Mapping):
            if not isinstance(right, Mapping) or left.keys() != right.keys():
                raise ValueError("native gdata track heads changed during iteration")
            return {
                head: torch.cat((left[head], right[head]), dim=0)
                for head in left
            }
        if isinstance(right, Mapping):
            raise ValueError("native gdata track representation changed during iteration")
        return torch.cat((left, right), dim=0)

    def _tokenize(self, sequence_codes: Any, tracks: Any, metadata: list[Any]):
        dna = self.tokenizer(sequence_codes)
        _np, torch, _F = _torch_modules()
        if not isinstance(dna, torch.Tensor):
            dna = torch.as_tensor(dna)
        if (
            dna.ndim != 3
            or dna.shape[0] != sequence_codes.shape[0]
            or dna.shape[1] != sequence_codes.shape[1]
        ):
            raise ValueError(
                "tokenizer must return shape (batch, sequence, dim); "
                f"got {tuple(dna.shape)} for input {tuple(sequence_codes.shape)}"
            )
        return dna, tracks, metadata

    def _next_irregular_batch(self):
        while (
            self._pending_sequence is None
            or self._pending_sequence.shape[0] < self.batch_size
        ):
            try:
                item = next(self.iterator)
            except StopIteration:
                break
            sequence, tracks, metadata = self._convert_raw_item(item)
            if self._pending_sequence is None:
                self._pending_sequence = sequence
                self._pending_tracks = tracks
                self._pending_metadata = metadata
            else:
                self._pending_sequence = torch_cat(
                    self._pending_sequence, sequence
                )
                self._pending_tracks = self._cat_tracks(
                    self._pending_tracks, tracks
                )
                self._pending_metadata.extend(metadata)

        if self._pending_sequence is None or self._pending_sequence.shape[0] == 0:
            raise StopIteration
        if self._pending_sequence.shape[0] < self.batch_size and self.drop_last:
            self._pending_sequence = None
            self._pending_tracks = None
            self._pending_metadata = []
            raise StopIteration

        count = min(self.batch_size, self._pending_sequence.shape[0])
        sequence = self._pending_sequence[:count]
        metadata = self._pending_metadata[:count]
        if isinstance(self._pending_tracks, Mapping):
            tracks = {
                head: value[:count]
                for head, value in self._pending_tracks.items()
            }
        else:
            tracks = self._pending_tracks[:count]

        if count == self._pending_sequence.shape[0]:
            self._pending_sequence = None
            self._pending_tracks = None
            self._pending_metadata = []
        else:
            self._pending_sequence = self._pending_sequence[count:]
            if isinstance(self._pending_tracks, Mapping):
                self._pending_tracks = {
                    head: value[count:]
                    for head, value in self._pending_tracks.items()
                }
            else:
                self._pending_tracks = self._pending_tracks[count:]
            self._pending_metadata = self._pending_metadata[count:]

        return self._tokenize(sequence, tracks, metadata)

    def __next__(self):
        if not self._use_rust_batch:
            return self._next_irregular_batch()

        try:
            item = self.iterator.next_batch(self.batch_size, self.drop_last)
        except RuntimeError as exc:
            if "multiple of segments_length" not in str(exc):
                raise
            self._use_rust_batch = False
            return self._next_irregular_batch()
        if item is None:
            raise StopIteration

        sequence_codes, tracks, metadata = self._convert_raw_item(item)
        return self._tokenize(sequence_codes, tracks, metadata)


def torch_cat(left: Any, right: Any):
    """Small lazy helper used by the irregular-batch fallback."""

    _np, torch, _F = _torch_modules()
    return torch.cat((left, right), dim=0)


def make_native_gdata_dataloader(
    iterator: Any,
    *,
    batch_size: int,
    tokenizer: Any | None = None,
    drop_last: bool = False,
) -> NativeGDataDataLoader:
    """Construct :class:`NativeGDataDataLoader` from a native iterator."""

    return NativeGDataDataLoader(
        iterator,
        batch_size=batch_size,
        tokenizer=tokenizer,
        drop_last=drop_last,
    )


__all__ = [
    "NativeGDataDataLoader",
    "make_native_gdata_dataloader",
    "_codes_to_one_hot",
]
