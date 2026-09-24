## Installation

```
pip install 'git+https://github.com/regulatory-genomics/genomics-dataloader.git#egg=genomics-dataloader'
```

[**Documentation**](https://lab.kaizhang.org/genomics-dataloader/version/dev/index.html)

## Native PyTorch batching

The center/split DLPack iterators also expose a package-native PyTorch
adapter.  It batches complete parent records in Rust when possible, converts
the returned DLPack values to CPU `bfloat16` tensors, and applies the default
DNA one-hot tokenizer unless a custom tokenizer is supplied:

```python
import gdata

iterator = loader_map.iter_bfloat16_dlpack_center_split(
    channels_last=False,
    mid=1_048_576,
    shift=2_048,
    segments_length=4,
    reverse_probability=0.5,
    # The default is "segment"; use "parent" to share one decision.
    reverse_granularity="segment",
    # For a GenomeDataLoaderMap, use an output-channel -> input-channel map.
    reverse_indices={"rna_seq": rna_reverse_indices},
)

for dna, tracks, metadata in iterator.dataloader(batch_size=8):
    # dna: (8, sequence_length, 4), float32
    # tracks: (8, track, sequence_length), bfloat16
    pass
```

The native iterator returns metadata entries of the form
`(segment_name, applied_shift, chunk_index, reverse)`.  In a synchronized
`GenomeDataLoaderMap`, the DNA sequence is materialized from the first head
only; the other heads decode their track values without allocating a second
DNA array.  The compressed records still contain a copy of DNA in each gdata
file, so the compressed bytes must still be read and decompressed per head.

`NativeGDataDataLoader` is also available as
`gdata.NativeGDataDataLoader`.  PyTorch is imported only when the adapter is
used; the core `gdata` reader remains usable without PyTorch.

The native center/split path uses a streaming worker queue: a completed parent
is forwarded immediately instead of waiting for the other workers in the same
refill batch.  Consequently, parent records can arrive in completion order;
for a `GenomeDataLoaderMap`, modalities are synchronized by their genomic
range before the shared augmentation plan is applied.

## Track-chunked gdata files

The gdata writers accept an optional `chunk_tracks` argument. When it is a
positive integer, each segment is stored as several independently compressed
track blocks, with at most that many tracks per block. The DNA sequence stays
inside each block (it is not moved to a separate sequence file), so the same
parent/segment reader APIs continue to work. A reader decodes the blocks in
parallel and reassembles the original track order; files created without this
option keep the legacy single-frame format and remain readable.

```python
builder = gdata.GenomeDataBuilder(
    "atac.gdata", "genome.fa", 1_048_576,
    resolution=1, chunk_tracks=64,
)
builder.add_files(track_files)
builder.finish()
```

The streaming builder and `convert_tfrecord_to_gdata(...,
chunk_tracks=64)` use the same format. Choose a block size according to the
number of tracks and available CPU parallelism; it does not change public
array shapes.

For synchronized multi-head loading, `GenomeDataLoaderMap` can receive one
total worker budget. It assigns workers proportionally to the number of
tracks in each head, with at least one worker per head:

```python
loader_map = gdata.GenomeDataLoaderMap(
    {"atac": atac_loader, "rna_seq": rna_loader},
    n_jobs=12,
)
```

For 256 ATAC and 768 RNA tracks this gives 3 and 9 workers, respectively. If
`n_jobs` is omitted, each loader's `GenomeDataLoader(n_jobs=...)` setting keeps
its historical meaning.
