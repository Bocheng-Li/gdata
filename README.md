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

iterator = loader.iter_bfloat16_dlpack_center_split(
    channels_last=False,
    mid=1_048_576,
    shift=2_048,
    segments_length=4,
)

for dna, tracks, metadata in iterator.dataloader(batch_size=8):
    # dna: (8, sequence_length, 4), float32
    # tracks: (8, track, sequence_length), bfloat16
    pass
```

`NativeGDataDataLoader` is also available as
`gdata.NativeGDataDataLoader`.  PyTorch is imported only when the adapter is
used; the core `gdata` reader remains usable without PyTorch.
