use anyhow::{ensure, Context, Result};
use bed_utils::bed::{BEDLike, GenomicRange};
use half::bf16;
use indexmap::IndexMap;
use itertools::Itertools;
use ndarray::{Array2, Array3, Axis};
use numpy::{PyArray2, PyArray3};
use pyo3::{prelude::*, py_run, types::PyDict};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use std::str::FromStr;
use std::{collections::HashSet, path::PathBuf};

use super::super::generic::{ParallelLoader, PrefethIterator};
use crate::dataloader::genome::data_store::{
    decode_nucleotide, DataStore, DataStoreBf16ParentRegionIter, DataStoreReadOptions,
};
use crate::dataloader::genome::dlpack::into_dlpack;

/// A parent record together with the arrays needed by the native augmentation
/// path.  The genomic range is retained so that the output coordinates can be
/// updated after the center crop and split.
type RegionBFloat16Record = (GenomicRange, u64, Array2<u8>, Array3<bf16>);

/// Metadata for one output segment.  The second field is the signed shift in
/// base pairs relative to the center of the parent record; the third field is
/// the segment's logical index before the random output ordering is applied.
type AugmentedSegmentMetadata = (String, i64, usize);

type AugmentedBFloat16Record = (Array2<u8>, Array3<bf16>, Vec<AugmentedSegmentMetadata>);

#[derive(Debug, Clone, Copy)]
struct CenterSplitConfig {
    /// Length of the centered window to retain, in base pairs.
    center_length: u32,
    /// Maximum symmetric shift requested by the caller, in base pairs.
    shift_width: u32,
    /// Number of equal output segments.
    num_segments: usize,
    /// Resolution of the values returned by the loader.
    resolution: u32,
}

/// Validate the options shared by the single- and multi-loader native
/// augmentation APIs.
fn center_split_config(
    store: &DataStore,
    center_length: Option<u32>,
    shift_width: u32,
    num_segments: Option<usize>,
) -> Result<CenterSplitConfig> {
    // The new operation is deliberately defined on a complete parent record.
    // Applying an existing split/trim/aggregation first would make the
    // available flank ambiguous and would also make coordinates inaccurate.
    ensure!(
        store.read_opts.split_size.is_none(),
        "center/split augmentation cannot be combined with window_size"
    );
    ensure!(
        store.read_opts.value_length.is_none(),
        "center/split augmentation cannot be combined with target_length"
    );
    ensure!(
        !store.has_aggregation(),
        "center/split augmentation requires the stored resolution"
    );
    ensure!(
        store.read_opts.shift_width == 0,
        "center/split augmentation requires random_shift=0 on the loader"
    );
    ensure!(
        store.read_opts.scale_value.is_none() && store.read_opts.clamp_value_max.is_none(),
        "center/split augmentation cannot be combined with scale or clamp"
    );

    let center_length = center_length.unwrap_or(store.sequence_length());
    let num_segments = num_segments.unwrap_or(1);
    ensure!(
        num_segments <= u32::MAX as usize,
        "segments_length is too large"
    );
    let num_segments_u32 = num_segments as u32;
    ensure!(center_length > 0, "center_length must be positive");
    ensure!(num_segments > 0, "segments_length must be positive");
    ensure!(
        center_length % num_segments_u32 == 0,
        "center_length must be divisible by segments_length"
    );
    let physical_parent_length = store
        .sequence_length()
        .checked_add(store.n_pad().saturating_mul(2))
        .ok_or_else(|| anyhow::anyhow!("parent sequence length overflows u32"))?;
    ensure!(
        center_length <= physical_parent_length,
        "center_length cannot exceed the parent sequence length"
    );
    ensure!(
        center_length % store.out_resolution == 0,
        "center_length must be a multiple of the output resolution"
    );
    ensure!(
        (center_length / num_segments_u32) % store.out_resolution == 0,
        "each output segment must be a multiple of the output resolution"
    );
    ensure!(
        shift_width % store.out_resolution == 0,
        "shift must be a multiple of the output resolution"
    );

    Ok(CenterSplitConfig {
        center_length,
        shift_width,
        num_segments,
        resolution: store.out_resolution,
    })
}

/// Crop a complete parent record around its center, apply one shared random
/// shift, and split the selected window into equal segments.
///
/// The requested shift is automatically limited by the actual extra sequence
/// present in the parent.  This is what permits `padding=0` gdata records to
/// provide augmentation as long as their stored sequence is longer than the
/// requested center window.
#[derive(Debug, Clone)]
struct CenterSplitPlan {
    start: usize,
    shift: i64,
    segment_length: usize,
    order: Vec<usize>,
}

fn make_center_split_plan(
    parent_length: usize,
    config: CenterSplitConfig,
    rng: &mut ChaCha12Rng,
) -> Result<CenterSplitPlan> {
    let center_length = config.center_length as usize;
    ensure!(
        parent_length >= center_length,
        "parent length {} is shorter than center_length {}",
        parent_length,
        center_length
    );
    let extra = parent_length - center_length;
    ensure!(
        parent_length % config.resolution as usize == 0,
        "parent length must be a multiple of the output resolution"
    );
    let extra_bins = extra / config.resolution as usize;
    let maximum_symmetric_shift = (extra_bins / 2) * config.resolution as usize;
    // A no-padding parent can still contain extra context.  Use as much of
    // the requested shift as the actual parent permits, instead of requiring
    // callers to know the physical record length in advance.
    let effective_shift = config.shift_width.min(maximum_symmetric_shift as u32);
    let shift_bins = (effective_shift / config.resolution) as i64;
    let shift = if shift_bins == 0 {
        0
    } else {
        rng.random_range(-shift_bins..=shift_bins) * config.resolution as i64
    };
    let center_start = (extra_bins / 2) * config.resolution as usize;
    let start = (center_start as i64 + shift) as usize;
    let end = start + center_length;
    ensure!(
        end <= parent_length,
        "computed center crop [{start}, {end}) exceeds parent length {parent_length}"
    );

    let mut order: Vec<usize> = (0..config.num_segments).collect();
    order.shuffle(rng);
    Ok(CenterSplitPlan {
        start,
        shift,
        segment_length: center_length / config.num_segments,
        order,
    })
}

fn apply_center_split_plan(
    region: GenomicRange,
    physical_start: u64,
    sequence: Array2<u8>,
    values: Array3<bf16>,
    config: CenterSplitConfig,
    channels_last: bool,
    plan: &CenterSplitPlan,
) -> Result<AugmentedBFloat16Record> {
    ensure!(
        sequence.ndim() == 2 && sequence.shape()[0] == 1,
        "native center/split expects a parent batch of one"
    );
    ensure!(
        values.ndim() == 3 && values.shape()[0] == 1,
        "native center/split expects values with batch size one"
    );

    let parent_length = sequence.shape()[1];
    let parent_value_length = parent_length / config.resolution as usize;
    let n_tracks = if channels_last {
        ensure!(
            values.shape()[1] == parent_value_length,
            "channels-last values do not match sequence length"
        );
        values.shape()[2]
    } else {
        ensure!(
            values.shape()[2] == parent_value_length,
            "channels-first values do not match sequence length"
        );
        values.shape()[1]
    };

    let output_sequence = split_sequence_with_plan(&sequence, config, plan);
    let output_values = split_values_with_plan(&values, config, channels_last, plan, n_tracks);
    let metadata = segment_metadata(&region, physical_start, plan);

    Ok((output_sequence, output_values, metadata))
}

fn split_sequence_with_plan(
    sequence: &Array2<u8>,
    config: CenterSplitConfig,
    plan: &CenterSplitPlan,
) -> Array2<u8> {
    let mut output = Array2::<u8>::zeros((config.num_segments, plan.segment_length));
    for (output_index, &chunk_index) in plan.order.iter().enumerate() {
        let chunk_start = plan.start + chunk_index * plan.segment_length;
        let chunk_end = chunk_start + plan.segment_length;
        output
            .slice_mut(ndarray::s![output_index, ..])
            .assign(&sequence.slice(ndarray::s![0, chunk_start..chunk_end]));
    }
    output
}

fn split_values_with_plan(
    values: &Array3<bf16>,
    config: CenterSplitConfig,
    channels_last: bool,
    plan: &CenterSplitPlan,
    n_tracks: usize,
) -> Array3<bf16> {
    let resolution = config.resolution as usize;
    let segment_value_length = plan.segment_length / resolution;
    let mut output = if channels_last {
        Array3::<bf16>::from_elem(
            (config.num_segments, segment_value_length, n_tracks),
            bf16::from_f32(0.0),
        )
    } else {
        Array3::<bf16>::from_elem(
            (config.num_segments, n_tracks, segment_value_length),
            bf16::from_f32(0.0),
        )
    };
    for (output_index, &chunk_index) in plan.order.iter().enumerate() {
        let chunk_start = (plan.start + chunk_index * plan.segment_length) / resolution;
        let chunk_end = chunk_start + segment_value_length;
        if channels_last {
            output
                .slice_mut(ndarray::s![output_index, .., ..])
                .assign(&values.slice(ndarray::s![0, chunk_start..chunk_end, ..]));
        } else {
            output
                .slice_mut(ndarray::s![output_index, .., ..])
                .assign(&values.slice(ndarray::s![0, .., chunk_start..chunk_end]));
        }
    }
    output
}

fn segment_metadata(
    region: &GenomicRange,
    physical_start: u64,
    plan: &CenterSplitPlan,
) -> Vec<AugmentedSegmentMetadata> {
    plan.order
        .iter()
        .map(|&chunk_index| {
            let chunk_start = plan.start + chunk_index * plan.segment_length;
            let genomic_start = physical_start + chunk_start as u64;
            let genomic_end = genomic_start + plan.segment_length as u64;
            (
                format!("{}:{}-{}", region.chrom(), genomic_start, genomic_end),
                plan.shift,
                chunk_index,
            )
        })
        .collect()
}

fn center_split_record(
    region: GenomicRange,
    physical_start: u64,
    sequence: Array2<u8>,
    values: Array3<bf16>,
    config: CenterSplitConfig,
    channels_last: bool,
    rng: &mut ChaCha12Rng,
) -> Result<AugmentedBFloat16Record> {
    ensure!(
        sequence.ndim() == 2 && sequence.shape()[0] == 1,
        "native center/split expects a parent batch of one"
    );
    let plan = make_center_split_plan(sequence.shape()[1], config, rng)?;
    apply_center_split_plan(
        region,
        physical_start,
        sequence,
        values,
        config,
        channels_last,
        &plan,
    )
}

/// Concatenate complete native parent outputs into one segment batch.
///
/// The center/split iterator intentionally yields all segments from one
/// parent together.  The Python training adapter, however, usually wants a
/// conventional segment batch.  Doing this concatenation in Rust avoids one
/// Python ``next()`` call and one Python-side stack operation per parent.
fn combine_augmented_bfloat16_records(
    records: Vec<AugmentedBFloat16Record>,
) -> Result<AugmentedBFloat16Record> {
    ensure!(!records.is_empty(), "cannot combine an empty record list");
    let first = records.first().unwrap();
    let segments_per_parent = first.0.shape()[0];
    let sequence_length = first.0.shape()[1];
    let value_dim_1 = first.1.shape()[1];
    let value_dim_2 = first.1.shape()[2];
    ensure!(segments_per_parent > 0, "native record has no segments");

    let total_segments: usize = records.iter().map(|record| record.0.shape()[0]).sum();
    let mut sequence = Array2::<u8>::zeros((total_segments, sequence_length));
    let mut values = Array3::<bf16>::from_elem(
        (total_segments, value_dim_1, value_dim_2),
        bf16::from_f32(0.0),
    );
    let mut metadata = Vec::with_capacity(total_segments);
    let mut output_start = 0;

    for (current_sequence, current_values, current_metadata) in records {
        let current_segments = current_sequence.shape()[0];
        ensure!(
            current_sequence.shape()[1] == sequence_length,
            "native batch contains incompatible sequence lengths"
        );
        ensure!(
            current_values.shape()
                == [current_segments, value_dim_1, value_dim_2],
            "native batch contains incompatible value shapes"
        );
        ensure!(
            current_metadata.len() == current_segments,
            "native batch metadata does not match the number of segments"
        );

        let output_end = output_start + current_segments;
        sequence
            .slice_mut(ndarray::s![output_start..output_end, ..])
            .assign(&current_sequence);
        values
            .slice_mut(ndarray::s![output_start..output_end, .., ..])
            .assign(&current_values);
        metadata.extend(current_metadata);
        output_start = output_end;
    }

    Ok((sequence, values, metadata))
}

#[cfg(test)]
mod center_split_tests {
    use super::*;
    use std::collections::HashSet;

    fn test_region() -> GenomicRange {
        GenomicRange::from_str("chr1:100-112").unwrap()
    }

    fn test_sequence() -> Array2<u8> {
        Array2::from_shape_fn((1, 12), |(_, i)| i as u8)
    }

    fn test_values() -> Array3<bf16> {
        Array3::from_shape_fn((1, 1, 12), |(_, _, i)| bf16::from_f32(i as f32))
    }

    #[test]
    fn shift_is_limited_by_the_available_symmetric_flank() {
        let config = CenterSplitConfig {
            center_length: 8,
            shift_width: 100,
            num_segments: 2,
            resolution: 1,
        };
        let mut rng = ChaCha12Rng::seed_from_u64(7);
        for _ in 0..100 {
            let plan = make_center_split_plan(12, config, &mut rng).unwrap();
            assert!((-2..=2).contains(&plan.shift));
            assert!(plan.start <= 4);
            assert!(plan.start + config.center_length as usize <= 12);
        }
    }

    #[test]
    fn center_crop_split_and_coordinates_are_consistent() {
        let config = CenterSplitConfig {
            center_length: 8,
            shift_width: 0,
            num_segments: 4,
            resolution: 1,
        };
        let mut rng = ChaCha12Rng::seed_from_u64(9);
        let (sequence, values, metadata) = center_split_record(
            test_region(),
            100,
            test_sequence(),
            test_values(),
            config,
            false,
            &mut rng,
        )
        .unwrap();

        assert_eq!(sequence.shape(), &[4, 2]);
        assert_eq!(values.shape(), &[4, 1, 2]);
        assert_eq!(metadata.len(), 4);
        let mut seen = HashSet::new();
        for (row, (name, shift, chunk_index)) in metadata.iter().enumerate() {
            assert_eq!(*shift, 0);
            assert!(seen.insert(*chunk_index));
            let chunk_start = 2 + chunk_index * 2;
            assert_eq!(sequence[[row, 0]], chunk_start as u8);
            assert_eq!(sequence[[row, 1]], (chunk_start + 1) as u8);
            assert_eq!(values[[row, 0, 0]].to_f32(), chunk_start as f32);
            assert_eq!(
                name,
                &format!("chr1:{}-{}", 100 + chunk_start, 100 + chunk_start + 2)
            );
        }
        assert_eq!(seen, HashSet::from([0, 1, 2, 3]));
    }

    #[test]
    fn values_are_split_in_bins_when_resolution_is_greater_than_one() {
        let config = CenterSplitConfig {
            center_length: 8,
            shift_width: 0,
            num_segments: 2,
            resolution: 2,
        };
        let sequence = Array2::from_shape_fn((1, 12), |(_, i)| i as u8);
        let values = Array3::from_shape_fn((1, 1, 6), |(_, _, i)| bf16::from_f32(i as f32));
        let mut rng = ChaCha12Rng::seed_from_u64(11);
        let (sequence, values, metadata) = center_split_record(
            test_region(),
            100,
            sequence,
            values,
            config,
            false,
            &mut rng,
        )
        .unwrap();
        assert_eq!(sequence.shape(), &[2, 4]);
        assert_eq!(values.shape(), &[2, 1, 2]);
        for (row, (_, _, chunk_index)) in metadata.iter().enumerate() {
            let value_start = 1 + chunk_index * 2;
            assert_eq!(values[[row, 0, 0]].to_f32(), value_start as f32);
        }
    }

    #[test]
    fn multi_record_uses_one_plan_for_all_modalities() {
        let config = CenterSplitConfig {
            center_length: 8,
            shift_width: 2,
            num_segments: 2,
            resolution: 1,
        };
        let mut values = IndexMap::new();
        values.insert("atac".to_string(), test_values());
        values.insert("dnase".to_string(), test_values());
        let mut rng = ChaCha12Rng::seed_from_u64(13);
        let (sequence, output, metadata) = center_split_multi_record(
            test_region(),
            100,
            test_sequence(),
            values,
            config,
            false,
            &mut rng,
        )
        .unwrap();
        assert_eq!(output["atac"], output["dnase"]);
        assert_eq!(sequence.shape(), &[2, 4]);
        assert_eq!(metadata.len(), 2);
        assert_eq!(metadata[0].1, metadata[1].1);
    }
}

/** A dataloader for genomic data, allowing for efficient retrieval of genomic
    sequences and their associated values.

    This object provides an iterator over genomic data chunks, enabling batch
    retrieval of genomic sequences and their associated values.
    The iterator yields tuples of (sequences, values).
    Sequences has shape (batch_size, sequence_length), and values has shape
    (batch_size, sequence_length / resolution, num_tracks).

    Parameters
    ----------
    location : Path
        The path to the genomic data directory.
    batch_size : int
        The number of genomic sequences to retrieve in each batch (default is 8).
    resolution : Optional[int]
        The resolution of the genomic data. If not provided, it defaults to the dataset's resolution.
        If the resolution is provided, it must be a multiple of the dataset's resolution.
        The values will be aggregated (by taking the average) to this resolution when it is
        higher than the dataset's resolution. Requesting the dataset's native resolution does
        not perform an identity aggregation pass.
    trim_target: Optional[int]
        Trim both ends of the target vector according to the `trim_target` parameter.
        As a result, the length of the values will be reduced by `2 * trim_target`.
        The unit of `trim_target` is base pairs, and it must be a multiple of the resolution.
        Note this only affects the values, not the sequences. The sequences will always
        have the full length as defined in the dataset.
        This is useful when you want to compute the loss on only the central part of the sequence.
        This is because the edges of the sequence may contain
        padding or other artifacts that should not be considered in the loss computation.
    scale : Optional[float]
        Scale the values by this factor. If not provided, no scaling is applied.
    clamp_max : Optional[float]
        Clamp the values to this maximum value. If not provided, no clamping is applied.
        If `scale` is also provided, the clamping will be applied after scaling.
        If neither `scale` nor `clamp_max` is provided, no value transformation or NaN scan is
        performed.
    window_size : Optional[int]
        The window size for retrieving genomic sequences. The loader's window size
        can be different from the underlying dataset's window size so that the same
        dataset can be used with different window sizes. However, there are two
        restrictions: (1) The dataset's window size must be a multiple of the loader's window size;
        (2) The loader's window size must be a multiple of the dataset's resolution.
    shuffle : bool
        If True, the data will be shuffled before being returned. Default is False.
    random_shift: int
        The maximum random shift (in base pairs) to apply to the start position of each sequence.
        The actual shift will be randomly chosen from the range [-random_shift, random_shift].
        This is useful for data augmentation, as it introduces variability in the sequences
        retrieved from the dataset.
    seq_as_string : bool
        If True, sequences will be returned as strings instead of numpy integer arrays.
        This is useful for cases where you want to work with the sequences as text,
        such as for visualization or text-based analysis.
    n_jobs: int
        The number of parallel jobs to use for loading data.
        This allows for asynchronous loading of data, improving performance during training or inference.
        But it will increase memory usage, the memory usage will be approximately
        `2 * n_jobs * memory_of_chunk`.
    random_seed : int
        The random seed for shuffling the data. Default is 2025.

    See Also
    --------
    GenomeDataBuilder
    GenomeDataLoaderMap

    Examples
    --------
    >>> from gdata import as GenomeDataLoader
    >>> loader = GenomeDataLoader("test_genome", trim_target=40_960)
    >>> region = 'chr11:35041782-35238390'
    >>> tracks = ['DNase:CD14-positive monocyte', 'DNase:keratinocyte', 'ChIP-H3K27ac:keratinocyte']
    >>> loader.plot(region, tracks, savefig="signal.png")

    .. image:: /_static/images/genome_signal.png
        :align: center
*/
#[pyclass]
#[derive(Debug, Clone)]
pub struct GenomeDataLoader {
    data_store: DataStore,
    subset: Option<Vec<GenomicRange>>,
    #[pyo3(get, set)]
    batch_size: usize,
    #[pyo3(get, set)]
    shuffle: bool,
    seq_as_string: bool,
    n_jobs: usize,
    random_seed: u64,
}

impl std::fmt::Display for GenomeDataLoader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        writeln!(
            f,
            "GenomeDataLoader ({} segments x {} tracks):",
            self.num_segments(),
            self.tracks().len()
        )?;
        write!(
            f,
            "    window_size = {}, resolution = {}, batch_size = {}, target_length = {}, padding = {}",
            self.window_size(),
            self.resolution(),
            self.batch_size,
            self.data_store
                .read_opts
                .value_length
                .unwrap_or(self.data_store.sequence_length()),
            self.data_store.n_pad(),
        )?;
        Ok(())
    }
}

impl GenomeDataLoader {
    pub fn num_segments(&self) -> usize {
        self.subset
            .as_ref()
            .map_or_else(|| self.data_store.num_segments(), |s| s.len())
    }

    pub fn num_batch(&self) -> usize {
        let mut n = self.num_segments();
        if let Some(split_size) = self.data_store.read_opts.split_size {
            n *= (self.data_store.sequence_length() / split_size) as usize
        }
        n / self.batch_size + if n % self.batch_size > 0 { 1 } else { 0 }
    }

    pub fn set_target_length(&mut self, target_length: u32) -> Result<()> {
        self.data_store.set_value_length(target_length)
    }

    pub fn set_window_size(&mut self, window_size: u32) -> Result<()> {
        self.data_store.set_split_size(window_size)
    }

    pub fn intersection(&self, regions: impl Iterator<Item = GenomicRange>) -> Self {
        let regions: HashSet<_> = regions.collect();
        let subset = if let Some(subset) = &self.subset {
            subset
                .iter()
                .filter(|x| regions.contains(x))
                .cloned()
                .collect()
        } else {
            self.data_store
                .segments()
                .filter(|x| regions.contains(x))
                .cloned()
                .collect()
        };
        let mut loader = self.clone();
        loader.subset = Some(subset);
        loader
    }

    pub fn difference(&self, regions: impl Iterator<Item = GenomicRange>) -> Self {
        let regions: HashSet<_> = regions.collect();
        let subset = if let Some(subset) = &self.subset {
            subset
                .iter()
                .filter(|x| !regions.contains(x))
                .cloned()
                .collect()
        } else {
            self.data_store
                .segments()
                .filter(|x| !regions.contains(x))
                .cloned()
                .collect()
        };
        let mut loader = self.clone();
        loader.subset = Some(subset);
        loader
    }

    fn ordered_regions(&self) -> Vec<GenomicRange> {
        let mut regions = self
            .subset
            .clone()
            .unwrap_or_else(|| self.data_store.segments().cloned().collect());
        if self.shuffle {
            let mut rng = ChaCha12Rng::seed_from_u64(self.random_seed);
            regions.shuffle(&mut rng);
        }
        regions
    }

    fn raw_region_bfloat16_iterator(
        &self,
        regions: Vec<GenomicRange>,
        channels_last: bool,
    ) -> ParallelLoader<DataStoreBf16ParentRegionIter, RegionBFloat16Record> {
        self.data_store
            .clone()
            .par_iter_bf16_parent_with_layout_and_regions(regions, self.n_jobs, channels_last)
    }

    fn center_split_config(
        &self,
        center_length: Option<u32>,
        shift_width: u32,
        num_segments: Option<usize>,
    ) -> Result<CenterSplitConfig> {
        center_split_config(&self.data_store, center_length, shift_width, num_segments)
    }

    pub fn iter(&mut self) -> GenomeDataLoaderIter {
        // Keep the default compatible with the historical loader while making
        // the queue depth tunable for large compressed records.  A larger
        // queue can hide bursty decompression latency, at the cost of holding
        // more decoded batches in memory.  This is intentionally an
        // environment variable so existing Python APIs and data files remain
        // compatible.
        let prefetch_multiplier = std::env::var("GDATA_PREFETCH_MULTIPLIER")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(2);
        let iter = PrefethIterator::new(
            self.data_store.par_iter(
                self.batch_size,
                self.n_jobs,
                self.shuffle,
                self.subset.as_ref().map(|x| x.as_slice()),
            ),
            self.n_jobs.saturating_mul(prefetch_multiplier),
        );

        GenomeDataLoaderIter {
            iter,
            seq_as_string: self.seq_as_string,
        }
    }

    /// Iterate over owned bfloat16 values exposed as DLPack capsules.
    ///
    /// The regular [`iter`](Self::iter) API is intentionally kept unchanged
    /// for NumPy users.  This opt-in iterator is used by the PyTorch training
    /// path and avoids both the Rust bfloat16-to-float32 conversion and the
    /// subsequent Python float32-to-bfloat16 conversion.
    pub fn iter_bfloat16_dlpack(&mut self) -> GenomeDataLoaderBFloat16DLPackIter {
        self.iter_bfloat16_dlpack_with_layout(true)
    }

    /// Iterate over native bfloat16 values with an explicit channel layout.
    ///
    /// `channels_last=true` yields values shaped `(batch, sequence, track)`.
    /// `channels_last=false` yields `(batch, track, sequence)` and lets the
    /// reader reuse the decoded track-major allocation when no crop or other
    /// sequence transform is needed.
    pub fn iter_bfloat16_dlpack_with_layout(
        &mut self,
        channels_last: bool,
    ) -> GenomeDataLoaderBFloat16DLPackIter {
        let prefetch_multiplier = std::env::var("GDATA_PREFETCH_MULTIPLIER")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(2);
        let iter = PrefethIterator::new(
            self.data_store.par_iter_bf16_with_layout(
                self.batch_size,
                self.n_jobs,
                self.shuffle,
                self.subset.as_ref().map(|x| x.as_slice()),
                channels_last,
            ),
            self.n_jobs.saturating_mul(prefetch_multiplier),
        );

        GenomeDataLoaderBFloat16DLPackIter {
            iter,
            seq_as_string: self.seq_as_string,
        }
    }

    /// Iterate over native bfloat16 data after a runtime center crop and
    /// equal split.  The parent record is not required to declare gdata
    /// padding: any extra bases physically present in the parent are used as
    /// the available shift flank.
    ///
    /// Each yielded item contains arrays for all output segments from one
    /// parent and a list of `(segment_name, applied_shift, chunk_index)`
    /// metadata tuples.  The low-level iterator reads one parent at a time so
    /// one shared shift/order plan is unambiguous.  It does this even when
    /// the loader was constructed with a larger ordinary `batch_size`; that
    /// setting only affects the legacy iterator.
    pub fn iter_bfloat16_dlpack_center_split(
        &self,
        channels_last: bool,
        center_length: Option<u32>,
        shift_width: u32,
        num_segments: Option<usize>,
    ) -> Result<GenomeDataLoaderAugmentedBFloat16DLPackIter> {
        let config = self.center_split_config(center_length, shift_width, num_segments)?;
        let seed = self.random_seed ^ 0x6a09e667f3bcc909;
        let raw = self.raw_region_bfloat16_iterator(self.ordered_regions(), channels_last);
        let mut rng = ChaCha12Rng::seed_from_u64(seed);
        let augmented = raw.map(move |(region, physical_start, sequence, values)| {
            center_split_record(
                region,
                physical_start,
                sequence,
                values,
                config,
                channels_last,
                &mut rng,
            )
            .expect("invalid center/split augmentation record")
        });
        let prefetch_multiplier = std::env::var("GDATA_PREFETCH_MULTIPLIER")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(2);
        Ok(GenomeDataLoaderAugmentedBFloat16DLPackIter {
            iter: PrefethIterator::new(
                augmented,
                self.n_jobs.max(1).saturating_mul(prefetch_multiplier),
            ),
            seq_as_string: self.seq_as_string,
            num_segments: config.num_segments,
        })
    }
}

#[pymethods]
impl GenomeDataLoader {
    #[new]
    #[pyo3(
        signature = (location, *,
            batch_size=8, resolution=None, target_length=None, scale=None, clamp_max=None,
            window_size=None, shuffle=false, random_shift=0, seq_as_string=false, n_jobs=8,
            random_seed=2025,
        ),
        text_signature = "($self, location, *,
            batch_size=8, resolution=None, target_length=None, scale=None, clamp_max=None,
            window_size=None, shuffle=False, random_shift=0, seq_as_string=False, n_jobs=8,
            random_seed=2025)"
    )]
    pub fn new(
        location: PathBuf,
        batch_size: usize,
        resolution: Option<u32>,
        target_length: Option<u32>,
        scale: Option<f32>,
        clamp_max: Option<f32>,
        window_size: Option<u32>,
        shuffle: bool,
        random_shift: u32,
        seq_as_string: bool,
        n_jobs: usize,
        random_seed: u64,
    ) -> Result<Self> {
        let store_opts = DataStoreReadOptions {
            shift_width: random_shift,
            value_length: target_length,
            split_size: window_size,
            read_resolution: resolution,
            scale_value: scale.map(|x| bf16::from_f32(x)),
            clamp_value_max: clamp_max.map(|x| bf16::from_f32(x)),
            rng: ChaCha12Rng::seed_from_u64(random_seed),
            ..Default::default()
        };

        let loader = Self {
            data_store: DataStore::open(location, store_opts)?,
            subset: None,
            batch_size,
            shuffle,
            seq_as_string,
            n_jobs,
            random_seed,
        };

        Ok(loader)
    }

    /** Returns the track names in the dataset.

       This method retrieves all keys from the dataset, which are typically the names of files
       containing genomic data. The keys are sorted alphabetically.

       Returns
       -------
       list[str]
           A sorted list of keys as strings.
    */
    #[getter]
    pub fn tracks(&self) -> Vec<String> {
        self.data_store
            .data_keys()
            .iter()
            .cloned()
            .collect::<Vec<_>>()
    }

    /** Returns the segments of the genome as a vector of strings.

       Returns
       -------
       list[str]
           A list of segment strings representing genomic ranges.
    */
    #[getter]
    pub fn segments(&self) -> Vec<String> {
        if let Some(subset) = &self.subset {
            subset.iter().map(|x| x.pretty_show()).collect()
        } else {
            self.data_store
                .segments()
                .map(|x| x.pretty_show())
                .collect()
        }
    }

    /** The length of input sequences in base pairs.

       This method returns the length of the input sequences in base pairs.
       It is determined by the `split_size` option in the read options of the data store.
       If `split_size` is not set, it defaults to the sequence length of the data store.

       Returns
       -------
       int
           The length of input sequences in base pairs.
    */
    #[getter]
    fn window_size(&self) -> u32 {
        self.data_store
            .read_opts
            .split_size
            .unwrap_or(self.data_store.sequence_length())
    }

    /** The length of the target vector in base pairs.

       Returns
       -------
       int
           The length of the target vector in base pairs.
    */
    #[getter]
    fn target_length(&self) -> u32 {
        self.data_store
            .read_opts
            .value_length
            .unwrap_or(self.window_size())
    }

    #[getter]
    fn resolution(&self) -> u32 {
        self.data_store.out_resolution
    }

    /** Returns the data indexer for accessing genomic data.

        This method provides an indexer that allows access to the genomic data
        associated with the sequences. It can be used to retrieve values for specific
        genomic regions and tracks.

        Returns
        -------
        _DataIndexer
            An indexer for accessing genomic data.
    */
    #[getter]
    fn data(slf: PyRef<'_, Self>) -> DataIndexer {
        DataIndexer(slf.into())
    }

    /** Creating a new genomic data loader in which the regions intersect with the specified ones.

        This method allows you to subset the genomic data by intersecting it with
        specified regions.
        The dataset will be updated to only include the regions that intersect with the provided ones.

        Parameters
        ----------
        regions : list[str]
            A list of genomic ranges in string format (e.g., 'chr1:1000-2000').

        See Also
        --------
        difference : For creating a loader with differing regions.

        Returns
        -------
        GenomeDataLoader
            A new instance of `GenomeDataLoader` that contains only the data for the specified regions.
    */
    #[pyo3(
        name = "intersection",
        signature = (regions),
        text_signature = "($self, regions)"
    )]
    fn intersection_py(&self, regions: Vec<String>) -> Self {
        self.intersection(
            regions.into_iter().map(|r| {
                GenomicRange::from_str(&r).expect(&format!("Invalid genomic range: {}", r))
            }),
        )
    }

    /** Creating a new genomic data loader in which the regions differ from the specified ones.

        This method allows you to subset the genomic data by removing the specified regions.

        Parameters
        ----------
        regions : list[str]
            A list of genomic ranges in string format (e.g., 'chr1:1000-2000').

        See Also
        --------
        intersection : For creating a loader with intersecting regions.

        Returns
        -------
        GenomeDataLoader
            A new instance of `GenomeDataLoader` that contains only the data for the regions
            that do not intersect with the specified ones.
    */
    #[pyo3(
        name = "difference",
        signature = (regions),
        text_signature = "($self, regions)"
    )]
    fn difference_py(&self, regions: Vec<String>) -> Self {
        self.difference(
            regions.into_iter().map(|r| {
                GenomicRange::from_str(&r).expect(&format!("Invalid genomic range: {}", r))
            }),
        )
    }

    /** Create a copy of the GenomeDataLoader.

        This method creates a new instance of `GenomeDataLoader` with the same configuration
        as the current instance. It is useful for creating independent copies of the loader
        that can be modified without affecting the original.

        Returns
        -------
        GenomeDataLoader
            A new instance of `GenomeDataLoader` with the same configuration.
    */
    fn copy(&self) -> Self {
        self.clone()
    }

    /** Plots the genomic signal for a specified region and tracks.

        This method generates a plot of the genomic signal for the specified region and tracks.
        If `savefig` is provided, it saves the plot to the specified file; otherwise, it displays the plot.

        Parameters
        ----------
        region : str
            The genomic region to plot, in the format 'chr:start-end'.
        tracks : list[str]
            A list of track names to plot.
        savefig : Optional[PathBuf]
            If provided, saves the plot to this file instead of displaying it.

        Returns
        -------
        None
    */
    #[pyo3(
        signature = (
            region, tracks, *, savefig=None,
        ),
        text_signature = "($self, region, tracks, *, savefig=None)"
    )]
    fn plot(
        slf: PyRef<'_, Self>,
        py: Python<'_>,
        region: &str,
        tracks: Bound<'_, PyAny>,
        savefig: Option<PathBuf>,
    ) -> Result<()> {
        let trim = (slf.window_size() - slf.target_length()) / 2;
        let mut data_indexer = Self::data(slf);
        let key = (region, &tracks).into_pyobject(py)?.into_any();
        let signal_values = data_indexer.__getitem__(py, key)?.1;
        let track_names = extract_string_list(tracks)?;

        let py_code = r#"
from matplotlib import pyplot as plt
import numpy as np
height_per_track = 1.5
width = 8

start, end = region.split(":")[1].split('-')
start = int(start) + trim
end = int(end) - trim
signal_values = np.squeeze(signal_values).T
n_tracks, n_points = signal_values.shape
fig, axes = plt.subplots(n_tracks, 1, figsize=[width, n_tracks * height_per_track], sharex=True)

if n_tracks == 1:
    axes = [axes]

for i, (ax, signal) in enumerate(zip(axes, signal_values)):
    ax.fill_between(range(n_points), 0, signal, color='black')
    ax.set_title(track_names[i], fontsize=7)
    ax.spines[['top', 'right']].set_visible(False)

axes[-1].set_xticks([0, n_points - 1])
axes[-1].set_xticklabels([str(start), str(end)])
axes[-1].set_xlabel(region)

plt.tight_layout()
if savefig is None:
    plt.show()
else:
    plt.savefig(savefig, dpi=300, bbox_inches='tight')
"#;
        py_run!(py, trim region track_names signal_values savefig, py_code);

        Ok(())
    }

    #[pyo3(
        signature = (*, bins=10000, limit=100, min=None, log=false, savefig=None),
        text_signature = "($self, *, bins=10000, limit=100, min=None, log=False, savefig=None)"
    )]
    fn hist(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'_>,
        bins: usize,
        limit: usize,
        min: Option<f32>,
        log: bool,
        savefig: Option<PathBuf>,
    ) -> Vec<((f32, f32), usize)> {
        let (lo, hi) = slf
            .iter()
            .take(limit)
            .flat_map(|(_, values)| {
                values.into_iter().flat_map(|mut x| {
                    if min.is_some() && x < min.unwrap() {
                        None
                    } else {
                        if log {
                            x = (x + 1.0).ln();
                        }
                        Some(x)
                    }
                })
            })
            .minmax()
            .into_option()
            .unwrap();

        let bin_size = (hi - lo) / bins as f32;
        let mut histogram = vec![0; bins];
        slf.iter().take(limit).for_each(|(_, values)| {
            for mut value in values.into_iter() {
                if min.is_some() && value < min.unwrap() {
                    continue;
                }
                if log {
                    value = (value + 1.0).ln();
                }
                let bin_index = ((value - lo) / bin_size).floor() as usize;
                if bin_index < bins {
                    histogram[bin_index] += 1;
                }
            }
        });

        let result: Vec<_> = histogram
            .into_iter()
            .enumerate()
            .flat_map(|(i, count)| {
                if count == 0 {
                    None
                } else {
                    let bin_start = lo + i as f32 * bin_size;
                    let bin_end = bin_start + bin_size;
                    Some(((bin_start, bin_end), count))
                }
            })
            .collect();

        let (xs, ys): (Vec<_>, Vec<_>) = result
            .iter()
            .map(|((x1, x2), y)| ((x1 + x2) / 2.0, *y))
            .unzip();

        let py_code = r#"
from matplotlib import pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(xs, ys, 'b-')
if log:
    ax.set_xlabel("log(value + 1)")
else:
    ax.set_xlabel("value")
plt.tight_layout()
if savefig is None:
    plt.show()
else:
    plt.savefig(savefig, dpi=300, bbox_inches='tight')
"#;
        py_run!(py, xs ys log savefig, py_code);
        result
    }

    fn __len__(&self) -> usize {
        self.num_batch()
    }

    fn __iter__(mut slf: PyRefMut<'_, Self>) -> GenomeDataLoaderIter {
        slf.iter()
    }

    /// Python-facing opt-in iterator which yields DLPack capsules containing
    /// native bfloat16 values.
    #[pyo3(name = "iter_bfloat16_dlpack")]
    #[pyo3(signature = (channels_last=true))]
    fn iter_bfloat16_dlpack_py(
        mut slf: PyRefMut<'_, Self>,
        channels_last: bool,
    ) -> GenomeDataLoaderBFloat16DLPackIter {
        slf.iter_bfloat16_dlpack_with_layout(channels_last)
    }

    /// Python-facing native center/crop/split iterator.
    #[pyo3(name = "iter_bfloat16_dlpack_center_split")]
    #[pyo3(signature = (channels_last=false, mid=None, shift=0, segments_length=None))]
    fn iter_bfloat16_dlpack_center_split_py(
        slf: PyRef<'_, Self>,
        channels_last: bool,
        mid: Option<u32>,
        shift: u32,
        segments_length: Option<usize>,
    ) -> PyResult<GenomeDataLoaderAugmentedBFloat16DLPackIter> {
        slf.iter_bfloat16_dlpack_center_split(channels_last, mid, shift, segments_length)
            .map_err(Into::into)
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }
}

#[pyclass]
pub struct GenomeDataLoaderIter {
    iter: PrefethIterator<(Array2<u8>, Array3<f32>)>,
    seq_as_string: bool,
}

impl Iterator for GenomeDataLoaderIter {
    type Item = (Array2<u8>, Array3<f32>);

    fn next(&mut self) -> Option<Self::Item> {
        let (mut seq, val) = self.iter.next()?;
        if self.seq_as_string {
            seq.mapv_inplace(|x| decode_nucleotide(x).unwrap());
        }
        Some((seq, val))
    }
}

#[pymethods]
impl GenomeDataLoaderIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'a>(
        mut slf: PyRefMut<'a, Self>,
        py: Python<'a>,
    ) -> Option<Bound<'a, pyo3::types::PyTuple>> {
        let (seq, values) = slf.next()?;
        let values = PyArray3::from_owned_array(py, values);
        let result = if slf.seq_as_string {
            let seq = seq_to_string(&seq);
            (seq, values).into_pyobject(py).unwrap()
        } else {
            let seq = PyArray2::from_owned_array(py, seq);
            (seq, values).into_pyobject(py).unwrap()
        };
        Some(result)
    }
}

/// Python iterator for the native bfloat16/DLPack path.
#[pyclass]
pub struct GenomeDataLoaderBFloat16DLPackIter {
    iter: PrefethIterator<(Array2<u8>, Array3<bf16>)>,
    seq_as_string: bool,
}

/// Python iterator for native center-cropped and split bfloat16 records.
#[pyclass]
pub struct GenomeDataLoaderAugmentedBFloat16DLPackIter {
    iter: PrefethIterator<AugmentedBFloat16Record>,
    seq_as_string: bool,
    num_segments: usize,
}

impl Iterator for GenomeDataLoaderAugmentedBFloat16DLPackIter {
    type Item = AugmentedBFloat16Record;

    fn next(&mut self) -> Option<Self::Item> {
        let (mut sequence, values, metadata) = self.iter.next()?;
        if self.seq_as_string {
            sequence.mapv_inplace(|x| decode_nucleotide(x).unwrap());
        }
        Some((sequence, values, metadata))
    }
}

impl GenomeDataLoaderAugmentedBFloat16DLPackIter {
    /// Consume enough parent records to form one conventional segment batch.
    ///
    /// ``batch_size`` is measured in output segments, not parent records.  A
    /// batch must be a multiple of ``num_segments`` because every parent is
    /// kept intact and contributes all of its segments.  This keeps the
    /// operation zero-ambiguity and lets the concatenation happen in Rust.
    fn next_batch_records(
        &mut self,
        batch_size: usize,
        drop_last: bool,
    ) -> Result<Option<AugmentedBFloat16Record>> {
        ensure!(batch_size > 0, "batch_size must be positive");
        ensure!(
            batch_size % self.num_segments == 0,
            "batch_size ({batch_size}) must be a multiple of segments_length ({})",
            self.num_segments
        );

        let parents_per_batch = batch_size / self.num_segments;
        let mut records = Vec::with_capacity(parents_per_batch);
        for _ in 0..parents_per_batch {
            let Some(record) = self.next() else {
                break;
            };
            records.push(record);
        }
        if records.is_empty() {
            return Ok(None);
        }

        let actual_segments: usize = records
            .iter()
            .map(|record| record.0.shape()[0])
            .sum();
        if actual_segments < batch_size && drop_last {
            return Ok(None);
        }
        Ok(Some(combine_augmented_bfloat16_records(records)?))
    }
}

#[pymethods]
impl GenomeDataLoaderAugmentedBFloat16DLPackIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'a>(
        mut slf: PyRefMut<'a, Self>,
        py: Python<'a>,
    ) -> PyResult<Option<Bound<'a, pyo3::types::PyTuple>>> {
        let Some((sequence, values, metadata)) = slf.next() else {
            return Ok(None);
        };
        let values = into_dlpack(py, values)?;
        let metadata = metadata.into_pyobject(py).unwrap();
        let result = if slf.seq_as_string {
            let sequence = seq_to_string(&sequence);
            (sequence, values, metadata).into_pyobject(py).unwrap()
        } else {
            let sequence = PyArray2::from_owned_array(py, sequence);
            (sequence, values, metadata).into_pyobject(py).unwrap()
        };
        Ok(Some(result))
    }

    /// Return a Rust-concatenated batch of segments as
    /// ``(sequence_numpy, values_dlpack, metadata)``.
    #[pyo3(signature = (batch_size, drop_last=false))]
    fn next_batch<'a>(
        mut slf: PyRefMut<'a, Self>,
        py: Python<'a>,
        batch_size: usize,
        drop_last: bool,
    ) -> PyResult<Option<Bound<'a, pyo3::types::PyTuple>>> {
        let Some((sequence, values, metadata)) = slf
            .next_batch_records(batch_size, drop_last)
            .map_err(PyErr::from)?
        else {
            return Ok(None);
        };
        let values = into_dlpack(py, values)?;
        let metadata = metadata.into_pyobject(py)?;
        let sequence = PyArray2::from_owned_array(py, sequence);
        Ok(Some((sequence, values, metadata).into_pyobject(py)?))
    }

    /// Wrap this raw iterator with the optional PyTorch tokenizer and batch
    /// adapter.  The low-level Rust API keeps tokenization out of the reader;
    /// this convenience method loads the adapter only when requested.
    #[pyo3(signature = (batch_size, tokenizer=None, drop_last=false))]
    fn dataloader<'a>(
        slf: Py<Self>,
        py: Python<'a>,
        batch_size: usize,
        tokenizer: Option<Py<PyAny>>,
        drop_last: bool,
    ) -> PyResult<Bound<'a, PyAny>> {
        let module = PyModule::import(py, "gdata")?;
        let class = module.getattr("NativeGDataDataLoader")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("batch_size", batch_size)?;
        kwargs.set_item("drop_last", drop_last)?;
        if let Some(tokenizer) = tokenizer {
            kwargs.set_item("tokenizer", tokenizer)?;
        }
        class.call((slf,), Some(&kwargs))
    }
}

impl Iterator for GenomeDataLoaderBFloat16DLPackIter {
    type Item = (Array2<u8>, Array3<bf16>);

    fn next(&mut self) -> Option<Self::Item> {
        let (mut seq, values) = self.iter.next()?;
        if self.seq_as_string {
            seq.mapv_inplace(|x| decode_nucleotide(x).unwrap());
        }
        Some((seq, values))
    }
}

#[pymethods]
impl GenomeDataLoaderBFloat16DLPackIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'a>(
        mut slf: PyRefMut<'a, Self>,
        py: Python<'a>,
    ) -> PyResult<Option<Bound<'a, pyo3::types::PyTuple>>> {
        let Some((seq, values)) = slf.next() else {
            return Ok(None);
        };
        let values = into_dlpack(py, values)?;
        let result = if slf.seq_as_string {
            let seq = seq_to_string(&seq);
            (seq, values).into_pyobject(py).unwrap()
        } else {
            let seq = PyArray2::from_owned_array(py, seq);
            (seq, values).into_pyobject(py).unwrap()
        };
        Ok(Some(result))
    }
}

#[pyclass]
struct DataIndexer(Py<GenomeDataLoader>);

#[pymethods]
impl DataIndexer {
    fn __getitem__<'a>(
        &'a mut self,
        py: Python<'a>,
        key: Bound<'a, PyAny>,
    ) -> Result<(Vec<String>, Bound<'a, PyArray3<f32>>)> {
        let mut loader = self.0.borrow_mut(py);
        let (seq_index, mut data_index): (Bound<'_, PyAny>, Bound<'_, PyAny>) = key.extract()?;
        if !data_index.is_instance_of::<pyo3::types::PyList>() {
            data_index = vec![data_index].into_pyobject(py)?.into_any();
        }

        let (seq, val) = if let Ok(region) = seq_index.extract::<String>() {
            let region = GenomicRange::from_str(&region).unwrap();
            loader.data_store.read(&region).unwrap()
        } else {
            let idx: usize = seq_index.extract()?;
            loader.data_store.read_at(idx).unwrap()
        };

        let idx: Vec<usize> = if let Ok(j_) = data_index.extract::<Vec<String>>() {
            j_.into_iter()
                .map(|x| loader.data_store.data_keys().get_index_of(&x).unwrap())
                .collect()
        } else {
            data_index.extract()?
        };

        let val: Array3<bf16> = val.into();
        let val = val.select(Axis(2), &idx).mapv(|x| x.to_f32());

        Ok((
            seq_to_string(&seq.into()),
            PyArray3::from_owned_array(py, val),
        ))
    }
}

/** A dictionary-like class for loading multiple genomic datasets simultaneously.

    This class allows you to load and iterate over multiple genomic datasets simultaneously,
    each identified by a unique tag. It ensures that all datasets share the same genomic segments,
    which is a requirement for consistent data processing.

    Note
    ----
    The tracks in different loaders may have different resolutions, but the segments must be the same.
    `GenomeDataLoaderMap` ensures that records correspond to the same genomic segments across all datasets,
    but there is no guarantee that the data values returned by different loaders will be aligned, e.,.,
    they may have different `resolution` or `trim_target`.

    Parameters
    ----------
    loaders: dict[str, GenomeDataLoader]
        A dictionary mapping tags to `GenomeDataLoader` instances.
    batch_size : Optional[int]
        Optional parameter to specify the batch size for loading genomic sequences.
        If not provided, it defaults to the minimum batch size across all loaders.
    target_length: Optional[int]
        Optional parameter to specify the target length for all loaders.
        If not provided, each loader will use its own target length.
    window_size : Optional[int]
        Optional parameter to specify the window size for all loaders.
    seq_as_string : bool
        If True, sequences will be returned as strings instead of numpy integer arrays.
        This is useful for cases where you want to work with the sequences as text,
        such as for visualization or text-based analysis.

    See Also
    --------
    GenomeDataLoader
    GenomeDataBuilder
*/
#[pyclass]
#[derive(Debug, Clone)]
pub struct GenomeDataLoaderMap(IndexMap<String, GenomeDataLoader>);

/// Synchronously pair region-preserving native iterators from all heads.  The
/// parent order is supplied by the map, so every modality consumes the same
/// genomic record before the shared center/split plan is applied.
struct MultiRegionBFloat16Iterator {
    iters: IndexMap<String, ParallelLoader<DataStoreBf16ParentRegionIter, RegionBFloat16Record>>,
}

impl Iterator for MultiRegionBFloat16Iterator {
    type Item = (
        GenomicRange,
        u64,
        Array2<u8>,
        IndexMap<String, Array3<bf16>>,
    );

    fn next(&mut self) -> Option<Self::Item> {
        let mut region = None;
        let mut physical_start = None;
        let mut sequence = None;
        let mut values = IndexMap::new();
        for (tag, iter) in self.iters.iter_mut() {
            let (current_region, current_physical_start, current_sequence, current_values) =
                iter.next()?;
            if let Some(expected) = region.as_ref() {
                assert_eq!(
                    expected, &current_region,
                    "all genome data loaders must yield the same parent region"
                );
            } else {
                region = Some(current_region.clone());
            }
            if let Some(expected) = physical_start {
                assert_eq!(
                    expected, current_physical_start,
                    "all genome data loaders must have the same parent origin"
                );
            } else {
                physical_start = Some(current_physical_start);
            }
            if let Some(expected) = sequence.as_ref() {
                assert_eq!(
                    expected, &current_sequence,
                    "all genome data loaders must yield the same DNA sequence"
                );
            } else {
                sequence = Some(current_sequence);
            }
            values.insert(tag.clone(), current_values);
        }
        Some((region?, physical_start?, sequence?, values))
    }
}

type MultiAugmentedBFloat16Record = (
    Array2<u8>,
    IndexMap<String, Array3<bf16>>,
    Vec<AugmentedSegmentMetadata>,
);

/// Concatenate synchronized multi-head parent outputs into one segment batch.
fn combine_multi_augmented_bfloat16_records(
    records: Vec<MultiAugmentedBFloat16Record>,
) -> Result<MultiAugmentedBFloat16Record> {
    ensure!(!records.is_empty(), "cannot combine an empty record list");
    let first = records.first().unwrap();
    let segments_per_parent = first.0.shape()[0];
    let sequence_length = first.0.shape()[1];
    ensure!(segments_per_parent > 0, "native record has no segments");

    let total_segments: usize = records.iter().map(|record| record.0.shape()[0]).sum();
    let mut sequence = Array2::<u8>::zeros((total_segments, sequence_length));
    let mut metadata = Vec::with_capacity(total_segments);
    let mut values = IndexMap::new();

    for (tag, first_values) in &first.1 {
        ensure!(
            first_values.shape()[0] == segments_per_parent,
            "multi-head values do not match the sequence segment count"
        );
        values.insert(
            tag.clone(),
            Array3::<bf16>::from_elem(
                (total_segments, first_values.shape()[1], first_values.shape()[2]),
                bf16::from_f32(0.0),
            ),
        );
    }

    let mut output_start = 0;
    for (current_sequence, current_values, current_metadata) in records {
        let current_segments = current_sequence.shape()[0];
        ensure!(
            current_sequence.shape()[1] == sequence_length,
            "native multi-head batch contains incompatible sequence lengths"
        );
        ensure!(
            current_metadata.len() == current_segments,
            "native multi-head metadata does not match the number of segments"
        );
        let output_end = output_start + current_segments;
        sequence
            .slice_mut(ndarray::s![output_start..output_end, ..])
            .assign(&current_sequence);

        ensure!(
            current_values.len() == values.len(),
            "native multi-head batch contains incompatible head sets"
        );
        for (tag, current_array) in current_values {
            let output_array = values
                .get_mut(&tag)
                .ok_or_else(|| anyhow::anyhow!("native batch is missing head {tag}"))?;
            ensure!(
                current_array.shape()[0] == current_segments
                    && current_array.shape()[1] == output_array.shape()[1]
                    && current_array.shape()[2] == output_array.shape()[2],
                "native multi-head batch contains incompatible shape for {tag}"
            );
            output_array
                .slice_mut(ndarray::s![output_start..output_end, .., ..])
                .assign(&current_array);
        }
        metadata.extend(current_metadata);
        output_start = output_end;
    }

    Ok((sequence, values, metadata))
}

fn center_split_multi_record(
    region: GenomicRange,
    physical_start: u64,
    sequence: Array2<u8>,
    values: IndexMap<String, Array3<bf16>>,
    config: CenterSplitConfig,
    channels_last: bool,
    rng: &mut ChaCha12Rng,
) -> Result<MultiAugmentedBFloat16Record> {
    ensure!(
        sequence.ndim() == 2 && sequence.shape()[0] == 1,
        "native center/split expects a parent batch of one"
    );
    let parent_length = sequence.shape()[1];
    let parent_value_length = parent_length / config.resolution as usize;
    let plan = make_center_split_plan(parent_length, config, rng)?;
    let output_sequence = split_sequence_with_plan(&sequence, config, &plan);
    let metadata = segment_metadata(&region, physical_start, &plan);
    let mut output_values = IndexMap::new();
    for (tag, current_values) in values {
        ensure!(
            current_values.ndim() == 3 && current_values.shape()[0] == 1,
            "native center/split expects values with batch size one"
        );
        let n_tracks = if channels_last {
            ensure!(
                current_values.shape()[1] == parent_value_length,
                "channels-last values do not match sequence length"
            );
            current_values.shape()[2]
        } else {
            ensure!(
                current_values.shape()[2] == parent_value_length,
                "channels-first values do not match sequence length"
            );
            current_values.shape()[1]
        };
        output_values.insert(
            tag,
            split_values_with_plan(&current_values, config, channels_last, &plan, n_tracks),
        );
    }
    Ok((output_sequence, output_values, metadata))
}

impl std::fmt::Display for GenomeDataLoaderMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        writeln!(
            f,
            "MultiGenomeDataLoader with keys: {}",
            self.0.keys().map(|x| "'".to_owned() + x + "'").join(", "),
        )?;
        Ok(())
    }
}

impl GenomeDataLoaderMap {
    pub fn len(&self) -> usize {
        self.0[0].num_batch()
    }

    pub fn iter(&mut self) -> MultiDataLoaderIter {
        let iter = self
            .0
            .iter_mut()
            .map(|(tag, loader)| (tag.clone(), loader.iter()))
            .collect();
        MultiDataLoaderIter(iter)
    }

    pub fn iter_bfloat16_dlpack(&mut self) -> MultiBFloat16DLPackIter {
        self.iter_bfloat16_dlpack_with_layout(true)
    }

    /// Layout-selectable native-bfloat16 iterator for synchronized loaders.
    pub fn iter_bfloat16_dlpack_with_layout(
        &mut self,
        channels_last: bool,
    ) -> MultiBFloat16DLPackIter {
        let iter = self
            .0
            .iter_mut()
            .map(|(tag, loader)| {
                (
                    tag.clone(),
                    loader.iter_bfloat16_dlpack_with_layout(channels_last),
                )
            })
            .collect();
        MultiBFloat16DLPackIter(iter)
    }

    /// Iterate over synchronized heads after one shared center crop, shift,
    /// and equal split plan has been applied to each parent.
    pub fn iter_bfloat16_dlpack_center_split(
        &self,
        channels_last: bool,
        center_length: Option<u32>,
        shift_width: u32,
        num_segments: Option<usize>,
    ) -> Result<MultiAugmentedBFloat16DLPackIter> {
        let first = self
            .0
            .values()
            .next()
            .expect("GenomeDataLoaderMap cannot be empty");
        let config = first.center_split_config(center_length, shift_width, num_segments)?;

        for loader in self.0.values() {
            let other = loader.center_split_config(center_length, shift_width, num_segments)?;
            ensure!(
                other.center_length == config.center_length
                    && other.num_segments == config.num_segments
                    && other.resolution == config.resolution,
                "synchronized loaders have incompatible center/split settings"
            );
        }

        let regions = first.ordered_regions();
        let raw_iters = self
            .0
            .iter()
            .map(|(tag, loader)| {
                (
                    tag.clone(),
                    loader.raw_region_bfloat16_iterator(regions.clone(), channels_last),
                )
            })
            .collect();
        let raw = MultiRegionBFloat16Iterator { iters: raw_iters };
        let mut rng = ChaCha12Rng::seed_from_u64(first.random_seed ^ 0x6a09e667f3bcc909);
        let augmented = raw.map(move |(region, physical_start, sequence, values)| {
            center_split_multi_record(
                region,
                physical_start,
                sequence,
                values,
                config,
                channels_last,
                &mut rng,
            )
            .expect("invalid synchronized center/split augmentation record")
        });
        let prefetch_multiplier = std::env::var("GDATA_PREFETCH_MULTIPLIER")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(2);
        Ok(MultiAugmentedBFloat16DLPackIter {
            iter: PrefethIterator::new(
                augmented,
                first.n_jobs.max(1).saturating_mul(prefetch_multiplier),
            ),
            seq_as_string: first.seq_as_string,
            num_segments: config.num_segments,
        })
    }
}

#[pymethods]
impl GenomeDataLoaderMap {
    #[new]
    #[pyo3(
        signature = (
            loaders, *, batch_size=None, target_length=None, window_size=None, seq_as_string=false,
        ),
        text_signature = "($self, loaders, *, batch_size=None, target_length=None, window_size=None, seq_as_string=False)"
    )]
    pub fn new(
        mut loaders: IndexMap<String, GenomeDataLoader>,
        batch_size: Option<usize>,
        target_length: Option<u32>,
        window_size: Option<u32>,
        seq_as_string: bool,
    ) -> Result<Self> {
        ensure!(
            !loaders.is_empty(),
            "At least one GenomeDataLoader must be provided"
        );
        ensure!(
            loaders.values().map(|loader| loader.segments()).all_equal(),
            "All genome data loaders must have the same segments"
        );

        let batch_size = batch_size.unwrap_or_else(|| {
            loaders
                .values()
                .map(|loader| loader.batch_size)
                .min()
                .unwrap()
        });

        loaders.values_mut().for_each(|loader| {
            loader.seq_as_string = seq_as_string;
            loader.batch_size = batch_size;
            target_length.map(|t| {
                loader.set_target_length(t).unwrap();
            });
            window_size.map(|w| {
                loader.set_window_size(w).unwrap();
            });
        });

        ensure!(
            loaders
                .values()
                .map(|loader| loader.window_size())
                .all_equal(),
            "All genome data loaders must have the same window size",
        );

        Ok(Self(loaders.into_iter().map(|(k, v)| (k, v)).collect()))
    }

    /** Returns a dictionary mapping dataset tags to the number of tracks.

      Returns
      -------
      dict[str, int]
          A dictionary where keys are dataset tags and values are the number of tracks.
    */
    #[getter]
    fn n_tracks(&self) -> IndexMap<String, usize> {
        self.0
            .iter()
            .map(|(k, v)| {
                let n_tracks = v.tracks().len();
                (k.clone(), n_tracks)
            })
            .collect()
    }

    /** Returns the segments of the genome as a vector of strings.

       Returns
       -------
       list[str]
           A list of segment strings representing genomic ranges.
    */
    #[getter]
    fn segments(&self) -> Vec<String> {
        self.0[0].segments()
    }

    /** batch size of the dataloader.
     */
    #[getter]
    fn batch_size(&self) -> usize {
        self.0[0].batch_size
    }

    /** Creates a new genomic data loader based on specified regions.

       This method allows you to subset the genomic data by intersecting it with
       specified regions across all loaders in the map.
       The dataset will be updated to only include the regions that intersect with the provided ones.

       Parameters
       ----------
       regions : list[str]
           A list of genomic ranges in string format (e.g., 'chr1:1000-2000').

       Returns
       -------
       GenomeDataLoaderMap
           A new instance of `GenomeDataLoaderMap` that contains only the data for the specified regions.
    */
    #[pyo3(
        signature = (regions),
        text_signature = "($self, regions)"
    )]
    fn intersection(&self, regions: Vec<String>) -> Result<Self> {
        let result = self
            .0
            .iter()
            .map(|(tag, loader)| {
                let new_loader = loader.intersection_py(regions.clone());
                Ok((tag.clone(), new_loader))
            })
            .collect::<Result<IndexMap<_, _>>>();
        Ok(Self(result?))
    }

    /** Creating a new genomic data loader in which the regions differ from the specified ones.

        This method allows you to subset the genomic data by removing the specified regions.

        Parameters
        ----------
        regions : list[str]
            A list of genomic ranges in string format (e.g., 'chr1:1000-2000').

        See Also
        --------
        intersection : For creating a loader with intersecting regions.

        Returns
        -------
        GenomeDataLoader
            A new instance of `GenomeDataLoader` that contains only the data for the regions
            that do not intersect with the specified ones.
    */
    #[pyo3(
        signature = (regions),
        text_signature = "($self, regions)"
    )]
    fn difference(&self, regions: Vec<String>) -> Result<Self> {
        let result = self
            .0
            .iter()
            .map(|(tag, loader)| {
                let new_loader = loader.difference_py(regions.clone());
                Ok((tag.clone(), new_loader))
            })
            .collect::<Result<IndexMap<_, _>>>();
        Ok(Self(result?))
    }

    /** Returns the keys of the dataloader.

       Returns
       -------
       list[str]
           A list of keys representing the genomic segments in the dataset.
    */
    fn keys(&self) -> Vec<String> {
        self.0.keys().cloned().collect()
    }

    fn __len__(&self) -> usize {
        self.len()
    }

    fn __iter__(mut slf: PyRefMut<'_, Self>) -> MultiDataLoaderIter {
        slf.iter()
    }

    /// Python-facing multi-head DLPack iterator.
    #[pyo3(name = "iter_bfloat16_dlpack")]
    #[pyo3(signature = (channels_last=true))]
    fn iter_bfloat16_dlpack_py(
        mut slf: PyRefMut<'_, Self>,
        channels_last: bool,
    ) -> MultiBFloat16DLPackIter {
        slf.iter_bfloat16_dlpack_with_layout(channels_last)
    }

    /// Python-facing synchronized center/crop/split iterator.
    #[pyo3(name = "iter_bfloat16_dlpack_center_split")]
    #[pyo3(signature = (channels_last=false, mid=None, shift=0, segments_length=None))]
    fn iter_bfloat16_dlpack_center_split_py(
        slf: PyRef<'_, Self>,
        channels_last: bool,
        mid: Option<u32>,
        shift: u32,
        segments_length: Option<usize>,
    ) -> PyResult<MultiAugmentedBFloat16DLPackIter> {
        slf.iter_bfloat16_dlpack_center_split(channels_last, mid, shift, segments_length)
            .map_err(PyErr::from)
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }
}

#[pyclass]
pub struct MultiDataLoaderIter(IndexMap<String, GenomeDataLoaderIter>);

impl Iterator for MultiDataLoaderIter {
    type Item = (Array2<u8>, IndexMap<String, Array3<f32>>);

    fn next(&mut self) -> Option<Self::Item> {
        let mut seqs = None;
        let data: Option<_> = self
            .0
            .iter_mut()
            .map(|(tag, iter)| {
                let (s, d) = iter.next()?;
                if let Some(s_) = seqs.as_ref() {
                    assert_eq!(s_, &s, "All sequences must be the same");
                } else {
                    seqs = Some(s);
                }
                Some((tag.clone(), d))
            })
            .collect();

        Some((seqs?, data?))
    }
}

#[pymethods]
impl MultiDataLoaderIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'a>(
        mut slf: PyRefMut<'a, Self>,
        py: Python<'a>,
    ) -> Option<Bound<'a, pyo3::types::PyTuple>> {
        let (seq, values) = slf.next()?;

        let values = values
            .into_iter()
            .map(|(tag, v)| (tag, PyArray3::from_owned_array(py, v)))
            .collect::<IndexMap<_, _>>();

        let result = if slf.0[0].seq_as_string {
            let seq = seq_to_string(&seq);
            (seq, values).into_pyobject(py).unwrap()
        } else {
            let seq = PyArray2::from_owned_array(py, seq);
            (seq, values).into_pyobject(py).unwrap()
        };
        Some(result)
    }
}

/// Synchronized multi-head iterator for the native bfloat16/DLPack path.
#[pyclass]
pub struct MultiBFloat16DLPackIter(IndexMap<String, GenomeDataLoaderBFloat16DLPackIter>);

impl Iterator for MultiBFloat16DLPackIter {
    type Item = (Array2<u8>, IndexMap<String, Array3<bf16>>);

    fn next(&mut self) -> Option<Self::Item> {
        let mut seqs = None;
        let data: Option<_> = self
            .0
            .iter_mut()
            .map(|(tag, iter)| {
                let (s, d) = iter.next()?;
                if let Some(s_) = seqs.as_ref() {
                    assert_eq!(s_, &s, "All sequences must be the same");
                } else {
                    seqs = Some(s);
                }
                Some((tag.clone(), d))
            })
            .collect();

        Some((seqs?, data?))
    }
}

#[pymethods]
impl MultiBFloat16DLPackIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'a>(
        mut slf: PyRefMut<'a, Self>,
        py: Python<'a>,
    ) -> PyResult<Option<Bound<'a, pyo3::types::PyTuple>>> {
        let Some((seq, values)) = slf.next() else {
            return Ok(None);
        };

        let values = values
            .into_iter()
            .map(|(tag, value)| Ok((tag, into_dlpack(py, value)?)))
            .collect::<PyResult<IndexMap<_, _>>>()?;

        let result = if slf.0[0].seq_as_string {
            let seq = seq_to_string(&seq);
            (seq, values).into_pyobject(py).unwrap()
        } else {
            let seq = PyArray2::from_owned_array(py, seq);
            (seq, values).into_pyobject(py).unwrap()
        };
        Ok(Some(result))
    }
}

/// Synchronized multi-head iterator for native center-cropped/split records.
#[pyclass]
pub struct MultiAugmentedBFloat16DLPackIter {
    iter: PrefethIterator<MultiAugmentedBFloat16Record>,
    seq_as_string: bool,
    num_segments: usize,
}

impl Iterator for MultiAugmentedBFloat16DLPackIter {
    type Item = MultiAugmentedBFloat16Record;

    fn next(&mut self) -> Option<Self::Item> {
        let (mut sequence, values, metadata) = self.iter.next()?;
        if self.seq_as_string {
            sequence.mapv_inplace(|x| decode_nucleotide(x).unwrap());
        }
        Some((sequence, values, metadata))
    }
}

impl MultiAugmentedBFloat16DLPackIter {
    fn next_batch_records(
        &mut self,
        batch_size: usize,
        drop_last: bool,
    ) -> Result<Option<MultiAugmentedBFloat16Record>> {
        ensure!(batch_size > 0, "batch_size must be positive");
        ensure!(
            batch_size % self.num_segments == 0,
            "batch_size ({batch_size}) must be a multiple of segments_length ({})",
            self.num_segments
        );

        let parents_per_batch = batch_size / self.num_segments;
        let mut records = Vec::with_capacity(parents_per_batch);
        for _ in 0..parents_per_batch {
            let Some(record) = self.next() else {
                break;
            };
            records.push(record);
        }
        if records.is_empty() {
            return Ok(None);
        }

        let actual_segments: usize = records
            .iter()
            .map(|record| record.0.shape()[0])
            .sum();
        if actual_segments < batch_size && drop_last {
            return Ok(None);
        }
        Ok(Some(combine_multi_augmented_bfloat16_records(records)?))
    }
}

#[pymethods]
impl MultiAugmentedBFloat16DLPackIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'a>(
        mut slf: PyRefMut<'a, Self>,
        py: Python<'a>,
    ) -> PyResult<Option<Bound<'a, pyo3::types::PyTuple>>> {
        let Some((sequence, values, metadata)) = slf.next() else {
            return Ok(None);
        };

        let values = values
            .into_iter()
            .map(|(tag, value)| Ok((tag, into_dlpack(py, value)?)))
            .collect::<PyResult<IndexMap<_, _>>>()?;
        let metadata = metadata.into_pyobject(py)?;
        let result = if slf.seq_as_string {
            let sequence = seq_to_string(&sequence);
            (sequence, values, metadata).into_pyobject(py)?
        } else {
            let sequence = PyArray2::from_owned_array(py, sequence);
            (sequence, values, metadata).into_pyobject(py)?
        };
        Ok(Some(result))
    }

    /// Return a Rust-concatenated synchronized multi-head batch.
    #[pyo3(signature = (batch_size, drop_last=false))]
    fn next_batch<'a>(
        mut slf: PyRefMut<'a, Self>,
        py: Python<'a>,
        batch_size: usize,
        drop_last: bool,
    ) -> PyResult<Option<Bound<'a, pyo3::types::PyTuple>>> {
        let Some((sequence, values, metadata)) = slf
            .next_batch_records(batch_size, drop_last)
            .map_err(PyErr::from)?
        else {
            return Ok(None);
        };

        let values = values
            .into_iter()
            .map(|(tag, value)| Ok((tag, into_dlpack(py, value)?)))
            .collect::<PyResult<IndexMap<_, _>>>()?;
        let metadata = metadata.into_pyobject(py)?;
        let sequence = PyArray2::from_owned_array(py, sequence);
        Ok(Some((sequence, values, metadata).into_pyobject(py)?))
    }

    /// Wrap this raw iterator with the package-level optional PyTorch
    /// tokenizer and batch adapter.
    #[pyo3(signature = (batch_size, tokenizer=None, drop_last=false))]
    fn dataloader<'a>(
        slf: Py<Self>,
        py: Python<'a>,
        batch_size: usize,
        tokenizer: Option<Py<PyAny>>,
        drop_last: bool,
    ) -> PyResult<Bound<'a, PyAny>> {
        let module = PyModule::import(py, "gdata")?;
        let class = module.getattr("NativeGDataDataLoader")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("batch_size", batch_size)?;
        kwargs.set_item("drop_last", drop_last)?;
        if let Some(tokenizer) = tokenizer {
            kwargs.set_item("tokenizer", tokenizer)?;
        }
        class.call((slf,), Some(&kwargs))
    }
}

/** This class combines multiple `GenomeDataLoaderMap` instances into a single loader.

    It allows for the simultaneous loading of genomic data from multiple species.
    The resulting loader will iterate over all datasets in an alternating fashion.

    Parameters
    ----------
    loaders: list[GenomeDataLoaderMap]
        A list of `GenomeDataLoaderMap` instances to combine.
    batch_size : Optional[int]
        Optional parameter to specify the batch size for loading genomic sequences.
    shuffle : Optional[bool]
        Optional parameter to specify whether to shuffle the data across all loaders.
*/
#[pyclass]
#[derive(Debug, Clone)]
pub struct CatGenomeDataLoader(Vec<GenomeDataLoaderMap>);

impl CatGenomeDataLoader {
    pub fn len(&self) -> usize {
        self.0.iter().map(|loader| loader.len()).sum()
    }

    pub fn iter(&mut self) -> MultiGenomeIter {
        let iters = self
            .0
            .iter_mut()
            .map(|loader| loader.iter())
            .collect::<Vec<_>>();
        MultiGenomeIter { iters, pos: 0 }
    }
}

#[pymethods]
impl CatGenomeDataLoader {
    #[new]
    #[pyo3(
        signature = (loaders, *, batch_size=None, shuffle=None),
        text_signature = "($self, loaders, *, batch_size=None, shuffle=None)"
    )]
    pub fn new(
        mut loaders: Vec<GenomeDataLoaderMap>,
        batch_size: Option<usize>,
        shuffle: Option<bool>,
    ) -> Result<Self> {
        ensure!(
            !loaders.is_empty(),
            "At least one GenomeDataLoaderMap must be provided"
        );

        if let Some(bs) = batch_size {
            loaders.iter_mut().for_each(|loader| {
                loader.0.values_mut().for_each(|l| l.batch_size = bs);
            });
        }

        if let Some(shuffle) = shuffle {
            loaders.iter_mut().for_each(|loader| {
                loader.0.values_mut().for_each(|l| l.shuffle = shuffle);
            });
        }

        Ok(Self(loaders))
    }

    /** Returns a dictionary mapping dataset tags to the number of tracks.

      Returns
      -------
      dict[str, int]
          A dictionary where keys are dataset tags and values are the number of tracks.
    */
    #[getter]
    fn n_tracks(&self) -> IndexMap<String, usize> {
        IndexMap::from_iter(
            self.0
                .iter()
                .flat_map(|loader| loader.n_tracks().into_iter()),
        )
    }

    fn __len__(&self) -> usize {
        self.len()
    }

    fn __iter__(mut slf: PyRefMut<'_, Self>) -> MultiGenomeIter {
        slf.iter()
    }
}

#[pyclass]
pub struct MultiGenomeIter {
    iters: Vec<MultiDataLoaderIter>,
    pos: usize,
}

impl Iterator for MultiGenomeIter {
    type Item = (Array2<u8>, IndexMap<String, Array3<f32>>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.iters.is_empty() {
            return None;
        }

        self.pos %= self.iters.len();
        if let Some(item) = self.iters[self.pos].next() {
            self.pos += 1;
            Some(item)
        } else {
            self.iters.remove(self.pos);
            self.next()
        }
    }
}

#[pymethods]
impl MultiGenomeIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'a>(
        mut slf: PyRefMut<'a, Self>,
        py: Python<'a>,
    ) -> Option<Bound<'a, pyo3::types::PyTuple>> {
        let (seq, values) = slf.next()?;
        let values = values
            .into_iter()
            .map(|(tag, v)| (tag, PyArray3::from_owned_array(py, v)))
            .collect::<IndexMap<_, _>>();

        let result = if slf.iters[0].0[0].seq_as_string {
            let seq = seq_to_string(&seq);
            (seq, values).into_pyobject(py).unwrap()
        } else {
            let seq = PyArray2::from_owned_array(py, seq);
            (seq, values).into_pyobject(py).unwrap()
        };
        Some(result)
    }
}

fn extract_string_list(str: Bound<'_, PyAny>) -> Result<Vec<String>> {
    if let Ok(s) = str.extract::<String>() {
        Ok(vec![s])
    } else {
        str.extract::<Vec<String>>()
            .with_context(|| "Failed to extract string list from PyAny")
    }
}

pub(crate) fn seq_to_string(seq: &Array2<u8>) -> Vec<String> {
    seq.rows()
        .into_iter()
        .map(|row| String::from_utf8(row.to_vec()).unwrap())
        .collect()
}
