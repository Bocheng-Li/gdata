use anyhow::{bail, ensure, Context, Result};
use bed_utils::bed::{BEDLike, GenomicRange};
use bincode::{Decode, Encode};
use half::bf16;
use indexmap::{IndexMap, IndexSet};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut3, Axis};
use noodles::core::Position;
use noodles::fasta::io::IndexedReader;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{Read, Seek, Write};
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::dataloader::generic::{split_n_with_batch_size, ParallelLoader, ReBatch};
use crate::w5z::W5Z;

use super::super::generic::{compress_data_zst, decompress_data_zst};

static PROFILE_ENABLED: AtomicBool = AtomicBool::new(false);
static PROFILE_METADATA_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_METADATA_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_FILE_READ_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_FILE_READ_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_ZSTD_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_ZSTD_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_BINCODE_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_BINCODE_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_POSTPROCESS_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_POSTPROCESS_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_VALUE_CONVERT_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_VALUE_CONVERT_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_SEQ_SHAPE_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_SEQ_SHAPE_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_VALUES_TRANSPOSE_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_VALUES_TRANSPOSE_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_SEQ_CROP_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_SEQ_CROP_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_VALUES_CROP_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_VALUES_CROP_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_AGGREGATE_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_AGGREGATE_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_SPLIT_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_SPLIT_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_TRIM_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_TRIM_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_TRANSFORM_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_TRANSFORM_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_FINAL_SEQ_LAYOUT_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_FINAL_SEQ_LAYOUT_COUNT: AtomicU64 = AtomicU64::new(0);
static PROFILE_FINAL_VALUES_LAYOUT_NS: AtomicU64 = AtomicU64::new(0);
static PROFILE_FINAL_VALUES_LAYOUT_COUNT: AtomicU64 = AtomicU64::new(0);

fn profile_start() -> Option<Instant> {
    PROFILE_ENABLED.load(Ordering::Relaxed).then(Instant::now)
}

fn profile_record(total: &AtomicU64, count: &AtomicU64, start: Option<Instant>) {
    if let Some(start) = start {
        total.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        count.fetch_add(1, Ordering::Relaxed);
    }
}

pub(crate) fn profile_reset() {
    PROFILE_METADATA_NS.store(0, Ordering::Relaxed);
    PROFILE_METADATA_COUNT.store(0, Ordering::Relaxed);
    PROFILE_FILE_READ_NS.store(0, Ordering::Relaxed);
    PROFILE_FILE_READ_COUNT.store(0, Ordering::Relaxed);
    PROFILE_ZSTD_NS.store(0, Ordering::Relaxed);
    PROFILE_ZSTD_COUNT.store(0, Ordering::Relaxed);
    PROFILE_BINCODE_NS.store(0, Ordering::Relaxed);
    PROFILE_BINCODE_COUNT.store(0, Ordering::Relaxed);
    PROFILE_POSTPROCESS_NS.store(0, Ordering::Relaxed);
    PROFILE_POSTPROCESS_COUNT.store(0, Ordering::Relaxed);
    PROFILE_VALUE_CONVERT_NS.store(0, Ordering::Relaxed);
    PROFILE_VALUE_CONVERT_COUNT.store(0, Ordering::Relaxed);
    PROFILE_SEQ_SHAPE_NS.store(0, Ordering::Relaxed);
    PROFILE_SEQ_SHAPE_COUNT.store(0, Ordering::Relaxed);
    PROFILE_VALUES_TRANSPOSE_NS.store(0, Ordering::Relaxed);
    PROFILE_VALUES_TRANSPOSE_COUNT.store(0, Ordering::Relaxed);
    PROFILE_SEQ_CROP_NS.store(0, Ordering::Relaxed);
    PROFILE_SEQ_CROP_COUNT.store(0, Ordering::Relaxed);
    PROFILE_VALUES_CROP_NS.store(0, Ordering::Relaxed);
    PROFILE_VALUES_CROP_COUNT.store(0, Ordering::Relaxed);
    PROFILE_AGGREGATE_NS.store(0, Ordering::Relaxed);
    PROFILE_AGGREGATE_COUNT.store(0, Ordering::Relaxed);
    PROFILE_SPLIT_NS.store(0, Ordering::Relaxed);
    PROFILE_SPLIT_COUNT.store(0, Ordering::Relaxed);
    PROFILE_TRIM_NS.store(0, Ordering::Relaxed);
    PROFILE_TRIM_COUNT.store(0, Ordering::Relaxed);
    PROFILE_TRANSFORM_NS.store(0, Ordering::Relaxed);
    PROFILE_TRANSFORM_COUNT.store(0, Ordering::Relaxed);
    PROFILE_FINAL_SEQ_LAYOUT_NS.store(0, Ordering::Relaxed);
    PROFILE_FINAL_SEQ_LAYOUT_COUNT.store(0, Ordering::Relaxed);
    PROFILE_FINAL_VALUES_LAYOUT_NS.store(0, Ordering::Relaxed);
    PROFILE_FINAL_VALUES_LAYOUT_COUNT.store(0, Ordering::Relaxed);
    PROFILE_ENABLED.store(true, Ordering::Relaxed);
}

pub(crate) fn profile_snapshot() -> Vec<(String, f64, u64)> {
    [
        (
            "metadata_read_s",
            &PROFILE_METADATA_NS,
            &PROFILE_METADATA_COUNT,
        ),
        (
            "file_read_s",
            &PROFILE_FILE_READ_NS,
            &PROFILE_FILE_READ_COUNT,
        ),
        ("zstd_decompress_s", &PROFILE_ZSTD_NS, &PROFILE_ZSTD_COUNT),
        (
            "bincode_decode_s",
            &PROFILE_BINCODE_NS,
            &PROFILE_BINCODE_COUNT,
        ),
        (
            "rust_postprocess_s",
            &PROFILE_POSTPROCESS_NS,
            &PROFILE_POSTPROCESS_COUNT,
        ),
        (
            "rust_bf16_to_f32_s",
            &PROFILE_VALUE_CONVERT_NS,
            &PROFILE_VALUE_CONVERT_COUNT,
        ),
        (
            "rust_seq_shape_s",
            &PROFILE_SEQ_SHAPE_NS,
            &PROFILE_SEQ_SHAPE_COUNT,
        ),
        (
            "rust_values_transpose_s",
            &PROFILE_VALUES_TRANSPOSE_NS,
            &PROFILE_VALUES_TRANSPOSE_COUNT,
        ),
        (
            "rust_seq_crop_s",
            &PROFILE_SEQ_CROP_NS,
            &PROFILE_SEQ_CROP_COUNT,
        ),
        (
            "rust_values_crop_copy_s",
            &PROFILE_VALUES_CROP_NS,
            &PROFILE_VALUES_CROP_COUNT,
        ),
        (
            "rust_aggregation_s",
            &PROFILE_AGGREGATE_NS,
            &PROFILE_AGGREGATE_COUNT,
        ),
        ("rust_split_s", &PROFILE_SPLIT_NS, &PROFILE_SPLIT_COUNT),
        ("rust_trim_s", &PROFILE_TRIM_NS, &PROFILE_TRIM_COUNT),
        (
            "rust_transform_s",
            &PROFILE_TRANSFORM_NS,
            &PROFILE_TRANSFORM_COUNT,
        ),
        (
            "rust_final_seq_layout_s",
            &PROFILE_FINAL_SEQ_LAYOUT_NS,
            &PROFILE_FINAL_SEQ_LAYOUT_COUNT,
        ),
        (
            "rust_final_values_layout_s",
            &PROFILE_FINAL_VALUES_LAYOUT_NS,
            &PROFILE_FINAL_VALUES_LAYOUT_COUNT,
        ),
    ]
    .into_iter()
    .map(|(name, total, count)| {
        (
            name.to_string(),
            total.load(Ordering::Relaxed) as f64 / 1e9,
            count.load(Ordering::Relaxed),
        )
    })
    .collect()
}

/// Dimension: (sequence, experiment)
#[derive(Debug, Clone, PartialEq)]
pub struct Values(Array3<bf16>);

impl From<Array3<bf16>> for Values {
    fn from(arr: Array3<bf16>) -> Self {
        // `read_bf16` already returns a standard-layout owned array.  Taking
        // ownership here is important: calling `as_standard_layout()` on an
        // owned array through a view would make a second, very large copy of
        // every parent record.
        Values(arr)
    }
}

impl From<ArrayView3<'_, bf16>> for Values {
    fn from(arr: ArrayView3<'_, bf16>) -> Self {
        Values(arr.as_standard_layout().to_owned())
    }
}

impl Into<Array3<bf16>> for Values {
    fn into(self) -> Array3<bf16> {
        self.0
    }
}

impl Into<Array3<f32>> for Values {
    fn into(self) -> Array3<f32> {
        self.0.mapv(|x| x.to_f32())
    }
}

#[derive(Debug, PartialEq)]
pub struct Sequence(Array2<u8>);

impl From<Array2<u8>> for Sequence {
    fn from(seq: Array2<u8>) -> Self {
        Sequence(seq.as_standard_layout().to_owned())
    }
}

impl From<ArrayView2<'_, u8>> for Sequence {
    fn from(arr: ArrayView2<'_, u8>) -> Self {
        Sequence(arr.as_standard_layout().to_owned())
    }
}

impl Into<Array2<u8>> for Sequence {
    fn into(self) -> Array2<u8> {
        self.0
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Offset((u64, u64));

#[derive(Debug, Decode, Encode)]
pub struct StoreMetadata {
    #[bincode(with_serde)]
    segment_index: IndexMap<GenomicRange, Offset>,
    #[bincode(with_serde)]
    data_keys: IndexSet<String>,
    sequence_length: u32,
    resolution: u32,
    padding: u32,
}

impl StoreMetadata {
    pub fn encode(self) -> Result<Vec<u8>> {
        let data = bincode::encode_to_vec(self, bincode::config::standard())?;
        let data = compress_data_zst(data, 9);
        Ok(data)
    }

    pub fn decode(buffer: &[u8]) -> Result<Self> {
        let mut data = decompress_data_zst(buffer);
        Ok(bincode::decode_from_slice(&mut data, bincode::config::standard())?.0)
    }
}

/// Options for reading from the data store.
#[derive(Debug, Clone)]
pub struct DataStoreReadOptions {
    /// Random shift to apply to the sequence from -shift_width to +shift_width.
    pub shift_width: u32,
    /// Length of the values to read.
    pub value_length: Option<u32>,
    /// Size of the split values.
    pub split_size: Option<u32>,
    /// Resolution to read the data at.
    pub read_resolution: Option<u32>,
    /// Scale value to apply to the values.
    pub scale_value: Option<bf16>,
    /// Maximum value to clamp the values to.
    pub clamp_value_max: Option<bf16>,
    /// Random number generator for shuffling.
    pub rng: ChaCha12Rng,
}

impl Default for DataStoreReadOptions {
    fn default() -> Self {
        Self {
            shift_width: 0,
            value_length: None,
            split_size: None,
            read_resolution: None,
            scale_value: None,
            clamp_value_max: None,
            rng: ChaCha12Rng::seed_from_u64(0),
        }
    }
}

#[derive(Debug)]
struct InnerStore {
    file: std::fs::File,
    metadata: StoreMetadata,
}

#[derive(Debug, Clone)]
pub struct DataStore {
    inner: Arc<InnerStore>,
    pub out_resolution: u32,
    pub read_opts: DataStoreReadOptions,
    aggregate_size: Option<u32>,
}

impl DataStore {
    pub fn open(path: impl AsRef<Path>, mut read_opts: DataStoreReadOptions) -> Result<Self> {
        let mut file = File::open(&path).context("Failed to open data store file")?;
        let metadata_start = profile_start();
        let metadata = read_metadata(&mut file)?;
        profile_record(
            &PROFILE_METADATA_NS,
            &PROFILE_METADATA_COUNT,
            metadata_start,
        );

        ensure!(
            read_opts.shift_width <= metadata.padding,
            "Shift must be less than or equal to padding"
        );

        let mut aggregate_size = None;
        let mut out_resolution = metadata.resolution;
        if let Some(read_resolution) = read_opts.read_resolution {
            ensure!(
                read_resolution % metadata.resolution == 0,
                "Read resolution must be a multiple of resolution"
            );
            let aggregation_factor = read_resolution / metadata.resolution;
            // Requesting the resolution at which the data is already stored
            // is an identity operation.  In particular, avoid routing
            // factor=1 through `aggregate_by_length`, which would otherwise
            // perform a full bf16 -> f64 -> mean -> bf16 pass over the array.
            if aggregation_factor > 1 {
                aggregate_size = Some(aggregation_factor);
            }
            out_resolution = read_resolution;
        }

        if let Some(v_len) = read_opts.value_length {
            ensure!(
                v_len % metadata.resolution == 0,
                "Value length must be a multiple of resolution"
            );
            ensure!(
                v_len <= read_opts.split_size.unwrap_or(metadata.sequence_length),
                "Value length must be less than or equal to sequence length"
            );
        }

        if let Some(s) = read_opts.split_size {
            ensure!(
                s % metadata.resolution == 0,
                "Split size must be a multiple of resolution"
            );

            if s == metadata.sequence_length {
                read_opts.split_size = None;
            }
        }

        let inner = Arc::new(InnerStore { file, metadata });

        Ok(Self {
            inner,
            out_resolution,
            aggregate_size,
            read_opts,
        })
    }

    pub fn segments(&self) -> impl Iterator<Item = &GenomicRange> {
        self.inner.metadata.segment_index.keys()
    }

    pub fn num_segments(&self) -> usize {
        self.inner.metadata.segment_index.len()
    }

    pub fn data_keys(&self) -> &IndexSet<String> {
        &self.inner.metadata.data_keys
    }

    pub fn sequence_length(&self) -> u32 {
        self.inner.metadata.sequence_length
    }

    pub fn output_length(&self) -> u32 {
        self.read_opts.split_size.unwrap_or(self.sequence_length())
    }

    pub fn n_pad(&self) -> u32 {
        self.inner.metadata.padding
    }

    pub fn resolution(&self) -> u32 {
        self.inner.metadata.resolution
    }

    /// Whether the requested reader resolution requires an aggregation pass.
    /// The native center/split path needs the stored resolution so that its
    /// base-pair crop can be shared exactly across modalities.
    pub(crate) fn has_aggregation(&self) -> bool {
        self.aggregate_size.is_some()
    }

    pub fn set_value_length(&mut self, value_length: u32) -> Result<()> {
        ensure!(
            value_length % self.out_resolution == 0,
            "Value length must be a multiple of resolution"
        );
        ensure!(
            value_length <= self.read_opts.split_size.unwrap_or(self.sequence_length()),
            "Value length must be less than or equal to sequence length"
        );

        self.read_opts.value_length = Some(value_length);
        Ok(())
    }

    pub fn set_split_size(&mut self, split_size: u32) -> Result<()> {
        ensure!(
            split_size % self.out_resolution == 0,
            "Split size must be a multiple of resolution"
        );

        if split_size == self.sequence_length() {
            self.read_opts.split_size = None;
        } else {
            self.read_opts.split_size = Some(split_size);
        }
        Ok(())
    }

    /// Read a region and keep the values in their on-disk bfloat16 type.
    ///
    /// The historical/default layout is `(batch, sequence, track)`.
    ///
    /// The returned array owns its allocation and is standard-layout with
    /// shape `(batch, sequence, track)`.  This is the zero-copy hand-off point
    /// used by the DLPack iterator.  In particular, do not turn this array
    /// into a view and then call `as_standard_layout().to_owned()` again: for
    /// the AlphaGenome-sized records that would copy hundreds of MiB per
    /// parent.
    pub fn read_bf16(&mut self, region: &GenomicRange) -> Option<(Sequence, Array3<bf16>)> {
        self.read_bf16_with_layout(region, true)
    }

    /// Read a region while choosing the returned values layout.
    ///
    /// `channels_last=true` returns `(batch, sequence, track)` and preserves
    /// the historical gdata/DLPack API.  `channels_last=false` returns
    /// `(batch, track, sequence)`.  The latter matches Conv1d-based model
    /// heads and can reuse the decoded on-disk `(track, sequence)` allocation
    /// directly when no sequence crop/aggregation/trim is requested.
    pub fn read_bf16_with_layout(
        &mut self,
        region: &GenomicRange,
        channels_last: bool,
    ) -> Option<(Sequence, Array3<bf16>)> {
        let offset = self.inner.metadata.segment_index.get(region)?;
        let mut buffer = vec![0; offset.0 .1 as usize];
        let file_read_start = profile_start();
        self.inner
            .file
            .read_exact_at(&mut buffer, offset.0 .0 as u64)
            .expect("read failed");
        profile_record(
            &PROFILE_FILE_READ_NS,
            &PROFILE_FILE_READ_COUNT,
            file_read_start,
        );

        // Deserialize the sequence and values
        let zstd_start = profile_start();
        let buffer = decompress_data_zst(&buffer);
        profile_record(&PROFILE_ZSTD_NS, &PROFILE_ZSTD_COUNT, zstd_start);
        let bincode_start = profile_start();
        let (seq, arr): (Vec<u8>, Array2<bf16>) =
            bincode::serde::decode_from_slice(&buffer, bincode::config::standard())
                .expect("decode failed")
                .0;
        profile_record(&PROFILE_BINCODE_NS, &PROFILE_BINCODE_COUNT, bincode_start);

        let postprocess_start = profile_start();
        let seq_shape_start = profile_start();
        let seq = Array1::from_vec(seq).insert_axis(Axis(0));
        profile_record(
            &PROFILE_SEQ_SHAPE_NS,
            &PROFILE_SEQ_SHAPE_COUNT,
            seq_shape_start,
        );
        // Apply random shifting
        let mut shift = self.read_opts.shift_width as i32;
        let res = self.resolution();
        if self.read_opts.shift_width != 0 {
            shift = self.read_opts.rng.random_range(-shift..=shift) / res as i32 * res as i32;
        }
        let seq_start = (shift + self.n_pad() as i32) as usize;
        let seq_end = seq_start + self.sequence_length() as usize;
        let arr_start = seq_start / res as usize;
        let arr_end = seq_end / res as usize;
        let seq_crop_start = profile_start();
        let mut seq = seq.slice(s![.., seq_start..seq_end]);
        profile_record(
            &PROFILE_SEQ_CROP_NS,
            &PROFILE_SEQ_CROP_COUNT,
            seq_crop_start,
        );

        let (n_tracks, n_values) = arr.dim();
        let values_crop_start = profile_start();
        // The decoded on-disk array is contiguous `(track, sequence)`.
        //
        // In channels-last mode we must transpose and materialise a new
        // `(batch, sequence, track)` array.  In channels-first mode we can
        // reshape the owned allocation to `(batch, track, sequence)` without
        // copying.  When the requested range is the complete parent, retain
        // that allocation instead of doing a full values copy.
        let mut arr = if channels_last {
            let values_transpose_start = profile_start();
            let arr_view = arr.view();
            let transposed = arr_view.t().insert_axis(Axis(0));
            profile_record(
                &PROFILE_VALUES_TRANSPOSE_NS,
                &PROFILE_VALUES_TRANSPOSE_COUNT,
                values_transpose_start,
            );
            transposed
                .slice(s![.., arr_start..arr_end, ..])
                .as_standard_layout()
                .to_owned()
        } else {
            let mut arr = arr
                .into_shape_with_order((1, n_tracks, n_values))
                .expect("decoded values must be contiguous");
            if arr_start != 0 || arr_end != n_values {
                arr = arr.slice(s![.., .., arr_start..arr_end]).to_owned();
            }
            arr
        };
        profile_record(
            &PROFILE_VALUES_CROP_NS,
            &PROFILE_VALUES_CROP_COUNT,
            values_crop_start,
        );

        // Apply value aggregation
        if let Some(agg) = self.aggregate_size {
            let aggregate_start = profile_start();
            arr = aggregate_by_length_layout(arr, agg as usize, channels_last);
            profile_record(
                &PROFILE_AGGREGATE_NS,
                &PROFILE_AGGREGATE_COUNT,
                aggregate_start,
            );
        }

        // Apply splitting
        if let Some(split) = self.read_opts.split_size {
            let split_start = profile_start();
            seq = split_sequence(seq, split as usize).unwrap();
            arr = split_data_layout(arr, (split / self.out_resolution) as usize, channels_last)
                .unwrap();
            profile_record(&PROFILE_SPLIT_NS, &PROFILE_SPLIT_COUNT, split_start);
        }

        // Trim the output values.  Keep an owned array whenever the requested
        // range is the complete array.  The previous implementation always
        // created a mutable slice and then copied that view into `Values`,
        // even when no trimming was requested.
        let trim_start_time = profile_start();
        let v_len = self.read_opts.value_length.unwrap_or(self.output_length());
        let trim_start = (self.output_length() - v_len) / self.out_resolution / 2;
        let trim_end = trim_start + (v_len / self.out_resolution);
        let arr_len = if channels_last {
            arr.shape()[1]
        } else {
            arr.shape()[2]
        };
        let trim_start = trim_start as usize;
        let trim_end = trim_end as usize;
        let mut arr = if trim_start == 0 && trim_end == arr_len {
            arr
        } else if channels_last {
            arr.slice(s![.., trim_start..trim_end, ..])
                .as_standard_layout()
                .to_owned()
        } else {
            arr.slice(s![.., .., trim_start..trim_end]).to_owned()
        };
        profile_record(&PROFILE_TRIM_NS, &PROFILE_TRIM_COUNT, trim_start_time);

        // Scaling, clamping, and NaN replacement are one optional transform
        // pass.  Do not scan every value when neither scale nor clamp was
        // requested.  NaN replacement is intentionally part of this same
        // opt-in path rather than an unconditional default operation.
        if self.read_opts.scale_value.is_some() || self.read_opts.clamp_value_max.is_some() {
            let transform_start = profile_start();
            transform(
                arr.view_mut(),
                self.read_opts.scale_value,
                self.read_opts.clamp_value_max,
            );
            profile_record(
                &PROFILE_TRANSFORM_NS,
                &PROFILE_TRANSFORM_COUNT,
                transform_start,
            );
        }

        profile_record(
            &PROFILE_POSTPROCESS_NS,
            &PROFILE_POSTPROCESS_COUNT,
            postprocess_start,
        );

        let seq_layout_start = profile_start();
        let seq: Sequence = seq.into();
        profile_record(
            &PROFILE_FINAL_SEQ_LAYOUT_NS,
            &PROFILE_FINAL_SEQ_LAYOUT_COUNT,
            seq_layout_start,
        );
        // The native-resolution training path is already an owned
        // standard-layout Array3, so this is only a move.  Keep a defensive
        // fallback for less common aggregation/splitting combinations; it
        // also preserves the layout guarantee required by ReBatch/DLPack.
        let values_layout_start = profile_start();
        let values = if arr.is_standard_layout() {
            arr
        } else {
            arr.as_standard_layout().to_owned()
        };
        profile_record(
            &PROFILE_FINAL_VALUES_LAYOUT_NS,
            &PROFILE_FINAL_VALUES_LAYOUT_COUNT,
            values_layout_start,
        );

        Some((seq, values))
    }

    /// Read the complete physical parent record without applying the normal
    /// loader windowing options.
    ///
    /// The ordinary reader intentionally returns the logical window described
    /// by the metadata (and removes the metadata padding first).  Native
    /// center/split augmentation has a different contract: it needs the
    /// complete record so that it can determine the available flank itself.
    /// In particular this is what makes a record written with
    /// ``padding=0`` but with a larger parent length usable for augmentation.
    /// The caller is responsible for validating that no aggregation, trim,
    /// split, scale, or random-shift options are active.
    pub fn read_parent_bf16_with_layout(
        &mut self,
        region: &GenomicRange,
        channels_last: bool,
    ) -> Option<(Sequence, Array3<bf16>)> {
        let offset = self.inner.metadata.segment_index.get(region)?;
        let mut buffer = vec![0; offset.0 .1 as usize];
        let file_read_start = profile_start();
        self.inner
            .file
            .read_exact_at(&mut buffer, offset.0 .0 as u64)
            .expect("read failed");
        profile_record(
            &PROFILE_FILE_READ_NS,
            &PROFILE_FILE_READ_COUNT,
            file_read_start,
        );

        let zstd_start = profile_start();
        let buffer = decompress_data_zst(&buffer);
        profile_record(&PROFILE_ZSTD_NS, &PROFILE_ZSTD_COUNT, zstd_start);

        let bincode_start = profile_start();
        let (seq, arr): (Vec<u8>, Array2<bf16>) =
            bincode::serde::decode_from_slice(&buffer, bincode::config::standard())
                .expect("decode failed")
                .0;
        profile_record(&PROFILE_BINCODE_NS, &PROFILE_BINCODE_COUNT, bincode_start);

        let sequence = Array2::from_shape_vec((1, seq.len()), seq)
            .expect("decoded sequence must be one-dimensional");
        let (n_tracks, n_values) = arr.dim();
        let values = if channels_last {
            arr.view()
                .t()
                .insert_axis(Axis(0))
                .as_standard_layout()
                .to_owned()
        } else {
            arr.into_shape_with_order((1, n_tracks, n_values))
                .expect("decoded values must be contiguous")
        };
        Some((Sequence(sequence), values))
    }

    /// Backwards-compatible bfloat16 wrapper used by the existing Rust API.
    pub fn read(&mut self, region: &GenomicRange) -> Option<(Sequence, Values)> {
        let (seq, values) = self.read_bf16(region)?;
        Some((seq, Values::from(values)))
    }

    /// Layout-selectable counterpart to [`DataStore::read`].
    pub fn read_with_layout(
        &mut self,
        region: &GenomicRange,
        channels_last: bool,
    ) -> Option<(Sequence, Array3<bf16>)> {
        self.read_bf16_with_layout(region, channels_last)
    }

    pub fn read_at(&mut self, i: usize) -> Option<(Sequence, Values)> {
        let region = self.inner.metadata.segment_index.get_index(i)?.0.clone();
        self.read(&region)
    }

    pub fn par_iter(
        &mut self,
        batch_size: usize,
        num_threads: usize,
        shuffle: bool,
        subset: Option<&[GenomicRange]>,
    ) -> impl Iterator<Item = (Array2<u8>, Array3<f32>)> {
        let mut segments = if let Some(s) = subset {
            assert!(
                s.iter()
                    .all(|r| self.inner.metadata.segment_index.contains_key(r)),
                "Some segments in the subset do not exist in the data store"
            );
            s.to_vec()
        } else {
            self.inner
                .metadata
                .segment_index
                .keys()
                .cloned()
                .collect::<Vec<_>>()
        };
        if shuffle {
            segments.shuffle(&mut self.read_opts.rng);
        }
        let iters = split_n_with_batch_size(&segments, num_threads, batch_size)
            .into_iter()
            .map(|chunk| {
                let iter = DataStoreIter {
                    segments: chunk.into(),
                    store: self.clone(),
                };
                ReBatch::new(iter, batch_size)
            })
            .collect::<Vec<_>>();
        ParallelLoader::new(iters)
    }

    /// Parallel iterator which preserves the stored bfloat16 values.
    ///
    /// This mirrors [`DataStore::par_iter`] but deliberately does not perform
    /// the bfloat16-to-float32 conversion.  The Python-facing loader exposes
    /// this path through a DLPack capsule so PyTorch can consume the owned
    /// buffer directly.
    pub fn par_iter_bf16(
        &mut self,
        batch_size: usize,
        num_threads: usize,
        shuffle: bool,
        subset: Option<&[GenomicRange]>,
    ) -> impl Iterator<Item = (Array2<u8>, Array3<bf16>)> {
        self.par_iter_bf16_with_layout(batch_size, num_threads, shuffle, subset, true)
    }

    /// Layout-selectable native-bfloat16 iterator.
    pub fn par_iter_bf16_with_layout(
        &mut self,
        batch_size: usize,
        num_threads: usize,
        shuffle: bool,
        subset: Option<&[GenomicRange]>,
        channels_last: bool,
    ) -> impl Iterator<Item = (Array2<u8>, Array3<bf16>)> {
        let mut segments = if let Some(s) = subset {
            assert!(
                s.iter()
                    .all(|r| self.inner.metadata.segment_index.contains_key(r)),
                "Some segments in the subset do not exist in the data store"
            );
            s.to_vec()
        } else {
            self.inner
                .metadata
                .segment_index
                .keys()
                .cloned()
                .collect::<Vec<_>>()
        };
        if shuffle {
            segments.shuffle(&mut self.read_opts.rng);
        }
        let iters = split_n_with_batch_size(&segments, num_threads, batch_size)
            .into_iter()
            .map(|chunk| {
                let iter = DataStoreBf16Iter {
                    segments: chunk.into(),
                    store: self.clone(),
                    channels_last,
                };
                ReBatch::new(iter, batch_size)
            })
            .collect::<Vec<_>>();
        ParallelLoader::new(iters)
    }

    /// Iterate over native-bfloat16 records while retaining the genomic range
    /// that produced each record.
    ///
    /// This is the low-level iterator used by the runtime center-crop/split
    /// augmentation path.  `regions` is already ordered by the caller; the
    /// method deliberately does not shuffle it so that multiple synchronized
    /// loaders can consume exactly the same parent order.
    pub fn par_iter_bf16_with_layout_and_regions(
        &mut self,
        regions: Vec<GenomicRange>,
        num_threads: usize,
        channels_last: bool,
    ) -> ParallelLoader<DataStoreBf16RegionIter, (GenomicRange, Array2<u8>, Array3<bf16>)> {
        let num_threads = num_threads.max(1);
        let iters = split_n_with_batch_size(&regions, num_threads, 1)
            .into_iter()
            .map(|chunk| DataStoreBf16RegionIter {
                segments: chunk.into(),
                store: self.clone(),
                channels_last,
            })
            .collect::<Vec<_>>();
        ParallelLoader::new(iters)
    }

    /// Region-preserving iterator over complete physical parent records.
    ///
    /// This is intentionally separate from the regular region iterator: the
    /// latter applies the logical window crop and is therefore not suitable
    /// for runtime center/shift augmentation.
    pub fn par_iter_bf16_parent_with_layout_and_regions(
        &mut self,
        regions: Vec<GenomicRange>,
        num_threads: usize,
        channels_last: bool,
    ) -> ParallelLoader<DataStoreBf16ParentRegionIter, (GenomicRange, u64, Array2<u8>, Array3<bf16>)>
    {
        let num_threads = num_threads.max(1);
        let iters = split_n_with_batch_size(&regions, num_threads, 1)
            .into_iter()
            .map(|chunk| DataStoreBf16ParentRegionIter {
                segments: chunk.into(),
                store: self.clone(),
                channels_last,
            })
            .collect::<Vec<_>>();
        ParallelLoader::new(iters)
    }
}

pub struct DataStoreIter {
    segments: VecDeque<GenomicRange>,
    store: DataStore,
}

impl Iterator for DataStoreIter {
    type Item = (Array2<u8>, Array3<f32>);

    fn next(&mut self) -> Option<Self::Item> {
        let segment = self.segments.pop_front()?;
        let (seq, values) = self.store.read_bf16(&segment).unwrap();
        let value_convert_start = profile_start();
        let values: Array3<f32> = values.mapv(|x| x.to_f32());
        profile_record(
            &PROFILE_VALUE_CONVERT_NS,
            &PROFILE_VALUE_CONVERT_COUNT,
            value_convert_start,
        );
        Some((seq.into(), values))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.segments.len();
        (len, Some(len))
    }
}

/// The native bfloat16 counterpart of [`DataStoreIter`].
pub struct DataStoreBf16Iter {
    segments: VecDeque<GenomicRange>,
    store: DataStore,
    channels_last: bool,
}

impl Iterator for DataStoreBf16Iter {
    type Item = (Array2<u8>, Array3<bf16>);

    fn next(&mut self) -> Option<Self::Item> {
        let segment = self.segments.pop_front()?;
        let (seq, values) = self
            .store
            .read_bf16_with_layout(&segment, self.channels_last)
            .unwrap();
        Some((seq.into(), values))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.segments.len();
        (len, Some(len))
    }
}

impl ExactSizeIterator for DataStoreBf16Iter {
    fn len(&self) -> usize {
        self.segments.len()
    }
}

/// Region-preserving counterpart to [`DataStoreBf16Iter`].
///
/// The regular iterator intentionally exposes only arrays for backwards
/// compatibility.  The native augmentation iterator needs the parent genomic
/// range in order to report the updated `chr:start-end` coordinates after a
/// center crop and split.
pub struct DataStoreBf16RegionIter {
    segments: VecDeque<GenomicRange>,
    store: DataStore,
    channels_last: bool,
}

impl Iterator for DataStoreBf16RegionIter {
    type Item = (GenomicRange, Array2<u8>, Array3<bf16>);

    fn next(&mut self) -> Option<Self::Item> {
        let segment = self.segments.pop_front()?;
        let (seq, values) = self
            .store
            .read_bf16_with_layout(&segment, self.channels_last)?;
        Some((segment, seq.into(), values))
    }
}

impl ExactSizeIterator for DataStoreBf16RegionIter {
    fn len(&self) -> usize {
        self.segments.len()
    }
}

/// Region-preserving iterator over complete physical parent records.
///
/// Unlike `DataStoreBf16RegionIter`, this intentionally bypasses the logical
/// window crop so the native center/split path can use both metadata padding
/// and any extra parent length when calculating a shift.
pub struct DataStoreBf16ParentRegionIter {
    segments: VecDeque<GenomicRange>,
    store: DataStore,
    channels_last: bool,
}

impl Iterator for DataStoreBf16ParentRegionIter {
    type Item = (GenomicRange, u64, Array2<u8>, Array3<bf16>);

    fn next(&mut self) -> Option<Self::Item> {
        let segment = self.segments.pop_front()?;
        let (seq, values) = self
            .store
            .read_parent_bf16_with_layout(&segment, self.channels_last)?;
        // The metadata range denotes the logical (unpadded) interval.  The
        // first base of the physical parent therefore starts at this origin.
        // `saturating_sub` handles chromosome-start records whose left flank
        // is represented by N bases outside the chromosome.
        let physical_start = segment.start().saturating_sub(self.store.n_pad() as u64);
        Some((segment, physical_start, seq.into(), values))
    }
}

impl ExactSizeIterator for DataStoreBf16ParentRegionIter {
    fn len(&self) -> usize {
        self.segments.len()
    }
}

impl ExactSizeIterator for DataStoreIter {
    fn len(&self) -> usize {
        self.segments.len()
    }
}

/// An on disk data store for storing genome sequences and their associated data values.
pub struct DataStoreBuilder {
    location: PathBuf,
    sequence_length: u32,
    resolution: u32,
    padding: u32,
    pub(crate) segments: IndexMap<GenomicRange, PathBuf>,
    pub(crate) data_keys: IndexSet<String>,
    // Number of segment value blocks written for each direct (non-W5Z) track.
    // This lets Python callers stream batches while finish() verifies that
    // every segment has a value block for every track.
    value_counts: HashMap<String, usize>,
}

impl DataStoreBuilder {
    /// Create an empty data store at the specified location.
    pub fn new(
        location: impl AsRef<Path>,
        sequence_length: u32,
        resolution: u32,
        padding: u32,
    ) -> Result<Self> {
        ensure!(
            sequence_length % resolution == 0,
            "window size must be a multiple of resolution"
        );
        ensure!(
            padding % resolution == 0,
            "Padding must be a multiple of resolution"
        );

        if !location.as_ref().exists() {
            std::fs::create_dir_all(&location)?;
        }

        Ok(Self {
            location: location.as_ref().to_path_buf(),
            segments: IndexMap::new(),
            data_keys: IndexSet::new(),
            value_counts: HashMap::new(),
            sequence_length,
            resolution,
            padding,
        })
    }

    /// Sequence length plus padding on both sides.
    pub fn total_sequence_length(&self) -> u32 {
        self.sequence_length + self.padding * 2
    }

    fn chroms(&self) -> Vec<String> {
        self.segments
            .keys()
            .map(|r| r.chrom().to_string())
            .unique()
            .collect()
    }

    fn add_seqs(
        &mut self,
        seqs: impl IndexedParallelIterator<Item = (GenomicRange, Vec<u8>)>,
    ) -> Result<()> {
        let seq_len = self.total_sequence_length() as usize;
        let chunk_size = (seqs.len() / 32).max(1);
        let files = seqs
            .chunks(chunk_size)
            .flat_map_iter(|chunk| {
                chunk.into_iter().map(|(range, seq)| {
                    assert!(
                        seq.len() == seq_len,
                        "Sequence length for range {} is {}, expected {}",
                        range.pretty_show(),
                        seq.len(),
                        seq_len,
                    );

                    let file_path = self.location.join(range.pretty_show());
                    let mut file = File::create_new(&file_path).expect(&format!(
                        "Failed to create sequence file at: {}",
                        file_path.display()
                    ));

                    let seq = compress_data_zst(seq, 5);
                    file.write_all(&(seq.len() as u64).to_le_bytes()).unwrap();
                    file.write_all(&seq)
                        .expect("Failed to write sequences to file");

                    (range, file_path)
                })
            })
            .collect::<Vec<_>>();
        files.into_iter().for_each(|(range, file)| {
            self.segments.insert(range, file);
        });
        Ok(())
    }

    pub fn add_segments(
        &mut self,
        segments: Vec<GenomicRange>,
        fasta: &mut IndexedReader<noodles::fasta::io::BufReader<File>>,
    ) -> Result<()> {
        let fasta = Arc::new(Mutex::new(fasta));
        let padding = self.padding as usize;
        let seq_len = self.total_sequence_length() as usize;
        let seqs = segments.into_par_iter().map(|segment| {
            let mut reader = fasta.lock().unwrap();
            let seq = get_seq(&mut reader, &segment, seq_len, padding).unwrap();
            (segment, seq)
        });
        self.add_seqs(seqs)
    }

    /// Append one sequence and all of its track values in a single pass.
    ///
    /// This is the streaming counterpart to `add_segments` + `add_values`:
    /// the caller supplies an already encoded sequence (A=0, C=1, G=2, T=3,
    /// N=4) and one float32 vector for every track.  The first segment fixes
    /// the track names and their order; every subsequent segment must provide
    /// exactly the same names in that order.  Data are written to the per-segment
    /// temporary file immediately, so no FASTA file or second TFRecord pass is
    /// required.
    pub fn add_segment(
        &mut self,
        range: GenomicRange,
        sequence: ArrayView1<'_, u8>,
        track_values: Vec<(String, ArrayView1<'_, f32>)>,
    ) -> Result<()> {
        let owned_values: Vec<(String, Vec<bf16>)> = track_values
            .into_iter()
            .map(|(key, values)| {
                (
                    key,
                    values.iter().map(|value| bf16::from_f32(*value)).collect(),
                )
            })
            .collect();
        self.add_segment_bf16(
            range,
            sequence,
            owned_values
                .iter()
                .map(|(key, values)| (key.clone(), values.as_slice()))
                .collect(),
        )
    }

    /// Append one sequence and already-decoded bfloat16 track values.
    ///
    /// This is kept as a Rust-level API so native TFRecord conversion can
    /// decode bfloat16 tensors without first materializing a float32 copy of
    /// every track.  The slices are copied/compressed before this method
    /// returns, so callers may reuse their buffers afterwards.
    pub(crate) fn add_segment_bf16(
        &mut self,
        range: GenomicRange,
        sequence: ArrayView1<'_, u8>,
        track_values: Vec<(String, &[bf16])>,
    ) -> Result<()> {
        let expected_sequence = self.total_sequence_length() as usize;
        ensure!(
            sequence.len() == expected_sequence,
            "sequence for {} has length {}, expected {}",
            range.pretty_show(),
            sequence.len(),
            expected_sequence
        );
        ensure!(!track_values.is_empty(), "at least one track is required");
        ensure!(
            !self.segments.contains_key(&range),
            "segment {} already exists",
            range.pretty_show()
        );

        for (key, values) in &track_values {
            ensure!(!key.is_empty(), "track name must not be empty");
            ensure!(
                values.len() == (self.total_sequence_length() / self.resolution) as usize,
                "track {} has {} values, expected {}",
                key,
                values.len(),
                self.total_sequence_length() / self.resolution
            );
        }

        if self.data_keys.is_empty() {
            for (key, _) in &track_values {
                ensure!(
                    !self.data_keys.contains(key),
                    "track {} was supplied more than once",
                    key
                );
                self.data_keys.insert(key.clone());
                self.value_counts.insert(key.clone(), 0);
            }
        } else {
            ensure!(
                track_values.len() == self.data_keys.len(),
                "segment {} supplies {} tracks, expected {}",
                range.pretty_show(),
                track_values.len(),
                self.data_keys.len()
            );
            for (index, (key, _)) in track_values.iter().enumerate() {
                let expected_key = self.data_keys.get_index(index).unwrap();
                ensure!(
                    key == expected_key,
                    "segment {} has track {} at position {}, expected {}",
                    range.pretty_show(),
                    key,
                    index,
                    expected_key
                );
            }
        }

        let file_path = self.location.join(range.pretty_show());
        let mut file = File::create_new(&file_path).with_context(|| {
            format!("Failed to create sequence file at: {}", file_path.display())
        })?;
        let encoded_sequence = compress_data_zst(sequence.to_vec(), 5);
        file.write_all(&(encoded_sequence.len() as u64).to_le_bytes())?;
        file.write_all(&encoded_sequence)?;

        let expected_values = (self.total_sequence_length() / self.resolution) as usize;
        for (_, values) in &track_values {
            debug_assert_eq!(values.len(), expected_values);
            let encoded = bincode::serde::encode_to_vec(values, bincode::config::standard())?;
            let encoded = compress_data_zst(encoded, 5);
            file.write_all(&(encoded.len() as u64).to_le_bytes())?;
            file.write_all(&encoded)?;
        }
        drop(file);

        self.segments.insert(range, file_path);
        for key in &self.data_keys {
            *self
                .value_counts
                .get_mut(key)
                .expect("track was registered") += 1;
        }
        Ok(())
    }

    /// Add a segment that was encoded by a worker thread.  The encoded bytes
    /// use the same per-segment layout as `add_segment_bf16`, so this method
    /// only performs ordered index bookkeeping and a single file write.
    pub(crate) fn add_encoded_segment(
        &mut self,
        range: GenomicRange,
        encoded: Vec<u8>,
        track_keys: Vec<String>,
    ) -> Result<()> {
        ensure!(!track_keys.is_empty(), "at least one track is required");
        ensure!(!encoded.is_empty(), "encoded segment must not be empty");
        ensure!(
            !self.segments.contains_key(&range),
            "segment {} already exists",
            range.pretty_show()
        );
        for key in &track_keys {
            ensure!(!key.is_empty(), "track name must not be empty");
        }

        if self.data_keys.is_empty() {
            for key in &track_keys {
                ensure!(
                    !self.data_keys.contains(key),
                    "track {} was supplied more than once",
                    key
                );
                self.data_keys.insert(key.clone());
                self.value_counts.insert(key.clone(), 0);
            }
        } else {
            ensure!(
                track_keys.len() == self.data_keys.len(),
                "segment {} supplies {} tracks, expected {}",
                range.pretty_show(),
                track_keys.len(),
                self.data_keys.len()
            );
            for (index, key) in track_keys.iter().enumerate() {
                let expected_key = self.data_keys.get_index(index).unwrap();
                ensure!(
                    key == expected_key,
                    "segment {} has track {} at position {}, expected {}",
                    range.pretty_show(),
                    key,
                    index,
                    expected_key
                );
            }
        }

        let file_path = self.location.join(range.pretty_show());
        let mut file = File::create_new(&file_path).with_context(|| {
            format!("Failed to create sequence file at: {}", file_path.display())
        })?;
        file.write_all(&encoded)?;
        drop(file);
        self.segments.insert(range, file_path);
        for key in &self.data_keys {
            *self
                .value_counts
                .get_mut(key)
                .expect("track was registered") += 1;
        }
        Ok(())
    }

    fn add_values(
        &mut self,
        key: impl Into<String>,
        data: impl IndexedParallelIterator<Item = (GenomicRange, Vec<bf16>)>,
    ) -> Result<()> {
        let key = key.into();
        let n_segments = self.segments.len();
        if self.data_keys.contains(&key) {
            bail!("Data key {} already exists", &key);
        }
        self.data_keys.insert(key.clone());

        let val_len = (self.total_sequence_length() / self.resolution) as usize;
        let chunk_size = (data.len() / 32).max(1);
        data.chunks(chunk_size).try_for_each(|chunk| {
            chunk.into_iter().try_for_each(|(range, values)| {
                ensure!(
                    values.len() == val_len,
                    "Values length for range {} is {}, expected {}",
                    range.pretty_show(),
                    values.len(),
                    val_len,
                );
                let file_path = self.segments.get(&range).ok_or_else(|| {
                    anyhow::anyhow!(
                        "No sequence for range {} exists. Add sequences first!",
                        range.pretty_show()
                    )
                })?;
                let mut file = std::fs::OpenOptions::new()
                    .append(true)
                    .open(file_path)
                    .with_context(|| {
                        format!("Failed to open data file at: {}", file_path.display())
                    })?;
                let values = bincode::serde::encode_to_vec(values, bincode::config::standard())?;
                let values = compress_data_zst(values, 5);
                file.write_all(&values.len().to_le_bytes())?;
                file.write_all(&values)?;
                Ok(())
            })
        })?;
        // add_w5z supplies exactly one value block per segment. Keep the
        // count for finish()'s consistency check, just as for streamed
        // add_segment_data calls.
        self.value_counts.insert(key, n_segments);
        Ok(())
    }

    /// Append a batch of already-segmented values for one track.
    ///
    /// ``values`` has shape ``(num_segments_in_batch, values_per_segment)``
    /// and must be supplied in exactly the same order as ``self.segments``.
    /// ``start_index`` makes it possible to stream a track in bounded-memory
    /// batches. The values are converted to bfloat16 and appended directly
    /// to the per-segment temporary files; no W5Z file is involved.
    pub fn add_segment_data(
        &mut self,
        key: impl Into<String>,
        start_index: usize,
        values: ndarray::ArrayView2<'_, f32>,
    ) -> Result<()> {
        let key = key.into();
        let n_segments = self.segments.len();
        ensure!(n_segments > 0, "no segments have been added");
        ensure!(
            start_index <= n_segments,
            "start_index {} exceeds {} segments",
            start_index,
            n_segments
        );
        ensure!(
            start_index + values.nrows() <= n_segments,
            "value batch [{}:{}) exceeds {} segments",
            start_index,
            start_index + values.nrows(),
            n_segments
        );
        ensure!(values.nrows() > 0, "value batch must not be empty");

        let expected_values = (self.total_sequence_length() / self.resolution) as usize;
        ensure!(
            values.ncols() == expected_values,
            "values have {} columns, expected {}",
            values.ncols(),
            expected_values
        );

        if start_index == 0 {
            if self.data_keys.contains(&key) {
                bail!("Data key {} already exists", key);
            }
            self.data_keys.insert(key.clone());
            self.value_counts.insert(key.clone(), 0);
        } else {
            ensure!(
                self.data_keys.contains(&key),
                "track {} must start with a batch at start_index=0",
                key
            );
        }

        let already_written = *self.value_counts.get(&key).unwrap_or(&0);
        ensure!(
            already_written == start_index,
            "track {} expects start_index {}, got {}",
            key,
            already_written,
            start_index
        );

        for (offset, row) in values.outer_iter().enumerate() {
            let segment_index = start_index + offset;
            let (_, file_path) = self
                .segments
                .get_index(segment_index)
                .ok_or_else(|| anyhow::anyhow!("invalid segment index {}", segment_index))?;
            let values: Vec<bf16> = row.iter().map(|value| bf16::from_f32(*value)).collect();
            let encoded = bincode::serde::encode_to_vec(&values, bincode::config::standard())?;
            let encoded = compress_data_zst(encoded, 5);
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .open(file_path)
                .with_context(|| format!("Failed to open data file at: {}", file_path.display()))?;
            file.write_all(&(encoded.len() as u64).to_le_bytes())?;
            file.write_all(&encoded)?;
        }
        self.value_counts.insert(key, start_index + values.nrows());
        Ok(())
    }

    pub fn add_w5z(&mut self, key: impl Into<String>, data: W5Z) -> Result<()> {
        let regions: Vec<_> = self.segments.keys().cloned().collect();
        let key = key.into();

        // Load the W5Z data
        let data = self
            .chroms()
            .into_par_iter()
            .map(|chr| {
                let v: Vec<_> = data
                    .get(&chr)?
                    .into_iter()
                    .map(|v| bf16::from_f32(v))
                    .collect();
                Ok((chr, v))
            })
            .collect::<Result<HashMap<_, _>>>()?;

        let padding = self.padding as isize;
        let seq_len = self.total_sequence_length() as isize;
        let resolution = self.resolution as usize;
        let values = regions.into_par_iter().map(|range| {
            let values = data.get(range.chrom()).unwrap();
            let start = range.start() as isize - padding;
            let end = start + seq_len;
            let mut output_vec = slice_pad(values, start, end, bf16::ZERO);

            if resolution > 1 {
                output_vec = output_vec
                    .chunks(resolution)
                    .map(|x| {
                        let m: average::Mean = x.iter().map(|x| f64::from(*x)).collect();
                        bf16::from_f64(m.mean())
                    })
                    .collect();
            }

            (range, output_vec)
        });

        self.add_values(key, values)?;
        Ok(())
    }

    pub fn finish(mut self, path: impl AsRef<Path>) -> Result<()> {
        for key in &self.data_keys {
            let count = self.value_counts.get(key).copied().unwrap_or(0);
            ensure!(
                count == self.segments.len(),
                "track {} has {} segment values, expected {}",
                key,
                count,
                self.segments.len()
            );
        }
        path.as_ref()
            .parent()
            .map(|p| std::fs::create_dir_all(p).unwrap());
        let mut store = File::create(&path).with_context(|| {
            format!(
                "Failed to create data store file at {}",
                path.as_ref().display()
            )
        })?;

        let style = ProgressStyle::with_template(
            "{msg}: [{elapsed}] {wide_bar:.cyan/blue} {percent}/100% (eta: {eta})",
        )
        .unwrap();
        let bar = ProgressBar::new(self.segments.len().div_ceil(32) as u64)
            .with_message("Compressing")
            .with_style(style);

        let n_keys = self.data_keys.len();
        let sizes: Vec<_> = self
            .segments
            .values_mut()
            .chunks(32)
            .into_iter()
            .flat_map(|chunk| {
                let (bytes, sizes): (Vec<_>, Vec<_>) = chunk
                    .collect::<Vec<_>>()
                    .into_par_iter()
                    .map(|file| compress_data_file(file, n_keys).unwrap())
                    .unzip();
                bytes.into_iter().for_each(|bytes| {
                    store.write_all(&bytes).unwrap();
                });
                bar.inc(1);
                sizes
            })
            .collect();
        bar.finish();

        let segment_index = self
            .segments
            .keys()
            .cloned()
            .zip(sizes.into_iter())
            .scan(0u64, |offset, (range, n_bytes)| {
                let o = Offset((*offset, n_bytes as u64));
                *offset += n_bytes as u64;
                Some((range, o))
            })
            .collect::<IndexMap<_, _>>();
        let offset = segment_index.last().unwrap().1 .0;

        let metadata = StoreMetadata {
            segment_index,
            data_keys: self.data_keys.clone(),
            sequence_length: self.sequence_length,
            resolution: self.resolution,
            padding: self.padding,
        };
        let metadata_byes = metadata.encode()?;
        store.write_all(&metadata_byes)?;
        let metadata_len = metadata_byes.len() as u32;

        // 8 + 4 = 12 bytes for the index metadata
        store.write_all(&(offset.0 + offset.1).to_le_bytes())?; // 8 bytes
        store.write_all(&metadata_len.to_le_bytes())?; // 4 bytes
        Ok(())
    }
}

/// Encode the intermediate per-segment representation.  This is independent
/// of `DataStoreBuilder` state and can therefore run concurrently in worker
/// threads before the ordered writer appends the segment to the store.
pub(crate) fn encode_segment_bf16(
    sequence: &[u8],
    track_values: &[(String, Vec<bf16>)],
    expected_sequence: usize,
    expected_values: usize,
) -> Result<Vec<u8>> {
    ensure!(
        sequence.len() == expected_sequence,
        "sequence has length {}, expected {}",
        sequence.len(),
        expected_sequence
    );
    ensure!(!track_values.is_empty(), "at least one track is required");
    for (key, values) in track_values {
        ensure!(!key.is_empty(), "track name must not be empty");
        ensure!(
            values.len() == expected_values,
            "track {} has {} values, expected {}",
            key,
            values.len(),
            expected_values
        );
    }
    let mut encoded = Vec::new();
    let sequence = compress_data_zst(sequence.to_vec(), 5);
    encoded.extend_from_slice(&(sequence.len() as u64).to_le_bytes());
    encoded.extend_from_slice(&sequence);
    for (_, values) in track_values {
        let values = bincode::serde::encode_to_vec(values, bincode::config::standard())?;
        let values = compress_data_zst(values, 5);
        encoded.extend_from_slice(&(values.len() as u64).to_le_bytes());
        encoded.extend_from_slice(&values);
    }
    Ok(encoded)
}

fn compress_data_file(file_path: impl AsRef<Path>, nrow: usize) -> Result<(Vec<u8>, usize)> {
    let mut file = File::open(&file_path).with_context(|| {
        format!(
            "Failed to open data file at: {}",
            file_path.as_ref().display()
        )
    })?;
    let mut n = [0; 8];
    file.read_exact(&mut n)?;
    let n = u64::from_le_bytes(n);

    let mut buf = vec![0; n as usize];
    file.read_exact(&mut buf)?;
    let seq = decompress_data_zst(&buf);

    let mut ncol = 0;
    let data: Vec<_> = std::iter::repeat_with(|| {
        let mut n = [0; 8];
        file.read_exact(&mut n).unwrap();
        let n = u64::from_le_bytes(n) as usize;

        let mut buf = vec![0; n];
        file.read_exact(&mut buf).unwrap();
        let values = decompress_data_zst(&buf);
        let values: Vec<bf16> =
            bincode::serde::decode_from_slice(&values, bincode::config::standard())
                .unwrap()
                .0;
        ncol = values.len();
        values
    })
    .take(nrow)
    .flatten()
    .collect();
    // Note the array dimension here is: (experiment, sequence).
    // This is key to get the best compression ratio.
    let arr = Array2::from_shape_vec((nrow, ncol), data)
        .map_err(|e| anyhow::anyhow!("Failed to create array: {}", e))?;

    let bytes = bincode::serde::encode_to_vec((seq, arr), bincode::config::standard())?;
    let bytes = compress_data_zst(bytes, 9);
    let n_bytes = bytes.len();
    Ok((bytes, n_bytes))
}

fn get_seq(
    reader: &mut IndexedReader<noodles::fasta::io::BufReader<File>>,
    region: &impl BEDLike,
    seq_len: usize,
    pad: usize,
) -> Result<Vec<u8>> {
    let mut start = region.start() as usize;
    let mut pad_left = 0;
    if pad > start {
        pad_left = pad - start;
        start = 0;
    } else {
        start -= pad;
    }
    let end = start + seq_len - pad_left;
    let interval = noodles::core::region::Region::new(
        region.chrom(),
        Position::try_from(start + 1)?..=Position::try_from(end)?,
    );

    let base_n = encode_nucleotide(b'N')?;
    let mut seq = vec![base_n; pad_left];
    seq.extend(
        reader
            .query(&interval)?
            .sequence()
            .as_ref()
            .iter()
            .map(|b| encode_nucleotide(*b).unwrap()),
    );
    seq.resize(seq_len, base_n);
    Ok(seq)
}

// Pad out-of-bounds slices with a specified value.
fn slice_pad<T: Clone>(values: &[T], start: isize, end: isize, pad_value: T) -> Vec<T> {
    let mut result = Vec::with_capacity((end - start) as usize);
    for i in start..end as isize {
        if i < 0 || i >= values.len() as isize {
            result.push(pad_value.clone());
        } else {
            result.push(values[i as usize].clone());
        }
    }
    result
}

fn encode_nucleotide(base: u8) -> Result<u8> {
    let b = match base {
        b'A' | b'a' => 0,
        b'C' | b'c' => 1,
        b'G' | b'g' => 2,
        b'T' | b't' => 3,
        b'N' | b'n' => 4,
        _ => bail!("Invalid DNA base: {}", base as char),
    };
    Ok(b)
}

pub(crate) fn decode_nucleotide(base: u8) -> Result<u8> {
    let b = match base {
        0 => b'A',
        1 => b'C',
        2 => b'G',
        3 => b'T',
        4 => b'N',
        _ => bail!("Invalid DNA base: {}", base),
    };
    Ok(b)
}

fn read_metadata(file: &mut std::fs::File) -> Result<StoreMetadata> {
    file.seek(std::io::SeekFrom::End(-12))?;

    let mut buffer = [0; 8];
    file.read_exact(&mut buffer)?;
    let start = u64::from_le_bytes(buffer);
    let mut buffer = [0; 4];
    file.read_exact(&mut buffer)?;
    let bytes_len = u32::from_le_bytes(buffer);

    file.seek(std::io::SeekFrom::Start(start))?;
    let mut buffer = vec![0; bytes_len as usize];
    file.read_exact(&mut buffer)?;
    Ok(StoreMetadata::decode(&buffer)?)
}

/// Array Helper

/// Aggregate the values along the sequence axis for channels-last arrays
/// `(batch, sequence, track)`.
fn aggregate_by_length(arr: Array3<bf16>, size: usize) -> Array3<bf16> {
    let (d, h, w) = arr.dim();
    if h % size != 0 {
        panic!(
            "Cannot aggregate values of length {} by size {}: length is not a multiple of size",
            h, size
        );
    }
    let num_chunks = h / size;
    let data = arr
        .into_shape_clone((d, num_chunks, size, w))
        .unwrap()
        .mapv(|x| x.to_f64())
        .mean_axis(Axis(2))
        .unwrap()
        .mapv(|x| bf16::from_f64(x));
    data
}

/// Aggregate values while preserving either supported output layout.
fn aggregate_by_length_layout(arr: Array3<bf16>, size: usize, channels_last: bool) -> Array3<bf16> {
    if channels_last {
        return aggregate_by_length(arr, size);
    }

    // Channels-first: (batch, track, sequence).
    let (d, w, h) = arr.dim();
    if h % size != 0 {
        panic!(
            "Cannot aggregate values of length {} by size {}: length is not a multiple of size",
            h, size
        );
    }
    let num_chunks = h / size;
    arr.into_shape_with_order((d, w, num_chunks, size))
        .unwrap()
        .mapv(|x| x.to_f64())
        .mean_axis(Axis(3))
        .unwrap()
        .mapv(|x| bf16::from_f64(x))
}

/// Split the values into consecutive chunks on the second dimension (the sequence).
/// The last chunk is dropped if it is smaller.
fn split_data(arr: Array3<bf16>, size: usize) -> Result<Array3<bf16>> {
    let (d, h, w) = arr.dim();

    if size == 0 || h % size != 0 {
        bail!(
            "Cannot split values into chunks of size {}: length {} is not a multiple of size",
            size,
            h
        );
    }

    let num_chunks = h / size;

    // Reshape to 4D to expose the chunks as a new dimension.
    // The shape becomes (d, num_chunks, chunk_height, w).
    // No axis permutation is needed because the dimensions are already in the correct order
    // to be collapsed.
    let intermediate_shape = (d, num_chunks, size, w);

    // Reshape again to the final 3D shape by collapsing the first two dimensions.
    let final_shape = (d * num_chunks, size, w);

    let result = arr
        .into_shape_clone(intermediate_shape)?
        .into_shape_clone(final_shape)?;

    Ok(result)
}

/// Split values while preserving either supported output layout.
fn split_data_layout(arr: Array3<bf16>, size: usize, channels_last: bool) -> Result<Array3<bf16>> {
    if channels_last {
        return split_data(arr, size);
    }

    // Channels-first input: (batch, track, sequence).  Expose the chunk axis,
    // move it next to batch, and materialise the resulting standard layout so
    // the final shape is (batch * chunks, track, chunk_sequence).
    let (d, w, h) = arr.dim();
    if size == 0 || h % size != 0 {
        bail!(
            "Cannot split values of length {} into chunks of size {}",
            h,
            size
        );
    }
    let num_chunks = h / size;
    let result = arr
        .into_shape_with_order((d, w, num_chunks, size))?
        .permuted_axes([0, 2, 1, 3])
        .as_standard_layout()
        .to_owned()
        .into_shape_with_order((d * num_chunks, w, size))?;
    Ok(result)
}

fn split_sequence(arr: ArrayView2<u8>, size: usize) -> Result<ArrayView2<u8>> {
    let (d, h) = arr.dim();

    if size == 0 || h % size != 0 {
        bail!(
            "Cannot split values into chunks of size {}: length {} is not a multiple of size",
            size,
            h
        );
    }

    let num_chunks = h / size;
    let intermediate_shape = (d, num_chunks, size);
    let final_shape = (d * num_chunks, size);

    let result = arr
        .into_shape_with_order(intermediate_shape)?
        .into_shape_with_order(final_shape)?;

    Ok(result)
}

/// Perform the optional in-place value transformation.
///
/// NaN replacement is part of this pass and is therefore only performed when
/// the caller has requested scaling or clamping.
fn transform(mut arr: ArrayViewMut3<bf16>, scale: Option<bf16>, clamp_max: Option<bf16>) {
    arr.map_inplace(|x| {
        if x.is_nan() {
            *x = bf16::from_f32(0.0); // Replace NaN with 0.0
        } else {
            if let Some(scale) = scale {
                *x *= scale;
            }
            if let Some(clamp_max) = clamp_max {
                if *x > clamp_max {
                    *x = clamp_max;
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;
    use ndarray::array;
    use rand::{
        distr::{Distribution, Uniform},
        Rng,
    };

    fn create_store(
        location: impl AsRef<Path>,
        sequence_length: u32,
        resolution: u32,
        padding: u32,
        n_segments: usize,
        n_data: usize,
    ) -> (Vec<Vec<u8>>, Array3<f32>) {
        let mut rng = rand::rng();
        let temp_dir = tempfile::tempdir().unwrap();
        let tmp = temp_dir.as_ref().join("store_builder");
        let mut store = DataStoreBuilder::new(&tmp, sequence_length, resolution, padding).unwrap();

        let random_regions = (0..n_segments)
            .map(|i| {
                GenomicRange::from_str(&format!(
                    "chr1:{}-{}",
                    i * sequence_length as usize + 1,
                    (i + 1) * sequence_length as usize
                ))
                .unwrap()
            })
            .collect::<Vec<_>>();
        let random_seqs = random_regions
            .iter()
            .map(|r| {
                (r.clone(), {
                    (0..(sequence_length + 2 * padding))
                        .map(|_| rng.random_range(0..5))
                        .collect::<Vec<u8>>()
                })
            })
            .collect::<Vec<_>>();
        store.add_seqs(random_seqs.clone().into_par_iter()).unwrap();

        let mut random_values = Vec::new();
        for i in 0..n_data {
            let key = format!("key{}", i + 1);
            let values = random_regions
                .iter()
                .map(|r| {
                    let between = Uniform::new(0.0, 100.0).unwrap();
                    (
                        r.clone(),
                        (0..((sequence_length + 2 * padding) / resolution))
                            .map(|_| bf16::from_f32(between.sample(&mut rng) as f32))
                            .collect::<Vec<bf16>>(),
                    )
                })
                .collect::<Vec<_>>();
            store
                .add_values(key, values.clone().into_par_iter())
                .unwrap();
            let values: Vec<_> = values
                .into_iter()
                .map(|(_, v)| Array1::from_vec(v))
                .collect();
            let values = ndarray::stack(
                Axis(0),
                &values.iter().map(|x| x.view()).collect::<Vec<_>>(),
            )
            .unwrap();
            random_values.push(values);
        }

        store.finish(location).unwrap();

        let random_values = ndarray::stack(
            Axis(2),
            &random_values.iter().map(|x| x.view()).collect::<Vec<_>>(),
        )
        .unwrap()
        .mapv(|x| x.to_f32());

        (
            random_seqs.into_iter().map(|(_, seq)| seq).collect(),
            random_values,
        )
    }

    #[test]
    fn test_arr_aggregation() {
        let arr = array![
            [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]],
            [[7.0], [8.0], [9.0], [10.0], [11.0], [12.0]]
        ];
        assert_eq!(arr.dim(), (2, 6, 1));

        let values = arr.mapv(bf16::from_f64);
        let aggregated = aggregate_by_length(values, 2).mapv(|x| x.to_f64());
        assert_eq!(aggregated.dim(), (2, 3, 1));
        assert_eq!(
            aggregated,
            array![[[1.5], [3.5], [5.5]], [[7.5], [9.5], [11.5]]]
        );
    }

    #[test]
    fn test_native_resolution_skips_aggregation() {
        let temp_dir = tempfile::tempdir().unwrap();
        let location = temp_dir.as_ref().join("store.gdata");
        create_store(&location, 64, 2, 4, 1, 1);

        let mut options = DataStoreReadOptions::default();
        options.read_resolution = Some(2);
        let store = DataStore::open(location, options).unwrap();

        assert_eq!(store.out_resolution, 2);
        assert_eq!(store.aggregate_size, None);
    }

    #[test]
    fn test_datastore() {
        let temp_dir = tempfile::tempdir().unwrap();
        let location = temp_dir.as_ref().join("store.gdata");
        let (seqs, array) = create_store(&location, 1024, 2, 8, 100, 10);

        let mut store = DataStore::open(location, DataStoreReadOptions::default()).unwrap();

        let (s, v) = store.read_at(2).unwrap();
        assert_eq!(s.0.into_raw_vec_and_offset().0, seqs[2][8..1024 + 8]);
        assert_eq!(v.0.shape(), [1, 512, 10]);
        assert_eq!(
            v.0.mapv(|x| x.to_f32()),
            array.slice(s![2..3, 8 / 2..(1024 + 8) / 2, ..]).to_owned()
        );

        let region = store
            .inner
            .metadata
            .segment_index
            .get_index(2)
            .unwrap()
            .0
            .clone();
        let (_, values_channels_first) = store.read_bf16_with_layout(&region, false).unwrap();
        assert_eq!(values_channels_first.shape(), [1, 10, 512]);
        assert!(values_channels_first.is_standard_layout());
        assert_eq!(
            values_channels_first.mapv(|x| x.to_f32()),
            array
                .slice(s![2..3, 8 / 2..(1024 + 8) / 2, ..])
                .permuted_axes([0, 2, 1])
                .to_owned()
        );

        let values_iter = store.par_iter(3, 2, false, None);
        let values = values_iter.map(|(_, values)| values).collect::<Vec<_>>();
        let values = ndarray::concatenate(
            Axis(0),
            &values.iter().map(|x| x.view()).collect::<Vec<_>>(),
        )
        .unwrap();

        assert_eq!(
            values,
            array.slice(s![.., 8 / 2..(1024 + 8) / 2, ..]).to_owned()
        );
    }

    #[test]
    fn test_shift() {
        let temp_dir = tempfile::tempdir().unwrap();
        let location = temp_dir.as_ref().join("store.gdata");
        let (seqs, array) = create_store(&location, 1024, 2, 8, 100, 10);

        let mut opt = DataStoreReadOptions::default();
        opt.shift_width = 4;
        let mut store = DataStore::open(location, opt).unwrap();

        let values_iter = store.par_iter(3, 2, false, None);
        let (s, values): (Vec<_>, Vec<_>) = values_iter.unzip();
        let s =
            ndarray::concatenate(Axis(0), &s.iter().map(|x| x.view()).collect::<Vec<_>>()).unwrap();
        let values = ndarray::concatenate(
            Axis(0),
            &values.iter().map(|x| x.view()).collect::<Vec<_>>(),
        )
        .unwrap();

        for i in 0..values.shape()[0] {
            let start: i32 = 8;
            let end: i32 = 1024 + 8;

            assert!((-4..=4).any(|shift| {
                let start = (start + shift) as usize;
                let end = (end + shift) as usize;
                values.slice(s![i, .., ..]) == array.slice(s![i, start / 2..end / 2, ..])
                    && s.slice(s![i, ..]).to_vec() == seqs[i][start..end]
            }));
        }
    }
}
