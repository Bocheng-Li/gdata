use anyhow::{bail, ensure, Context, Result};
use bed_utils::bed::{BEDLike, GenomicRange};
use bincode::{Decode, Encode};
use half::bf16;
use indexmap::{IndexMap, IndexSet};
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use ndarray::{s, Array1, Array2, Array3, ArrayView2, ArrayView3, ArrayViewMut3, Axis};
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
use std::sync::{Arc, Mutex};

use crate::dataloader::generic::{split_n_with_batch_size, ParallelLoader, ReBatch};
use crate::w5z::W5Z;

use super::super::generic::{compress_data_zst, decompress_data_zst};

/// Dimension: (sequence, experiment)
#[derive(Debug, Clone, PartialEq)]
pub struct Values(Array3<bf16>);

impl From<Array3<bf16>> for Values {
    fn from(arr: Array3<bf16>) -> Self {
        Values(arr.as_standard_layout().to_owned())
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
    store_pad: bool,
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
    /// Additional padding to extend sequences beyond segment boundaries.
    pub sequence_padding: u32,
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
            sequence_padding: 0,
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
    pub fn store_pad(&self) -> bool {
        self.inner.metadata.store_pad
    }
    pub fn open(path: impl AsRef<Path>, mut read_opts: DataStoreReadOptions) -> Result<Self> {
        let mut file = File::open(&path).context("Failed to open data store file")?;
        let metadata = read_metadata(&mut file)?;

        if metadata.store_pad {
            ensure!(
                read_opts.shift_width <= metadata.padding,
                "Shift must be less than or equal to padding"
            );
        }

        let mut aggregate_size = None;
        let mut out_resolution = metadata.resolution;
        if let Some(read_resolution) = read_opts.read_resolution {
            ensure!(
                read_resolution % metadata.resolution == 0,
                "Read resolution must be a multiple of resolution"
            );
            aggregate_size = Some(read_resolution / metadata.resolution);
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

    pub fn read(&mut self, region: &GenomicRange) -> Option<(Sequence, Values)> {
        let offset = self.inner.metadata.segment_index.get(region)?;
        let mut buffer = vec![0; offset.0 .1 as usize];
        self.inner
            .file
            .read_exact_at(&mut buffer, offset.0 .0 as u64)
            .expect("read failed");

        // Deserialize the sequence and values
        let buffer = decompress_data_zst(&buffer);
        let (seq, arr): (Vec<u8>, Array2<bf16>) =
            bincode::serde::decode_from_slice(&buffer, bincode::config::standard())
                .expect("decode failed")
                .0;
        let seq = Array1::from_vec(seq).insert_axis(Axis(0));
        // arr need to be transposed to match the expected shape (sequence, experiment)
        let arr = arr.t().insert_axis(Axis(0));

        // Apply random shifting
        let mut shift = self.read_opts.shift_width as i32;
        let res = self.resolution();
        if self.read_opts.shift_width != 0 {
            shift = self.read_opts.rng.random_range(-shift..=shift) / res as i32 * res as i32;
        }
        
        // Calculate sequence boundaries with additional padding
        let padding = self.read_opts.sequence_padding as i32;
        let base_start = (shift + self.n_pad() as i32) as i32;
        let seq_start = (base_start - padding).max(0) as usize;
        let seq_len_with_padding = self.sequence_length() as usize + (2 * padding) as usize;
        let seq_end = (seq_start + seq_len_with_padding).min(seq.len_of(Axis(1)));
        
        let arr_start = if self.inner.metadata.store_pad {
            seq_start / res as usize
        } else {
            0
        };
        let arr_len_base = (self.sequence_length() / res) as usize;
        let arr_len_with_padding = if self.inner.metadata.store_pad {
            // When store_pad=true, we can extend values according to sequence padding
            arr_len_base + (2 * padding / res as i32) as usize
        } else {
            // When store_pad=false, values don't have padding, so keep original length
            arr_len_base
        };
        let arr_end = (arr_start + arr_len_with_padding).min(arr.len_of(Axis(1)));
        
        let mut seq = seq.slice(s![.., seq_start..seq_end]);
        let mut arr = arr.slice(s![.., arr_start..arr_end, ..]).to_owned();

        // Apply value aggregation
        if let Some(agg) = self.aggregate_size {
            arr = aggregate_by_length(arr, agg as usize);
        }

        // Apply splitting
        if let Some(split) = self.read_opts.split_size {
            seq = split_sequence(seq, split as usize).unwrap();
            arr = split_data(arr, (split / self.out_resolution) as usize).unwrap()
        }

        // Trim the output values
        let v_len = self.read_opts.value_length.unwrap_or(self.output_length());
        let trim_start = (self.output_length() - v_len) / self.out_resolution / 2;
        let trim_end = trim_start + (v_len / self.out_resolution);
        let mut arr = arr.slice_mut(s![.., trim_start as usize..trim_end as usize, ..]);

        // Aplly scaling and clamping
        transform(
            arr.view_mut(),
            self.read_opts.scale_value,
            self.read_opts.clamp_value_max,
        );

        Some((seq.into(), arr.view().into()))
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
}

pub struct DataStoreIter {
    segments: VecDeque<GenomicRange>,
    store: DataStore,
}

impl Iterator for DataStoreIter {
    type Item = (Array2<u8>, Array3<f32>);

    fn next(&mut self) -> Option<Self::Item> {
        let segment = self.segments.pop_front()?;
        let (seq, values) = self.store.read(&segment).unwrap();
        Some((seq.into(), values.into()))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.segments.len();
        (len, Some(len))
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
    store_pad: bool,
    pub(crate) segments: IndexMap<GenomicRange, PathBuf>,
    pub(crate) data_keys: IndexSet<String>,
}

impl DataStoreBuilder {
    /// Create an empty data store at the specified location.
    pub fn new(
        location: impl AsRef<Path>,
        sequence_length: u32,
        resolution: u32,
        padding: u32,
        store_pad: bool,
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
            sequence_length,
            resolution,
            padding,
            store_pad,
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

    fn add_values(
        &mut self,
        key: impl Into<String>,
        data: impl IndexedParallelIterator<Item = (GenomicRange, Vec<bf16>)>,
    ) -> Result<()> {
        let key = key.into();
        if self.data_keys.contains(&key) {
            bail!("Data key {} already exists", &key);
        }
        self.data_keys.insert(key);

        let val_len = if self.store_pad {
            (self.total_sequence_length() / self.resolution) as usize
        } else {
            (self.sequence_length / self.resolution) as usize
        };
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
        })
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
        let seq_len_total = self.total_sequence_length() as isize;
        let seq_len_core = self.sequence_length as isize;
        let resolution = self.resolution as usize;
        let store_pad = self.store_pad;
        let values = regions.into_par_iter().map(move |range| {
            let values = data.get(range.chrom()).unwrap();
            let (start, end) = if store_pad {
                let start = range.start() as isize - padding;
                let end = start + seq_len_total;
                (start, end)
            } else {
                let start = range.start() as isize;
                let end = start + seq_len_core;
                (start, end)
            };
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
            store_pad: self.store_pad,
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

/// Aggregate the values along the sequence axis (axis 1).
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

/// Perform in-place scaling and clamping on the values.
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
        let mut store = DataStoreBuilder::new(&tmp, sequence_length, resolution, padding, true).unwrap();

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
