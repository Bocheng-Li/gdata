//! Native reader for the gzip-compressed TFRecord files used by AlphaGenome.
//!
//! The public records are ordinary TFRecord frames containing a
//! `tf.train.Example`.  Feature values are serialized `TensorProto` messages,
//! but pulling TensorFlow in just to read them would add a very large runtime
//! dependency.  This module therefore implements the small protobuf subset
//! needed by these files.  A sequential shard reader feeds a bounded worker
//! pool for parallel decode/encoding, and a single writer commits completed
//! records in TFRecord order within each input file.

use anyhow::{bail, ensure, Context, Result};
use crossbeam_channel::{bounded, select};
use flate2::read::GzDecoder;
use half::bf16;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;

use super::data_store::{encode_segment_bf16, DataStoreBuilder};
use bed_utils::bed::GenomicRange;

#[derive(Debug)]
enum Feature<'a> {
    Bytes(&'a [u8]),
    Ints(Vec<u64>),
}

#[derive(Debug, Clone)]
struct TensorProto<'a> {
    dtype: u32,
    shape: Vec<usize>,
    content: Option<&'a [u8]>,
}

#[derive(Debug, Clone, Copy)]
enum WireValue<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
    Fixed32(()),
    Fixed64(()),
}

fn varint(data: &[u8], pos: &mut usize) -> Result<u64> {
    let mut value = 0u64;
    for shift in (0..70).step_by(7) {
        let byte = *data
            .get(*pos)
            .ok_or_else(|| anyhow::anyhow!("truncated protobuf varint"))?;
        *pos += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte < 0x80 {
            return Ok(value);
        }
    }
    bail!("protobuf varint is too long")
}

fn fields<'a>(
    data: &'a [u8],
    mut callback: impl FnMut(u32, u8, WireValue<'a>) -> Result<()>,
) -> Result<()> {
    let mut pos = 0usize;
    while pos < data.len() {
        let key = varint(data, &mut pos)?;
        let number = (key >> 3) as u32;
        let wire = (key & 7) as u8;
        ensure!(number != 0, "protobuf field number is zero");
        let value = match wire {
            0 => WireValue::Varint(varint(data, &mut pos)?),
            1 => {
                let end = pos.checked_add(8).context("protobuf fixed64 overflow")?;
                ensure!(end <= data.len(), "truncated protobuf fixed64 field");
                let value = WireValue::Fixed64(());
                pos = end;
                value
            }
            2 => {
                let size = usize::try_from(varint(data, &mut pos)?)
                    .context("protobuf length does not fit usize")?;
                let end = pos.checked_add(size).context("protobuf length overflow")?;
                ensure!(end <= data.len(), "truncated protobuf bytes field");
                let value = WireValue::Bytes(&data[pos..end]);
                pos = end;
                value
            }
            5 => {
                let end = pos.checked_add(4).context("protobuf fixed32 overflow")?;
                ensure!(end <= data.len(), "truncated protobuf fixed32 field");
                let value = WireValue::Fixed32(());
                pos = end;
                value
            }
            other => bail!("unsupported protobuf wire type {other}"),
        };
        callback(number, wire, value)?;
    }
    Ok(())
}

fn packed_varints(data: &[u8]) -> Result<Vec<u64>> {
    let mut pos = 0;
    let mut values = Vec::new();
    while pos < data.len() {
        values.push(varint(data, &mut pos)?);
    }
    Ok(values)
}

fn parse_feature(data: &[u8]) -> Result<Option<Feature<'_>>> {
    let mut output = None;
    fields(data, |number, wire, value| {
        match (number, wire, value) {
            // Feature.bytes_list.value[0].
            (1, 2, WireValue::Bytes(bytes_list)) if output.is_none() => {
                fields(bytes_list, |n, w, value| {
                    if n == 1 && w == 2 {
                        if let WireValue::Bytes(item) = value {
                            output = Some(Feature::Bytes(item));
                        }
                    }
                    Ok(())
                })?;
            }
            // Feature.int64_list.value, which can be packed or unpacked.
            (3, 2, WireValue::Bytes(int_list)) if output.is_none() => {
                let mut values = Vec::new();
                fields(int_list, |n, w, value| {
                    if n != 1 {
                        return Ok(());
                    }
                    match (w, value) {
                        (0, WireValue::Varint(v)) => values.push(v),
                        (2, WireValue::Bytes(packed)) => values.extend(packed_varints(packed)?),
                        _ => bail!("invalid Int64List value wire type"),
                    }
                    Ok(())
                })?;
                output = Some(Feature::Ints(values));
            }
            _ => {}
        }
        Ok(())
    })?;
    Ok(output)
}

fn parse_example<'a>(
    record: &'a [u8],
    wanted: &HashSet<&str>,
) -> Result<HashMap<String, Feature<'a>>> {
    let mut result = HashMap::new();
    fields(record, |number, wire, value| {
        if number != 1 || wire != 2 {
            return Ok(());
        }
        let WireValue::Bytes(features) = value else {
            return Ok(());
        };
        fields(features, |entry_number, entry_wire, entry_value| {
            if entry_number != 1 || entry_wire != 2 {
                return Ok(());
            }
            let WireValue::Bytes(entry) = entry_value else {
                return Ok(());
            };
            let mut key: Option<String> = None;
            let mut feature: Option<&[u8]> = None;
            fields(entry, |n, w, value| {
                match (n, w, value) {
                    (1, 2, WireValue::Bytes(bytes)) => {
                        key = Some(std::str::from_utf8(bytes)?.to_owned())
                    }
                    (2, 2, WireValue::Bytes(bytes)) => feature = Some(bytes),
                    _ => {}
                }
                Ok(())
            })?;
            if let (Some(key), Some(feature)) = (key, feature) {
                if wanted.contains(key.as_str()) {
                    if let Some(value) = parse_feature(feature)? {
                        result.insert(key, value);
                    }
                }
            }
            Ok(())
        })?;
        Ok(())
    })?;
    Ok(result)
}

fn require_feature<'a>(
    features: &'a HashMap<String, Feature<'a>>,
    name: &str,
) -> Result<&'a Feature<'a>> {
    features
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("record is missing feature {name:?}"))
}

fn tensor_proto(data: &[u8]) -> Result<TensorProto<'_>> {
    let mut dtype = None;
    let mut shape = None;
    let mut content = None;
    fields(data, |number, wire, value| {
        match (number, wire, value) {
            (1, 0, WireValue::Varint(v)) => dtype = Some(u32::try_from(v)?),
            (2, 2, WireValue::Bytes(shape_proto)) => shape = Some(tensor_shape(shape_proto)?),
            (4, 2, WireValue::Bytes(bytes)) => content = Some(bytes),
            _ => {}
        }
        Ok(())
    })?;
    Ok(TensorProto {
        dtype: dtype.ok_or_else(|| anyhow::anyhow!("TensorProto is missing dtype"))?,
        shape: shape.ok_or_else(|| anyhow::anyhow!("TensorProto is missing shape"))?,
        content,
    })
}

fn tensor_shape(data: &[u8]) -> Result<Vec<usize>> {
    let mut shape = Vec::new();
    fields(data, |number, wire, value| {
        if number != 2 || wire != 2 {
            return Ok(());
        }
        let WireValue::Bytes(dim) = value else {
            return Ok(());
        };
        let mut size = None;
        fields(dim, |n, w, value| {
            if n == 1 && w == 0 {
                if let WireValue::Varint(v) = value {
                    size = Some(usize::try_from(v).context("TensorShape dimension overflow")?);
                }
            }
            Ok(())
        })?;
        shape.push(size.ok_or_else(|| anyhow::anyhow!("TensorShape dimension has no size"))?);
        Ok(())
    })?;
    Ok(shape)
}

fn tensor_count(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |count, dim| {
        count
            .checked_mul(*dim)
            .context("TensorProto shape is too large")
    })
}

fn bytes_feature<'a>(features: &'a HashMap<String, Feature<'a>>, name: &str) -> Result<&'a [u8]> {
    match require_feature(features, name)? {
        Feature::Bytes(value) => Ok(value),
        Feature::Ints(_) => bail!("feature {name:?} is not a bytes list"),
    }
}

fn int_feature(features: &HashMap<String, Feature<'_>>, name: &str) -> Result<u64> {
    match require_feature(features, name)? {
        Feature::Ints(values) => values
            .first()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("feature {name:?} has no values")),
        Feature::Bytes(_) => bail!("feature {name:?} is not an int64 list"),
    }
}

fn decode_mask(data: &[u8], channels: usize) -> Result<Vec<bool>> {
    let tensor = tensor_proto(data)?;
    ensure!(
        tensor_count(&tensor.shape)? == channels,
        "mask element count does not match channels"
    );
    let content = tensor
        .content
        .context("mask TensorProto has no tensor_content")?;
    let mut mask = Vec::with_capacity(channels);
    match tensor.dtype {
        10 => {
            ensure!(
                content.len() >= channels,
                "mask tensor_content is truncated"
            );
            mask.extend(content[..channels].iter().map(|v| *v != 0));
        }
        3 | 9 | 22 | 23 => {
            let width = if tensor.dtype == 3 || tensor.dtype == 22 {
                4
            } else {
                8
            };
            ensure!(
                content.len() >= channels * width,
                "mask tensor_content is truncated"
            );
            for chunk in content.chunks_exact(width).take(channels) {
                let value = if width == 4 {
                    u32::from_le_bytes(chunk.try_into().unwrap()) as u64
                } else {
                    u64::from_le_bytes(chunk.try_into().unwrap())
                };
                mask.push(value != 0);
            }
        }
        dtype => bail!("unsupported mask TensorProto dtype {dtype}"),
    }
    Ok(mask)
}

fn decode_dna(data: &[u8], start: usize, length: usize) -> Result<Vec<u8>> {
    let tensor = tensor_proto(data)?;
    ensure!(
        tensor.shape.len() == 2 && tensor.shape[1] == 4,
        "DNA tensor must have shape [length, 4]"
    );
    let source_length = tensor.shape[0];
    ensure!(
        start
            .checked_add(length)
            .is_some_and(|end| end <= source_length),
        "DNA crop exceeds tensor length"
    );
    let count = tensor_count(&tensor.shape)?;
    let content = tensor
        .content
        .context("DNA TensorProto has no tensor_content")?;
    let mut sequence = Vec::with_capacity(length);
    match tensor.dtype {
        1 => {
            ensure!(
                content.len() >= count * 4,
                "DNA tensor_content is truncated"
            );
            for row in start..start + length {
                let mut best = 0usize;
                let mut best_value = f32::NEG_INFINITY;
                let mut sum = 0.0f32;
                for channel in 0..4 {
                    let offset = (row * 4 + channel) * 4;
                    let value = f32::from_le_bytes(content[offset..offset + 4].try_into().unwrap());
                    sum += value;
                    if value > best_value {
                        best_value = value;
                        best = channel;
                    }
                }
                sequence.push(if sum > 0.0 { best as u8 } else { 4 });
            }
        }
        14 => {
            ensure!(
                content.len() >= count * 2,
                "DNA tensor_content is truncated"
            );
            for row in start..start + length {
                let mut best = 0usize;
                let mut best_value = f32::NEG_INFINITY;
                let mut sum = 0.0f32;
                for channel in 0..4 {
                    let offset = (row * 4 + channel) * 2;
                    let value = bf16::from_bits(u16::from_le_bytes(
                        content[offset..offset + 2].try_into().unwrap(),
                    ))
                    .to_f32();
                    sum += value;
                    if value > best_value {
                        best_value = value;
                        best = channel;
                    }
                }
                sequence.push(if sum > 0.0 { best as u8 } else { 4 });
            }
        }
        dtype => bail!("unsupported DNA TensorProto dtype {dtype}"),
    }
    Ok(sequence)
}

fn decode_target(
    data: &[u8],
    start: usize,
    length: usize,
    channels: usize,
) -> Result<Vec<Vec<bf16>>> {
    let tensor = tensor_proto(data)?;
    ensure!(
        tensor.shape.len() == 2 && tensor.shape[1] == channels,
        "target tensor must have shape [length, {channels}]"
    );
    let source_length = tensor.shape[0];
    ensure!(
        start
            .checked_add(length)
            .is_some_and(|end| end <= source_length),
        "target crop exceeds tensor length"
    );
    let count = tensor_count(&tensor.shape)?;
    let content = tensor
        .content
        .context("target TensorProto has no tensor_content")?;
    let mut tracks = (0..channels)
        .map(|_| Vec::with_capacity(length))
        .collect::<Vec<_>>();
    match tensor.dtype {
        14 => {
            ensure!(
                content.len() >= count * 2,
                "target tensor_content is truncated"
            );
            for row in start..start + length {
                for channel in 0..channels {
                    let offset = (row * channels + channel) * 2;
                    tracks[channel].push(bf16::from_bits(u16::from_le_bytes(
                        content[offset..offset + 2].try_into().unwrap(),
                    )));
                }
            }
        }
        1 => {
            ensure!(
                content.len() >= count * 4,
                "target tensor_content is truncated"
            );
            for row in start..start + length {
                for channel in 0..channels {
                    let offset = (row * channels + channel) * 4;
                    tracks[channel].push(bf16::from_f32(f32::from_le_bytes(
                        content[offset..offset + 4].try_into().unwrap(),
                    )));
                }
            }
        }
        dtype => bail!("unsupported target TensorProto dtype {dtype}"),
    }
    Ok(tracks)
}

fn modality_config(modality: &str) -> Result<(&'static str, &'static str, usize)> {
    let normalized = modality.to_ascii_uppercase().replace('-', "_");
    match normalized.as_str() {
        "ATAC" => Ok(("atac", "atac_mask", 256)),
        "DNASE" => Ok(("dnase", "dnase_mask", 384)),
        "RNA" | "RNA_SEQ" | "RNASEQ" => Ok(("rna_seq", "rna_seq_mask", 768)),
        _ => bail!("unknown modality {modality:?}; expected ATAC, DNASE, or RNA_SEQ"),
    }
}

fn canonical_modality(modality: &str) -> Result<&'static str> {
    let normalized = modality.to_ascii_uppercase().replace('-', "_");
    match normalized.as_str() {
        "ATAC" => Ok("ATAC"),
        "DNASE" => Ok("DNASE"),
        "RNA" | "RNA_SEQ" | "RNASEQ" => Ok("RNA_SEQ"),
        _ => bail!("unknown modality {modality:?}; expected ATAC, DNASE, or RNA_SEQ"),
    }
}

fn read_record(gzip: &mut GzDecoder<File>) -> Result<Option<Vec<u8>>> {
    let mut header = [0u8; 12];
    if gzip.read(&mut header[..1])? == 0 {
        return Ok(None);
    }
    gzip.read_exact(&mut header[1..])
        .context("truncated TFRecord header")?;
    let length = u64::from_le_bytes(header[..8].try_into().unwrap());
    let length = usize::try_from(length).context("TFRecord payload is too large")?;
    let mut payload = vec![0u8; length];
    gzip.read_exact(&mut payload)?;
    let mut footer = [0u8; 4];
    gzip.read_exact(&mut footer)?;
    Ok(Some(payload))
}

struct RecordFrame {
    file_index: usize,
    record_index: usize,
    payload: Vec<u8>,
    permit: PayloadPermit,
}

/// A permit held for the complete lifetime of one payload/result.
///
/// The permit must not be returned when decoding finishes: an encoded result
/// can still occupy hundreds of megabytes while it waits for the ordered
/// writer.  Returning it only when this guard is dropped bounds both decoded
/// payloads and out-of-order encoded results.
struct PayloadPermit {
    tx: crossbeam_channel::Sender<()>,
}

impl Drop for PayloadPermit {
    fn drop(&mut self) {
        let _ = self.tx.send(());
    }
}

struct EncodedRecord {
    file_index: usize,
    record_index: usize,
    chromosome: String,
    start: u64,
    end: u64,
    target_length: usize,
    encoded: Vec<u8>,
    track_keys: Vec<String>,
}

enum ReaderMessage {
    Done { file_index: usize, count: usize },
    Error(anyhow::Error),
}

enum WorkerMessage {
    Record(EncodedRecord, PayloadPermit),
    Error(anyhow::Error),
}

fn worker_count(explicit: Option<usize>) -> usize {
    explicit
        .or_else(|| {
            std::env::var("GDATA_NUM_THREADS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
        })
        .or_else(|| {
            std::env::var("RAYON_NUM_THREADS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
        })
        .filter(|value| *value > 0)
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|value| value.get())
                .unwrap_or(1)
        })
        .max(1)
        .min(16)
}

fn decode_and_encode_record(
    payload: Vec<u8>,
    file_index: usize,
    record_index: usize,
    target_name: &'static str,
    mask_name: &'static str,
    channels: usize,
    modality_prefix: &'static str,
    wanted: &HashSet<&str>,
) -> Result<EncodedRecord> {
    let features = parse_example(&payload, wanted)?;
    let dna_proto = bytes_feature(&features, "dna_sequence")?;
    let target_proto = bytes_feature(&features, target_name)?;
    let mask_proto = bytes_feature(&features, mask_name)?;
    let dna_tensor = tensor_proto(dna_proto)?;
    ensure!(
        dna_tensor.shape.len() == 2 && dna_tensor.shape[1] == 4,
        "DNA tensor must have shape [length, 4]"
    );
    let source_length = dna_tensor.shape[0];
    let target_tensor = tensor_proto(target_proto)?;
    ensure!(
        source_length >= target_tensor.shape[0],
        "DNA length is shorter than target length"
    );
    let target_length = target_tensor.shape[0];
    ensure!(target_length > 0, "target length must be positive");
    let crop_start = if source_length > target_length {
        (source_length - target_length) / 2
    } else {
        0
    };
    let output_start = int_feature(&features, "interval/start")? + crop_start as u64;
    let output_end = output_start + target_length as u64;
    ensure!(
        target_tensor.shape.len() == 2 && target_tensor.shape[1] == channels,
        "target tensor has unexpected shape"
    );
    let sequence = decode_dna(dna_proto, crop_start, target_length)?;
    let mut tracks = decode_target(target_proto, 0, target_length, channels)?;
    let mask = decode_mask(mask_proto, channels)?;
    for (index, valid) in mask.into_iter().enumerate() {
        if !valid {
            tracks[index].fill(bf16::ZERO);
        }
    }
    let chromosome = match require_feature(&features, "interval/chromosome")? {
        Feature::Bytes(bytes) => std::str::from_utf8(bytes)?.to_owned(),
        Feature::Ints(_) => bail!("interval/chromosome is not a bytes feature"),
    };
    let track_keys = (0..channels)
        .map(|index| format!("{modality_prefix}:track_{index:04}"))
        .collect::<Vec<_>>();
    let track_values = track_keys.iter().cloned().zip(tracks).collect::<Vec<_>>();
    let encoded = encode_segment_bf16(&sequence, &track_values, target_length, target_length)?;
    Ok(EncodedRecord {
        file_index,
        record_index,
        chromosome,
        start: output_start,
        end: output_end,
        target_length,
        encoded,
        track_keys,
    })
}

fn convert_paths(
    paths: &[PathBuf],
    output: &Path,
    modality: &str,
    split: &str,
    temp_dir: Option<&Path>,
    max_records: Option<usize>,
    num_threads: Option<usize>,
    overwrite: bool,
    chunk_tracks: Option<usize>,
) -> Result<usize> {
    ensure!(!paths.is_empty(), "at least one TFRecord path is required");
    ensure!(
        max_records != Some(0),
        "max_records must be positive when provided"
    );
    let (target_name, mask_name, channels) = modality_config(modality)?;
    let modality_prefix = canonical_modality(modality)?;
    let split = split.to_ascii_lowercase();
    ensure!(
        split == "train" || split == "valid" || split == "validation",
        "split must be train or valid"
    );
    if output.exists() && !overwrite {
        bail!(
            "output already exists: {}; pass overwrite=True to replace it",
            output.display()
        );
    }
    if overwrite && output.exists() {
        std::fs::remove_file(output)
            .with_context(|| format!("failed to remove {}", output.display()))?;
    }
    let temp_parent = temp_dir.unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(temp_parent)?;
    let tmp = tempfile::Builder::new()
        .prefix("gdata_tfrecord_")
        .tempdir_in(temp_parent)?;
    let partial = output.with_extension("gdata.partial");
    if partial.exists() {
        if overwrite {
            std::fs::remove_file(&partial)?;
        } else {
            bail!(
                "incomplete output already exists: {}; pass overwrite=True",
                partial.display()
            );
        }
    }
    let mut builder: Option<DataStoreBuilder> = None;
    let wanted: HashSet<&str> = [
        "dna_sequence",
        target_name,
        mask_name,
        "interval/chromosome",
        "interval/start",
        "interval/end",
    ]
    .into_iter()
    .collect();
    let n_workers = worker_count(num_threads);
    // A TFRecord payload is hundreds of MB to several GB.  Keep the queue
    // deliberately small so parallelism cannot turn into an unbounded memory
    // multiplier (especially for the 768-channel RNA-seq records).
    let queue_capacity = n_workers.saturating_mul(2).clamp(2, 8);
    let (task_tx, task_rx) = bounded::<RecordFrame>(queue_capacity);
    let (result_tx, result_rx) = bounded::<WorkerMessage>(queue_capacity);
    let (control_tx, control_rx) = bounded::<ReaderMessage>(paths.len().max(1));
    // A reader must acquire a permit before allocating a decompressed
    // TFRecord payload.  Without this semaphore, one reader per shard could
    // simultaneously hold hundreds of large train records in memory before
    // the bounded task queue has a chance to apply back-pressure.
    let max_inflight = n_workers;
    let (permit_tx, permit_rx) = bounded::<()>(max_inflight);
    for _ in 0..max_inflight {
        permit_tx.send(()).expect("permit channel is open");
    }
    let failed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop = Arc::new(AtomicBool::new(false));

    // Use one sequential reader for the shards.  Decode/encode remains fully
    // parallel in the worker pool, while this ordering prevents later shards
    // from filling the reorder buffer while an earlier shard is still being
    // read.  In particular, it keeps the number of large encoded results in
    // memory bounded by `max_inflight` rather than by the number of shards.
    let mut reader_handles = Vec::with_capacity(1);
    let task_tx_reader = task_tx.clone();
    let control_tx_reader = control_tx.clone();
    let failed_reader = failed.clone();
    let stop_reader = stop.clone();
    let permit_rx_reader = permit_rx.clone();
    let permit_tx_reader = permit_tx.clone();
    let paths_reader = paths.to_vec();
    reader_handles.push(thread::spawn(move || {
        let outcome = (|| -> Result<()> {
            for (file_index, path) in paths_reader.into_iter().enumerate() {
                if failed_reader.load(Ordering::Relaxed) || stop_reader.load(Ordering::Relaxed) {
                    break;
                }
                let file = File::open(&path)
                    .with_context(|| format!("failed to open TFRecord {}", path.display()))?;
                let mut gzip = GzDecoder::new(file);
                let mut count = 0usize;
                loop {
                    if failed_reader.load(Ordering::Relaxed) || stop_reader.load(Ordering::Relaxed)
                    {
                        break;
                    }
                    permit_rx_reader
                        .recv()
                        .map_err(|_| anyhow::anyhow!("TFRecord permit queue closed"))?;
                    let permit = PayloadPermit {
                        tx: permit_tx_reader.clone(),
                    };
                    let Some(payload) = read_record(&mut gzip)? else {
                        break;
                    };
                    if task_tx_reader
                        .send(RecordFrame {
                            file_index,
                            record_index: count,
                            payload,
                            permit,
                        })
                        .is_err()
                    {
                        return Err(anyhow::anyhow!("TFRecord worker queue closed"));
                    }
                    count += 1;
                }
                let _ = control_tx_reader.send(ReaderMessage::Done { file_index, count });
            }
            Ok(())
        })();
        if let Err(error) = outcome {
            let _ = control_tx_reader.send(ReaderMessage::Error(error));
        }
    }));
    drop(task_tx);
    drop(control_tx);

    let mut worker_handles = Vec::with_capacity(n_workers);
    for _ in 0..n_workers {
        let task_rx = task_rx.clone();
        let result_tx = result_tx.clone();
        let wanted = wanted.clone();
        let failed = failed.clone();
        let stop = stop.clone();
        worker_handles.push(thread::spawn(move || {
            for frame in task_rx.iter() {
                if failed.load(Ordering::Relaxed) || stop.load(Ordering::Relaxed) {
                    continue;
                }
                let permit = frame.permit;
                let result = decode_and_encode_record(
                    frame.payload,
                    frame.file_index,
                    frame.record_index,
                    target_name,
                    mask_name,
                    channels,
                    modality_prefix,
                    &wanted,
                );
                let message = match result {
                    Ok(record) => WorkerMessage::Record(record, permit),
                    Err(error) => {
                        failed.store(true, Ordering::Relaxed);
                        WorkerMessage::Error(error)
                    }
                };
                if result_tx.send(message).is_err() {
                    break;
                }
            }
        }));
    }
    drop(result_tx);

    let mut pending =
        std::collections::BTreeMap::<(usize, usize), (EncodedRecord, PayloadPermit)>::new();
    let mut file_counts = vec![None; paths.len()];
    let mut next_file = 0usize;
    let mut next_record = 0usize;
    let mut records = 0usize;
    let mut result_open = true;
    let mut control_open = true;
    let run_result = (|| -> Result<usize> {
        while next_file < paths.len() {
            let mut progressed = true;
            while progressed {
                progressed = false;
                if let Some((record, _permit)) = pending.remove(&(next_file, next_record)) {
                    let name = format!("{}:{}-{}", record.chromosome, record.start, record.end);
                    if builder.is_none() {
                        builder = Some(DataStoreBuilder::new_with_chunk_tracks(
                            &tmp,
                            record.target_length as u32,
                            1,
                            0,
                            chunk_tracks,
                        )?);
                    }
                    let segment = GenomicRange::from_str(&name)
                        .map_err(|e| anyhow::anyhow!("invalid generated segment {name}: {e:?}"))?;
                    builder
                        .as_mut()
                        .expect("builder should be initialized")
                        .add_encoded_segment(segment, record.encoded, record.track_keys)?;
                    records += 1;
                    if max_records.is_some_and(|limit| records >= limit) {
                        stop.store(true, Ordering::Relaxed);
                        return Ok(records);
                    }
                    next_record += 1;
                    progressed = true;
                } else if next_file < paths.len()
                    && file_counts[next_file].is_some_and(|count| next_record >= count)
                {
                    next_file += 1;
                    next_record = 0;
                    progressed = true;
                }
            }
            if next_file >= paths.len() {
                break;
            }
            select! {
                recv(result_rx) -> message => match message {
                    Ok(WorkerMessage::Record(record, permit)) => {
                        pending.insert((record.file_index, record.record_index), (record, permit));
                    }
                    Ok(WorkerMessage::Error(error)) => return Err(error),
                    Err(_) => { result_open = false; }
                },
                recv(control_rx) -> message => match message {
                    Ok(ReaderMessage::Done { file_index, count }) => file_counts[file_index] = Some(count),
                    Ok(ReaderMessage::Error(error)) => return Err(error),
                    Err(_) => { control_open = false; }
                },
            }
            if !result_open && !control_open && next_file < paths.len() {
                bail!("TFRecord reader workers exited before conversion completed");
            }
        }
        ensure!(records > 0, "no TFRecord records were converted");
        Ok(records)
    })();

    // Closing the receive ends makes workers/readers terminate promptly on an
    // error, then join every thread before returning to Python.
    // Release permits held by out-of-order results before joining readers;
    // otherwise a reader waiting for a permit could keep the join blocked.
    drop(pending);
    drop(result_rx);
    drop(control_rx);
    for handle in reader_handles {
        let _ = handle.join();
    }
    for handle in worker_handles {
        let _ = handle.join();
    }
    let records = run_result?;
    ensure!(records > 0, "no TFRecord records were converted");
    builder
        .take()
        .ok_or_else(|| anyhow::anyhow!("no records were converted"))?
        .finish(&partial)?;
    std::fs::rename(&partial, output)
        .with_context(|| format!("failed to finalize {}", output.display()))?;
    Ok(records)
}

/// Convert one or more gzip TFRecord files directly to a gdata file.
///
/// This native path performs TFRecord framing, protobuf/TensorProto decoding,
/// DNA encoding, mask handling, and gdata compression in one pass.  Record
/// decoding/encoding uses multiple CPU threads; `GDATA_NUM_THREADS` (or
/// `RAYON_NUM_THREADS`) controls the worker count.  It avoids TensorFlow,
/// FASTA, intermediate W5Z files, and Python tensor copies.
#[pyfunction]
#[pyo3(signature = (paths, output, modality, split, *, temp_dir=None, max_records=None, num_threads=None, overwrite=false, chunk_tracks=None))]
pub fn convert_tfrecord_to_gdata(
    paths: Vec<PathBuf>,
    output: PathBuf,
    modality: String,
    split: String,
    temp_dir: Option<PathBuf>,
    max_records: Option<usize>,
    num_threads: Option<usize>,
    overwrite: bool,
    chunk_tracks: Option<usize>,
) -> Result<usize> {
    convert_paths(
        &paths,
        &output,
        &modality,
        &split,
        temp_dir.as_deref(),
        max_records,
        num_threads,
        overwrite,
        chunk_tracks,
    )
}
