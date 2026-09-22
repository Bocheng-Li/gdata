use anyhow::{Context, Result};
use bed_utils::bed::{BEDLike, GenomicRange};
use indexmap::IndexMap;
use indicatif::{ProgressIterator, ProgressStyle};
use itertools::Itertools;
use noodles::fasta::{
    fai::Index,
    io::{indexed_reader::Builder, IndexedReader},
};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::{BTreeMap, HashSet};
use std::io::BufReader;
use std::path::PathBuf;
use std::str::FromStr;
use std::{fs::File, io::BufRead, path::Path};
use tempfile::TempDir;

use super::data_store::DataStoreBuilder;
use crate::w5z::W5Z;

/** Represents a builder for genomic data, allowing for the creation and management of genomic datasets.

    This struct provides methods to create a new genomic dataset, open an existing dataset,
    and manage genomic data chunks. It supports operations like adding files, retrieving chromosome
    information, and iterating over data chunks.

    Parameters
    ----------
    location
        The directory where the genomic data will be stored.
    genome_fasta
        The path to the FASTA file containing the genome sequences.
    window_size
        The size of the genomic windows to be processed.
    segments
        Optional list of genomic segments to include in the dataset.
        The genomic segments should be provided as strings in the format "chrom:start-end".
        If None, the entire genome will be used.
    step_size
        The step size for sliding the window across the genome (default is None, which uses `window_size`).
    resolution
        The resolution of the stored genomic data (default is 1).
    padding
        The amount of padding to add around each genomic segment (default is 0).
        Extra padding allows shifting the window within the padded region during
        data loading.
    chroms
        A list of chromosomes to include in the dataset. If None, all chromosomes in the FASTA file will be used.
    temp_dir
        Optional temporary directory for intermediate files. If None, a system temporary directory will be used.

    See Also
    --------
    GenomeDataLoader

    Examples
    --------
    >>> from gdata import as GenomeDataBuilder
    >>> regions = ["chr11:35041782-35238390", "chr11:35200000-35300000"]
    >>> tracks = {'DNase:CD14-positive monocyte': 'ENCSR464ETX.w5z', 'DNase:keratinocyte': 'ENCSR000EPQ.w5z'}
    >>> builder = GenomeDataBuilder("genome.gdata", 'genome.fa.gz', segments=regions, window_size=196_608, resolution=128)
    >>> builder.add_files(tracks)
    >>> builder.finish()
*/
#[pyclass]
pub struct GenomeDataBuilder {
    location: PathBuf,
    store_builder: Option<(DataStoreBuilder, TempDir)>,
}

impl GenomeDataBuilder {
    fn store(&self) -> &DataStoreBuilder {
        &self
            .store_builder
            .as_ref()
            .expect("data store has been moved")
            .0
    }

    fn store_mut(&mut self) -> &mut DataStoreBuilder {
        &mut self
            .store_builder
            .as_mut()
            .expect("data store has been moved")
            .0
    }

    fn create_streaming(
        location: PathBuf,
        window_size: u32,
        resolution: u32,
        padding: u32,
        temp_dir: Option<PathBuf>,
    ) -> Result<Self> {
        let tmp_dir = if let Some(dir) = temp_dir {
            tempfile::Builder::new()
                .prefix("gdata_tmp")
                .tempdir_in(dir)?
        } else {
            tempfile::Builder::new().prefix("gdata_tmp").tempdir()?
        };
        let store_builder = DataStoreBuilder::new(&tmp_dir, window_size, resolution, padding)?;
        Ok(Self {
            location,
            store_builder: Some((store_builder, tmp_dir)),
        })
    }
}

#[pymethods]
impl GenomeDataBuilder {
    #[new]
    #[pyo3(
        signature = (
            location, genome_fasta, window_size, *, segments=None, step_size=None, resolution=32,
            padding=0, chroms=None, temp_dir=None,
        ),
        text_signature = "($self, location, genome_fasta, window_size, *, segments=None,
            step_size=None, resolution=32, padding=0, chroms=None, temp_dir=None)"
    )]
    pub fn new(
        location: PathBuf,
        genome_fasta: PathBuf,
        window_size: u32,
        segments: Option<Vec<String>>,
        step_size: Option<u32>,
        resolution: u32,
        padding: u32,
        chroms: Option<Vec<String>>,
        temp_dir: Option<PathBuf>,
    ) -> Result<Self> {
        let tmp_dir = if let Some(dir) = temp_dir {
            tempfile::Builder::new()
                .prefix("gdata_tmp")
                .tempdir_in(dir)?
        } else {
            tempfile::Builder::new().prefix("gdata_tmp").tempdir()?
        };
        let mut store_builder = DataStoreBuilder::new(&tmp_dir, window_size, resolution, padding)?;
        let mut fasta_reader = open_fasta(genome_fasta)?;

        // Retrieve chromosome sizes from the FASTA index
        let mut chrom_sizes: BTreeMap<String, u64> = fasta_reader
            .index()
            .as_ref()
            .iter()
            .map(|rec| (rec.name().to_string(), rec.length()))
            .collect();
        if let Some(chroms) = chroms {
            let chroms: HashSet<_> = chroms.into_iter().collect();
            chrom_sizes.retain(|chrom, _| chroms.contains(chrom));
        }

        let segments: Vec<_> = if let Some(s) = segments {
            let mut all_chroms = HashSet::new();
            let s = s
                .into_iter()
                .map(|s| {
                    let mut g = GenomicRange::from_str(&s).unwrap();
                    all_chroms.insert(g.chrom().to_string());
                    expand_segment(&mut g, window_size as u64, &chrom_sizes);
                    g
                })
                .unique() // Ensure segments are unique
                .collect();
            chrom_sizes.retain(|chrom, _| all_chroms.contains(chrom));
            s
        } else {
            let step_size = step_size.unwrap_or(window_size);
            get_genome_segments(&chrom_sizes, window_size as u64, step_size as u64)
                .flat_map(|(_, iter)| iter)
                .collect()
        };
        store_builder.add_segments(segments, &mut fasta_reader)?;

        Ok(Self {
            location,
            store_builder: Some((store_builder, tmp_dir)),
        })
    }

    /// Create a builder for one-pass conversion from already segmented data.
    ///
    /// Unlike ``GenomeDataBuilder(...)``, this constructor does not require a
    /// FASTA file or a complete segment list up front.  Call ``add_segment``
    /// once per record, then call ``finish``.
    #[staticmethod]
    #[pyo3(
        signature = (location, window_size, *, resolution=1, padding=0, temp_dir=None),
        text_signature = "(location, window_size, *, resolution=1, padding=0, temp_dir=None)"
    )]
    pub fn streaming(
        location: PathBuf,
        window_size: u32,
        resolution: u32,
        padding: u32,
        temp_dir: Option<PathBuf>,
    ) -> Result<Self> {
        Self::create_streaming(location, window_size, resolution, padding, temp_dir)
    }

    /** Returns the keys (track names) in the dataset.

       This method retrieves all keys from the dataset, which are typically the names of files
       containing genomic data.

       Returns
       -------
       list[str]
           A list of keys as strings.
    */
    pub fn tracks(&self) -> Vec<String> {
        self.store().data_keys.iter().cloned().collect::<Vec<_>>()
    }

    /** Returns the segments of the genome as a vector of strings.

       Returns
       -------
       list[str]
           A list of segment strings representing genomic ranges.
    */
    fn segments(&self) -> Vec<String> {
        self.store()
            .segments
            .keys()
            .map(|x| x.pretty_show())
            .collect()
    }

    /** Adds w5z files to the dataset.

       This method processes a batch of files, each associated with a key, and adds them to the genomic data.

       Parameters
       ----------
       files : dict[str, Path]
           A dictionary mapping keys to file paths.
    */
    #[pyo3(
        signature = (files),
        text_signature = "($self, files)",
    )]
    pub fn add_files(&mut self, py: Python<'_>, files: IndexMap<String, PathBuf>) -> Result<()> {
        let style = ProgressStyle::with_template(
            "Adding files: [{elapsed}] {wide_bar:.cyan/blue} {human_pos}/{human_len} (eta: {eta})",
        )
        .unwrap();
        files
            .into_iter()
            .progress_with_style(style)
            .try_for_each(|(key, path)| {
                py.check_signals()?;
                self.add_file(&key, path)?;
                Ok(())
            })
    }

    /** Adds a single file to the dataset.

       This method processes a file associated with a key and adds it to the genomic data.
       The data is read from a W5Z file, and stored in chunks. Each chunk has the
       shape (num_segments, num_columns), where num_columns is determined by the
       resolution. The values are averaged if the resolution is greater than 1.

       Parameters
       ----------
       key : str
           The key associated with the file.
       w5z : Path
           The path to the W5Z file containing genomic data.
    */
    #[pyo3(
        signature = (key, path),
        text_signature = "($self, key, path)",
    )]
    pub fn add_file(&mut self, key: &str, path: PathBuf) -> Result<()> {
        let w5z = W5Z::open(path)?;
        self.store_mut().add_w5z(key, w5z)
    }

    /** Adds already-segmented values directly, without creating a W5Z file.

       Parameters
       ----------
       key : str
           Track name stored in the resulting gdata file.
       values : numpy.ndarray
           A float32 array with shape ``(batch_segments, values_per_segment)``.
           Rows must correspond to consecutive entries in ``segments``.
       start_index : int
           Index of the first segment represented by ``values``. The first
           batch for a track must use 0; subsequent batches must immediately
           follow the previous batch.

       ``values_per_segment`` is ``(window_size + 2 * padding) / resolution``.
       This method is intended for streaming conversion: callers can submit a
       small batch at a time and avoid holding a complete track in memory.
    */
    #[pyo3(
        signature = (key, values, start_index=0),
        text_signature = "($self, key, values, start_index=0)",
    )]
    pub fn add_segment_data(
        &mut self,
        key: &str,
        values: PyReadonlyArray2<'_, f32>,
        start_index: usize,
    ) -> Result<()> {
        self.store_mut()
            .add_segment_data(key, start_index, values.as_array())
    }

    /** Add one already-segmented sequence and all track values.

       This method is intended for one-pass TFRecord conversion.  ``segment``
       must be a unique ``chrom:start-end`` string whose length equals the
       builder window size.  ``sequence`` is a one-dimensional uint8 array
       using the encoding A=0, C=1, G=2, T=3, N=4.  ``track_values`` is a
       dictionary mapping each track name to a contiguous or strided float32
       vector of ``(window_size + 2 * padding) / resolution`` values.

       The first call fixes the track names and their order.  Every subsequent
       call must provide exactly the same names in that order and one vector
       per track.  Data
       are appended immediately to temporary segment files; no FASTA, W5Z, or
       second pass over the source records is needed.
    */
    #[pyo3(
        signature = (segment, sequence, track_values),
        text_signature = "($self, segment, sequence, track_values)"
    )]
    pub fn add_segment(
        &mut self,
        segment: &str,
        sequence: PyReadonlyArray1<'_, u8>,
        track_values: &Bound<'_, PyDict>,
    ) -> Result<()> {
        let range = GenomicRange::from_str(segment)
            .map_err(|error| anyhow::anyhow!("invalid segment range {segment}: {error:?}"))?;
        let mut arrays = Vec::with_capacity(track_values.len());
        for (py_key, py_value) in track_values.iter() {
            let key: String = py_key.extract()?;
            let values: PyReadonlyArray1<'_, f32> = py_value.extract()?;
            arrays.push((key, values));
        }
        let views = arrays
            .iter()
            .map(|(key, values)| (key.clone(), values.as_array()))
            .collect();
        self.store_mut()
            .add_segment(range, sequence.as_array(), views)
    }

    /** Finalizes the dataset creation.

       This method finalizes the dataset by writing all data to the specified location.
       After calling this method, the builder cannot be used to add more data.

       Returns
       -------
       None
    */
    pub fn finish(&mut self) -> Result<()> {
        let (store_builder, tmp) = self
            .store_builder
            .take()
            .expect("data store has been moved");
        store_builder.finish(&self.location)?;
        tmp.close()?;
        Ok(())
    }
}

fn expand_segment(
    segment: &mut GenomicRange,
    window_size: u64,
    chrom_sizes: &BTreeMap<String, u64>,
) {
    if segment.len() < window_size {
        let start = segment
            .start()
            .saturating_sub((window_size - segment.len()) / 2);
        let end = start + window_size;
        segment.set_start(start);
        segment.set_end(end);
    }
    let size = chrom_sizes.get(segment.chrom()).unwrap();
    segment.set_end(segment.end().min(*size));
}

/// Return the segments of the genome as an iterator.
fn get_genome_segments(
    chrom_sizes: &BTreeMap<String, u64>,
    window_size: u64,
    step_size: u64,
) -> impl Iterator<Item = (String, impl Iterator<Item = GenomicRange> + '_)> + '_ {
    chrom_sizes.iter().map(move |(chrom, &size)| {
        let mut start = 0;
        let iter = std::iter::from_fn(move || {
            if start >= size {
                return None;
            }
            let end = (start + window_size).min(size);
            let range = GenomicRange::new(chrom, start, end);
            start += step_size;
            Some(range)
        });
        (chrom.clone(), iter)
    })
}

fn open_fasta(
    genome_fasta: impl AsRef<Path>,
) -> Result<IndexedReader<noodles::fasta::io::BufReader<File>>> {
    let fai = PathBuf::from(format!("{}.fai", genome_fasta.as_ref().display()));
    let index = if fai.exists() {
        noodles::fasta::fai::io::Reader::new(BufReader::new(File::open(&fai)?))
            .read_index()
            .context("Failed to read FASTA index")?
    } else {
        let reader = noodles::fasta::io::reader::Builder::default()
            .build_from_path(&genome_fasta)
            .with_context(|| {
                format!(
                    "Failed to open FASTA file: {}",
                    genome_fasta.as_ref().display()
                )
            })?;
        create_index(reader.into_inner())?
    };

    let reader = Builder::default()
        .set_index(index)
        .build_from_path(&genome_fasta)?;
    Ok(reader)
}

fn create_index<R: BufRead>(reader: R) -> Result<Index> {
    let mut index = Vec::new();
    let mut reader = noodles::fasta::io::Indexer::new(reader);
    while let Some(record) = reader.index_record()? {
        index.push(record);
    }
    Ok(Index::from(index))
}
