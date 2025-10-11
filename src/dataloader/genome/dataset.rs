use anyhow::{ensure, Result};
use bed_utils::bed::GenomicRange;
use indexmap::IndexMap;
use ndarray::{s, Array1, Array2, Array3};
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::Bound;
use rand::SeedableRng;
use std::path::PathBuf;
use std::str::FromStr;

use super::data_store::{decode_nucleotide, DataStore, DataStoreReadOptions};

#[pyclass]
#[derive(Debug, Clone)]
pub struct GenomeDataset {
    data_store: DataStore,
    subset: Option<Vec<GenomicRange>>,
    #[pyo3(get, set)]
    seq_as_string: bool,
}

impl GenomeDataset {
    fn items_per_segment(&self) -> usize {
        let window = self
            .data_store
            .read_opts
            .split_size
            .unwrap_or(self.data_store.sequence_length());
        (self.data_store.sequence_length() / window) as usize
    }

    fn num_segments(&self) -> usize {
        self.subset
            .as_ref()
            .map_or_else(|| self.data_store.num_segments(), |s| s.len())
    }

    fn total_len(&self) -> usize {
        self.num_segments() * self.items_per_segment()
    }
}

#[pymethods]
impl GenomeDataset {
    #[new]
    #[pyo3(
        signature = (location, *,
            resolution=None, target_length=None, scale=None, clamp_max=None,
            window_size=None, random_shift=0, seq_as_string=false, random_seed=2025,
        ),
        text_signature = "($self, location, *, resolution=None, target_length=None, scale=None, clamp_max=None, window_size=None, random_shift=0, seq_as_string=False, random_seed=2025)"
    )]
    pub fn new(
        location: PathBuf,
        resolution: Option<u32>,
        target_length: Option<u32>,
        scale: Option<f32>,
        clamp_max: Option<f32>,
        window_size: Option<u32>,
        random_shift: u32,
        seq_as_string: bool,
        random_seed: u64,
    ) -> Result<Self> {
        let store_opts = DataStoreReadOptions {
            shift_width: random_shift,
            value_length: target_length,
            split_size: window_size,
            read_resolution: resolution,
            scale_value: scale.map(|x| half::bf16::from_f32(x)),
            clamp_value_max: clamp_max.map(|x| half::bf16::from_f32(x)),
            rng: rand_chacha::ChaCha12Rng::seed_from_u64(random_seed),
        };

        Ok(Self {
            data_store: DataStore::open(location, store_opts)?,
            subset: None,
            seq_as_string,
        })
    }

    #[getter]
    pub fn tracks(&self) -> Vec<String> {
        self.data_store.data_keys().iter().cloned().collect()
    }

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

    #[getter]
    fn window_size(&self) -> u32 {
        self.data_store
            .read_opts
            .split_size
            .unwrap_or(self.data_store.sequence_length())
    }

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

    pub fn set_target_length(&mut self, target_length: u32) -> Result<()> {
        self.data_store.set_value_length(target_length)
    }

    pub fn set_window_size(&mut self, window_size: u32) -> Result<()> {
        self.data_store.set_split_size(window_size)
    }

    #[pyo3(
        name = "intersection",
        signature = (regions),
        text_signature = "($self, regions)"
    )]
    fn intersection_py(&self, regions: Vec<String>) -> Self {
        let regions = regions
            .into_iter()
            .map(|r| GenomicRange::from_str(&r).expect("Invalid genomic range"))
            .collect::<Vec<_>>();
        let mut new_self = self.clone();
        if let Some(subset) = &self.subset {
            let set = std::collections::HashSet::<_>::from_iter(regions.into_iter());
            new_self.subset = Some(
                subset
                    .iter()
                    .filter(|x| set.contains(x))
                    .cloned()
                    .collect(),
            );
        } else {
            let set = std::collections::HashSet::<_>::from_iter(regions.into_iter());
            new_self.subset = Some(
                self.data_store
                    .segments()
                    .filter(|x| set.contains(x))
                    .cloned()
                    .collect(),
            );
        }
        new_self
    }

    #[pyo3(
        name = "difference",
        signature = (regions),
        text_signature = "($self, regions)"
    )]
    fn difference_py(&self, regions: Vec<String>) -> Self {
        let regions = regions
            .into_iter()
            .map(|r| GenomicRange::from_str(&r).expect("Invalid genomic range"))
            .collect::<Vec<_>>();
        let mut new_self = self.clone();
        if let Some(subset) = &self.subset {
            let set = std::collections::HashSet::<_>::from_iter(regions.into_iter());
            new_self.subset = Some(
                subset
                    .iter()
                    .filter(|x| !set.contains(x))
                    .cloned()
                    .collect(),
            );
        } else {
            let set = std::collections::HashSet::<_>::from_iter(regions.into_iter());
            new_self.subset = Some(
                self.data_store
                    .segments()
                    .filter(|x| !set.contains(x))
                    .cloned()
                    .collect(),
            );
        }
        new_self
    }

    fn __len__(&self) -> usize {
        self.total_len()
    }

    fn __getitem__<'a>(
        &'a mut self,
        py: Python<'a>,
        idx: usize,
    ) -> PyResult<Bound<'a, pyo3::types::PyTuple>> {
        if idx >= self.total_len() {
            return Err(pyo3::exceptions::PyIndexError::new_err("Index out of range"));
        }

        let per_seg = self.items_per_segment();
        let seg_index = idx / per_seg;
        let within = idx % per_seg;

        let (seq, values) = if let Some(subset) = &self.subset {
            self.data_store
                .read(&subset[seg_index])
                .expect("Failed to read segment")
        } else {
            self.data_store
                .read_at(seg_index)
                .expect("Failed to read segment")
        };

        let seq: Array2<u8> = seq.into();
        let mut seq_row: Array1<u8> = seq.slice(s![within, ..]).to_owned();

        let values: Array3<f32> = values.into();
        let values_row = values.slice(s![within, .., ..]).to_owned();

        let result = if self.seq_as_string {
            seq_row.mapv_inplace(|x| decode_nucleotide(x).unwrap());
            let s = String::from_utf8(seq_row.to_vec()).unwrap();
            (s, PyArray2::from_owned_array(py, values_row)).into_pyobject(py)?
        } else {
            let s = PyArray1::from_owned_array(py, seq_row);
            (s, PyArray2::from_owned_array(py, values_row)).into_pyobject(py)?
        };
        Ok(result)
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct GenomeDatasetMap(IndexMap<String, GenomeDataset>);

impl GenomeDatasetMap {
    fn get_item_internal<'a>(
        &'a mut self,
        py: Python<'a>,
        idx: usize,
        seq_as_string: bool,
    ) -> PyResult<Bound<'a, pyo3::types::PyTuple>> {
        let (_, first_ds) = self.0.get_index(0).expect("Empty dataset map");
        let per_seg = first_ds.items_per_segment();
        let total_len = first_ds.total_len();
        if idx >= total_len {
            return Err(pyo3::exceptions::PyIndexError::new_err("Index out of range"));
        }

        let seg_index = idx / per_seg;
        let within = idx % per_seg;

        // Read from first dataset to get sequence row
        let mut seq_row: Array1<u8> = {
            let (_, first_ds_mut) = self.0.get_index_mut(0).expect("Empty dataset map");
            let (s, _v) = if let Some(subset) = &first_ds_mut.subset {
                first_ds_mut
                    .data_store
                    .read(&subset[seg_index])
                    .expect("Failed to read segment")
            } else {
                first_ds_mut
                    .data_store
                    .read_at(seg_index)
                    .expect("Failed to read segment")
            };
            let s: Array2<u8> = s.into();
            s.slice(s![within, ..]).to_owned()
        };

        // Build values for each dataset
        let mut values_map: IndexMap<String, Py<PyArray2<f32>>> = IndexMap::new();
        for (tag, ds) in self.0.iter_mut() {
            let (s, v) = if let Some(subset) = &ds.subset {
                ds.data_store
                    .read(&subset[seg_index])
                    .expect("Failed to read segment")
            } else {
                ds.data_store
                    .read_at(seg_index)
                    .expect("Failed to read segment")
            };
            // Optional: assert sequences consistent
            let s: Array2<u8> = s.into();
            let v: Array3<f32> = v.into();
            let v_row = v.slice(s![within, .., ..]).to_owned();
            values_map.insert(tag.clone(), PyArray2::from_owned_array(py, v_row).unbind());

            // sanity: ensure sequence alignment for first and others
            #[allow(unused)]
            {
                let seq_cmp: Array1<u8> = s.slice(s![within, ..]).to_owned();
                debug_assert_eq!(seq_row, seq_cmp, "All sequences must be the same");
            }
        }

        let result = if seq_as_string {
            seq_row.mapv_inplace(|x| decode_nucleotide(x).unwrap());
            let s = String::from_utf8(seq_row.to_vec()).unwrap();
            (s, values_map).into_pyobject(py)?
        } else {
            let s = PyArray1::from_owned_array(py, seq_row);
            (s, values_map).into_pyobject(py)?
        };
        Ok(result)
    }
}

#[pymethods]
impl GenomeDatasetMap {
    #[new]
    #[pyo3(
        signature = (loaders, *, target_length=None, window_size=None, seq_as_string=false),
        text_signature = "($self, loaders, *, target_length=None, window_size=None, seq_as_string=False)"
    )]
    pub fn new(
        mut loaders: IndexMap<String, GenomeDataset>,
        target_length: Option<u32>,
        window_size: Option<u32>,
        seq_as_string: bool,
    ) -> Result<Self> {
        ensure!(!loaders.is_empty(), "At least one GenomeDataset must be provided");
        ensure!(
            loaders.values().map(|ds| ds.segments()).all(|s| s == loaders.values().next().unwrap().segments()),
            "All genome datasets must have the same segments"
        );

        loaders.values_mut().for_each(|ds| {
            ds.seq_as_string = seq_as_string;
            if let Some(t) = target_length {
                ds.set_target_length(t).unwrap();
            }
            if let Some(w) = window_size {
                ds.set_window_size(w).unwrap();
            }
        });

        ensure!(
            loaders
                .values()
                .map(|ds| ds.window_size())
                .all(|w| w == loaders.values().next().unwrap().window_size()),
            "All genome datasets must have the same window size",
        );

        Ok(Self(loaders))
    }

    fn __len__(&self) -> usize {
        let (_, ds) = self.0.get_index(0).expect("Empty dataset map");
        ds.total_len()
    }

    fn __getitem__<'a>(
        &'a mut self,
        py: Python<'a>,
        idx: usize,
    ) -> PyResult<Bound<'a, pyo3::types::PyTuple>> {
        self.get_item_internal(py, idx, self.0.get_index(0).unwrap().1.seq_as_string)
    }

    #[getter]
    fn n_tracks(&self) -> IndexMap<String, usize> {
        self.0
            .iter()
            .map(|(k, v)| (k.clone(), v.tracks().len()))
            .collect()
    }

    #[getter]
    fn segments(&self) -> Vec<String> {
        self.0.get_index(0).unwrap().1.segments()
    }

    #[pyo3(
        signature = (regions),
        text_signature = "($self, regions)"
    )]
    fn intersection(&self, regions: Vec<String>) -> Result<Self> {
        let result = self
            .0
            .iter()
            .map(|(tag, ds)| {
                let new_ds = ds.intersection_py(regions.clone());
                Ok((tag.clone(), new_ds))
            })
            .collect::<Result<IndexMap<_, _>>>();
        Ok(Self(result?))
    }

    #[pyo3(
        signature = (regions),
        text_signature = "($self, regions)"
    )]
    fn difference(&self, regions: Vec<String>) -> Result<Self> {
        let result = self
            .0
            .iter()
            .map(|(tag, ds)| {
                let new_ds = ds.difference_py(regions.clone());
                Ok((tag.clone(), new_ds))
            })
            .collect::<Result<IndexMap<_, _>>>();
        Ok(Self(result?))
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct CatGenomeDataset(Vec<GenomeDatasetMap>);

#[pymethods]
impl CatGenomeDataset {
    #[new]
    pub fn new(loaders: Vec<GenomeDatasetMap>) -> Result<Self> {
        ensure!(!loaders.is_empty(), "At least one GenomeDatasetMap must be provided");
        Ok(Self(loaders))
    }

    fn __len__(&self) -> usize {
        self.0.iter().map(|ds| ds.__len__()).sum()
    }

    fn __getitem__<'a>(
        &'a mut self,
        py: Python<'a>,
        mut idx: usize,
    ) -> PyResult<Bound<'a, pyo3::types::PyTuple>> {
        for loader in self.0.iter_mut() {
            let l = loader.__len__();
            if idx < l {
                return loader.__getitem__(py, idx);
            } else {
                idx -= l;
            }
        }
        Err(pyo3::exceptions::PyIndexError::new_err("Index out of range"))
    }
}


