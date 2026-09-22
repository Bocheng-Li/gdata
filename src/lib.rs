pub mod dataloader;
pub mod utils;
pub mod w5z;

use pyo3::prelude::*;
use std::io::Write;

#[cfg(not(target_env = "msvc"))]
use tikv_jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

/// A Python module implemented in Rust.
#[pymodule]
fn gdata(m: &Bound<'_, PyModule>) -> PyResult<()> {
    env_logger::builder()
        .format(|buf, record| {
            let timestamp = buf.timestamp();
            let style = buf.default_level_style(record.level());
            writeln!(
                buf,
                "[{timestamp} {style}{}{style:#}] {}",
                record.level(),
                record.args()
            )
        })
        .filter_level(log::LevelFilter::Info)
        .try_init()
        .unwrap();

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<w5z::W5Z>()?;
    m.add_class::<dataloader::generic::PyArrayDataLoader>()?;
    m.add_class::<dataloader::genome::GenomeDataBuilder>()?;
    m.add_class::<dataloader::genome::GenomeDataLoader>()?;
    m.add_class::<dataloader::genome::GenomeDataLoaderAugmentedBFloat16DLPackIter>()?;
    m.add_class::<dataloader::genome::GenomeDataLoaderBFloat16DLPackIter>()?;
    m.add_class::<dataloader::genome::GenomeDataLoaderMap>()?;
    m.add_class::<dataloader::genome::MultiAugmentedBFloat16DLPackIter>()?;
    m.add_class::<dataloader::genome::MultiBFloat16DLPackIter>()?;
    m.add_class::<dataloader::genome::CatGenomeDataLoader>()?;
    m.add_function(wrap_pyfunction!(
        dataloader::genome::convert_tfrecord_to_gdata,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(dataloader::genome::profile_reset, m)?)?;
    m.add_function(wrap_pyfunction!(dataloader::genome::profile_snapshot, m)?)?;

    utils::register_utils(m)?;

    Ok(())
}
