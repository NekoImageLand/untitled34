#[cfg(feature = "cosine-sim")]
pub mod cosine_sim;
#[cfg(feature = "hnsw")]
pub mod hnsw;
#[cfg(feature = "neko-uuid")]
pub mod neko_uuid;
#[cfg(any(feature = "opendal-data-compat", feature = "opendal-ext"))]
pub mod opendal;
#[cfg(feature = "point-explorer")]
pub mod point_explorer;
#[cfg(feature = "qdrant-ext")]
pub mod qdrant;
#[cfg(feature = "shared-structure")]
pub mod structure;

#[cfg(feature = "pyo3")]
mod pyo3 {
    use crate::structure::{NekoPoint, NekoPointText};
    use pyo3::prelude::*;

    #[pymodule]
    fn shared(_: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
        #[cfg(feature = "point-explorer-pyo3")]
        {
            m.add_wrapped(pyo3::wrap_pymodule!(
                crate::point_explorer::pyo3::point_explorer
            ))?;
        }
        #[cfg(feature = "hnsw-pyo3")]
        {
            m.add_wrapped(pyo3::wrap_pymodule!(crate::hnsw::pyo3::hnsw))?;
        }
        m.add_class::<NekoPoint>()?;
        m.add_class::<NekoPointText>()?;
        Ok(())
    }
}
