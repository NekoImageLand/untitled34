#[cfg(feature = "cosine-sim")]
pub mod cosine_sim;
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
    use pyo3::prelude::*;
    #[pymodule]
    fn shared(m: &Bound<'_, PyModule>) -> PyResult<()> {
        #[cfg(feature = "point-explorer-pyo3")]
        {
            m.add_class::<crate::point_explorer::pyo3::PyPointExplorer>()?;
        }
        Ok(())
    }
}
