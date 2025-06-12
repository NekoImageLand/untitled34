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
    use crate::point_explorer::pyo3::point_explorer;
    use crate::structure::{NekoPoint, NekoPointText};
    use pyo3::prelude::*;

    #[pymodule]
    fn shared(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
        #[cfg(feature = "point-explorer-pyo3")]
        {
            let pe = PyModule::new(py, "point_explorer")?;
            point_explorer(py, &pe)?;
            m.add_submodule(&pe)?;
        }
        m.add_class::<NekoPoint>()?;
        m.add_class::<NekoPointText>()?;
        Ok(())
    }
}
