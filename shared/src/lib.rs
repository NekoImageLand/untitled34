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
    use pyo3::py_run;

    macro_rules! add_submodule {
        ($py:expr, $m:expr, $name:literal, $func:path) => {
            let submodule = PyModule::new($py, $name)?;
            py_run!(
                $py,
                submodule,
                concat!(
                    "import sys\n",
                    "sys.modules['shared.",
                    $name,
                    "'] = submodule"
                )
            );
            $func($py, &submodule)?;
            $m.add_submodule(&submodule)?;
        };
    }

    #[pymodule]
    fn shared(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
        #[cfg(feature = "point-explorer-pyo3")]
        add_submodule!(
            py,
            m,
            "point_explorer",
            crate::point_explorer::pyo3::point_explorer
        );
        #[cfg(feature = "hnsw-pyo3")]
        add_submodule!(py, m, "hnsw", crate::hnsw::pyo3::hnsw);
        m.add_class::<NekoPoint>()?;
        m.add_class::<NekoPointText>()?;
        Ok(())
    }
}
