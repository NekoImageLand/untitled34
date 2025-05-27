#[cfg(feature = "cosine-sim")]
pub mod cosine_sim;
#[cfg(any(feature = "opendal-data-compat", feature = "opendal-ext"))]
pub mod opendal;
#[cfg(feature = "qdrant-ext")]
pub mod qdrant;
#[cfg(feature = "shared-structure")]
pub mod structure;
