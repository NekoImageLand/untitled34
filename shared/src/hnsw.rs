use hnsw_rs::prelude::*;
use rayon::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::fmt::Debug;
use std::path::Path;
use std::sync::atomic::AtomicBool;
#[cfg(feature = "pyo3")]
use {
    ::pyo3::prelude::*,
    pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods},
};

#[derive(Debug, Copy, Clone)]
#[cfg_attr(
    feature = "pyo3",
    gen_stub_pyclass,
    pyclass(module = "shared.hnsw", get_all)
)]
pub struct HnswSearchResult {
    point_id: usize,
    distance: f32,
}

#[cfg_attr(feature = "pyo3", gen_stub_pymethods, pymethods)]
impl HnswSearchResult {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }
}

#[derive(Default)]
pub struct HnswStorage {
    io: HnswIo,
}

impl HnswStorage {
    pub fn open<P: AsRef<Path>>(dir: P, basename: &str) -> Self {
        let io = HnswIo::new(dir.as_ref(), basename);
        HnswStorage { io }
    }

    pub fn load<V, D>(&mut self) -> Hnsw<'_, V, D>
    where
        V: Serialize + DeserializeOwned + Clone + Debug + Default + Send + Sync + 'static,
        D: Distance<V> + Default + Send + Sync,
    {
        self.io.load_hnsw().unwrap()
    }
}

pub struct HnswIndex<'a, V, D>
where
    V: Serialize + DeserializeOwned + Clone + Debug + Default + Send + Sync + 'static,
    D: Distance<V> + Default + Send + Sync,
{
    inner: Hnsw<'a, V, D>,
    search_mode_flag: AtomicBool,
}

impl<'a, V, D> HnswIndex<'a, V, D>
where
    V: Serialize + DeserializeOwned + Clone + Debug + Default + Send + Sync + 'static,
    D: Distance<V> + Default + Send + Sync,
{
    pub fn new(
        max_nb_connection: usize,
        max_elements: usize,
        max_layer: usize,
        ef_construction: usize,
        distance: D,
    ) -> Self {
        let inner = Hnsw::new(
            max_nb_connection,
            max_elements,
            max_layer,
            ef_construction,
            distance,
        );
        HnswIndex {
            inner,
            search_mode_flag: AtomicBool::new(false),
        }
    }

    pub fn new_from_storage(storage: &mut HnswStorage) -> HnswIndex<'_, V, D> {
        let inner = storage.load();
        HnswIndex {
            inner,
            search_mode_flag: AtomicBool::new(false),
        }
    }

    fn check_insert(&mut self) {
        if self
            .search_mode_flag
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            self.inner.set_extend_candidates(false);
        }
    }

    pub fn insert(&mut self, points: &[(&Vec<V>, usize)]) {
        self.check_insert();
        self.inner.parallel_insert(&points);
    }

    fn check_search(&mut self) {
        if !self
            .search_mode_flag
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            self.inner.set_searching_mode(true);
            self.search_mode_flag
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    pub fn search(&mut self, query: &[V], k: usize, ef: usize) -> Vec<HnswSearchResult> {
        self.check_search();
        let res = self.inner.search(query, k, ef);
        res.into_iter()
            .map(|n| HnswSearchResult {
                point_id: n.d_id,
                distance: n.distance,
            })
            .collect()
    }

    // TODO: indicatif
    pub fn search_batch(
        &mut self,
        queries: &[Vec<V>],
        k: usize,
        ef: usize,
    ) -> Vec<Vec<HnswSearchResult>> {
        self.check_search();
        queries
            .par_iter()
            .map(|query| {
                let res = self.inner.search(query, k, ef);
                res.into_iter()
                    .map(|n| HnswSearchResult {
                        point_id: n.d_id,
                        distance: n.distance,
                    })
                    .collect()
            })
            .collect()
    }
}

#[cfg(feature = "hnsw-pyo3")]
pub mod pyo3 {
    use crate::hnsw::{HnswIndex, HnswSearchResult, HnswStorage};
    use hnsw_rs::prelude::DistHamming;
    use pyo3::prelude::*;
    use pyo3_stub_gen::define_stub_info_gatherer;
    use pyo3_stub_gen_derive::*;

    macro_rules! define_py_hnsw {
        ($storage_struct:ident, $index_struct:ident, $V:ty, $D:ty) => {
            #[gen_stub_pyclass]
            #[pyclass(module = "shared.hnsw")]
            pub struct $storage_struct {
                inner: Option<HnswStorage>,
            }

            #[gen_stub_pymethods]
            #[pymethods]
            impl $storage_struct {
                #[new]
                pub fn new(base_path: &str, base_filename: &str) -> Self {
                    let inner = Some(HnswStorage::open(base_path, base_filename));
                    $storage_struct { inner }
                }

                pub fn load(&mut self) -> PyResult<$index_struct> {
                    let storage = self.inner.take().ok_or_else(|| {
                        pyo3::exceptions::PyRuntimeError::new_err("storage already loaded")
                    })?;
                    let storage_ref: &'static mut HnswStorage = Box::leak(Box::new(storage));
                    let inner_index = HnswIndex::new_from_storage(storage_ref);
                    Ok($index_struct { inner: inner_index })
                }
            }

            #[gen_stub_pyclass]
            #[pyclass(module = "shared.hnsw")]
            pub struct $index_struct {
                inner: HnswIndex<'static, $V, $D>,
            }

            #[gen_stub_pymethods]
            #[pymethods]
            impl $index_struct {
                #[new]
                pub fn new(
                    max_nb_connection: usize,
                    max_elements: usize,
                    max_layer: usize,
                    ef_construction: usize,
                ) -> Self {
                    let distance = <$D>::default();
                    let inner = HnswIndex::new(
                        max_nb_connection,
                        max_elements,
                        max_layer,
                        ef_construction,
                        distance,
                    );
                    $index_struct { inner }
                }

                pub fn insert(&mut self, points: Vec<(Vec<$V>, usize)>) -> PyResult<()> {
                    let refs: Vec<(&Vec<$V>, usize)> = points.iter().map(|p| (&p.0, p.1)).collect();
                    self.inner.insert(&refs);
                    Ok(())
                }

                pub fn search(
                    &mut self,
                    query: Vec<$V>,
                    k: usize,
                    ef: usize,
                ) -> PyResult<Vec<HnswSearchResult>> {
                    let results = self.inner.search(&query, k, ef);
                    Ok(results)
                }

                pub fn search_batch(
                    &mut self,
                    queries: Vec<Vec<$V>>,
                    k: usize,
                    ef: usize,
                ) -> PyResult<Vec<Vec<HnswSearchResult>>> {
                    let batch = self.inner.search_batch(&queries, k, ef);
                    Ok(batch)
                }
            }
        };
    }

    define_py_hnsw!(HnswStorageU8Hamming, HnswIndexU8Hamming, u8, DistHamming);

    #[pymodule]
    pub fn hnsw(_: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<HnswStorageU8Hamming>()?;
        m.add_class::<HnswIndexU8Hamming>()?;
        m.add_class::<HnswSearchResult>()?;
        Ok(())
    }

    define_stub_info_gatherer!(stub_info);
}
