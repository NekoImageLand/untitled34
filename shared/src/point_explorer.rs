use crate::cosine_sim::{Cosine, cosine_sim};
use crate::structure::{NekoPoint, NekoPointExt};
use indexmap::IndexMap;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::fs;
use std::hash::Hash;
use std::path::PathBuf;
use url::Url;
use uuid::Uuid;

// TODO: more intuitive error
#[derive(Debug, thiserror::Error)]
pub enum PointExplorerError {
    #[error("Failed to read file: {0}")]
    PathNotFound(String),
    #[error(transparent)]
    SerdePickleError(#[from] serde_pickle::Error),
    #[error("Bincode encode error: {0:?}")]
    BinCodeSerdeEncodeError(bincode::error::EncodeError),
    #[error("Bincode decode error: {0:?}")]
    BinCodeSerdeDecodeError(bincode::error::DecodeError),
    #[error("Point with ID {0} not found")]
    PointNotFound(Uuid),
}

pub type PointExplorerResult<T> = Result<T, PointExplorerError>;

#[derive(Clone, Debug)]
pub struct PointExplorerBuilder {
    capacity: Option<usize>,
    point_explorer_path: Option<String>,
    metadata_path: Option<String>,
    metadata_ext_path: Option<String>,
    point_uri_prefix_map: Option<HashMap<String, String>>,
}

impl PointExplorerBuilder {
    pub fn new() -> Self {
        Self {
            capacity: None,
            point_explorer_path: None,
            metadata_path: None,
            metadata_ext_path: None,
            point_uri_prefix_map: None,
        }
    }

    pub fn capacity(mut self, capacity: usize) -> Self {
        self.capacity = Some(capacity);
        self
    }

    pub fn path<P: Into<String>>(mut self, path: P) -> Self {
        self.point_explorer_path = Some(path.into());
        self
    }

    pub fn metadata_path<P: Into<String>>(mut self, path: P) -> Self {
        self.metadata_path = Some(path.into());
        self
    }

    pub fn metadata_ext_path<P: Into<String>>(mut self, path: P) -> Self {
        self.metadata_ext_path = Some(path.into());
        self
    }

    pub fn point_url_prefix<P: Into<String>>(mut self, key: P, prefix: P) -> Self {
        self.point_uri_prefix_map = match self.point_uri_prefix_map {
            Some(mut map) => {
                map.insert(key.into(), prefix.into());
                Some(map)
            }
            None => {
                let mut map = HashMap::new();
                map.insert(key.into(), prefix.into());
                Some(map)
            }
        };
        self
    }

    pub fn build<T, const D: usize>(self) -> PointExplorerResult<PointExplorer<T, D>>
    where
        T: Copy + Debug + Default + Serialize + DeserializeOwned,
        [T; D]: for<'a> TryFrom<&'a [T]>,
        for<'a> <[T; D] as TryFrom<&'a [T]>>::Error: Debug,
    {
        let mut explorer = if let Some(path) = self.point_explorer_path {
            PointExplorer::load(&path).map_err(PointExplorerError::from)?
        } else if let Some(cap) = self.capacity {
            PointExplorer::with_capacity(cap)
        } else {
            PointExplorer::new()
        };
        // TODO: load builtin self.metadata_ext_path & self.point_explorer_path
        // TODO: overwrite warn (use tracing)
        if let Some(meta_path) = self.metadata_path {
            explorer.load_metadata(&meta_path)?;
        }
        if let Some(ext_path) = self.metadata_ext_path {
            explorer.load_metadata_ext(&ext_path)?;
        }
        if let Some(prefix) = self.point_uri_prefix_map {
            explorer.load_points_uri_prefix(&prefix);
        }
        Ok(explorer)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
enum PointUri {
    Path(PathBuf),
    Url(Url),
}

#[allow(dead_code)]
#[serde_as]
#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize", deserialize = "T: DeserializeOwned",))]
pub struct PointExplorer<T, const D: usize>
where
    T: Copy + Debug + Default + Serialize + DeserializeOwned,
    [T; D]: for<'a> TryFrom<&'a [T]>,
    for<'a> <[T; D] as TryFrom<&'a [T]>>::Error: Debug,
{
    #[serde_as(as = "IndexMap<_, [_; D]>")]
    point_vector_map: IndexMap<Uuid, [T; D]>,
    /// Deprecated
    #[serde(skip)]
    point_uri_prefix: Option<PointUri>,
    #[serde(default)]
    point_uri_prefix_map: Option<HashMap<String, PointUri>>,
    #[serde(skip)]
    point_metadata: Option<HashMap<Uuid, NekoPoint>>,
    #[serde(default)]
    point_metadata_path: Option<PathBuf>,
    #[serde(skip)]
    point_metadata_ext: Option<HashMap<Uuid, NekoPointExt>>,
    #[serde(default)]
    point_metadata_ext_path: Option<PathBuf>,
}

impl<T, const D: usize> Display for PointExplorer<T, D>
where
    T: Copy + Debug + Default + Serialize + DeserializeOwned,
    [T; D]: for<'a> TryFrom<&'a [T]>,
    for<'a> <[T; D] as TryFrom<&'a [T]>>::Error: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[inline]
        fn display_hashmap<K, V>(path: &Option<PathBuf>, map: &Option<HashMap<K, V>>) -> String {
            format!(
                "path = {:?}, inner = {}",
                path,
                match map {
                    Some(m) => format!("Some(len = {})", m.len()),
                    None => "None".to_string(),
                }
            )
        }

        f.debug_struct("PointExplorer")
            .field(
                "point_vector_map",
                &format!("len = {}", self.point_vector_map.len()),
            )
            .field("dim", &D)
            .field(
                "point_metadata",
                &display_hashmap(&self.point_metadata_path, &self.point_metadata),
            )
            .field(
                "point_metadata_ext",
                &display_hashmap(&self.point_metadata_ext_path, &self.point_metadata_ext),
            )
            .field("point_uri_prefix_map", &self.point_uri_prefix_map)
            .finish()
    }
}

impl<T, const D: usize> PointExplorer<T, D>
where
    T: Copy + Debug + Default + Serialize + DeserializeOwned,
    [T; D]: for<'a> TryFrom<&'a [T]>,
    for<'a> <[T; D] as TryFrom<&'a [T]>>::Error: Debug,
{
    fn new() -> Self {
        Self::default()
    }

    fn with_capacity(capacity: usize) -> Self {
        Self {
            point_vector_map: IndexMap::with_capacity(capacity),
            point_metadata: None,
            point_metadata_path: None,
            point_metadata_ext: None,
            point_metadata_ext_path: None,
            point_uri_prefix: None,
            point_uri_prefix_map: None,
        }
    }

    fn load(path: &str) -> PointExplorerResult<Self> {
        let data =
            fs::read(path).map_err(|_| PointExplorerError::PathNotFound(path.to_string()))?;
        let explorer: PointExplorer<T, D> =
            bincode::serde::decode_from_slice(&data, bincode::config::standard())
                .map_err(PointExplorerError::BinCodeSerdeDecodeError)?
                .0;
        Ok(explorer)
    }

    fn load_metadata(&mut self, path: &str) -> PointExplorerResult<()> {
        let data =
            fs::read(path).map_err(|_| PointExplorerError::PathNotFound(path.to_string()))?;
        let metadata: HashMap<Uuid, NekoPoint> =
            serde_pickle::from_slice(&data, serde_pickle::DeOptions::default())
                .map_err(PointExplorerError::SerdePickleError)?;
        self.point_metadata = Some(metadata);
        self.point_metadata_path = Some(PathBuf::from(path));
        Ok(())
    }

    fn load_metadata_ext(&mut self, path: &str) -> PointExplorerResult<()> {
        let data =
            fs::read(path).map_err(|_| PointExplorerError::PathNotFound(path.to_string()))?;
        let metadata_ext: HashMap<Uuid, NekoPointExt> =
            serde_pickle::from_slice(&data, serde_pickle::DeOptions::default())
                .map_err(PointExplorerError::SerdePickleError)?;
        self.point_metadata_ext = Some(metadata_ext);
        self.point_metadata_ext_path = Some(PathBuf::from(path));
        Ok(())
    }

    pub fn load_points_uri_prefix(&mut self, prefix: &HashMap<String, String>) {
        self.point_uri_prefix_map = Some(
            prefix
                .iter()
                .map(|(k, v)| {
                    (
                        k.to_owned(),
                        match Url::parse(v) {
                            Ok(url) if !url.cannot_be_a_base() => PointUri::Url(url),
                            _ => PointUri::Path(PathBuf::from(v)),
                        },
                    )
                })
                .collect(),
        );
    }

    pub fn save(&self, path: &str) -> PointExplorerResult<()> {
        let data = bincode::serde::encode_to_vec(self, bincode::config::standard())
            .map_err(PointExplorerError::BinCodeSerdeEncodeError)?;
        fs::write(path, data).map_err(|_| PointExplorerError::PathNotFound(path.to_string()))?;
        Ok(())
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.point_vector_map.len()
    }

    #[inline]
    pub fn iter(&self) -> indexmap::map::Iter<'_, Uuid, [T; D]> {
        self.point_vector_map.iter()
    }

    pub fn insert<K, V>(&mut self, key_like: K, vec_like: V)
    where
        K: Borrow<Uuid>,
        V: AsRef<[T]>,
    {
        let id = *key_like.borrow();
        let slice: &[T] = vec_like.as_ref();
        debug_assert_eq!(slice.len(), D, "Vector must be of length {}", D);
        let arr: [T; D] = slice.try_into().expect("Vector length must match D");
        self.point_vector_map.insert(id, arr);
    }

    pub fn extend<I, K, V>(&mut self, points: I)
    where
        I: IntoIterator<Item = (K, V)>,
        K: Borrow<Uuid>,
        V: AsRef<[T]>,
    {
        let iter = points.into_iter();
        let (_, higher) = iter.size_hint();
        self.point_vector_map.reserve(higher.unwrap_or_default());
        self.point_vector_map
            .extend(iter.map(|(key_like, vec_like)| {
                let id = *key_like.borrow();
                let slice = vec_like.as_ref();
                debug_assert_eq!(slice.len(), D, "Vector must be of length {}", D);
                let arr: [T; D] = slice.try_into().expect("Vector length must match D");
                (id, arr)
            }));
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.point_vector_map.is_empty()
    }

    #[inline]
    pub fn get_vector(&self, point_id: &Uuid) -> Option<&[T; D]> {
        self.point_vector_map.get(point_id)
    }

    #[inline]
    pub fn contains(&self, point_id: &Uuid) -> bool {
        self.point_vector_map.contains_key(point_id)
    }

    #[inline]
    pub fn remove(&mut self, point_id: &Uuid) -> Option<[T; D]> {
        self.point_vector_map.shift_remove(point_id)
    }

    #[inline]
    pub fn clear(&mut self) {
        self.point_vector_map.clear();
    }

    #[inline]
    pub fn index2uuid(&self, index: usize) -> Option<&Uuid> {
        self.point_vector_map.get_index(index).map(|(id, _)| id)
    }

    #[inline]
    pub fn uuid2index(&self, point_id: &Uuid) -> Option<usize> {
        self.point_vector_map
            .get_full(point_id)
            .map(|(idx, _, _)| idx)
    }

    pub fn get_point_metadata(&self, point_id: &Uuid) -> Option<&NekoPoint> {
        self.point_metadata.as_ref()?.get(point_id)
    }

    pub fn get_point_uri(&self, pm_prefix: &str, point_id: &Uuid) -> Option<String> {
        let prefix = self.point_uri_prefix_map.as_ref()?.get(pm_prefix)?;
        let point = self.point_metadata_ext.as_ref()?.get(point_id)?;
        let filename = format!("{}.{}", point_id, point.ext());
        match prefix {
            PointUri::Url(base) => base.join(&filename).ok().map(|u| u.into()),
            PointUri::Path(base) => {
                let mut path = base.clone();
                path.push(filename);
                Some(path.to_string_lossy().into_owned())
            }
        }
    }
}

impl<T, const D: usize> PointExplorer<T, D>
where
    T: Copy + Debug + Default + Serialize + DeserializeOwned + Cosine,
    [T; D]: for<'a> TryFrom<&'a [T]>,
    for<'a> <[T; D] as TryFrom<&'a [T]>>::Error: Debug,
{
    pub fn get_cosine_sim(&self, point_id: (&Uuid, &Uuid)) -> PointExplorerResult<f32> {
        let (id_a, id_b) = point_id;
        let vector_a = self
            .point_vector_map
            .get(id_a)
            .ok_or(PointExplorerError::PointNotFound(*id_a))?;
        let vector_b = self
            .point_vector_map
            .get(id_b)
            .ok_or(PointExplorerError::PointNotFound(*id_b))?;
        Ok(cosine_sim(vector_a, vector_b))
    }
}

// TODO: impl hamming distance for u8

#[cfg(feature = "point-explorer-pyo3")]
pub mod pyo3 {
    use crate::point_explorer::{PointExplorer, PointExplorerBuilder, PointExplorerError};
    use pyo3::exceptions::{PyIOError, PyKeyError, PyValueError};
    use pyo3::prelude::*;
    use pyo3_stub_gen::{define_stub_info_gatherer, derive::*};

    impl From<PointExplorerError> for PyErr {
        fn from(err: PointExplorerError) -> PyErr {
            match err {
                PointExplorerError::PathNotFound(msg) => PyIOError::new_err(msg),
                PointExplorerError::SerdePickleError(e) => {
                    PyValueError::new_err(format!("Serde Pickle Error: {}", e))
                }
                PointExplorerError::BinCodeSerdeEncodeError(e) => {
                    PyValueError::new_err(e.to_string())
                }
                PointExplorerError::BinCodeSerdeDecodeError(e) => {
                    PyValueError::new_err(e.to_string())
                }
                PointExplorerError::PointNotFound(id) => {
                    PyKeyError::new_err(format!("Point with ID {} not found", id))
                }
            }
        }
    }

    #[gen_stub_pyclass]
    #[pyclass(module = "shared.point_explorer")]
    pub struct PyPointExplorerBuilder {
        builder: PointExplorerBuilder,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyPointExplorerBuilder {
        #[new]
        pub fn new() -> Self {
            PyPointExplorerBuilder {
                builder: PointExplorerBuilder::new(),
            }
        }

        pub fn capacity<'a>(
            mut slf: PyRefMut<'a, Self>,
            capacity: usize,
        ) -> PyResult<PyRefMut<'a, Self>> {
            slf.builder = slf.builder.clone().capacity(capacity);
            Ok(slf)
        }

        pub fn path<'a>(mut slf: PyRefMut<'a, Self>, path: String) -> PyResult<PyRefMut<'a, Self>> {
            slf.builder = slf.builder.clone().path(path);
            Ok(slf)
        }

        pub fn metadata_path<'a>(
            mut slf: PyRefMut<'a, Self>,
            path: String,
        ) -> PyResult<PyRefMut<'a, Self>> {
            slf.builder = slf.builder.clone().metadata_path(path);
            Ok(slf)
        }

        pub fn metadata_ext_path<'a>(
            mut slf: PyRefMut<'a, Self>,
            path: String,
        ) -> PyResult<PyRefMut<'a, Self>> {
            slf.builder = slf.builder.clone().metadata_ext_path(path);
            Ok(slf)
        }

        pub fn point_uri_prefix<'a>(
            mut slf: PyRefMut<'a, Self>,
            key: String,
            prefix: String,
        ) -> PyResult<PyRefMut<'a, Self>> {
            slf.builder = slf.builder.clone().point_url_prefix(key, prefix);
            Ok(slf)
        }

        pub fn build_f32d768(&self) -> PyResult<PyPointExplorerF32D768> {
            let explorer = self.builder.clone().build::<f32, 768>()?;
            Ok(PyPointExplorerF32D768 { inner: explorer })
        }

        pub fn build_u8d32(&self) -> PyResult<PyPointExplorerU8D32> {
            let explorer = self.builder.clone().build::<u8, 32>()?;
            Ok(PyPointExplorerU8D32 { inner: explorer })
        }

        pub fn build_u8d128(&self) -> PyResult<PyPointExplorerU8D128> {
            let explorer = self.builder.clone().build::<u8, 128>()?;
            Ok(PyPointExplorerU8D128 { inner: explorer })
        }
    }

    macro_rules! py_point_explorer_impl {
        ($name:ident, $scalar:ty, $dim:expr) => {
            #[gen_stub_pyclass]
            #[pyclass(module = "shared.point_explorer")]
            pub struct $name {
                pub(crate) inner: PointExplorer<$scalar, $dim>,
            }

            #[gen_stub_pymethods]
            #[pymethods]
            impl $name {
                #[new]
                #[pyo3(signature=(capacity=None))]
                pub fn new(capacity: Option<usize>) -> Self {
                    let inner = match capacity {
                        Some(cap) => PointExplorer::<$scalar, $dim>::with_capacity(cap),
                        None => PointExplorer::<$scalar, $dim>::new(),
                    };
                    Self { inner }
                }

                pub fn contains(&self, point_id: String) -> PyResult<bool> {
                    let uuid = uuid::Uuid::parse_str(&point_id)
                        .map_err(|e| PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
                    Ok(self.inner.contains(&uuid))
                }

                pub fn remove(&mut self, point_id: &str) -> PyResult<Option<Vec<$scalar>>> {
                    let uuid = uuid::Uuid::parse_str(point_id)
                        .map_err(|e| PyValueError::new_err(format!("Invalid UUID: {e}")))?;
                    Ok(self.inner.remove(&uuid).map(|v| v.to_vec()))
                }

                pub fn clear(&mut self) {
                    self.inner.clear();
                }

                pub fn len(&self) -> usize {
                    self.inner.len()
                }

                pub fn is_empty(&self) -> bool {
                    self.inner.is_empty()
                }

                pub fn index2uuid(&self, index: usize) -> Option<String> {
                    self.inner.index2uuid(index).map(|uuid| uuid.to_string())
                }

                pub fn uuid2index(&self, point_id: String) -> PyResult<Option<usize>> {
                    let uuid = uuid::Uuid::parse_str(&point_id)
                        .map_err(|e| PyValueError::new_err(format!("Invalid UUID: {e}")))?;
                    Ok(self.inner.uuid2index(&uuid))
                }

                pub fn get_all_ids(&self) -> Vec<String> {
                    self.inner.iter().map(|(id, _)| id.to_string()).collect()
                }

                pub fn get_all_vectors(&self) -> Vec<Vec<$scalar>> {
                    self.inner.iter().map(|(_, v)| v.to_vec()).collect()
                }

                pub fn get_items(&self) -> Vec<(String, Vec<$scalar>)> {
                    self.inner
                        .iter()
                        .map(|(id, vector)| (id.to_string(), vector.to_vec()))
                        .collect()
                }

                pub fn __len__(&self) -> usize {
                    self.len()
                }

                pub fn __bool__(&self) -> bool {
                    !self.is_empty()
                }

                pub fn __contains__(&self, point_id: String) -> PyResult<bool> {
                    self.contains(point_id)
                }

                pub fn __repr__(&self) -> String {
                    format!("{}", self.inner)
                }

                pub fn __iter__(slf: PyRef<'_, Self>) -> PyPointExplorerIterator {
                    PyPointExplorerIterator {
                        ids: slf.get_all_ids(),
                        index: 0,
                    }
                }

                // pub fn get_cosine_similarity(&self, id_a: &str, id_b: &str) -> PyResult<f32> {
                //     let a = Uuid::parse_str(id_a)
                //         .map_err(|e| PyValueError::new_err(format!("Invalid UUID id_a: {e}")))?;
                //     let b = Uuid::parse_str(id_b)
                //         .map_err(|e| PyValueError::new_err(format!("Invalid UUID id_b: {e}")))?;
                //     self.inner
                //         .get_cosine_similarity((&a, &b))
                //         .map_err(PyErr::from)
                // }

                pub fn get_vector(&self, point_id: String) -> PyResult<Option<Vec<$scalar>>> {
                    let uuid = uuid::Uuid::parse_str(&point_id)
                        .map_err(|e| PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
                    Ok(self.inner.get_vector(&uuid).map(|v| v.to_vec()))
                }

                pub fn get_point_metadata(
                    &self,
                    point_id: &str,
                ) -> PyResult<Option<crate::structure::NekoPoint>> {
                    let uuid = uuid::Uuid::parse_str(point_id)
                        .map_err(|e| PyValueError::new_err(format!("Invalid UUID: {e}")))?;
                    Ok(self.inner.get_point_metadata(&uuid).cloned())
                }

                pub fn get_point_uri(
                    &self,
                    pm_key: &str,
                    point_id: &str,
                ) -> PyResult<Option<String>> {
                    let uuid = uuid::Uuid::parse_str(point_id)
                        .map_err(|e| PyValueError::new_err(format!("Invalid UUID: {e}")))?;
                    Ok(self.inner.get_point_uri(pm_key, &uuid))
                }
            }
        };
    }

    py_point_explorer_impl!(PyPointExplorerF32D768, f32, 768);
    py_point_explorer_impl!(PyPointExplorerU8D32, u8, 32);
    py_point_explorer_impl!(PyPointExplorerU8D128, u8, 128);

    #[gen_stub_pyclass]
    #[pyclass(module = "shared.point_explorer")]
    pub struct PyPointExplorerIterator {
        ids: Vec<String>,
        index: usize,
    }

    #[gen_stub_pymethods]
    #[pymethods]
    impl PyPointExplorerIterator {
        pub fn __iter__(s: PyRef<'_, Self>) -> PyRef<'_, Self> {
            s
        }

        pub fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<String> {
            if slf.index < slf.ids.len() {
                let id = slf.ids[slf.index].clone();
                slf.index += 1;
                Some(id)
            } else {
                None
            }
        }
    }

    pub fn point_explorer(_: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyPointExplorerBuilder>()?;
        m.add_class::<PyPointExplorerF32D768>()?;
        m.add_class::<PyPointExplorerU8D32>()?;
        m.add_class::<PyPointExplorerU8D128>()?;
        m.add_class::<PyPointExplorerIterator>()?;
        Ok(())
    }

    define_stub_info_gatherer!(stub_info);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_pickle::{DeOptions, SerOptions};
    use uuid::Uuid;

    const EPS: f32 = 1e-6;

    fn make_unit_vector(dim: usize, pos: usize) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[pos] = 1.0;
        v
    }

    #[test]
    fn insert_and_similarity() {
        let mut explorer: PointExplorer<f32, 768> = PointExplorer::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let v1 = vec![1.0; 768];
        let v2 = vec![1.0; 768];
        explorer.insert(&id1, &v1);
        explorer.insert(&id2, &v2);
        assert_eq!(explorer.len(), 2);
        let sim = explorer.get_cosine_sim((&id1, &id2)).unwrap();
        assert!((sim - 1.0).abs() < EPS);
    }

    #[test]
    fn batch_insert_and_error_handling() {
        let mut explorer: PointExplorer<f32, 768> = PointExplorer::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let v1 = make_unit_vector(768, 0);
        let v2 = make_unit_vector(768, 1);
        explorer.extend([(&id1, &v1), (&id2, &v2)]);
        let uuid_except_1 = explorer.index2uuid(1);
        assert_eq!(uuid_except_1, Some(&id2));
        let sim = explorer.get_cosine_sim((&id1, &id2)).unwrap();
        assert!(sim.abs() < EPS);
        let missing = Uuid::new_v4();
        let err = explorer.get_cosine_sim((&id1, &missing)).unwrap_err();
        assert!(matches!(err, PointExplorerError::PointNotFound(_)));
    }

    #[test]
    fn test_index2uuid() {
        let mut explorer: PointExplorer<f32, 768> = PointExplorer::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        explorer.insert(&id1, &vec![0.0; 768]);
        explorer.insert(&id2, &vec![1.0; 768]);
        assert_eq!(explorer.index2uuid(0), Some(&id1));
        assert_eq!(explorer.index2uuid(1), Some(&id2));
        assert_eq!(explorer.index2uuid(2), None);
    }

    #[test]
    fn serialize_deserialize_simple() {
        let mut explorer: PointExplorer<f32, 768> = PointExplorer::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let v1 = vec![1.0; 768];
        let v2 = vec![2.0; 768];
        explorer.insert(&id1, &v1);
        explorer.insert(&id2, &v2);
        let pre_sim = explorer.get_cosine_sim((&id1, &id2)).unwrap();
        let bytes = serde_pickle::to_vec(&explorer, SerOptions::default()).unwrap();
        let decoded: PointExplorer<f32, 768> =
            serde_pickle::from_slice(&bytes, DeOptions::default()).unwrap();
        let post_sim = decoded.get_cosine_sim((&id1, &id2)).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(pre_sim, post_sim);
        let bytes = bincode::serde::encode_to_vec(&explorer, bincode::config::standard()).unwrap();
        let decoded: PointExplorer<f32, 768> =
            bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                .unwrap()
                .0;
        let post_sim = decoded.get_cosine_sim((&id1, &id2)).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(pre_sim, post_sim);
    }

    #[test]
    fn serialize_deserialize_large_random() {
        use rand::{Rng, SeedableRng};
        use rand_pcg::Pcg64;
        let mut explorer: PointExplorer<f32, 768> = PointExplorer::new();
        let mut rng = Pcg64::seed_from_u64(42);
        let dim = 768;
        let n = 1000;
        let mut ids = Vec::with_capacity(n);
        for _ in 0..n {
            let id = Uuid::new_v4();
            let v: Vec<f32> = (0..dim).map(|_| rng.random()).collect();
            explorer.insert(&id, &v);
            ids.push((id, v));
        }
        for _ in 0..100 {
            let i = rng.random_range(0..n);
            let j = rng.random_range(0..n);
            let (id1, _) = &ids[i];
            let (id2, _) = &ids[j];
            let expected = explorer.get_cosine_sim((id1, id2)).unwrap();
            let bytes = serde_pickle::to_vec(&explorer, SerOptions::default()).unwrap();
            let decoded: PointExplorer<f32, 768> =
                serde_pickle::from_slice(&bytes, DeOptions::default()).unwrap();
            let actual = decoded.get_cosine_sim((id1, id2)).unwrap();
            assert!((expected - actual).abs() < EPS, "pair {}-{} differs", i, j);
            let bytes =
                bincode::serde::encode_to_vec(&explorer, bincode::config::standard()).unwrap();
            let decoded: PointExplorer<f32, 768> =
                bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
                    .unwrap()
                    .0;
            let actual = decoded.get_cosine_sim((id1, id2)).unwrap();
            assert!((expected - actual).abs() < EPS, "pair {}-{} differs", i, j);
        }
    }

    #[test]
    fn test_resource_prefix() {
        let url = "https://example.com/resources/";
        let unix_path = "/path/to/resources/";
        // FIXME: currently, c:/xxx will be parsed as URL
        let windows_path = "C:\\path\\to\\resources\\";
        let pe = PointExplorerBuilder::new()
            .point_url_prefix("url", url)
            .point_url_prefix("unix", unix_path)
            .point_url_prefix("windows", windows_path)
            .build::<u8, 32>()
            .unwrap();
        assert_eq!(
            pe.point_uri_prefix_map.as_ref().unwrap().get("url"),
            Some(&PointUri::Url(Url::parse(url).unwrap()))
        );
        assert_eq!(
            pe.point_uri_prefix_map.as_ref().unwrap().get("unix"),
            Some(&PointUri::Path(PathBuf::from(unix_path)))
        );
        assert_eq!(
            pe.point_uri_prefix_map.as_ref().unwrap().get("windows"),
            Some(&PointUri::Path(PathBuf::from(windows_path)))
        );
    }
}
