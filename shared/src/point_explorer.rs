use crate::cosine_sim::cosine_sim;
use crate::structure::{NekoPoint, NekoPointExt};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::fs;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum PointExplorerError {
    #[error("Failed to read file: {0}")]
    PathNotFound(String),
    #[error(transparent)]
    SerdeError(#[from] serde_pickle::Error),
    #[error("Point with ID {0} not found")]
    PointNotFound(Uuid),
}

pub type PointExplorerResult<T> = Result<T, PointExplorerError>;

#[derive(Clone, Debug)]
pub struct PointExplorerBuilder {
    capacity: Option<usize>,
    data_path: Option<String>,
    metadata_path: Option<String>,
    metadata_ext_path: Option<String>,
    point_url_prefix: Option<String>,
}

impl PointExplorerBuilder {
    pub fn new() -> Self {
        Self {
            capacity: None,
            data_path: None,
            metadata_path: None,
            metadata_ext_path: None,
            point_url_prefix: None,
        }
    }

    pub fn capacity(mut self, capacity: usize) -> Self {
        self.capacity = Some(capacity);
        self
    }

    pub fn data_path<P: Into<String>>(mut self, path: P) -> Self {
        self.data_path = Some(path.into());
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

    pub fn point_url_prefix<P: Into<String>>(mut self, prefix: P) -> Self {
        self.point_url_prefix = Some(prefix.into());
        self
    }

    pub fn build(self) -> PointExplorerResult<PointExplorer> {
        let mut explorer = if let Some(path) = self.data_path {
            PointExplorer::new_from_path(&path)?
        } else if let Some(cap) = self.capacity {
            PointExplorer::with_capacity(cap)
        } else {
            PointExplorer::new()
        };
        if let Some(meta_path) = self.metadata_path {
            explorer.load_metadata(&meta_path)?;
        }
        if let Some(ext_path) = self.metadata_ext_path {
            explorer.load_metadata_ext(&ext_path)?;
        }
        if let Some(prefix) = self.point_url_prefix {
            explorer.point_url_prefix = Some(prefix);
        }
        Ok(explorer)
    }
}

#[serde_as]
#[derive(Default, Debug, Serialize, Deserialize)]
pub struct PointExplorer {
    #[serde_as(as = "IndexMap<_, [_; 768]>")]
    point_vector_map: IndexMap<Uuid, [f32; 768]>,
    point_metadata: Option<HashMap<Uuid, NekoPoint>>,
    point_metadata_ext: Option<HashMap<Uuid, NekoPointExt>>,
    point_url_prefix: Option<String>,
}

impl Display for PointExplorer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PointExplorer")
            .field("inner_size", &self.point_vector_map.len())
            .finish()
    }
}

impl PointExplorer {
    fn new() -> Self {
        Self::default()
    }

    fn with_capacity(capacity: usize) -> Self {
        Self {
            point_vector_map: IndexMap::with_capacity(capacity),
            point_metadata: None,
            point_metadata_ext: None,
            point_url_prefix: None,
        }
    }

    fn new_from_path(path: &str) -> PointExplorerResult<Self> {
        let data =
            fs::read(path).map_err(|_| PointExplorerError::PathNotFound(path.to_string()))?;
        let explorer: PointExplorer =
            serde_pickle::from_slice(&data, serde_pickle::DeOptions::default())
                .map_err(PointExplorerError::SerdeError)?;
        Ok(explorer)
    }

    fn load_metadata(&mut self, path: &str) -> PointExplorerResult<()> {
        let data =
            fs::read(path).map_err(|_| PointExplorerError::PathNotFound(path.to_string()))?;
        let metadata: HashMap<Uuid, NekoPoint> =
            serde_pickle::from_slice(&data, serde_pickle::DeOptions::default())
                .map_err(PointExplorerError::SerdeError)?;
        self.point_metadata = Some(metadata);
        Ok(())
    }

    fn load_metadata_ext(&mut self, path: &str) -> PointExplorerResult<()> {
        let data =
            fs::read(path).map_err(|_| PointExplorerError::PathNotFound(path.to_string()))?;
        let metadata_ext: HashMap<Uuid, NekoPointExt> =
            serde_pickle::from_slice(&data, serde_pickle::DeOptions::default())
                .map_err(PointExplorerError::SerdeError)?;
        self.point_metadata_ext = Some(metadata_ext);
        Ok(())
    }

    pub fn save(&self, path: &str) -> PointExplorerResult<()> {
        let data = serde_pickle::to_vec(self, serde_pickle::SerOptions::default())
            .map_err(PointExplorerError::SerdeError)?;
        fs::write(path, data).map_err(|_| PointExplorerError::PathNotFound(path.to_string()))?;
        Ok(())
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.point_vector_map.len()
    }

    #[inline]
    pub fn iter(&self) -> indexmap::map::Iter<'_, Uuid, [f32; 768]> {
        self.point_vector_map.iter()
    }

    pub fn insert(&mut self, point_id: &Uuid, vector: &[f32]) {
        debug_assert_eq!(vector.len(), 768, "Vector must be of length 768");
        self.point_vector_map.insert(
            *point_id,
            vector.try_into().expect("Vector must be of length 768"),
        );
    }

    pub fn extend(&mut self, points: &[(&Uuid, &[f32])]) {
        self.point_vector_map.reserve(points.len());
        self.point_vector_map
            .extend(points.iter().copied().map(|(id, vector)| {
                debug_assert_eq!(vector.len(), 768, "Vector must be of length 768");
                let arr = vector.try_into().expect("Vector must be of length 768");
                (*id, arr)
            }));
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.point_vector_map.is_empty()
    }

    #[inline]
    pub fn get_vector(&self, point_id: &Uuid) -> Option<&[f32; 768]> {
        self.point_vector_map.get(point_id)
    }

    #[inline]
    pub fn contains(&self, point_id: &Uuid) -> bool {
        self.point_vector_map.contains_key(point_id)
    }

    #[inline]
    pub fn remove(&mut self, point_id: &Uuid) -> Option<[f32; 768]> {
        self.point_vector_map.shift_remove(point_id)
    }

    #[inline]
    pub fn clear(&mut self) {
        self.point_vector_map.clear();
    }

    #[inline]
    pub fn index2uuid(&self, index: usize) -> Option<Uuid> {
        self.point_vector_map.get_index(index).map(|(id, _)| *id)
    }

    pub fn get_similarity(&self, point_id: (&Uuid, &Uuid)) -> PointExplorerResult<f32> {
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

    pub fn get_point_metadata(&self, point_id: &Uuid) -> Option<&NekoPoint> {
        self.point_metadata.as_ref()?.get(point_id)
    }

    pub fn get_point_url(&self, point_id: &Uuid) -> Option<String> {
        if self.point_url_prefix.is_none() && self.point_metadata_ext.is_none() {
            return None;
        }
        self.point_metadata_ext
            .as_ref()?
            .get(point_id)
            .and_then(|point| {
                Some(format!(
                    "{}{}{}",
                    self.point_url_prefix.as_deref().unwrap(),
                    point_id,
                    point.ext(),
                ))
            })
    }
}

#[cfg(feature = "point-explorer-pyo3")]
pub(super) mod pyo3 {
    use crate::point_explorer::{PointExplorer, PointExplorerBuilder, PointExplorerError};
    use crate::structure::NekoPoint;
    use pyo3::exceptions::{PyIOError, PyKeyError, PyValueError};
    use pyo3::prelude::*;
    use uuid::Uuid;

    impl From<PointExplorerError> for PyErr {
        fn from(err: PointExplorerError) -> PyErr {
            match err {
                PointExplorerError::PathNotFound(msg) => PyIOError::new_err(msg),
                PointExplorerError::SerdeError(e) => PyValueError::new_err(e.to_string()),
                PointExplorerError::PointNotFound(id) => {
                    PyKeyError::new_err(format!("Point with ID {} not found", id))
                }
            }
        }
    }

    #[pyclass]
    pub struct PyPointExplorerBuilder {
        builder: PointExplorerBuilder,
    }

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

        pub fn data_path<'a>(
            mut slf: PyRefMut<'a, Self>,
            path: String,
        ) -> PyResult<PyRefMut<'a, Self>> {
            slf.builder = slf.builder.clone().data_path(path);
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

        pub fn point_url_prefix<'a>(
            mut slf: PyRefMut<'a, Self>,
            prefix: String,
        ) -> PyResult<PyRefMut<'a, Self>> {
            slf.builder = slf.builder.clone().point_url_prefix(prefix);
            Ok(slf)
        }

        pub fn build(&self) -> PyResult<PyPointExplorer> {
            let explorer = self.builder.clone().build().map_err(PyErr::from)?;
            Ok(PyPointExplorer { inner: explorer })
        }
    }

    #[pyclass]
    pub struct PyPointExplorer {
        inner: PointExplorer,
    }

    #[pymethods]
    impl PyPointExplorer {
        #[new]
        #[pyo3(signature=(capacity=None))]
        pub fn new(capacity: Option<usize>) -> Self {
            // TODO: add more args?
            let inner = match capacity {
                Some(cap) => PointExplorer::with_capacity(cap),
                None => PointExplorer::new(),
            };
            PyPointExplorer { inner }
        }

        pub fn save(&self, path: String) -> PyResult<()> {
            self.inner.save(&path).map_err(|e| e.into())
        }

        pub fn insert(&mut self, point_id: String, vector: Vec<f32>) -> PyResult<()> {
            let uuid = Uuid::parse_str(&point_id)
                .map_err(|e| PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
            if vector.len() != 768 {
                return Err(PyValueError::new_err(format!(
                    "Vector must be of length 768, got {}",
                    vector.len()
                )));
            }
            self.inner.insert(&uuid, &vector);
            Ok(())
        }

        pub fn extend(&mut self, points: Vec<(String, Vec<f32>)>) -> PyResult<()> {
            let mut parsed_points = Vec::with_capacity(points.len());
            for (id_str, vector) in points {
                let uuid = Uuid::parse_str(&id_str)
                    .map_err(|e| PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
                if vector.len() != 768 {
                    return Err(PyValueError::new_err(format!(
                        "Vector must be of length 768, got {}",
                        vector.len()
                    )));
                }
                parsed_points.push((uuid, vector));
            }
            let points_refs: Vec<(&Uuid, &[f32])> = parsed_points
                .iter()
                .map(|(id, vec)| (id, vec.as_slice()))
                .collect();
            self.inner.extend(&points_refs);
            Ok(())
        }

        pub fn contains(&self, point_id: String) -> PyResult<bool> {
            let uuid = Uuid::parse_str(&point_id)
                .map_err(|e| PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
            Ok(self.inner.contains(&uuid))
        }

        pub fn remove(&mut self, point_id: String) -> PyResult<Option<Vec<f32>>> {
            let uuid = Uuid::parse_str(&point_id)
                .map_err(|e| PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
            Ok(self.inner.remove(&uuid).map(|arr| arr.to_vec()))
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

        pub fn index_to_uuid(&self, index: usize) -> Option<String> {
            self.inner.index2uuid(index).map(|uuid| uuid.to_string())
        }

        pub fn get_all_ids(&self) -> Vec<String> {
            self.inner.iter().map(|(id, _)| id.to_string()).collect()
        }

        pub fn get_all_vectors(&self) -> Vec<Vec<f32>> {
            self.inner
                .iter()
                .map(|(_, vector)| vector.to_vec())
                .collect()
        }

        pub fn get_items(&self) -> Vec<(String, Vec<f32>)> {
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
            format!("PyPointExplorer(size={})", self.len())
        }

        pub fn __str__(&self) -> String {
            format!("PointExplorer with {} points", self.len())
        }

        pub fn __iter__(slf: PyRef<'_, Self>) -> PyPointExplorerIterator {
            PyPointExplorerIterator {
                ids: slf.get_all_ids(),
                index: 0,
            }
        }

        pub fn get_similarity(&self, id_a: String, id_b: String) -> PyResult<f32> {
            let uuid_a = Uuid::parse_str(&id_a)
                .map_err(|e| PyValueError::new_err(format!("Invalid UUID for id_a: {}", e)))?;
            let uuid_b = Uuid::parse_str(&id_b)
                .map_err(|e| PyValueError::new_err(format!("Invalid UUID for id_b: {}", e)))?;
            self.inner
                .get_similarity((&uuid_a, &uuid_b))
                .map_err(|e| e.into())
        }

        pub fn get_vector(&self, point_id: String) -> PyResult<Option<Vec<f32>>> {
            let uuid = Uuid::parse_str(&point_id)
                .map_err(|e| PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
            Ok(self.inner.get_vector(&uuid).map(|v| v.to_vec()))
        }

        pub fn get_point_metadata(&self, point_id: String) -> PyResult<Option<NekoPoint>> {
            let uuid = Uuid::parse_str(&point_id)
                .map_err(|e| PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
            Ok(self.inner.get_point_metadata(&uuid).cloned())
        }

        pub fn get_point_url(&self, point_id: String) -> PyResult<Option<String>> {
            let uuid = Uuid::parse_str(&point_id)
                .map_err(|e| PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
            Ok(self.inner.get_point_url(&uuid))
        }
    }

    #[pyclass]
    pub struct PyPointExplorerIterator {
        ids: Vec<String>,
        index: usize,
    }

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
        let mut explorer = PointExplorer::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let v1 = vec![1.0; 768];
        let v2 = vec![1.0; 768];
        explorer.insert(&id1, &v1);
        explorer.insert(&id2, &v2);
        assert_eq!(explorer.len(), 2);
        let sim = explorer.get_similarity((&id1, &id2)).unwrap();
        assert!((sim - 1.0).abs() < EPS);
    }

    #[test]
    fn batch_insert_and_error_handling() {
        let mut explorer = PointExplorer::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let v1 = make_unit_vector(768, 0);
        let v2 = make_unit_vector(768, 1);
        explorer.extend(&[(&id1, &v1), (&id2, &v2)]);
        let uuid_except_1 = explorer.index2uuid(1);
        assert_eq!(uuid_except_1, Some(id2));
        let sim = explorer.get_similarity((&id1, &id2)).unwrap();
        assert!(sim.abs() < EPS);
        let missing = Uuid::new_v4();
        let err = explorer.get_similarity((&id1, &missing)).unwrap_err();
        assert!(matches!(err, PointExplorerError::PointNotFound(_)));
    }

    #[test]
    fn test_index2uuid() {
        let mut explorer = PointExplorer::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        explorer.insert(&id1, &vec![0.0; 768]);
        explorer.insert(&id2, &vec![1.0; 768]);
        assert_eq!(explorer.index2uuid(0), Some(id1));
        assert_eq!(explorer.index2uuid(1), Some(id2));
        assert_eq!(explorer.index2uuid(2), None);
    }

    #[test]
    fn serialize_deserialize_simple() {
        let mut explorer = PointExplorer::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let v1 = vec![1.0; 768];
        let v2 = vec![2.0; 768];
        explorer.insert(&id1, &v1);
        explorer.insert(&id2, &v2);
        let pre_sim = explorer.get_similarity((&id1, &id2)).unwrap();
        let bytes = serde_pickle::to_vec(&explorer, SerOptions::default()).unwrap();
        let decoded: PointExplorer =
            serde_pickle::from_slice(&bytes, DeOptions::default()).unwrap();
        let post_sim = decoded.get_similarity((&id1, &id2)).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(pre_sim, post_sim);
    }

    #[test]
    fn serialize_deserialize_large_random() {
        use rand::{Rng, SeedableRng};
        use rand_pcg::Pcg64;
        let mut explorer = PointExplorer::new();
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
            let expected = explorer.get_similarity((id1, id2)).unwrap();
            let bytes = serde_pickle::to_vec(&explorer, SerOptions::default()).unwrap();
            let decoded: PointExplorer =
                serde_pickle::from_slice(&bytes, DeOptions::default()).unwrap();
            let actual = decoded.get_similarity((id1, id2)).unwrap();
            assert!((expected - actual).abs() < EPS, "pair {}-{} differs", i, j);
        }
    }
}
