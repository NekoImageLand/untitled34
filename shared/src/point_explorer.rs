use crate::cosine_sim::cosine_sim;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::collections::{HashMap, hash_map};
use std::fmt::{Debug, Display};
use uuid::Uuid;

#[serde_as]
#[derive(Default, Debug, Serialize, Deserialize)]
pub struct PointExplorer {
    #[serde_as(as = "HashMap<_, [_; 768]>")]
    point_vector_map: HashMap<Uuid, [f32; 768]>,
}

impl Display for PointExplorer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PointExplorer")
            .field("inner_size", &self.point_vector_map.len())
            .finish()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PointExplorerError {
    #[error("Point with ID {0} not found")]
    PointNotFound(Uuid),
}

pub type PointExplorerResult<T> = Result<T, PointExplorerError>;

impl PointExplorer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            point_vector_map: HashMap::with_capacity(capacity),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.point_vector_map.len()
    }

    #[inline]
    pub fn iter(&self) -> hash_map::Iter<'_, Uuid, [f32; 768]> {
        self.point_vector_map.iter()
    }

    #[inline]
    pub fn insert(&mut self, point_id: &Uuid, vector: &[f32]) {
        debug_assert_eq!(vector.len(), 768, "Vector must be of length 768");
        self.point_vector_map.insert(
            *point_id,
            vector.try_into().expect("Vector must be of length 768"),
        );
    }

    pub fn insert_batch(&mut self, points: &[(&Uuid, &[f32])]) {
        self.point_vector_map.reserve(points.len());
        for &(point_id, vector) in points {
            self.insert(point_id, vector);
        }
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use bincode::{config, serde};
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
        let mut explorer = PointExplorer::with_capacity(2);
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let v1 = make_unit_vector(768, 0);
        let v2 = make_unit_vector(768, 1);
        explorer.insert_batch(&[(&id1, &v1), (&id2, &v2)]);
        let sim = explorer.get_similarity((&id1, &id2)).unwrap();
        assert!(sim.abs() < EPS);
        let missing = Uuid::new_v4();
        let err = explorer.get_similarity((&id1, &missing)).unwrap_err();
        assert!(matches!(err, PointExplorerError::PointNotFound(_)));
    }

    #[test]
    fn serialize_deserialize_round_trip() {
        let mut explorer = PointExplorer::new();
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let v = make_unit_vector(768, 0);
        explorer.insert(&id1, &v);
        explorer.insert(&id2, &v);
        let expected_sim = explorer.get_similarity((&id1, &id2)).unwrap();
        let cfg = config::standard();
        let bytes = serde::encode_to_vec(&explorer, cfg).expect("serialize");
        let decoded: PointExplorer = serde::decode_from_slice(&bytes, config::standard())
            .expect("deserialize")
            .0;
        assert_eq!(explorer.len(), decoded.len());
        let sim_after = decoded.get_similarity((&id1, &id2)).unwrap();
        assert!((expected_sim - sim_after).abs() < EPS);
    }
}
