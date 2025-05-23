use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NekoPoint {
    pub id: Uuid,
    pub height: usize,
    pub weight: usize,
    pub size: Option<usize>,
    pub categories: Option<Vec<String>>,
    pub text_info: Option<NekoPointText>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NekoPointText {
    pub text: String,
    pub text_vector: Vec<f32>,
}
