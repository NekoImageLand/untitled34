use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// P1
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
    pub text_vector: Vec<f32>, // 768 Dimension
}

/// P2
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WrongExtFile {
    pub path: String,
    pub expected_ext: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FailedExtFile {
    pub path: String,
    pub error: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum TriageFile {
    Wrong(WrongExtFile),
    Failed(FailedExtFile),
}
