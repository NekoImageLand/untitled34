use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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
    pub text_vector: Vec<f32>,
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

/// P3
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum EntryMode {
    FILE,
    DIR,
    Unknown,
}

impl From<opendal::EntryMode> for EntryMode {
    fn from(mode: opendal::EntryMode) -> Self {
        match mode {
            opendal::EntryMode::FILE => EntryMode::FILE,
            opendal::EntryMode::DIR => EntryMode::DIR,
            opendal::EntryMode::Unknown => EntryMode::Unknown,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct BytesContentRange(pub Option<u64>, pub Option<u64>, pub Option<u64>);

impl From<opendal::raw::BytesContentRange> for BytesContentRange {
    fn from(raw: opendal::raw::BytesContentRange) -> Self {
        let size = raw.size();
        if let Some(ri) = raw.range_inclusive() {
            BytesContentRange(Some(*ri.start()), Some(*ri.end()), size)
        } else {
            BytesContentRange(None, None, size)
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Metadata {
    pub mode: EntryMode,
    pub is_current: Option<bool>,
    pub is_deleted: bool,
    pub cache_control: Option<String>,
    pub content_disposition: Option<String>,
    pub content_length: Option<u64>,
    pub content_md5: Option<String>,
    pub content_range: Option<BytesContentRange>,
    pub content_type: Option<String>,
    pub content_encoding: Option<String>,
    pub etag: Option<String>,
    pub last_modified: Option<DateTime<Utc>>,
    pub version: Option<String>,
    pub user_metadata: Option<HashMap<String, String>>,
}

impl From<opendal::Metadata> for Metadata {
    fn from(m: opendal::Metadata) -> Self {
        Metadata {
            mode: m.mode().into(),
            is_current: m.is_current(),
            is_deleted: m.is_deleted(),
            cache_control: m.cache_control().map(String::from),
            content_disposition: m.content_disposition().map(String::from),
            content_length: Some(m.content_length()), // TODO: 0?
            content_md5: m.content_md5().map(String::from),
            content_range: m.content_range().map(BytesContentRange::from),
            content_type: m.content_type().map(String::from),
            content_encoding: m.content_encoding().map(String::from),
            etag: m.etag().map(String::from),
            last_modified: m.last_modified().map(DateTime::<Utc>::from),
            version: m.version().map(String::from),
            user_metadata: m.user_metadata().map(|hm| hm.clone()),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Entry {
    pub path: String,
    pub metadata: Metadata,
}

impl From<opendal::Entry> for Entry {
    fn from(e: opendal::Entry) -> Self {
        Entry {
            path: e.path().to_string(),
            metadata: e.metadata().clone().into(),
        }
    }
}
