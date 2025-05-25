use chrono::{DateTime, Utc};
use opendal::{Operator, services};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::ops::Deref;
use std::time::Duration;

#[cfg(feature = "opendal-data-compat")]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum EntryMode {
    FILE,
    DIR,
    Unknown,
}

#[cfg(feature = "opendal-ext")]
impl From<opendal::EntryMode> for EntryMode {
    fn from(mode: opendal::EntryMode) -> Self {
        match mode {
            opendal::EntryMode::FILE => EntryMode::FILE,
            opendal::EntryMode::DIR => EntryMode::DIR,
            opendal::EntryMode::Unknown => EntryMode::Unknown,
        }
    }
}

#[cfg(feature = "opendal-data-compat")]
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
pub struct BytesContentRange(pub Option<u64>, pub Option<u64>, pub Option<u64>);

#[cfg(feature = "opendal-ext")]
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

#[cfg(feature = "opendal-data-compat")]
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

#[cfg(feature = "opendal-ext")]
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

#[cfg(feature = "opendal-data-compat")]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Entry {
    pub path: String,
    pub metadata: Metadata,
}

#[cfg(feature = "opendal-ext")]
impl From<opendal::Entry> for Entry {
    fn from(e: opendal::Entry) -> Self {
        Entry {
            path: e.path().to_string(),
            metadata: e.metadata().clone().into(),
        }
    }
}

#[cfg(feature = "opendal-ext")]
pub struct GenShinOperator {
    pub op: Operator,
}

#[cfg(feature = "opendal-ext")]
impl Deref for GenShinOperator {
    type Target = Operator;

    fn deref(&self) -> &Self::Target {
        &self.op
    }
}

#[cfg(feature = "opendal-ext")]
use opendal::layers::{ConcurrentLimitLayer, RetryLayer, TracingLayer};
impl GenShinOperator {
    pub fn new() -> Result<Self, anyhow::Error> {
        let builder = services::S3::default()
            .bucket(&env::var("S3_BUCKET")?)
            .access_key_id(&env::var("S3_ACCESS_KEY")?)
            .secret_access_key(&env::var("S3_SECRET_ACCESS_KEY")?)
            .endpoint(&env::var("S3_ENDPOINT")?)
            .region(&env::var("S3_REGION")?);
        let op = Operator::new(builder)?
            .layer(TracingLayer)
            .layer(
                RetryLayer::default()
                    .with_max_times(20)
                    .with_factor(1.5)
                    .with_min_delay(Duration::from_millis(50))
                    .with_max_delay(Duration::from_millis(20000)),
            )
            .layer(ConcurrentLimitLayer::new(4096))
            .finish();
        Ok(GenShinOperator { op })
    }
}
