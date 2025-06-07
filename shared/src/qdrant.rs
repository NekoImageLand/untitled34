use qdrant_client::config::CompressionEncoding;
use qdrant_client::{Qdrant, QdrantBuilder, QdrantError};
use std::env;
use std::ops::Deref;
use std::time::Duration;

pub type QdrantResult<T> = Result<T, QdrantError>; // TODO: extend it using thiserror

pub struct GenShinQdrantClient(Qdrant);

impl Deref for GenShinQdrantClient {
    type Target = Qdrant;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl GenShinQdrantClient {
    pub fn new() -> anyhow::Result<Self> {
        let mut config = QdrantBuilder::from_url(&env::var("QDRANT_URL")?)
            .compression(Some(CompressionEncoding::Gzip));
        config.api_key = match env::var("QDRANT_API_KEY") {
            Ok(key) => Some(key),
            Err(_) => None,
        };
        config.timeout = match env::var("QDRANT_TIMEOUT") {
            Ok(key) => Duration::from_secs(key.parse()?),
            Err(_) => Duration::from_secs(3600),
        };
        config.check_compatibility = true;
        Ok(GenShinQdrantClient(config.build()?))
    }
}
