use shared::qdrant::GenShinQdrantClient;
use std::ops::Deref;

struct Stage0GenshinQdrantClient {
    client: GenShinQdrantClient,
    collection_name: String,
    worker_num: usize,
}

impl Deref for Stage0GenshinQdrantClient {
    type Target = GenShinQdrantClient;

    fn deref(&self) -> &Self::Target {
        &self.client
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    Ok(())
}
