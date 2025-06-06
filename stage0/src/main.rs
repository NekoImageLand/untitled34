use clap::Parser;
use shared::qdrant::GenShinQdrantClient;
use std::env;
use std::ops::Deref;
use std::sync::Arc;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

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

impl Stage0GenshinQdrantClient {
    pub fn new(collection_name: &str, worker_num: usize) -> anyhow::Result<Self> {
        let client = GenShinQdrantClient::new()?;
        Ok(Stage0GenshinQdrantClient {
            client,
            collection_name: collection_name.to_string(),
            worker_num,
        })
    }
}

#[derive(Parser, Debug)]
#[command(name = "Stage11", version)]
struct Cli {
    #[arg(long, default_value = "16")]
    worker_num: usize,
    #[arg(long, default_value = "qdrant_point_reset_errors")]
    save_result_prefix: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let stdout = tracing_subscriber::fmt::layer().with_filter(EnvFilter::new("info"));
    let file_appender = RollingFileAppender::new(Rotation::HOURLY, "logs", "stage0.log");
    let file = tracing_subscriber::fmt::layer()
        .with_writer(file_appender)
        .with_filter(EnvFilter::new("info"));
    tracing_subscriber::registry()
        .with(stdout)
        .with(file)
        .init();
    let collection_name = env::var("QDRANT_COLLECTION_NAME")?;
    let client = Arc::new(Stage0GenshinQdrantClient::new(
        &collection_name,
        cli.worker_num,
    )?);
    Ok(())
}
