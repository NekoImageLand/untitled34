use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use mimalloc::MiMalloc;
use qdrant_client::qdrant::vectors_output::VectorsOptions as VectorsOptionsOutput;
use qdrant_client::qdrant::{PointId, ScrollPointsBuilder, point_id};
use shared::point_explorer::{PointExplorer, PointExplorerBuilder};
use shared::qdrant::{GenShinQdrantClient, QdrantResult};
use std::env;
use std::ops::Deref;
use std::sync::Arc;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};
use uuid::Uuid;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

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

    pub async fn fetch_point_num(self: Arc<Self>) -> QdrantResult<u64> {
        let collection_info = self.client.collection_info(&self.collection_name).await?;
        Ok(collection_info.result.unwrap().points_count.unwrap())
    }

    pub async fn fetch_all_points(
        self: Arc<Self>,
        pre_num: usize,
    ) -> QdrantResult<Vec<(Uuid, Vec<f32>)>> {
        let pb = ProgressBar::new(pre_num as u64);
        let style = ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap();
        pb.set_style(style);
        pb.set_message("Overwriting Qdrant payload...");
        let mut offset: Option<PointId> = None;
        let mut out: Vec<(Uuid, Vec<f32>)> = Vec::with_capacity(pre_num);
        loop {
            let mut sc = ScrollPointsBuilder::new(&self.collection_name)
                .limit(1000)
                .with_payload(false)
                .with_vectors(true);
            if let Some(ov) = offset {
                sc = sc.offset(ov);
            }
            let resp = self.client.scroll(sc).await?;
            let size = resp.result.len();
            offset = resp.next_page_offset.to_owned();
            out.extend(resp.result.into_iter().filter_map(|mut p| {
                let uuid =
                    p.id.as_ref()
                        .and_then(|pid| pid.point_id_options.as_ref())
                        .and_then(|opt| match opt {
                            point_id::PointIdOptions::Uuid(s) => Some(Uuid::parse_str(s).ok()?),
                            _ => None,
                        })?;
                let vectors = p.vectors.take()?;
                let named = match vectors.vectors_options? {
                    VectorsOptionsOutput::Vectors(named) => named,
                    _ => return None,
                };
                let vec = named
                    .vectors
                    .into_iter()
                    .find(|(k, _)| k == "image_vector")?
                    .1
                    .data;
                Some((uuid, vec))
            }));
            pb.inc(size as u64);
            if offset.is_none() {
                break;
            }
        }
        Ok(out)
    }
}

#[derive(Parser, Debug)]
#[command(name = "Stage0", version)]
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
    let point_num = client.clone().fetch_point_num().await?;
    let points = client.clone().fetch_all_points(point_num as usize).await?;
    tracing::info!("Found {} points", points.len());
    let mut point_explorer: PointExplorer<f32, 768> =
        PointExplorerBuilder::new().capacity(points.len()).build()?;
    point_explorer.extend(points);
    tracing::info!("Saving {} points into PointExplorer", point_explorer.len());
    point_explorer.save("qdrant_point_explorer_250611.pkl")?; // TODO: with metadata?
    Ok(())
}
