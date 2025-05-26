use clap::Parser;
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use qdrant_client::Payload;
use qdrant_client::QdrantError;
use qdrant_client::qdrant::{PointsIdsList, PointsOperationResponse, SetPayloadPointsBuilder};
use serde::{Deserialize, Serialize};
use serde_json::json;
use shared::qdrant::GenShinQdrantClient;
use shared::structure::WrongExtFile;
use std::env;
use std::fs::File;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RenameOp {
    point_id: String,
    src: String,
    dst: String,
    target_ext: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct FailedRenameOp {
    #[serde(flatten)]
    op: RenameOp,
    error: String,
}

struct Stage8GenshinQdrantClient {
    client: GenShinQdrantClient,
    collection_name: String,
    dry_run: bool,
    worker_num: usize,
    url_prefix: String,
}

impl Deref for Stage8GenshinQdrantClient {
    type Target = GenShinQdrantClient;

    fn deref(&self) -> &Self::Target {
        &self.client
    }
}

impl Stage8GenshinQdrantClient {
    pub fn new(
        collection_name: &str,
        dry_run: bool,
        worker_num: usize,
        url_prefix: &str,
    ) -> anyhow::Result<Self> {
        let client = GenShinQdrantClient::new()?;
        Ok(Self {
            client,
            collection_name: collection_name.to_owned(),
            dry_run,
            worker_num,
            url_prefix: url_prefix.to_owned(),
        })
    }

    async fn set_payload_task(
        self: Arc<Self>,
        ops: &[RenameOp],
    ) -> anyhow::Result<Option<Vec<FailedRenameOp>>> {
        let pb = ProgressBar::new(ops.len() as u64);
        let style = ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?;
        pb.set_style(style);
        pb.set_message("Overwriting Qdrant payload...");
        let mut stream = futures::stream::iter(ops.into_iter().map(|op| {
            let client = self.clone();
            let pb = pb.clone();
            async move {
                let triage = client.set_payload_atomic(op).await;
                pb.inc(1);
                (op, triage)
            }
        }))
        .buffer_unordered(self.worker_num);
        let mut failed_tasks = Vec::new();
        while let Some((op, res)) = stream.next().await {
            match res {
                Ok(Some(res)) => {
                    tracing::debug!("Point {} overwritten successfully: {:?}", op.point_id, res);
                }
                Err(e) => {
                    tracing::error!("Failed to overwrite point {}: {}", op.point_id, e);
                    failed_tasks.push(FailedRenameOp {
                        op: op.clone(),
                        error: e.to_string(),
                    });
                }
                _ => {} // already handled
            }
        }
        pb.finish_with_message("Done");
        if failed_tasks.is_empty() {
            Ok(None)
        } else {
            Ok(Some(failed_tasks))
        }
    }

    async fn set_payload_atomic(
        self: Arc<Self>,
        op: &RenameOp,
    ) -> Result<Option<PointsOperationResponse>, QdrantError> {
        let url = format!("{}/{}.{}", &self.url_prefix, &op.point_id, &op.target_ext);
        let payload = Payload::try_from(json!({
            "format": op.target_ext.to_owned(),
            "url": url,
        }))?;
        if self.dry_run {
            tracing::info!(
                "Dry run: would overwrite point {} with URL {}, Payload: {:?}",
                &op.point_id,
                &url,
                &payload
            );
            return Ok(None);
        }
        self.client
            .set_payload(
                SetPayloadPointsBuilder::new(&self.collection_name, payload)
                    .points_selector(PointsIdsList {
                        ids: vec![op.point_id.to_owned().into()],
                    })
                    .wait(true),
            )
            .await
            .map(Some)
    }
}

#[derive(Parser, Debug)]
#[command(name = "Stage8", version)]
struct Cli {
    #[arg(long)]
    wrong_ext_file_list: PathBuf,
    #[arg(long, default_value = "false")]
    dry_run: bool,
    #[arg(long, default_value = "16")]
    worker_num: usize,
    #[arg(long, default_value = "qdrant_point_rename_errors")]
    save_result_prefix: String,
    #[arg(long, default_value = "http://127.0.0.1:10000/nekoimg/NekoImage")]
    url_prefix: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let stdout = tracing_subscriber::fmt::layer().with_filter(EnvFilter::new("info"));
    let file_appender = RollingFileAppender::new(Rotation::HOURLY, "logs", "stage8.log");
    let file = tracing_subscriber::fmt::layer()
        .with_writer(file_appender)
        .with_filter(EnvFilter::new("info"));
    tracing_subscriber::registry()
        .with(stdout)
        .with(file)
        .init();
    let cli = Cli::parse();
    let collection_name = env::var("QDRANT_COLLECTION_NAME")?;
    let client = Arc::new(Stage8GenshinQdrantClient::new(
        &collection_name,
        cli.dry_run,
        cli.worker_num,
        &cli.url_prefix,
    )?);
    let need_rename_filelist = File::open(&cli.wrong_ext_file_list)?;
    let need_rename_filelist =
        serde_json::from_reader::<File, Vec<WrongExtFile>>(need_rename_filelist)?;
    let rename_ops = need_rename_filelist
        .into_iter()
        .filter_map(|file| {
            let left = file.path.split('.').next()?;
            let point_id = left.split('/').last()?;
            Some(RenameOp {
                point_id: point_id.to_owned(),
                dst: format!("{}.{}", left, file.expected_ext.as_str()),
                src: file.path,
                target_ext: file.expected_ext,
            })
        })
        .collect::<Vec<_>>();
    let res = client.set_payload_task(&rename_ops).await?;
    if let Some(failed_tasks) = res {
        let filename = format!(
            "{}_{}.json",
            cli.save_result_prefix,
            chrono::Local::now().format("%Y%m%d_%H%M%S")
        );
        let failed_file = File::create(&filename)?;
        serde_json::to_writer_pretty(failed_file, &failed_tasks)?;
        tracing::error!(
            "Some tasks failed, details saved to {}. Total failed tasks: {}",
            &filename,
            failed_tasks.len()
        );
    } else {
        tracing::info!("All tasks completed successfully.");
    }
    Ok(())
}
