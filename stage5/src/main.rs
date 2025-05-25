use anyhow::Result;
use bytes::Buf;
use clap::Parser;
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use opendal::layers::{ConcurrentLimitLayer, RetryLayer, TracingLayer};
use opendal::{Operator, services};
use shared::{FailedExtFile, TriageFile, WrongExtFile};
use std::cmp::min;
use std::env;
use std::io::Write;
use std::ops::Deref;
use std::sync::Arc;
use tokio::time::Duration;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{EnvFilter, prelude::*};

pub struct GenShinOperator(pub Operator);

impl Deref for GenShinOperator {
    type Target = Operator;
    fn deref(&self) -> &Operator {
        &self.0
    }
}

impl GenShinOperator {
    pub async fn load_or_init_save_file_list(
        &self,
        cache_file_path: &str,
        list_path: &str,
    ) -> Result<Vec<shared::Entry>> {
        let res = match std::fs::File::open(cache_file_path) {
            Ok(file) => {
                tracing::info!("Loading existing file list: opendal_list_file.bin");
                let mut reader = std::io::BufReader::new(file);
                let entries: Vec<shared::Entry> =
                    bincode::serde::decode_from_reader(&mut reader, bincode::config::standard())?;
                tracing::info!("Loaded result from file: {:?}", &entries.len());
                Ok::<Vec<shared::Entry>, anyhow::Error>(entries)
            }
            Err(_) => {
                tracing::info!("File not found, creating new file list");
                let mut file = std::fs::File::create(cache_file_path)?;
                let res = self.list(list_path).await?;
                tracing::info!("Fetched result from s3, len = {:?}", &res.len());
                let res: Vec<shared::Entry> = res.into_iter().map(shared::Entry::from).collect();
                let serialized =
                    bincode::serde::encode_to_vec::<
                        &Vec<shared::Entry>,
                        bincode::config::Configuration,
                    >(&res.clone().into(), bincode::config::standard())?;
                file.write_all(&serialized)?;
                tracing::info!("Saved result: {:?}", &file);
                Ok::<Vec<shared::Entry>, anyhow::Error>(res)
            }
        }?;
        tracing::info!("Got result, len = {:?}", &res.len());
        Ok(res)
    }

    pub async fn verify(
        self: Arc<Self>,
        entries: Vec<shared::Entry>,
        worker_num: usize,
    ) -> Result<(Vec<WrongExtFile>, Vec<FailedExtFile>)> {
        let pb = ProgressBar::new(entries.len() as u64);
        let style = ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?;
        pb.set_style(style);
        pb.set_message("Validating extensions...");
        let mut stream = futures::stream::iter(entries.into_iter().map(|entry| {
            let op = self.clone();
            let pb = pb.clone();
            async move {
                let triage = op.verify_single_ext(entry).await?;
                pb.inc(1);
                Ok::<_, anyhow::Error>(triage)
            }
        }))
        .buffer_unordered(worker_num);
        let mut all_wrong = Vec::new();
        let mut all_failed = Vec::new();
        while let Some(res) = stream.next().await {
            if let Ok(Some(triage)) = res {
                match triage {
                    TriageFile::Wrong(w) => all_wrong.push(w),
                    TriageFile::Failed(f) => all_failed.push(f),
                }
            }
        }
        pb.finish_with_message("Validation complete");
        tracing::info!(
            "Validation complete：wrong_ext = {}, failed = {}",
            all_wrong.len(),
            all_failed.len()
        );
        Ok((all_wrong, all_failed))
    }

    pub async fn verify_single_ext(
        self: Arc<Self>,
        file: shared::Entry,
    ) -> Result<Option<TriageFile>> {
        let path = file.path;
        let len = file.metadata.content_length.unwrap_or_default();
        match self.read_with(&path).range(0..=min(len, 8192)).await {
            Ok(buf) => match infer::get(buf.chunk()) {
                Some(kind) => {
                    let inferred_ext = kind.extension();
                    let ori_ext = path.split('.').last().unwrap_or_default();
                    if inferred_ext != ori_ext {
                        tracing::debug!(
                            "verify_single_ext: File {:?} has wrong ext: {}, expected: {}",
                            path,
                            inferred_ext,
                            ori_ext
                        );
                        return Ok(Some(TriageFile::Wrong(WrongExtFile {
                            path: path.clone(),
                            expected_ext: inferred_ext.to_string(),
                        })));
                    }
                    Ok(None)
                }
                None => {
                    tracing::debug!(
                        "verify_single_ext: Failed to infer file type for: {:?}",
                        path
                    );
                    Ok(Some(TriageFile::Failed(FailedExtFile {
                        path: path.clone(),
                        error: "infer::get returned None".into(),
                    })))
                }
            },
            Err(e) => {
                tracing::debug!("verify_single_ext: Error reading {:?}: {}", path, e);
                Ok(Some(TriageFile::Failed(FailedExtFile {
                    path: path.clone(),
                    error: format!("read error: {}", e),
                })))
            }
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "Stage5", version)]
struct Cli {
    #[arg(long, default_value = "opendal_list_file.bin")]
    file_list_cache: String,
    #[arg(long, default_value = "/")]
    file_list_path: String, // it is non-recursive
    #[arg(short, long, default_value = "16")]
    worker_num: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let stdout_layer = tracing_subscriber::fmt::layer().with_filter(EnvFilter::new("info"));
    let file_appender = RollingFileAppender::new(Rotation::HOURLY, "logs", "stage5_opendal.log");
    let file_layer = tracing_subscriber::fmt::layer()
        .with_writer(file_appender)
        .with_filter(EnvFilter::new("warn"));
    tracing_subscriber::registry()
        .with(stdout_layer)
        .with(file_layer)
        .init();
    let builder = services::S3::default()
        .bucket(&env::var("S3_BUCKET")?)
        .access_key_id(&env::var("S3_ACCESS_KEY")?)
        .secret_access_key(&env::var("S3_SECRET_ACCESS_KEY")?)
        .endpoint(&env::var("S3_ENDPOINT")?)
        .region(&env::var("S3_REGION")?);
    let op = GenShinOperator(
        Operator::new(builder)?
            .layer(TracingLayer)
            .layer(
                RetryLayer::default()
                    .with_max_times(20)
                    .with_factor(1.5)
                    .with_min_delay(Duration::from_millis(50))
                    .with_max_delay(Duration::from_millis(20000)),
            )
            .layer(ConcurrentLimitLayer::new(4096))
            .finish(),
    );
    let res = op
        .load_or_init_save_file_list(&cli.file_list_cache, &cli.file_list_path)
        .await?;
    let (wrong_ext_files, failed_ext_files) = Arc::new(op).verify(res, 32).await?;
    tracing::info!(
        "Verification complete! wrong_ext_files: {}, failed_ext_files: {}",
        wrong_ext_files.len(),
        failed_ext_files.len()
    );
    // save it!
    let mut file = std::fs::File::create("wrong_ext_files.json")?;
    let serialized = serde_json::to_string_pretty(&wrong_ext_files)?;
    file.write_all(serialized.as_bytes())?;
    let mut file = std::fs::File::create("failed_ext_files.json")?;
    let serialized = serde_json::to_string_pretty(&failed_ext_files)?;
    file.write_all(serialized.as_bytes())?;
    tracing::info!("Saved wrong_ext_files.json and failed_ext_files.json");
    Ok(())
}
