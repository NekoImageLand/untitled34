use anyhow::Result;
use bytes::Buf;
use clap::Parser;
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use shared::opendal::GenShinOperator;
use shared::structure::{FailedExtFile, TriageFile, WrongExtFile};
use std::cmp::min;
use std::fs::File;
use std::io::Write;
use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

pub struct Stage6Operator {
    op: GenShinOperator,
    worker_num: usize,
}

impl Deref for Stage6Operator {
    type Target = GenShinOperator;

    fn deref(&self) -> &Self::Target {
        &self.op
    }
}

impl Stage6Operator {
    pub fn new(worker_num: usize) -> Result<Self> {
        let op = GenShinOperator::new()?;
        Ok(Self { op, worker_num })
    }

    pub async fn verify(
        self: Arc<Self>,
        entries: Vec<shared::opendal::Entry>,
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
        file: shared::opendal::Entry,
    ) -> Result<Option<TriageFile>> {
        let path = file.path;
        let len = file.metadata.content_length.unwrap_or_default();
        match self.op.read_with(&path).range(0..min(len, 8192 + 1)).await {
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
#[command(name = "Stage6", version)]
struct Cli {
    #[arg(long, default_value = "opendal_list_file.bin")]
    filelist_checkpoint_path: String,
    #[arg(short, long, default_value = "16")]
    worker_num: usize,
    #[arg(long)]
    include_exclude_file: Option<PathBuf>,
    #[arg(long)]
    include_files: Option<Vec<String>>,
    #[arg(long)]
    exclude_files: Option<Vec<String>>,
    #[arg(short, long, default_value = "ext_files")]
    save_result_prefix: String,
}

#[derive(Deserialize, Default)]
struct FilterConfig {
    include_files: Option<Vec<String>>,
    exclude_files: Option<Vec<String>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let stdout = tracing_subscriber::fmt::layer().with_filter(EnvFilter::new("info"));
    let file_appender = RollingFileAppender::new(Rotation::HOURLY, "logs", "stage6.log");
    let file = tracing_subscriber::fmt::layer()
        .with_writer(file_appender)
        .with_filter(EnvFilter::new("info"));
    tracing_subscriber::registry()
        .with(stdout)
        .with(file)
        .init();

    let cli = Cli::parse();
    let op = Stage6Operator::new(cli.worker_num)?;
    let checkpoint = File::open(cli.filelist_checkpoint_path)?;
    let mut reader = std::io::BufReader::new(checkpoint);
    let entries: Vec<shared::opendal::Entry> =
        bincode::serde::decode_from_reader(&mut reader, bincode::config::standard())?;
    let mut cfg = if let Some(path) = cli.include_exclude_file.as_ref() {
        let file = File::open(path)?;
        serde_json::from_reader(file)?
    } else {
        FilterConfig::default()
    };
    if cli.include_files.is_some() {
        cfg.include_files = cli.include_files.clone();
    }
    if cli.exclude_files.is_some() {
        cfg.exclude_files = cli.exclude_files.clone();
    }

    let entries: Vec<shared::opendal::Entry> = match (&cfg.include_files, &cfg.exclude_files) {
        (None, None) => entries.into_iter().collect(),
        _ => entries
            .into_iter()
            .filter(|entry| {
                cli.include_files
                    .as_ref()
                    .map_or(true, |inc| inc.iter().any(|f| entry.path.contains(f)))
                    && cli
                        .exclude_files
                        .as_ref()
                        .map_or(true, |exc| !exc.iter().any(|f| entry.path.contains(f)))
            })
            .collect(),
    };
    tracing::info!("Loaded {} entries from checkpoint", entries.len());

    let (wrong_ext_files, failed_ext_files) = Arc::new(op).verify(entries, cli.worker_num).await?;
    tracing::info!(
        "Verification complete! wrong_ext_files: {}, failed_ext_files: {}",
        wrong_ext_files.len(),
        failed_ext_files.len()
    );
    let mut file = File::create(format!("{}_wrong.json", &cli.save_result_prefix))?;
    let serialized = serde_json::to_string_pretty(&wrong_ext_files)?;
    file.write_all(serialized.as_bytes())?;
    let mut file = File::create(format!("{}_failed.json", &cli.save_result_prefix))?;
    let serialized = serde_json::to_string_pretty(&failed_ext_files)?;
    file.write_all(serialized.as_bytes())?;
    tracing::info!(
        "Saved results to {}_wrong.json and {}_failed.json",
        &cli.save_result_prefix,
        &cli.save_result_prefix
    );
    Ok(())
}
