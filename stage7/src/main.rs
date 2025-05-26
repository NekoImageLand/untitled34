use anyhow::Result;
use clap::Parser;
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use shared::opendal::GenShinOperator;
use shared::structure::WrongExtFile;
use std::borrow::Cow;
use std::collections::HashSet;
use std::ops::Deref;
use std::sync::Arc;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

#[derive(Serialize, Deserialize)]
#[serde(transparent)]
struct RenameFailedTask(WrongExtFile);

pub struct Stage7Operator {
    op: GenShinOperator,
    dry_run: bool,
    worker_num: usize,
    skip_ext_pairs: HashSet<(Cow<'static, str>, Cow<'static, str>)>,
}

impl Deref for Stage7Operator {
    type Target = GenShinOperator;

    fn deref(&self) -> &Self::Target {
        &self.op
    }
}

impl Stage7Operator {
    fn new(
        dry_run: bool,
        worker_num: usize,
        skip_ext_pairs: HashSet<(Cow<'static, str>, Cow<'static, str>)>,
    ) -> Result<Self> {
        let op = GenShinOperator::new()?;
        Ok(Self {
            op,
            dry_run,
            worker_num,
            skip_ext_pairs,
        })
    }

    async fn rename_task(
        self: Arc<Self>,
        files: Vec<WrongExtFile>,
    ) -> Result<Option<Vec<RenameFailedTask>>> {
        let pb = ProgressBar::new(files.len() as u64);
        let style = ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?;
        pb.set_style(style);
        pb.set_message("Renaming extensions...");
        let mut stream = futures::stream::iter(files.into_iter().map(|file| {
            let op = self.clone();
            let pb = pb.clone();
            async move {
                let triage = op.rename_single_task(file).await?;
                pb.inc(1);
                Ok::<_, anyhow::Error>(triage)
            }
        }))
        .buffer_unordered(self.worker_num);
        let mut failed_tasks = Vec::new();
        while let Some(res) = stream.next().await {
            match res {
                Ok(Some(task)) => failed_tasks.push(task),
                Ok(None) => {}
                Err(e) => {
                    tracing::error!("Error: {}", e);
                }
            }
        }
        pb.finish_with_message("Done");
        if failed_tasks.is_empty() {
            Ok(None)
        } else {
            Ok(Some(failed_tasks))
        }
    }

    async fn rename_single_task(
        self: Arc<Self>,
        file: WrongExtFile,
    ) -> Result<Option<RenameFailedTask>> {
        let wrong_ext = file.path.split('.').last().unwrap();
        let right_ext = &file.expected_ext;
        let wrong_file_path = &file.path;
        let right_file_path = format!(
            "{}.{}",
            file.path.split('.').next().unwrap(),
            file.expected_ext.as_str()
        );
        if self
            .skip_ext_pairs
            .contains(&(Cow::Borrowed(wrong_ext), Cow::Borrowed(right_ext)))
        {
            tracing::warn!(
                "Skipping rename from {} to {}",
                wrong_file_path,
                right_file_path
            );
            return Ok::<_, anyhow::Error>(None);
        }
        if self.dry_run {
            tracing::info!("Dry run: {} -> {}", wrong_file_path, right_file_path);
            return Ok(None);
        }
        match self
            .rename_atomic_task(&wrong_file_path, &right_file_path)
            .await
        {
            Ok(_) => {
                tracing::debug!("Renamed {} to {}", wrong_file_path, right_file_path);
                Ok(None)
            }
            Err(e) => {
                tracing::error!("Failed to rename {}: {}", wrong_file_path, e);
                Ok(Some(RenameFailedTask(file)))
            }
        }
    }

    async fn rename_atomic_task(self: Arc<Self>, src: &str, dst: &str) -> Result<()> {
        self.op.copy(src, dst).await?;
        self.op.delete(src).await?;
        Ok(())
    }
}

#[derive(Parser, Debug)]
#[command(name = "Stage7", version)]
struct Cli {
    #[arg(long, default_value = "ext_files_wrong.json")]
    wrong_file: String,
    #[arg(long, default_value = "16")]
    worker_num: usize,
    #[arg(long, default_value = "false")]
    dry_run: bool,
    #[arg(long, default_value = "ext_files_rename")]
    save_result_prefix: String,
    /// Skip renaming for these extensions
    /// Example: --skip-ext-pair jpeg jpg --skip-ext-pair png jpg
    #[arg(long,
          number_of_values = 2,
          value_names = &["FROM","TO"],
          action = clap::ArgAction::Append)]
    skip_ext_pair: Option<Vec<String>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let stdout = tracing_subscriber::fmt::layer().with_filter(EnvFilter::new("info"));
    let file_appender = RollingFileAppender::new(Rotation::HOURLY, "logs", "stage7.log");
    let file = tracing_subscriber::fmt::layer()
        .with_writer(file_appender)
        .with_filter(EnvFilter::new("info"));
    tracing_subscriber::registry()
        .with(stdout)
        .with(file)
        .init();
    let cli = Cli::parse();
    let skip_ext_pairs: HashSet<(Cow<'static, str>, Cow<'static, str>)> = cli
        .skip_ext_pair
        .unwrap_or_default()
        .chunks(2)
        .filter_map(|chunk| {
            if chunk.len() == 2 {
                Some((Cow::Owned(chunk[0].clone()), Cow::Owned(chunk[1].clone())))
            } else {
                None
            }
        })
        .collect();
    let op = Stage7Operator::new(cli.dry_run, cli.worker_num, skip_ext_pairs)?;
    let file = std::fs::File::open(cli.wrong_file)?;
    let files: Vec<WrongExtFile> = serde_json::from_reader(file)?;
    tracing::info!("Loaded {} files", files.len());
    let failed_tasks = Arc::new(op).rename_task(files).await?;
    if let Some(tasks) = failed_tasks {
        let save_path = format!("{}_failed.json", cli.save_result_prefix);
        tracing::info!("Saved failed tasks to {}", &save_path);
        let file = std::fs::File::create(save_path)?;
        serde_json::to_writer(file, &tasks)?;
    } else {
        tracing::info!("All tasks succeeded");
    }
    Ok(())
}
