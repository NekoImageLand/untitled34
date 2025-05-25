use anyhow::Result;
use clap::Parser;
use shared::opendal::GenShinOperator;
use std::ops::Deref;
use std::path::Path;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{EnvFilter, prelude::*};

pub struct Stage5Operator(GenShinOperator);

impl Deref for Stage5Operator {
    type Target = GenShinOperator;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Stage5Operator {
    pub async fn filelist(&self, list_path: &str) -> Result<Vec<shared::opendal::Entry>> {
        let res = self.op.list(list_path).await?;
        tracing::info!("Fetched result from s3, len = {:?}", &res.len());
        let res: Vec<shared::opendal::Entry> =
            res.into_iter().map(shared::opendal::Entry::from).collect();
        Ok(res)
    }
}

#[derive(Parser, Debug)]
#[command(name = "Stage5", version)]
struct Cli {
    #[arg(long, default_value = "/")]
    filelist_bucket_path: String, // it is non-recursive
    #[arg(long, default_value = "opendal_list_file.bin")]
    filelist_checkpoint_path: String,
    #[arg(short, long, default_value = "false")]
    overwrite: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let stdout = tracing_subscriber::fmt::layer().with_filter(EnvFilter::new("info"));
    let file_appender = RollingFileAppender::new(Rotation::HOURLY, "logs", "stage5.log");
    let file = tracing_subscriber::fmt::layer()
        .with_writer(file_appender)
        .with_filter(EnvFilter::new("debug"));
    tracing_subscriber::registry()
        .with(stdout)
        .with(file)
        .init();

    let cli = Cli::parse();
    let checkpoint = Path::new(&cli.filelist_checkpoint_path);
    if checkpoint.exists() && !cli.overwrite {
        tracing::warn!("Checkpoint exists, skipping.");
        return Ok(());
    }
    if checkpoint.exists() {
        tracing::warn!("Overwriting existing checkpoint.");
    } else {
        tracing::info!("Creating new checkpoint.");
    }

    let op = Stage5Operator(GenShinOperator::new()?);
    let entries = op.filelist(&cli.filelist_bucket_path).await?;
    tracing::info!(
        "Saving {} entries to {}",
        entries.len(),
        cli.filelist_checkpoint_path
    );
    let res: Vec<shared::opendal::Entry> = entries
        .into_iter()
        .map(shared::opendal::Entry::from)
        .collect();
    let serialized = bincode::serde::encode_to_vec::<
        &Vec<shared::opendal::Entry>,
        bincode::config::Configuration,
    >(&res.clone().into(), bincode::config::standard())?;
    std::fs::write(&cli.filelist_checkpoint_path, &serialized)?;
    Ok(())
}
