use clap::Parser;
use image::imageops::FilterType;
use image_hasher::HasherConfig;
use indicatif::{ProgressBar, ProgressStyle};
use mimalloc::MiMalloc;
use rayon::iter::Either;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use shared::point_explorer::{PointExplorerBuilder, PointExplorerError};
use shared::structure::{NekoPointExt, NekoPointExtResource};
use std::collections::HashMap;
use std::path::PathBuf;
use std::str::FromStr;
use std::{env, fs};
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};
use uuid::Uuid;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    src_dir: PathBuf,
}

#[derive(Debug, thiserror::Error, Serialize, Deserialize)]
enum Stage16Error {
    #[error("IO error: {0}")]
    IoError(String),
    #[error("Image Error: {0}")]
    ImageError(String),
    #[error("UUID Parse Error: {0}")]
    UUidError(String),
    #[error("Point Explorer Error: {0}")]
    #[serde(skip)]
    PointExplorerError(#[from] PointExplorerError),
}

fn main() -> anyhow::Result<()> {
    let stdout = tracing_subscriber::fmt::layer().with_filter(EnvFilter::new(
        env::var("STDOUT_LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
    ));
    let file_appender = RollingFileAppender::new(Rotation::HOURLY, "logs", "stage16.log");
    let file = tracing_subscriber::fmt::layer()
        .with_writer(file_appender)
        .with_filter(EnvFilter::new(
            env::var("FILE_LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
        ));
    tracing_subscriber::registry()
        .with(stdout)
        .with(file)
        .init();
    let args = Args::parse();
    let all_files: Vec<PathBuf> = walkdir::WalkDir::new(&args.src_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .map(|e| e.into_path())
        .collect();
    let hasher = HasherConfig::new()
        .hash_alg(image_hasher::HashAlg::Median)
        .resize_filter(FilterType::Lanczos3)
        .preproc_dct()
        .hash_size(16, 16)
        .to_hasher();
    let pb = ProgressBar::new(all_files.len() as u64);
    let style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?;
    pb.set_style(style);
    pb.set_message("Working...");
    // HashMap<Uuid, NekoPointExt>
    let (final_res_ok, final_res_err): (Vec<(Uuid, Vec<u8>, NekoPointExt)>, Vec<Stage16Error>) =
        all_files
            .into_par_iter()
            .map(|file| {
                pb.inc(1);
                let file_path = file
                    .to_str()
                    .ok_or_else(|| Stage16Error::IoError("Invalid file path".to_string()))?;
                let file_id = file
                    .file_stem()
                    .and_then(|os| os.to_str())
                    .ok_or_else(|| Stage16Error::IoError("Invalid file stem".to_string()))?;
                let file_id = Uuid::from_str(file_id)
                    .map_err(|_| Stage16Error::UUidError(file_id.to_string()))?;
                let img =
                    image::open(&file).map_err(|e| Stage16Error::ImageError(e.to_string()))?;
                let hash = hasher.hash_image(&img);
                let ext = NekoPointExt {
                    source: Some(NekoPointExtResource::Local(String::from(file_path))),
                };
                Ok((file_id, hash.as_bytes().to_vec(), ext))
            })
            .partition_map(
                |res: Result<(Uuid, Vec<u8>, NekoPointExt), Stage16Error>| match res {
                    Ok(v) => Either::Left(v),
                    Err(err) => Either::Right(err),
                },
            );
    let (final_res_size, final_err_size) = (final_res_ok.len(), final_res_err.len());
    let mut point_explorer = PointExplorerBuilder::new()
        .capacity(final_res_ok.len())
        .build::<u8, 32>()?;
    let (point_pairs, ext_pairs): (Vec<(&Uuid, &Vec<u8>)>, Vec<(&Uuid, &NekoPointExt)>) =
        final_res_ok
            .iter()
            .map(|(id, hash, ext)| ((id, hash), (id, ext)))
            .unzip();
    point_explorer.extend(point_pairs);
    pb.finish();
    tracing::info!(
        "Processed {} files successfully, {} errors encountered",
        final_res_size,
        final_err_size
    );
    let timestamp = chrono::Utc::now().format("%Y%m%d%H%M%S").to_string();
    // serde ext_pairs to HashMap<Uuid, NekoPointExt>
    let ext_map: HashMap<Uuid, NekoPointExt> = ext_pairs
        .into_iter()
        .map(|(id, ext)| (id.clone(), ext.clone()))
        .collect();
    let ext_name = format!("stage16_ext_map_{}.pkl", timestamp);
    let ext_pkl = serde_pickle::to_vec(&ext_map, serde_pickle::SerOptions::default())
        .map_err(|e| Stage16Error::IoError(e.to_string()))?;
    fs::write(&ext_name, ext_pkl).map_err(|e| Stage16Error::IoError(e.to_string()))?;
    // final_res_err
    if !final_res_err.is_empty() {
        let err_name = format!("stage16_err_image_vec_{}.json", timestamp);
        let f = serde_json::to_string(&final_res_err)
            .map_err(|e| Stage16Error::IoError(e.to_string()))?;
        fs::write(&err_name, f.as_bytes()).map_err(|e| Stage16Error::IoError(e.to_string()))?;
    }
    // final
    let pe_name = format!("stage16_point_explorer_{}.bin", timestamp);
    point_explorer
        .save(&pe_name)
        .map_err(|e| Stage16Error::PointExplorerError(e))?;
    Ok(())
}
