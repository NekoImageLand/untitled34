use clap::{ArgAction, ArgGroup, Parser};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use shared::neko_uuid::NekoUuid;
use shared::structure::WrongExtFile;
use std::cmp::min;
use std::io::Write;
use std::path::PathBuf;
use std::{env, fs};
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};
use walkdir::WalkDir;

#[derive(Debug, Copy, Clone, Default)]
enum Op {
    #[default]
    Copy,
    Move,
}

#[derive(Debug, Parser)]
#[command(group(ArgGroup::new("Op").args(&["copy", "r#move"]).multiple(false)))]
struct Args {
    #[arg(long, value_delimiter = ',')]
    #[arg(value_parser = clap::value_parser!(PathBuf))]
    src_paths: Vec<PathBuf>,
    #[arg(long)]
    dst_path: PathBuf,
    #[arg(long, group = "Op", action = ArgAction::SetTrue, help = "Copy files")]
    copy: bool,
    #[arg(long = "move", group = "Op", action = ArgAction::SetTrue, help = "Move files")]
    r#move: bool,
    #[arg(long, default_value = "true")]
    check_ext: bool,
}

#[derive(Debug, thiserror::Error, Serialize, Deserialize)]
enum Stage15Error {
    #[error("Failed to infer file {0} type!")]
    InferError(PathBuf),
    #[error("Failed to copy or move file {0} to {1}: {2}")]
    IOError(PathBuf, PathBuf, String),
}

type Stage15Result<T> = Result<T, Stage15Error>;

fn main() -> anyhow::Result<()> {
    let stdout = tracing_subscriber::fmt::layer().with_filter(EnvFilter::new(
        env::var("STDOUT_LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
    ));
    let file_appender = RollingFileAppender::new(Rotation::HOURLY, "logs", "stage15.log");
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
    let op = if args.r#move { Op::Move } else { Op::Copy };
    let mut all_files = Vec::new();
    for src in &args.src_paths {
        match src {
            src if src.is_dir() => {
                for entry in WalkDir::new(src)
                    .into_iter()
                    .filter_map(Result::ok)
                    .filter(|e| e.file_type().is_file())
                {
                    all_files.push(entry.into_path());
                }
            }
            src if src.is_file() => {
                all_files.push(src.clone());
            }
            _ => {
                tracing::error!(
                    "Source path {} is neither a file nor a directory",
                    src.display()
                );
            }
        }
    }
    tracing::info!(
        "Found {} files in {} directories",
        all_files.len(),
        args.src_paths.len()
    );
    let files_len = all_files.len();
    let neko_uuid = NekoUuid::new();
    let pb = ProgressBar::new(files_len as u64);
    let style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?;
    pb.set_style(style);
    pb.set_message("Working...");
    let res: Vec<Stage15Result<Option<WrongExtFile>>> = all_files
        .into_par_iter()
        .map(|file| {
            pb.inc(1);
            let src_path = file;
            let src_path_ext = src_path
                .extension()
                .and_then(|ext| ext.to_str())
                .unwrap_or_default();
            let file_contents = fs::read(&src_path).map_err(|e| {
                Stage15Error::IOError(src_path.clone(), PathBuf::new(), e.to_string())
            })?;
            let target_filename = neko_uuid.generate(file_contents.as_slice());
            let mut dst_path = args.dst_path.join(target_filename.to_string());
            dst_path.set_extension(src_path_ext);
            let mut maybe_wrong_ext: Option<WrongExtFile> = None;
            if args.check_ext {
                let file_infer_ext =
                    match infer::get(&file_contents[0..min(file_contents.len(), 8192 + 1)]) {
                        Some(typ) => typ.extension(),
                        _ => return Err(Stage15Error::InferError(src_path)),
                    };
                if src_path_ext != file_infer_ext {
                    tracing::debug!(
                        "File {} has extension {}, but inferred as {}",
                        src_path.display(),
                        src_path_ext,
                        file_infer_ext
                    );
                    dst_path.set_extension(file_infer_ext);
                    maybe_wrong_ext = Some(WrongExtFile {
                        path: dst_path.to_string_lossy().to_string(), // stage8 need it
                        expected_ext: file_infer_ext.to_string(),
                    });
                }
            }
            match &op {
                Op::Copy => {
                    fs::copy(&src_path, &dst_path).map_err(|e| {
                        Stage15Error::IOError(src_path.clone(), dst_path.clone(), e.to_string())
                    })?;
                    return Ok(maybe_wrong_ext);
                }
                Op::Move => fs::rename(&src_path, &dst_path).map_err(|e| {
                    Stage15Error::IOError(src_path.clone(), dst_path.clone(), e.to_string())
                })?,
            }
            Ok(maybe_wrong_ext)
        })
        .collect();
    pb.finish_with_message("Done!");
    let (wrong_ext_files, failed_res): (Vec<WrongExtFile>, Vec<Stage15Error>) = res
        .into_iter()
        .fold((Vec::new(), Vec::new()), |(mut wrong, mut error), r| {
            if let Ok(Some(w)) = r {
                wrong.push(w);
            } else if let Err(e) = r {
                error.push(e);
            }
            (wrong, error)
        });
    if !failed_res.is_empty() {
        let timestamp = chrono::Utc::now().format("%Y%m%d%H%M%S").to_string();
        let name = format!("stage15_failed_files_{}.json", timestamp);
        tracing::error!(
            "Found {} files with errors, saving to {}",
            failed_res.len(),
            &name
        );
        let f = serde_json::to_string(&failed_res)?;
        let mut file = fs::File::create(&name)?;
        file.write_all(f.as_bytes())?;
    }
    if !wrong_ext_files.is_empty() {
        let timestamp = chrono::Utc::now().format("%Y%m%d%H%M%S").to_string();
        let name = format!("stage15_wrong_ext_files_{}.json", timestamp);
        tracing::warn!(
            "Found {} files with wrong extensions, saving to {}",
            wrong_ext_files.len(),
            &name
        );
        let f = serde_json::to_string(&wrong_ext_files)?;
        let mut file = fs::File::create(&name)?;
        file.write_all(f.as_bytes())?;
    }
    tracing::info!(
        "Successfully processed {} files, which errors: {}",
        files_len,
        failed_res.len()
    );
    Ok(())
}
