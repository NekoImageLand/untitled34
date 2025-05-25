use clap::Parser;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::{fs, path::PathBuf};
use walkdir::WalkDir;

#[derive(Parser, Debug)]
#[command(
    name = "ext-checker",
    version,
    about = "Scan a directory and verify file extensions against their content type"
)]
struct Cli {
    #[arg(short, long)]
    path: PathBuf,
    #[arg(short, long)]
    recursive: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    println!("Scanning directory: {:?}", cli.path);

    let walker = if cli.recursive {
        WalkDir::new(&cli.path)
    } else {
        WalkDir::new(&cli.path).max_depth(1)
    };

    let paths: Vec<PathBuf> = walker
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .collect();
    println!("Number of files found: {}", paths.len());

    let pb = ProgressBar::new(paths.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})",
        )?
        .progress_chars("#>-"),
    );

    let records: Vec<(Option<shared::WrongExtFile>, Option<shared::FailedExtFile>)> = paths
        .into_par_iter()
        .progress_with(pb)
        .map(|path| {
            let path_str = path.to_string_lossy().into_owned();
            match infer::get_from_path(&path) {
                Ok(Some(kind)) => {
                    let detected = kind.extension().to_string();
                    let actual = path
                        .extension()
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_ascii_lowercase());
                    if actual.as_deref() != Some(detected.as_str()) {
                        (
                            Some(shared::WrongExtFile {
                                path: path_str,
                                expected_ext: detected,
                            }),
                            None,
                        )
                    } else {
                        (None, None)
                    }
                }
                Ok(None) => (
                    None,
                    Some(shared::FailedExtFile {
                        path: path_str,
                        error: "Unknown file type".into(),
                    }),
                ),
                Err(e) => (
                    None,
                    Some(shared::FailedExtFile {
                        path: path_str,
                        error: e.to_string(),
                    }),
                ),
            }
        })
        .collect();

    let wrongs: Vec<shared::WrongExtFile> = records.iter().filter_map(|(w, _)| w.clone()).collect();
    let fails: Vec<shared::FailedExtFile> = records.iter().filter_map(|(_, f)| f.clone()).collect();

    fs::write("wrong_files.json", serde_json::to_string_pretty(&wrongs)?)?;
    fs::write("failed_files.json", serde_json::to_string_pretty(&fails)?)?;

    println!(
        "Done: {} mismatches (wrong_files.json), {} failures (failed_files.json)",
        wrongs.len(),
        fails.len()
    );
    Ok(())
}
