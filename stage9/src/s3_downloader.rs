use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use shared::opendal::GenShinOperator;
use std::ops::Deref;
use thiserror::Error;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

#[derive(Debug)]
struct Stage9OpenDALOperator {
    op: GenShinOperator,
    worker_num: usize,
    save_path: String,
    overwrite: bool,
}

#[derive(Debug)]
pub struct DownloadErrorFile<'a> {
    file: &'a Uuid,
    error: String,
}

#[derive(Debug, Error)]
pub enum DownloadError<'a> {
    #[error("Some files failed to download: {0:?}")]
    Final(Vec<DownloadErrorFile<'a>>),
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}

impl Deref for Stage9OpenDALOperator {
    type Target = GenShinOperator;

    fn deref(&self) -> &Self::Target {
        &self.op
    }
}

impl Stage9OpenDALOperator {
    fn new(worker_num: usize, save_path: &str, overwrite: bool) -> Result<Self, anyhow::Error> {
        let op = GenShinOperator::new()?;
        Ok(Self {
            op,
            worker_num,
            overwrite,
            save_path: save_path.to_string(),
        })
    }

    async fn download_files<'a>(&self, file_list: &'a [&'a Uuid]) -> Result<(), DownloadError<'a>> {
        let pb = ProgressBar::new(file_list.len() as u64);
        let style = ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .map_err(|e| DownloadError::Internal(e.into()))?;
        pb.set_style(style);
        pb.set_message("Downloading S3 files...");
        let mut stream = futures::stream::iter(file_list.iter().map(|file| {
            let op = self;
            let pb = pb.clone();
            async move {
                let triage = op.download_file_atomic(file).await;
                pb.inc(1);
                triage
            }
        }))
        .buffer_unordered(self.worker_num);
        let mut failed_tasks = Vec::new();
        while let Some(result) = stream.next().await {
            match result {
                Ok(_) => continue,
                Err(e) => {
                    tracing::error!("Error downloading file: {}", e.error);
                    failed_tasks.push(e)
                }
            }
        }
        pb.finish_with_message("Download completed");
        match failed_tasks.is_empty() {
            true => Ok(()),
            false => Err(DownloadError::Final(failed_tasks)),
        }
    }

    async fn download_file_atomic<'a>(&self, file: &'a Uuid) -> Result<(), DownloadErrorFile<'a>> {
        let file_name = file.to_string();
        let local_path = format!("{}/{}.gif", self.save_path, file_name);
        let s3_path = format!("NekoImage/{}.gif", file_name);
        match fs::try_exists(&local_path).await {
            Ok(true) if !self.overwrite => {
                tracing::warn!(
                    "File {} already exists and overwrite is not allowed",
                    local_path
                );
                return Ok(());
            },
            Err(e) => {
                return Err(DownloadErrorFile {
                    file,
                    error: e.to_string(),
                });
            }
            _ => {}
        }
        let mut buffer = Vec::<u8>::new();
        let mut stream = self
            .op
            .read(&s3_path)
            .await
            .map_err(|e| DownloadErrorFile {
                file,
                error: e.to_string(),
            })?;
        while let Some(chunk_res) = StreamExt::next(&mut stream).await {
            let chunk = chunk_res.map_err(|e| DownloadErrorFile {
                file,
                error: e.to_string(),
            })?;
            buffer.extend_from_slice(&chunk);
        }
        let mut fs_file = fs::File::create(&local_path)
            .await
            .map_err(|e| DownloadErrorFile {
                file,
                error: e.to_string(),
            })?;
        fs_file
            .write_all(&buffer)
            .await
            .map_err(|e| DownloadErrorFile {
                file,
                error: e.to_string(),
            })?;
        fs_file.flush().await.map_err(|e| DownloadErrorFile {
            file,
            error: e.to_string(),
        })?;
        Ok(())
    }
}

pub struct S3Downloader {
    op: Stage9OpenDALOperator,
    runtime: tokio::runtime::Runtime,
}

impl S3Downloader {
    pub fn new(worker_num: usize, save_path: &str, overwrite: bool) -> anyhow::Result<Self> {
        let op = Stage9OpenDALOperator::new(worker_num, save_path, overwrite)?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(op.worker_num)
            .enable_all()
            .build()
            .expect("Failed to create Tokio runtime");
        Ok(Self { op, runtime })
    }

    pub fn download_files<'a>(&self, file_list: &'a [&'a Uuid]) -> Result<(), DownloadError<'a>> {
        self.runtime.block_on(self.op.download_files(file_list))
    }
}
