use mimalloc::MiMalloc;
use ndarray::Array2;
use petal_clustering::{Fit, Optics};
use petal_neighbors::distance::Hamming;
use rand::prelude::*;
use rand::rng;
use shared::point_explorer::{PointExplorer, PointExplorerBuilder};
use std::collections::HashSet;
use std::env;
use std::path::PathBuf;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};
use uuid::Uuid;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() -> anyhow::Result<()> {
    let stdout = tracing_subscriber::fmt::layer().with_filter(EnvFilter::new(
        env::var("STDOUT_LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
    ));
    let file_appender = RollingFileAppender::new(Rotation::HOURLY, "logs", "stage19.log");
    let file = tracing_subscriber::fmt::layer()
        .with_writer(file_appender)
        .with_filter(EnvFilter::new(
            env::var("FILE_LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
        ));
    tracing_subscriber::registry()
        .with(stdout)
        .with(file)
        .init();
    let point_explorer: PointExplorer<u8, 32> = PointExplorerBuilder::new()
        .path(env::var("stage19_POINT_MAP")?)
        .metadata_ext_path(env::var("stage19_POINT_EXT")?)
        .point_url_prefix(env::var("stage19_POINT_URL_PREFIX")?)
        .build()?;
    let pre_knn: HashSet<Uuid> = serde_pickle::from_slice(
        &std::fs::read(env::var("stage19_POINT_KNN")?)?,
        serde_pickle::DeOptions::default(),
    )?;
    let pre_knn_vecs = pre_knn
        .iter()
        .map(|id| {
            point_explorer
                .get_vector(id)
                .expect("Failed to get point from PointExplorer")
                .map(|v| v as f32)
        })
        .collect::<Vec<_>>();
    let vecs = Array2::from_shape_vec(
        (pre_knn_vecs.len(), 32),
        pre_knn_vecs.into_iter().flatten().collect(),
    )?;
    let mut opt = Optics::new(10.0, 2, Hamming::default());
    let res = opt.fit(&vecs, None);
    tracing::info!("Optics clustering result: {:?}", res);
    // save res
    let res = serde_pickle::to_vec(&res, serde_pickle::SerOptions::default())?;
    let file_name = format!(
        "stage19_optics_results{}.pkl",
        chrono::Utc::now().timestamp()
    );
    std::fs::write(&file_name, res)?;
    tracing::info!("Saved clustering results to {}", file_name);
    Ok(())
}
