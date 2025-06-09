use chrono::Local;
use ndarray::Array2;
use pacmap::fit_transform;
use shared::point_explorer::{PointExplorer, PointExplorerBuilder};
use std::io::Write;
use std::{env, fs};
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::prelude::*;
use tracing_subscriber::{EnvFilter, Layer};

fn main() -> anyhow::Result<()> {
    let stdout = tracing_subscriber::fmt::layer().with_filter(EnvFilter::new(
        env::var("STDOUT_LOG_LEVEL").unwrap_or_else(|_| "debug".to_string()),
    ));
    let file_appender = RollingFileAppender::new(Rotation::HOURLY, "logs", "stage12.log");
    let file = tracing_subscriber::fmt::layer()
        .with_writer(file_appender)
        .with_filter(EnvFilter::new(
            env::var("FILE_LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
        ));
    tracing_subscriber::registry()
        .with(stdout)
        .with(file)
        .init();
    let points = PointExplorerBuilder::new()
        .data_path("img_sim_clean_new.pkl")
        .build()?;
    let n = points.len();
    let mut points_vec = Vec::with_capacity(n * 768);
    for (_, vector) in points.iter() {
        points_vec.extend_from_slice(vector);
    }
    let arr2: Array2<f32> = Array2::from_shape_vec((n, 768), points_vec)?;
    tracing::info!(
        "Loaded {} points with shape {:?}",
        points.len(),
        arr2.shape()
    );
    let config = pacmap::Configuration::builder()
        .embedding_dimensions(10)
        .mid_near_ratio(0.5)
        .far_pair_ratio(2.0)
        .override_neighbors(15)
        .seed(1145141919810)
        .learning_rate(1.0)
        .num_iters((200, 200, 500))
        .snapshots(vec![
            100, 200, // Midpoint of attraction phase
            300, 400, // Midpoint of local structure
            500, 600, 700, 800, // Midpoint of global structure
        ])
        .build();
    let (embedding, snap) = fit_transform(arr2.view(), config)?;
    tracing::info!(
        "Successfully computed embedding with shape: {:?}",
        embedding.shape()
    );
    // save it
    let ts = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let emb_fname = format!("embedding_{}.pkl", ts);
    let snap_fname = format!("snapshots_{}.pkl", ts);
    let mut f_emb = fs::File::create(&emb_fname)?;
    let ser_emb = serde_pickle::to_vec(&embedding, serde_pickle::SerOptions::default())?;
    f_emb.write_all(&ser_emb)?;
    tracing::info!("Saved embedding to {}", emb_fname);
    let mut f_snap = fs::File::create(&snap_fname)?;
    let ser_snap = serde_pickle::to_vec(&snap, serde_pickle::SerOptions::default())?;
    f_snap.write_all(&ser_snap)?;
    tracing::info!("Saved snapshots to {}", snap_fname);
    Ok(())
}
