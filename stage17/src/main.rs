use hnsw_rs::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use mimalloc::MiMalloc;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use shared::point_explorer::{PointExplorer, PointExplorerBuilder};
use std::collections::HashSet;
use std::env;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::prelude::*;
use uuid::Uuid;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Debug, Serialize, Deserialize)]
struct SearchResult {
    uri: String,
    distance: f32,
}

fn hnsw_query(
    hnsw: &Hnsw<u8, DistHamming>,
    point_explorer: &PointExplorer<u8, 32>,
    query_ids: &[&str],
) -> Vec<Vec<SearchResult>> {
    let queries: Vec<(&str, [u8; 32])> = query_ids
        .iter()
        .map(|&id_str| {
            let uuid = Uuid::from_str(id_str).expect("invalid UUID in query_ids");
            let vec = *point_explorer.get_vector(&uuid).expect("point not found");
            (id_str, vec)
        })
        .collect();
    let mut all_results = Vec::new();
    for (id_str, query_vec) in queries {
        tracing::debug!("Querying for point id = {}", id_str);
        let mut result = Vec::new();
        let neighbors = hnsw.search(&query_vec, 200, 500);
        for n in neighbors {
            let id = point_explorer.index2uuid(n.d_id).unwrap();
            let uri = point_explorer.get_point_uri(id).unwrap_or_default();
            let res = SearchResult {
                uri,
                distance: n.distance,
            };
            tracing::debug!("Success query for id {}, res: {:?}", id_str, res);
            result.push(res);
        }
        all_results.push(result);
    }
    all_results
}

fn query(
    hnsw: &Hnsw<u8, DistHamming>,
    point_explorer: &PointExplorer<u8, 32>,
) -> anyhow::Result<()> {
    // query sample
    let res = hnsw_query(
        &hnsw,
        &point_explorer,
        &[
            "fd1faa7e-d9e2-5712-913d-bb72ba7447cd", // you
            "b43f2ec7-b950-5259-9b19-e1656ce60213", // 24
            "cd29d18a-8d92-5d12-807c-d8f9031e877c", // jenny
            "e2f6453d-3553-5db8-81c3-0c4a8762ffbd",
            "00008799-ad39-5f58-959b-12ae973421f7",
            "00068fdf-7c5a-57d9-b5c2-9a9845271f9d", // very simple
        ],
    );
    // save res
    let res = serde_pickle::to_vec(&res, serde_pickle::SerOptions::default())?;
    std::fs::write(PathBuf::from("stage17_query_results.pkl"), res)?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let stdout = tracing_subscriber::fmt::layer().with_filter(EnvFilter::new(
        env::var("STDOUT_LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
    ));
    let file_appender = RollingFileAppender::new(Rotation::HOURLY, "logs", "stage17.log");
    let file = tracing_subscriber::fmt::layer()
        .with_writer(file_appender)
        .with_filter(EnvFilter::new(
            env::var("FILE_LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
        ));
    tracing_subscriber::registry()
        .with(stdout)
        .with(file)
        .init();
    // stage16_point_explorer_20250611083440.pkl
    let point_explorer: PointExplorer<u8, 32> = PointExplorerBuilder::new()
        .path(env::var("STAGE17_POINT_MAP")?)
        .metadata_ext_path(env::var("STAGE17_POINT_EXT")?)
        .point_url_prefix(env::var("STAGE17_POINT_URL_PREFIX")?)
        .build()?;
    let all_vecs: Vec<Vec<u8>> = point_explorer
        .iter()
        .map(|(_id, arr)| arr.to_vec())
        .collect();
    let all_ids: Vec<&Uuid> = point_explorer.iter().map(|(id, _arr)| id).collect();
    let data: Vec<(&Vec<u8>, usize)> = all_vecs
        .iter()
        .enumerate()
        .map(|(idx, v)| (v, idx))
        .collect();
    tracing::info!("Successfully loaded {} points", data.len());
    let hnsw_base = env::var("STAGE17_HNSW_BASENAME").unwrap_or("stage17_hnsw".to_string());
    let hnsw_data = PathBuf::from(&hnsw_base).with_extension("hnsw.data");
    let hnsw_graph = PathBuf::from(&hnsw_base).with_extension("hnsw.graph");
    let hnsw_exists = hnsw_data.exists() && hnsw_graph.exists();
    let mut maybe_hnsw_io = if hnsw_exists {
        tracing::info!("Loading existing HNSW index from {}", hnsw_base);
        Some(HnswIo::new(Path::new("."), &hnsw_base))
    } else {
        tracing::info!("{} not found, Creating new HNSW index", hnsw_base);
        None
    };
    let mut hnsw = match maybe_hnsw_io {
        Some(ref mut hnsw_io) => hnsw_io.load_hnsw()?,
        None => {
            let mut hnsw = Hnsw::<u8, DistHamming>::new(48, data.len(), 16, 600, DistHamming);
            hnsw.set_extend_candidates(false);
            tracing::info!("Building HNSW index with {} points", data.len());
            hnsw.parallel_insert(&data);
            tracing::info!("Successfully built HNSW index with {} points", data.len());
            hnsw
        }
    };
    // debug
    hnsw.dump_layer_info();
    hnsw.set_searching_mode(true);
    let pb = ProgressBar::new(all_ids.len() as u64);
    let style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?;
    pb.set_style(style);
    pb.set_message("Working...");
    let points_knn_set = all_ids
        .into_par_iter()
        .flat_map(|id| {
            pb.inc(1);
            let id_index = point_explorer.uuid2index(id).expect("point not found");
            let vec = point_explorer.get_vector(id).expect("point not found");
            let neighbors = hnsw.search(vec, 200, 500);
            neighbors
                .iter()
                .filter(|n| n.distance <= 0.625 && n.d_id != id_index) // filter by distance threshold
                .map(|n| point_explorer.index2uuid(n.d_id).unwrap())
                .collect::<Vec<_>>()
        })
        .collect::<HashSet<&Uuid>>();
    pb.finish_with_message("KNN search completed");
    tracing::info!("Found {} unique points in KNN search", points_knn_set.len());
    // save knn set
    let knn_set_path = PathBuf::from(format!(
        "stage17_knn_set_{}.pkl",
        chrono::Utc::now().timestamp()
    ));
    let knn_set_data = serde_pickle::to_vec(&points_knn_set, serde_pickle::SerOptions::default())?;
    std::fs::write(knn_set_path, knn_set_data)?;
    // save hnsw
    if !hnsw_exists {
        tracing::info!("Saving HNSW index to {}", hnsw_base);
        let file_name = format!("stage17_hnsw_{}", chrono::Utc::now().timestamp());
        hnsw.file_dump(Path::new("."), &file_name)?;
    }
    Ok(())
}
