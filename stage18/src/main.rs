use mimalloc::MiMalloc;
use ndarray::Array2;
use petal_clustering::{Fit, Optics};
use petal_neighbors::distance::Hamming;
use rand::prelude::*;
use rand::rng;
use shared::point_explorer::{PointExplorer, PointExplorerBuilder};
use std::collections::{HashMap, HashSet};
use std::env;
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
    let file_appender = RollingFileAppender::new(Rotation::HOURLY, "logs", "stage18.log");
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
        .path(env::var("STAGE18_POINT_MAP")?)
        .metadata_ext_path(env::var("STAGE18_POINT_EXT")?)
        .point_url_prefix(env::var("STAGE18_POINT_URL_PREFIX")?)
        .build()?;

    let point_explorer_keys = point_explorer.iter().map(|(k, _)| k).collect::<Vec<_>>();

    let hm_list = [
        "d1e89323-b7e2-5504-b7f5-ee7a4af642f7",
        "b6d9d291-773d-5f0a-9e75-a0a39e9c57ab",
        "a5f86426-c0fa-58ca-970e-77161c487ad4",
        "a7ea18cb-99fc-50a0-bbd4-44df5de010cc",
        "9a6b963e-5932-513c-a026-0e58457cfb69",
        "47a3709f-7572-52dc-b8d0-c4d3f07a2304",
        "9a3fc1e1-8972-5ce2-b1b6-08e1fb86bca5",
        "5d5c8589-689e-5196-9256-341929af13e4",
        "7729d7de-7fe5-5dd4-8860-b768df94628a",
        "a1e784ea-f983-5504-a503-6d01b2c3db53",
        "d2f2a6cb-c2ae-5aaa-b789-b808907bf3c6",
        "b40198bb-130f-51e6-b010-3593d51e7263",
        "7dcc8718-235a-5b72-87de-77d745b773c1",
        "fb8a9788-5165-5029-b876-6aee08c72afa",
        "74a248de-5bdf-57ce-9cbc-dff82f2380c5",
        "9291db30-255c-563d-819f-16356a39cba5",
        "070b8538-c85c-59c2-a0a2-69eb85063956",
        "b7977195-30f4-562d-82c9-aa374bdee1a0",
        "833623f2-e285-50f4-836d-1ec4bdc84833",
        "eff0216d-1c2f-5da7-a245-7c076d8bca5d",
        "c88cf8ec-1b89-5a80-b82c-d9f21625a7ac",
        "6bfb3911-fe2f-5ef3-89d9-a2b07074847b",
        "47e133c6-2523-50e1-b8d1-60348f90563c",
        "2c09f5b1-1e28-5a14-8f4e-1e2a0ab11473",
        "c61ed21f-375c-55e4-be4a-cbaff2988ba9",
        "afadf350-c46a-5c75-ad23-9fa8fa053b41",
        "3610ab55-8fb1-5efa-9b66-2e641d1cc42a",
        "99dcd8be-3491-51cd-b5be-da5bbfd03b56",
        "dd7ad935-3e17-544e-aeda-f2293cc95a63",
        "b43f2ec7-b950-5259-9b19-e1656ce60213",
        "815ed065-cf39-5fba-a867-4a7774a4de15",
        "257718f1-e701-5cb8-aab3-7c7d4ded8424",
    ];

    let hm_list_2 = [
        "cd29d18a-8d92-5d12-807c-d8f9031e877c",
        "5a21ca1a-0c16-5099-8488-5e4218a974a2",
        "4b3d7de0-4b3f-5dbc-bdcc-8e74c1f5415a",
        "24b40206-80b0-5a80-b80b-5f3e8a151495",
        "cf097eab-ca60-5c43-bd3d-35db46ea7a49",
    ];

    let mut first_batch: Vec<Uuid> = hm_list
        .iter()
        .chain(hm_list_2.iter())
        .map(|s| Uuid::parse_str(s).expect("invalid UUID in hm_list"))
        .collect();
    let exclude: HashSet<Uuid> = first_batch.iter().cloned().collect();
    let mut remaining: Vec<&Uuid> = point_explorer_keys
        .iter()
        .filter_map(|&id| {
            if !exclude.contains(id) {
                Some(id)
            } else {
                None
            }
        })
        .collect();
    let mut thread_rng = rng();
    remaining.shuffle(&mut thread_rng);
    let sample_200: Vec<&Uuid> = remaining.into_iter().take(200).collect();
    let combined_uuids: Vec<&Uuid> = first_batch
        .iter()
        .chain(sample_200.iter().map(|&uuid| uuid))
        .collect();
    let data: Vec<f32> = combined_uuids
        .iter()
        .filter_map(|uuid| point_explorer.get_vector(uuid))
        .flat_map(|point_data| point_data.iter().map(|&byte| byte as f32))
        .collect();
    let vecs: Array2<f32> = Array2::from_shape_vec((combined_uuids.len(), 32), data)
        .expect("Failed to create Array2 from data");
    let mut opt = Optics::new(10.0, 2, Hamming::default());
    let (clusters_map, noises) = opt.fit(&vecs, None);
    let uuid_clusters: HashMap<usize, Vec<&Uuid>> = clusters_map
        .into_iter()
        .map(|(cluster_id, indices)| {
            let uuids = indices.into_iter().map(|idx| combined_uuids[idx]).collect();
            (cluster_id, uuids)
        })
        .collect();
    tracing::info!("Optics clustering result: {:?}", uuid_clusters);
    Ok(())
}
