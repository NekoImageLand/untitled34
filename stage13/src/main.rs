use chrono::Local;
use ndarray::Array2;
use petal_clustering::{Dbscan, Fit, HDbscan};
use petal_neighbors::distance::{Cosine, Euclidean};
use std::io::Write;
use std::{env, fs};

fn main() -> anyhow::Result<()> {
    let points_file = fs::read(env::var("POINT_AFTER_PACMAP_PATH")?)?;
    let points: Array2<f32> =
        serde_pickle::from_slice(&points_file, serde_pickle::DeOptions::default())?;
    println!(
        "Successfully loaded {} points with shape {:?}",
        points.len(),
        points.shape()
    );
    let mut dbscan = Dbscan::new(0.015, 2, Cosine::default());
    let final_res = dbscan.fit(&points, None);
    println!(
        "DBSCAN clustering completed with {} clusters.",
        final_res.0.len()
    );
    // save final_res
    let ts = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let res_fname = format!("dbscan_result_{}.pkl", ts);
    let mut f_res = fs::File::create(&res_fname)?;
    let ser_res = serde_pickle::to_vec(&final_res, serde_pickle::SerOptions::default())?;
    f_res.write_all(&ser_res)?;
    let mut hdbscan = HDbscan {
        alpha: 1.,
        min_samples: 2,
        min_cluster_size: 2,
        metric: Euclidean::default(), // TODO:
        boruvka: true,
    };
    let final_res = hdbscan.fit(&points, None);
    println!(
        "HDBSCAN clustering completed with {} clusters.",
        final_res.0.len()
    );
    // save final_res
    let ts = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let res_fname = format!("hdbscan_result_{}.pkl", ts);
    let mut f_res = fs::File::create(&res_fname)?;
    let ser_res = serde_pickle::to_vec(&final_res, serde_pickle::SerOptions::default())?;
    f_res.write_all(&ser_res)?;
    Ok(())
}
