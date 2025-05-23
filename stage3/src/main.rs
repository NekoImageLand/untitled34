use plotters::prelude::*;
use shared::NekoPoint;
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let global_clusters_data = std::fs::read(r"global_clusters.pkl")?;
    let global_clusters: Vec<HashSet<Uuid>> =
        serde_pickle::from_slice(&global_clusters_data, Default::default())?;

    let pf = std::fs::read(r"points_map.bin")?;
    let metadata: (HashMap<Uuid, NekoPoint>, _) =
        bincode::serde::decode_from_slice(&pf, bincode::config::standard())?;
    let _metadata_map = metadata.0;

    println!("Loaded global clusters, count = {}", global_clusters.len());

    let mut sizes: Vec<usize> = global_clusters
        .iter()
        .map(|cluster| cluster.len())
        .filter(|&l| l > 1)
        .collect();
    sizes.sort_unstable();
    let count = sizes.len();

    let min = *sizes.first().unwrap();
    let max = *sizes.last().unwrap();
    let sum: usize = sizes.iter().sum();
    let mean = sum as f64 / count as f64;
    let median = if count % 2 == 0 {
        (sizes[count / 2 - 1] + sizes[count / 2]) as f64 / 2.0
    } else {
        sizes[count / 2] as f64
    };

    // Mode calculation
    let mut freq: HashMap<usize, usize> = HashMap::new();
    for &size in &sizes {
        *freq.entry(size).or_insert(0) += 1;
    }
    let mode = freq
        .into_iter()
        .max_by_key(|&(_, freq)| freq)
        .map(|(size, _)| size)
        .unwrap_or(0);

    println!("Cluster size stats (excluding size=1):");
    println!("  Count  = {}", count);
    println!("  Min    = {}", min);
    println!("  Max    = {}", max);
    println!("  Mean   = {:.2}", mean);
    println!("  Median = {:.2}", median);
    println!("  Mode   = {}", mode);

    plot_distribution(&sizes, "cluster_size_distribution.png")?;
    println!("Saved size distribution plot to cluster_size_distribution.png");

    Ok(())
}

fn plot_distribution(sizes: &[usize], output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Count frequencies per size
    let mut freq_map: HashMap<usize, usize> = HashMap::new();
    for &size in sizes {
        *freq_map.entry(size).or_insert(0) += 1;
    }
    let mut data: Vec<(usize, usize)> = freq_map.into_iter().collect();
    data.sort_by_key(|&(size, _)| size);

    // Prepare drawing area
    let max_size = data.iter().map(|&(s, _)| s).max().unwrap();
    let max_count = data.iter().map(|&(_, c)| c).max().unwrap();
    let root = BitMapBackend::new(output_path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Cluster Size Distribution", ("sans-serif", 40))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0usize..max_size, 0usize..(max_count + 5))?;

    chart.configure_mesh().draw()?;

    chart.draw_series(
        data.iter()
            .map(|&(size, count)| Rectangle::new([(size, 0), (size + 1, count)], BLUE.filled())),
    )?;

    Ok(())
}
