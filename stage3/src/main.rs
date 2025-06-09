use clap::Parser;
use plotters::prelude::*;
use shared::structure::NekoPoint;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use uuid::Uuid;

#[derive(Parser)]
#[clap(author, version, about = "Cluster analysis with optional UUID lookup")]
struct Args {
    #[clap(short, long, default_value = "clusters.bin")]
    clusters: PathBuf,
    #[clap(short = 'm', long, default_value = "points_map.bin")]
    points_map: PathBuf,
    #[clap(short, long)]
    uuid: Option<Uuid>,
    #[clap(short, long, default_value = "cluster_size_distribution.png")]
    output: String,
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    // Load clusters
    let cluster_data = std::fs::read(&args.clusters)?;
    let global_clusters: Vec<HashSet<Uuid>> =
        bincode::serde::decode_from_slice(&cluster_data, bincode::config::standard())?.0;
    println!("Loaded global clusters, count = {}", global_clusters.len());

    // Compute sizes of clusters with more than one member
    let mut sizes: Vec<usize> = global_clusters
        .iter()
        .map(HashSet::len)
        .filter(|&len| len > 1)
        .collect();
    sizes.sort_unstable();

    let count = sizes.len();
    let min = *sizes.first().unwrap_or(&0);
    let max = *sizes.last().unwrap_or(&0);
    let sum: usize = sizes.iter().sum();
    let mean = sum as f64 / count as f64;
    let median = if count % 2 == 0 {
        (sizes[count / 2 - 1] + sizes[count / 2]) as f64 / 2.0
    } else {
        sizes[count / 2] as f64
    };
    let mode = sizes
        .iter()
        .fold(HashMap::new(), |mut acc, &v| {
            *acc.entry(v).or_insert(0) += 1;
            acc
        })
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
    println!("Sizes vector: {:?}", sizes);

    // Plot distribution
    plot_distribution(&sizes, &args.output)?;
    println!("Saved size distribution plot to {}", args.output);

    Ok(())
}

fn plot_distribution(sizes: &[usize], output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut freq_map: HashMap<usize, usize> = HashMap::new();
    for &size in sizes {
        *freq_map.entry(size).or_insert(0) += 1;
    }
    let mut data: Vec<(usize, usize)> = freq_map.into_iter().collect();
    data.sort_by_key(|&(size, _)| size);

    let max_size = data.iter().map(|&(s, _)| s).max().unwrap_or(0);
    let max_count = data.iter().map(|&(_, c)| c).max().unwrap_or(0);

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
