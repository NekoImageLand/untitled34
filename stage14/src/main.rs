use indicatif::{ProgressBar, ProgressStyle};
use petgraph::unionfind::UnionFind;
use shared::cosine_sim::cosine_sim;
use shared::point_explorer::PointExplorerBuilder;
use shared::structure::IMAGE_SIM_THRESHOLD;
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

fn main() -> anyhow::Result<()> {
    let file_path = "img_sim_clean_new.pkl";

    println!("Loading data from '{}'...", file_path);
    let pe = PointExplorerBuilder::new().data_path(file_path).build()?;
    let n = pe.len();
    if n == 0 {
        println!("No points found in the file. Exiting.");
        return Ok(());
    }
    println!("Successfully loaded {} points.", n);
    let mut uf = UnionFind::<usize>::new(n);
    let vectors: Vec<_> = pe.iter().map(|(_, v)| v).collect();
    let total_pairs = if n > 1 { (n * (n - 1)) / 2 } else { 0 };

    let pb = ProgressBar::new(total_pairs as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) - ETA: {eta}")?
            .progress_chars("#>-"),
    );
    println!(
        "\nStarting clustering of {} pairs with threshold {}...",
        total_pairs, IMAGE_SIM_THRESHOLD
    );

    for i in 0..n {
        for j in (i + 1)..n {
            let vector_i = vectors[i];
            let vector_j = vectors[j];
            let similarity = cosine_sim(vector_i, vector_j);
            if similarity >= IMAGE_SIM_THRESHOLD {
                uf.union(i, j);
            }
            pb.inc(1);
        }
    }
    pb.finish_with_message("Clustering complete!");

    println!("\nExtracting cluster results...");
    let mut clusters: HashMap<usize, HashSet<Uuid>> = HashMap::new();
    for i in 0..n {
        let root = uf.find_mut(i);
        let uuid = pe.index2uuid(i).expect("Index should be valid");
        clusters.entry(root).or_default().insert(uuid);
    }
    let result_clusters: Vec<HashSet<Uuid>> = clusters.into_values().collect();

    println!("\n--- Clustering Finished ---");
    println!("Found {} clusters.", result_clusters.len());
    let mut sorted_clusters: Vec<_> = result_clusters.iter().collect();
    sorted_clusters.sort_by_key(|a| std::cmp::Reverse(a.len()));
    for (i, cluster) in sorted_clusters.iter().enumerate() {
        println!("\nCluster #{} (Size: {})", i + 1, cluster.len());
        for (j, uuid) in cluster.iter().take(5).enumerate() {
            println!("  - Member {}: {}", j + 1, uuid);
        }
        if cluster.len() > 5 {
            println!("  - ... and {} more members.", cluster.len() - 5);
        }
    }
    let r = bincode::serde::encode_to_vec(result_clusters, bincode::config::standard())
        .map_err(|e| anyhow::anyhow!("Failed to serialize clusters: {}", e))?;
    std::fs::write("clusters.bin", r)
        .map_err(|e| anyhow::anyhow!("Failed to write clusters to file: {}", e))?;
    Ok(())
}
