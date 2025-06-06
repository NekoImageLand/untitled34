use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

const THRESHOLD: f32 = 0.985;

fn cluster_chunk(ids: &[Uuid], sim_map: &HashMap<Uuid, HashMap<Uuid, f32>>) -> Vec<HashSet<Uuid>> {
    let mut clusters: Vec<HashSet<Uuid>> = Vec::new(); // a b c d e
    for &id in ids {
        let mut placed = false;
        for cl in clusters.iter_mut() {
            let ok = cl.iter().all(|&other| {
                let s = sim_map
                    .get(&id)
                    .and_then(|m| m.get(&other))
                    .copied()
                    .or_else(|| sim_map.get(&other).and_then(|m| m.get(&id)).copied())
                    .unwrap_or(0.0);
                s > THRESHOLD
            });
            if ok {
                cl.insert(id);
                placed = true;
                break;
            }
        }
        if !placed {
            let mut newc = HashSet::new();
            newc.insert(id);
            clusters.push(newc);
        }
    }
    clusters
}

fn merge_cluster(
    local: HashSet<Uuid>,
    global: &mut Vec<HashSet<Uuid>>,
    sim_map: &HashMap<Uuid, HashMap<Uuid, f32>>,
) {
    for g in global.iter_mut() {
        let ok = local.iter().all(|&i| {
            g.iter().all(|&j| {
                let s = sim_map
                    .get(&i)
                    .and_then(|m| m.get(&j))
                    .copied()
                    .or_else(|| sim_map.get(&j).and_then(|m| m.get(&i)).copied())
                    .unwrap_or(0.0);
                s > THRESHOLD
            })
        });
        if ok {
            g.extend(local.into_iter());
            return;
        }
    }
    global.push(local);
}

pub fn main() {
    let data = std::fs::read(
        r"C:\Users\pk5ls\Desktop\Projects\NekoImages\NekoImageGallery\merge\img_sim_clean.pkl",
    )
    .unwrap();
    // TODO:
    let sim_map: HashMap<Uuid, HashMap<Uuid, f32>> =
        serde_pickle::from_slice(&data, Default::default()).unwrap();

    let all_ids: Vec<Uuid> = sim_map.keys().copied().collect();
    let chunk_size = 20000;
    let chunks: Vec<&[Uuid]> = all_ids.chunks(chunk_size).collect();

    let m = MultiProgress::new();
    let pb_local = m.add(ProgressBar::new(chunks.len() as u64));
    let style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("#>-");
    pb_local.set_style(style.clone());
    pb_local.set_message("Local clustering");

    let local_vec: Vec<Vec<HashSet<Uuid>>> = chunks
        .par_iter()
        .map(|&chunk| {
            let res = cluster_chunk(chunk, &sim_map);
            pb_local.inc(1);
            res
        })
        .collect();
    pb_local.finish_with_message("Local clustering done");

    let all_local_clusters: Vec<HashSet<Uuid>> = local_vec.into_iter().flatten().collect();
    let mut global_clusters = Vec::new();
    let pb_merge = m.add(ProgressBar::new(0));
    pb_merge.set_length(all_local_clusters.len() as u64);
    pb_merge.set_style(style);
    pb_merge.set_message("Global merging");
    for lc in all_local_clusters {
        merge_cluster(lc, &mut global_clusters, &sim_map);
        pb_merge.inc(1);
    }
    pb_merge.finish_with_message("Global merging done");

    let mut saved_file = std::fs::File::create(r"../global_clusters.pkl").unwrap();
    serde_pickle::to_writer(&mut saved_file, &global_clusters, Default::default()).unwrap();

    println!("最终得到 {} 个簇", global_clusters.len());
}
