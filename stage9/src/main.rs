mod s3_downloader;

use anyhow::Result;
use rayon::prelude::*;
use shared::cosine_sim::cosine_sim;
use shared::structure::{NekoPoint, NekoPointExt, NekoPointExtResource};
use std::collections::{HashMap, HashSet};
use std::fs;
use uuid::Uuid;

const TEXT_SIM_THRESHOLD: f32 = 0.9;

fn find_text_anomalies<'a>(
    text_points: &[&'a Uuid],
    points_metadata: &HashMap<Uuid, (NekoPoint, NekoPointExt)>,
) -> Option<Vec<&'a Uuid>> {
    let mut id_vec_pairs = Vec::with_capacity(text_points.len());
    for &id in text_points {
        if let Some((pt, _)) = points_metadata.get(&id) {
            if let Some(ref txt) = pt.text_info {
                id_vec_pairs.push((id, txt.text_vector.as_slice()));
            }
        }
    }
    let mut anomalies = None;
    for &(id, vec_i) in &id_vec_pairs {
        let is_anomaly = id_vec_pairs
            .iter()
            .filter(|&&(other_id, _)| other_id != id)
            .all(|&(_, vec_j)| cosine_sim(vec_i, vec_j) < TEXT_SIM_THRESHOLD);
        if is_anomaly {
            if anomalies.is_none() {
                anomalies = Some(Vec::new());
            }
            anomalies.as_mut().unwrap().push(id);
        }
    }
    anomalies
}

fn extract_clusters<'a>(
    points_clusters: &'a [HashSet<Uuid>],
    points_metadata: &'a HashMap<Uuid, (NekoPoint, NekoPointExt)>,
) -> Vec<(
    Option<Vec<&'a Uuid>>,
    Option<Vec<&'a Uuid>>,
    Option<&'a Uuid>,
    Option<Vec<&'a Uuid>>,
)> {
    points_clusters
        .par_iter()
        .map(|cursor| {
            let cursor_ref: HashSet<&Uuid> = cursor.iter().collect();
            // stage1
            let text_points = cursor
                .iter()
                .filter(|id| {
                    points_metadata
                        .get(id)
                        .map(|(pt, _)| pt.text_info.is_some())
                        .unwrap_or(false)
                })
                .fold(None, |opt_vec: Option<Vec<&Uuid>>, id| {
                    Some(match opt_vec {
                        Some(mut v) => {
                            v.push(id);
                            v
                        }
                        None => vec![id],
                    })
                });
            let text_points_size = text_points.as_ref().map_or(0, |v| v.len());
            let text_anomalies = text_points
                .as_ref()
                .and_then(|tp| find_text_anomalies(tp, &points_metadata));
            let text_anomalies_set: HashSet<&Uuid> = text_anomalies
                .as_deref()
                .unwrap_or(&[])
                .iter()
                .copied()
                .collect();
            let non_text_anomalies_set: HashSet<&Uuid> = cursor_ref
                .difference(&text_anomalies_set)
                .copied()
                .collect();
            if text_points_size == cursor.len() && text_points == text_anomalies {
                // println!("All points in the current cluster have textual dissimilarity, skip!");
                return (
                    text_anomalies,
                    None,
                    None,
                    Some(non_text_anomalies_set.into_iter().collect()),
                );
            }
            // stage2
            let mut gif_points_in_left_points: Option<HashSet<&Uuid>> = None;
            let mut non_gif_points_in_left_points: Option<HashSet<&Uuid>> = None;
            for &id in non_text_anomalies_set.iter() {
                let is_gif = points_metadata
                    .get(id)
                    .map(|(_, ex)| ex.ext() == "gif")
                    .unwrap_or(false);
                match is_gif {
                    true => {
                        if gif_points_in_left_points.is_none() {
                            gif_points_in_left_points = Some(HashSet::new());
                        }
                        gif_points_in_left_points.as_mut().unwrap().insert(id);
                    }
                    false => {
                        if non_gif_points_in_left_points.is_none() {
                            non_gif_points_in_left_points = Some(HashSet::new());
                        }
                        non_gif_points_in_left_points.as_mut().unwrap().insert(id);
                    }
                }
            }
            // stage3 (Option<HashSet<&NeedTriageGifs>>, Option<&KeptNonGif>)
            let gif_spilt: (Option<HashSet<&Uuid>>, Option<&Uuid>) =
                match (gif_points_in_left_points, non_gif_points_in_left_points) {
                    (Some(gif), Some(_none_gif)) => (Some(gif), None),
                    (Some(gif), None) => (Some(gif), None),
                    (None, non_gif) => {
                        let maybe_biggest_non_gif = non_gif.and_then(|hs| {
                            hs.iter()
                                .max_by_key(|&&id| {
                                    points_metadata
                                        .get(&id)
                                        .map(|(pt, _)| pt.size.unwrap_or_default())
                                        .unwrap_or(0)
                                })
                                .cloned()
                        });
                        (None, maybe_biggest_non_gif)
                    }
                };
            // stage4 return it!
            // Return (1) Vec<(Option<Vec<KeptTextAnomaliesPic>>, (2) Option<Vec<NeedTriageGifs>>,
            // (3) Option<KeptNonGif>, (4) Option<Vec<OtherNeedDeletePics>>)>
            // Now we calculate Option<Vec<OtherNeedDeletePics>>
            // HashSet<OtherNeedDeletePics> = <HashSet>cursor_refs - <HashSet>text_anomalies - <HashSet>gif_spilt.0 - <Uuid>gif_spilt.1
            let mut delete_set: HashSet<&Uuid> = gif_spilt.0.as_ref().map_or_else(
                || non_text_anomalies_set.iter().copied().collect(),
                |gif_set| {
                    non_text_anomalies_set
                        .difference(gif_set)
                        .copied()
                        .collect()
                },
            );
            if let Some(id) = gif_spilt.1 {
                delete_set.remove(id);
            }
            return (
                text_anomalies.map(|v| v.into_iter().collect()),
                gif_spilt.0.map(|v| v.into_iter().collect()),
                gif_spilt.1,
                Some(delete_set.into_iter().collect::<Vec<&Uuid>>()).filter(|v| !v.is_empty()),
            );
        })
        .collect()
}

fn main() -> Result<()> {
    let points_clusters: Vec<HashSet<Uuid>> =
        serde_pickle::from_slice(&fs::read(r"global_clusters.pkl")?, Default::default())?;
    let points_metadata = fs::read(r"points_map.bin")?;
    let points_metadata: HashMap<Uuid, NekoPoint> =
        bincode::serde::decode_from_slice(&points_metadata, bincode::config::standard())?.0;
    let s3_file_data = fs::read(r"opendal_list_file_after_rename.bin")?;
    let s3_file_data: Vec<shared::opendal::Entry> =
        bincode::serde::decode_from_slice(&s3_file_data, bincode::config::standard())?.0;
    println!("Successfully loaded data from files.");
    let s3_pre_map: HashMap<String, String> = s3_file_data
        .into_iter()
        .map(|entry| {
            let key = entry.to_point().to_string();
            let val = entry.path;
            (key, val)
        })
        .collect();
    println!("S3 map: {:?}", s3_pre_map.len());
    let points_metadata: HashMap<Uuid, (NekoPoint, NekoPointExt)> = points_metadata
        .into_iter()
        .map(|(id, point)| {
            let file_path = s3_pre_map.get(&point.id.to_string()).unwrap().clone();
            let ext = NekoPointExt {
                source: Some(NekoPointExtResource::LocalPath(format!(
                    "stage9_temp/{}.{}",
                    point.id,
                    file_path.rsplit('.').next().unwrap()
                ))),
                file_path,
            };
            (id, (point, ext))
        })
        .collect();
    println!("S3 metadata: {:?}", points_metadata.len());
    // Vec<(Option<Vec<KeptTextAnomaliesPic>>, Option<Vec<NeedTriageGifs>>, Option<KeptNonGif>, Option<Vec<OtherNeedDeletePics>>)>
    let res = extract_clusters(&points_clusters, &points_metadata);
    // TODO:
    let all_kept_text_anomalies: Vec<&Vec<&Uuid>> = res
        .iter()
        .filter_map(|(opt_text, _, _, _)| opt_text.as_ref())
        .collect();
    let all_need_triage_gifs: Vec<&Vec<&Uuid>> = res
        .iter()
        .filter_map(|(_, opt_gifs, _, _)| opt_gifs.as_ref())
        .collect();
    let all_need_triage_gifs_flat: Vec<&Uuid> = all_need_triage_gifs
        .iter()
        .flat_map(|v| v.iter())
        .copied()
        .collect();
    let all_kept_non_gif: Vec<&Uuid> = res.iter().filter_map(|(_, _, opt_ng, _)| *opt_ng).collect();
    println!("Successfully loaded data from files.");
    println!(
        "all_kept_text_anomalies: {:?}",
        all_kept_text_anomalies.len()
    );
    println!("all_need_triage_gifs: {:?}", all_need_triage_gifs.len());
    println!(
        "all_need_triage_gifs_flat: {:?}",
        all_need_triage_gifs_flat.len()
    );
    println!("all_kept_non_gif: {:?}", all_kept_non_gif.len());
    Ok(())
}
