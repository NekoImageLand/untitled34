mod clip_worker;
mod gif_worker;
mod s3_downloader;
mod structure;

use crate::clip_worker::ClipWorker;
use crate::gif_worker::GifWorker;
use crate::s3_downloader::S3Downloader;
use crate::structure::{
    TEXT_SIM_THRESHOLD, TriageGif, TriageGifGroupsClipStageReq, TriageGifGroupsGifStageReq,
};
use anyhow::Result;
use candle_core::DType;
use candle_transformers::models::clip::ClipConfig;
use half::bf16;
use mimalloc::MiMalloc;
use rayon::prelude::*;
use shared::cosine_sim::cosine_sim;
use shared::structure::{NekoPoint, NekoPointExt, NekoPointExtResource};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::{env, fs};
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};
use uuid::Uuid;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn find_text_anomalies<'a>(
    text_points: &[&'a Uuid],
    points_metadata: &HashMap<Uuid, (NekoPoint, NekoPointExt)>,
) -> Option<Vec<&'a Uuid>> {
    let mut id_vec_pairs = Vec::with_capacity(text_points.len());
    for &id in text_points {
        if let Some((pt, _)) = points_metadata.get(id) {
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
    Option<Vec<&'a Uuid>>, // Option<Vec<KeptTextAnomaliesPic>>
    Option<Vec<&'a Uuid>>, // Option<Vec<NeedTriageGifs>>
    Option<&'a Uuid>,      // Option<KeptNonGif>
    Option<Vec<&'a Uuid>>, // Option<Vec<OtherNeedDeletePics>>
)> {
    points_clusters
        .par_iter()
        .map(|cursor| {
            let cursor_ref: HashSet<&Uuid> = cursor.iter().collect();
            // stage1
            let mut only_test_uuids: Vec<&Uuid> = cursor
                .iter()
                .filter(|id| {
                    points_metadata
                        .get(id)
                        .map(|(pt, _)| pt.text_info.is_some())
                        .unwrap_or(false)
                })
                .collect();
            let text_points = (!only_test_uuids.is_empty()).then_some(only_test_uuids);
            let text_points_size = text_points.as_ref().map_or(0, |v| v.len());
            let text_anomalies = text_points
                .as_ref()
                .and_then(|tp| find_text_anomalies(tp, points_metadata));
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
                    (Some(gif), _) => {
                        // Is a GIF group considered an orphan group? (i.e., a group containing only one GIF)
                        match gif.len() {
                            0 => {
                                if cfg!(debug_assertions) {
                                    panic!("Gif points_in_left_points is empty");
                                }
                                (Some(gif), None)
                            }
                            1 => {
                                let id = gif.iter().next().cloned();
                                (None, id)
                            }
                            _ => (Some(gif), None),
                        }
                    }
                    (None, non_gif) => {
                        let maybe_biggest_non_gif = non_gif.and_then(|hs| {
                            hs.iter()
                                .max_by_key(|&&id| {
                                    points_metadata
                                        .get(id)
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
    let stdout = tracing_subscriber::fmt::layer().with_filter(EnvFilter::new("info"));
    let file_appender = RollingFileAppender::new(Rotation::HOURLY, "logs", "stage9.log");
    let file = tracing_subscriber::fmt::layer()
        .with_writer(file_appender)
        .with_filter(EnvFilter::new("info"));
    tracing_subscriber::registry()
        .with(stdout)
        .with(file)
        .init();
    let points_clusters: Vec<HashSet<Uuid>> =
        serde_pickle::from_slice(&fs::read(r"global_clusters.pkl")?, Default::default())?;
    let points_metadata = fs::read(r"points_map.bin")?;
    let points_metadata_ex: HashMap<Uuid, NekoPoint> =
        bincode::serde::decode_from_slice(&points_metadata, bincode::config::standard())?.0;
    let s3_file_data = fs::read(r"opendal_list_file_after_rename.bin")?;
    let s3_file_data: Vec<shared::opendal::Entry> =
        bincode::serde::decode_from_slice(&s3_file_data, bincode::config::standard())?.0;
    tracing::info!("Successfully loaded data from files.");
    let s3_pre_map: HashMap<String, shared::opendal::Entry> = s3_file_data
        .into_iter()
        .map(|entry| {
            let key = entry.to_point().to_string();
            let val = entry;
            (key, val)
        })
        .collect();
    tracing::info!("S3 map: {:?}", s3_pre_map.len());
    let points_metadata: HashMap<Uuid, (NekoPoint, NekoPointExt)> = points_metadata_ex
        .into_iter()
        .map(|(id, mut point)| {
            let entry = s3_pre_map.get(&point.id.to_string()).unwrap().clone();
            let file_path = entry.path;
            let file_size = entry.metadata.content_length.unwrap_or_default() as usize;
            point.size = Some(file_size); // unhappy patching...
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
    tracing::info!("S3 metadata: {:?}", points_metadata.len());
    // Vec<(Option<Vec<KeptTextAnomaliesPic>>, Option<Vec<NeedTriageGifs>>, Option<KeptNonGif>, Option<Vec<OtherNeedDeletePics>>)>
    let res = extract_clusters(&points_clusters, &points_metadata);
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
    let all_kept_non_gif_path_map: HashMap<&Uuid, String> = all_need_triage_gifs_flat
        .iter()
        .map(|&uuid| (uuid, format!("nekoimg_stage9_gifs/{}.gif", uuid)))
        .collect();
    let all_kept_non_gif_path_ref: Vec<(&Uuid, &str)> = all_kept_non_gif_path_map
        .iter()
        .map(|(&uuid, path)| (uuid, path.as_str()))
        .collect();
    let all_kept_non_gif: Vec<&Uuid> = res.iter().filter_map(|(_, _, opt_ng, _)| *opt_ng).collect();
    tracing::info!("Successfully loaded data from files.");
    tracing::info!(
        "all_kept_text_anomalies: {:?}",
        all_kept_text_anomalies.len()
    );
    tracing::info!("all_need_triage_gifs: {:?}", all_need_triage_gifs.len());
    tracing::info!(
        "all_need_triage_gifs_flat: {:?}",
        all_need_triage_gifs_flat.len()
    );
    tracing::info!("all_kept_non_gif, len = {:?}", all_kept_non_gif.len());
    // Now, we need download all_need_triage_gifs_flat from S3
    tracing::info!("Starting S3 download for triage GIFs...");
    let triage_gif_downloader = S3Downloader::new(20, false)?;
    let download_result =
        triage_gif_downloader.download_files(all_kept_non_gif_path_ref.as_slice());
    match download_result {
        Ok(_) => tracing::info!("Successfully downloaded all triage GIFs."),
        Err(e) => tracing::error!("Failed to download triage GIFs: {}", e),
    }
    // Now, Refine GIFs
    tracing::info!("Starting refining GIFs...");
    let clip_config = ClipConfig::baai_bge_vl_large();
    let refine_gif_worker = GifWorker::new(clip_config.image_size as u32); // in
    let triage_req: TriageGifGroupsGifStageReq = all_need_triage_gifs
        .iter()
        .map(|uuid_group| {
            uuid_group
                .iter()
                .map(|&uuid| {
                    let path = all_kept_non_gif_path_map
                        .get(uuid)
                        .expect("Path must be present for GIFs");
                    let size = points_metadata.get(uuid).and_then(|(p, _)| p.size).unwrap();
                    TriageGif {
                        id: uuid,
                        path,
                        size,
                    }
                })
                .collect::<Vec<TriageGif>>()
        })
        .collect();
    serde_json::to_string(&triage_req).map(|s| fs::write("triage_gifs_req.json", s))??;
    let refine_gif_res = refine_gif_worker.process(&triage_req)?;
    serde_json::to_string(&refine_gif_res).map(|s| fs::write("refine_gifs_res.json", s))??;
    tracing::info!("Refine GIFs result: {:?}", refine_gif_res.len());
    // Calculate all gif embeddings
    let clip_res: TriageGifGroupsClipStageReq = refine_gif_res
        .into_iter()
        .filter_map(|pair| pair.prepare_clip_gif_pair)
        .collect();
    let model_path = PathBuf::from(env::var("CLIP_MODEL_PATH")?);
    let worker = ClipWorker::new(model_path.to_str().unwrap(), clip_config, DType::BF16, true)?;
    let clip_res = worker.get_images_embedding_adapted::<bf16>(clip_res)?;
    let serde_clip_res = serde_json::to_string(&clip_res)?;
    fs::write("clip_embeddings.json", serde_clip_res)?;
    tracing::info!("Clip embeddings calculated!");
    Ok(())
}
