use anyhow::Result;
use core::arch::x86_64::*;
use rayon::prelude::*;
use shared::structure::NekoPoint;
use std::collections::{HashMap, HashSet};
use std::fs;
use uuid::Uuid;

#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { cosine_sim_avx2(a, b) }
    } else {
        let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
        let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    }
}

#[inline]
#[target_feature(enable = "avx2,fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn cosine_sim_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut sum_dot = _mm256_setzero_ps();
    let mut sum_a2 = _mm256_setzero_ps();
    let mut sum_b2 = _mm256_setzero_ps();
    let chunks = len / 8;
    for i in 0..chunks {
        let pa = a.as_ptr().add(i * 8);
        let pb = b.as_ptr().add(i * 8);
        let va = _mm256_loadu_ps(pa);
        let vb = _mm256_loadu_ps(pb);
        sum_dot = _mm256_fmadd_ps(va, vb, sum_dot);
        sum_a2 = _mm256_fmadd_ps(va, va, sum_a2);
        sum_b2 = _mm256_fmadd_ps(vb, vb, sum_b2);
    }
    #[inline(always)]
    unsafe fn hsum256(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps::<1>(v);
        let lo = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum128);
        let sums = _mm_add_ps(sum128, shuf);
        let shuf2 = _mm_movehl_ps(shuf, sums);
        let sums2 = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(sums2)
    }
    let mut dot = hsum256(sum_dot);
    let mut a2 = hsum256(sum_a2);
    let mut b2 = hsum256(sum_b2);
    for i in (chunks * 8)..len {
        let ai = *a.get_unchecked(i);
        let bi = *b.get_unchecked(i);
        dot += ai * bi;
        a2 += ai * ai;
        b2 += bi * bi;
    }
    dot / (a2.sqrt() * b2.sqrt())
}

struct NekoPointExt {
    file_path: String,
}

impl NekoPointExt {
    #[inline]
    fn ext(&self) -> &str {
        self.file_path.rsplit('.').next().unwrap()
    }
}

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
            (id, (point, NekoPointExt { file_path }))
        })
        .collect();
    println!("S3 metadata: {:?}", points_metadata.len());
    // Vec<(Vec<KeptTextAnomaliesPic>, Vec<NeedTriageGifs>, KeptNonGif)>
    let res: Vec<(Option<Vec<&Uuid>>, Option<Vec<&Uuid>>, Option<&Uuid>)> = points_clusters
        .par_iter()
        .map(|cursor| {
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
            let text_anomalies = match text_points {
                Some(text_points) => find_text_anomalies(&text_points, &points_metadata),
                None => None,
            };
            if let Some(anomalies) = &text_anomalies {
                println!(
                    "Cluster with {} (in {}) text anomalies: {:?}",
                    anomalies.len(),
                    cursor.len(),
                    &anomalies
                );
            }
            // stage2
            let cursor_except_text_anomalies: Vec<&Uuid> = match &text_anomalies {
                Some(anomalies) if !anomalies.is_empty() => {
                    cursor.iter().filter(|id| !anomalies.contains(id)).collect()
                }
                _ => cursor.iter().collect(),
            };
            // All points in the current cluster have textual dissimilarity, skip!
            if cursor_except_text_anomalies.is_empty() {
                return (text_anomalies, None, None);
            }
            let (gif_points, non_gif_points) = cursor_except_text_anomalies.into_iter().fold(
                (None::<Vec<&Uuid>>, None::<Vec<&Uuid>>),
                |(mut gp, mut ngp), id| {
                    let is_gif = points_metadata
                        .get(id)
                        .map(|(_, ex)| ex.ext() == "gif")
                        .unwrap_or(false);
                    if is_gif {
                        match gp {
                            Some(ref mut v) => v.push(id),
                            None => gp = Some(vec![id]),
                        }
                    } else {
                        match ngp {
                            Some(ref mut v) => v.push(id),
                            None => ngp = Some(vec![id]),
                        }
                    }
                    (gp, ngp)
                },
            );
            // stage3
            let gif_spilt: (Option<Vec<&Uuid>>, Option<&Uuid>) = match (gif_points, non_gif_points)
            {
                (Some(gif), Some(_none_gif)) => (Some(gif), None),
                (Some(gif), None) => (Some(gif), None),
                (None, Some(non_gif)) => {
                    let biggest_non_gif = non_gif
                        .iter()
                        .max_by_key(|&&id| {
                            points_metadata
                                .get(id)
                                .map(|(pt, _)| pt.size.unwrap_or_default())
                                .unwrap_or(0)
                        })
                        .cloned();
                    (None, biggest_non_gif)
                }
                (None, None) => {
                    panic!("No gif points or non-gif points were given")
                }
            };
            (text_anomalies, gif_spilt.0, gif_spilt.1)
        })
        .collect();
    // TODO:
    Ok(())
}
