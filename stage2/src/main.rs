use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use opendal::layers::{LoggingLayer, RetryLayer};
use opendal::{Operator, services};
use prost::Message;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::vectors_output::VectorsOptions as VectorsOptionsOutput;
use qdrant_client::qdrant::with_payload_selector::SelectorOptions as SelectorOptionsPayload;
use qdrant_client::qdrant::with_vectors_selector::SelectorOptions;
use qdrant_client::qdrant::{GetPointsBuilder, GetResponse, PointId, VectorsSelector};
use qdrant_client::qdrant::{point_id, value};
use shared::structure::{NekoPoint, NekoPointText};
use std::collections::HashMap;
use std::collections::HashSet;
use std::io::{Read, Write};
use std::time::Duration;
use uuid::Uuid;

fn extract_point(pb: ProgressBar, points: GetResponse) -> HashMap<Uuid, NekoPoint> {
    let mut points_map: HashMap<Uuid, NekoPoint> = HashMap::new();
    for raw in points.result.into_iter() {
        let id = raw
            .id
            .and_then(|pid| pid.point_id_options)
            .map(|opt| match opt {
                point_id::PointIdOptions::Uuid(s) => Uuid::parse_str(&s).unwrap(),
                point_id::PointIdOptions::Num(n) => Uuid::from_u128(n as u128),
            })
            .unwrap();
        let height = raw.payload.get("height").unwrap().as_integer().unwrap() as usize;
        let weight = raw.payload.get("width").unwrap().as_integer().unwrap() as usize;
        let categories = match raw.payload.get("categories").and_then(|v| v.kind.clone()) {
            Some(value::Kind::ListValue(list)) => Some(
                list.values
                    .iter()
                    .filter_map(|item| {
                        if let Some(value::Kind::StringValue(s)) = item.kind.clone() {
                            Some(s)
                        } else {
                            None
                        }
                    })
                    .collect(),
            ),
            _ => None,
        };
        let text_info = raw.vectors.and_then(|vectors| {
            if let Some(VectorsOptionsOutput::Vectors(named)) = vectors.vectors_options {
                named.vectors.get("text_contain_vector").and_then(|v| {
                    raw.payload
                        .get("ocr_text")
                        .and_then(|t| t.as_str().map(|s| s.to_string()))
                        .map(|txt| NekoPointText {
                            text: txt,
                            text_vector: v.data.clone(),
                        })
                })
            } else {
                None
            }
        });
        let pt = NekoPoint {
            id,
            height,
            weight,
            categories,
            text_info,
            size: None,
        };
        points_map.insert(pt.id, pt);
        pb.inc(1);
    }
    points_map
}

// TODO:
async fn fill_pic_size(map: HashMap<Uuid, NekoPoint>) -> Result<(), Box<dyn std::error::Error>> {
    let builder = services::S3::default()
        .bucket("nekoimg")
        .access_key_id("")
        .secret_access_key("")
        .endpoint("");
    let op = Operator::new(builder)?
        .layer(LoggingLayer::default())
        .layer(
            RetryLayer::default()
                .with_max_times(3)
                .with_factor(1.5)
                .with_min_delay(Duration::from_millis(50))
                .with_max_delay(Duration::from_millis(200)),
        )
        .finish();
    Ok(())
}

#[tokio::main]
pub async fn main() {
    let global_clusters = std::fs::read(r"global_clusters.pkl").unwrap();
    let global_clusters: Vec<HashSet<Uuid>> =
        serde_pickle::from_slice(&global_clusters, Default::default()).unwrap();
    let point_set: HashSet<String> = global_clusters
        .iter()
        .flat_map(|c| c.iter())
        .map(|uuid| uuid.to_string())
        .collect();
    let point_list: Vec<PointId> = point_set
        .into_iter()
        .map(|s: String| PointId::from(s))
        .collect();
    println!("Got point_list, len={:?}", point_list.len());
    let points;
    /// Desperate
    match std::fs::File::open(r"!Desperate") {
        Ok(file) => {
            println!("File already exists, skipping download.");
            // read file and decode
            let mut reader = std::io::BufReader::new(file);
            let mut data = Vec::new();
            reader.read_to_end(&mut data).unwrap();
            points = GetResponse::decode(data.as_slice()).unwrap();
        }
        Err(_) => {
            println!("File not found, fetching...");
            let client = Qdrant::from_url("http://localhost:6334")
                .timeout(std::time::Duration::from_secs(3600))
                .build()
                .unwrap();
            points = client
                .get_points(
                    GetPointsBuilder::new("nekoimg", point_list)
                        .timeout(3600)
                        .with_vectors(SelectorOptions::Include(VectorsSelector::from(vec![
                            "text_contain_vector".to_string(),
                        ])))
                        .with_payload(SelectorOptionsPayload::Enable(true))
                        .build(),
                )
                .await
                .unwrap();
        }
    }
    println!("Got points, {:?}", points.result.len());
    let m = MultiProgress::new();
    let pb_local = m.add(ProgressBar::new(points.result.len() as u64));
    let style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("#>-");
    pb_local.set_style(style.clone());
    pb_local.set_message("extract_point");
    let points_map = extract_point(pb_local, points);
    println!("Got points, {:?}", points_map.len());
    let mut saved_file = std::fs::File::create(r"points_map.bin").unwrap();
    let serialized =
        bincode::serde::encode_to_vec(&points_map, bincode::config::standard()).unwrap();
    saved_file.write_all(&serialized).unwrap();
}
