use clap::Parser;
use futures::StreamExt;
use futures::future::join_all;
use indicatif::{ProgressBar, ProgressStyle};
use qdrant_client::qdrant::{
    DeletePointsBuilder, PointsIdsList, PointsOperationResponse, SetPayloadPointsBuilder,
};
use qdrant_client::{Payload, QdrantError};
use serde::Serialize;
use serde_json::json;
use shared::qdrant::GenShinQdrantClient;
use shared::structure::{FinalClassification, NekoPoint};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::ops::Deref;
use std::sync::Arc;
use std::{env, fs};
use tokio::join;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize)]
struct ReSetPointTask<'a> {
    keep_point_list: Vec<&'a Uuid>,
    discard_point_list: Vec<&'a Uuid>,
    transfer_tag_list: Vec<Vec<&'a str>>,
}

#[derive(Debug, Serialize)]
struct FailedReSetPointTask<'a> {
    #[serde(flatten)]
    task: ReSetPointTask<'a>,
    error: String,
}

struct Stage11GenshinQdrantClient {
    client: GenShinQdrantClient,
    collection_name: String,
    dry_run: bool,
    worker_num: usize,
    url_prefix: String,
}

impl Deref for Stage11GenshinQdrantClient {
    type Target = GenShinQdrantClient;

    fn deref(&self) -> &Self::Target {
        &self.client
    }
}

impl Stage11GenshinQdrantClient {
    pub fn new(
        collection_name: &str,
        dry_run: bool,
        worker_num: usize,
        url_prefix: &str,
    ) -> anyhow::Result<Self> {
        let client = GenShinQdrantClient::new()?;
        Ok(Self {
            client,
            collection_name: collection_name.to_owned(),
            dry_run,
            worker_num,
            url_prefix: url_prefix.to_owned(),
        })
    }

    async fn set_reset_point_task<'a>(
        self: Arc<Self>,
        tasks: &'a [ReSetPointTask<'a>],
    ) -> anyhow::Result<Option<Vec<FailedReSetPointTask<'a>>>> {
        let pb = ProgressBar::new(tasks.len() as u64);
        let style = ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?;
        pb.set_style(style);
        pb.set_message("Overwriting Qdrant payload...");
        let mut stream = futures::stream::iter(tasks.into_iter().map(|op| {
            let client = self.clone();
            let pb = pb.clone();
            async move {
                let triage = client.set_reset_point_task_atomic(op).await;
                pb.inc(1);
                (op, triage)
            }
        }))
        .buffer_unordered(self.worker_num);
        let mut failed_tasks = Vec::new();
        while let Some((tasks, res)) = stream.next().await {
            match res {
                Some(res) => {
                    res.into_iter().for_each(|result| match result {
                        Ok(_) => {}
                        Err(e) => {
                            tracing::error!("Failed to overwrite task: {}", e);
                            failed_tasks.push(FailedReSetPointTask {
                                task: tasks.clone(),
                                error: e.to_string(),
                            });
                        }
                    });
                }
                _ => {}
            }
        }
        pb.finish_with_message("Done");
        if failed_tasks.is_empty() {
            Ok(None)
        } else {
            Ok(Some(failed_tasks))
        }
    }

    async fn set_reset_point_task_atomic<'a>(
        self: Arc<Self>,
        task: &'a ReSetPointTask<'a>,
    ) -> Option<Vec<Result<PointsOperationResponse, QdrantError>>> {
        let keep_point_ids: Vec<&Uuid> = task.keep_point_list.iter().cloned().collect();
        let delete_point_ids: Vec<&Uuid> = task.discard_point_list.iter().cloned().collect();
        let payload = task
            .keep_point_list
            .iter()
            .zip(task.transfer_tag_list.iter())
            .map(|(_, tags)| {
                Payload::try_from(json!({
                    "categories": tags,
                }))
                .expect("Failed to create payload")
            })
            .collect::<Vec<_>>();
        if self.dry_run {
            tracing::info!(
                "Dry run: would overwrite points {:?} with Payload: {:?}",
                task.keep_point_list,
                payload
            );
            return None;
        }
        let add_ops = join_all(
            keep_point_ids
                .into_iter()
                .zip(payload)
                .map(|(id, payload)| {
                    self.client.set_payload(
                        SetPayloadPointsBuilder::new(&self.collection_name, payload)
                            .points_selector(PointsIdsList {
                                ids: vec![id.to_string().into()],
                            })
                            .wait(true),
                    )
                }),
        );
        let del_ops = join_all(delete_point_ids.into_iter().map(|id| {
            self.client.delete_points(
                DeletePointsBuilder::new(&self.collection_name)
                    .points(PointsIdsList {
                        ids: vec![id.to_string().into()],
                    })
                    .wait(true),
            )
        }));
        let res = join!(add_ops, del_ops);
        let res = res
            .0
            .into_iter()
            .chain(res.1.into_iter())
            .collect::<Vec<_>>();
        Some(res)
    }
}

fn into_keep_tags<'a>(
    uuid: &'a Uuid,
    tags_sets: &mut Vec<HashSet<&'a str>>,
    metadata: &'a HashMap<Uuid, NekoPoint>,
) {
    if let Some(categories) = metadata.get(uuid).and_then(|p| p.categories.as_ref()) {
        let mut tags = HashSet::with_capacity(categories.len());
        categories.iter().for_each(|tag| {
            tags.insert(tag.as_str());
        });
        tags_sets.push(tags);
    }
}

fn into_duplicate_tags<'a>(
    uuid: &'a Uuid,
    tags_set: &mut HashSet<&'a str>,
    metadata: &'a HashMap<Uuid, NekoPoint>,
) {
    if let Some(categories) = metadata.get(uuid).and_then(|p| p.categories.as_ref()) {
        categories.iter().for_each(|tag| {
            tags_set.insert(tag.as_str());
        });
    }
}

#[derive(Parser, Debug)]
#[command(name = "Stage11", version)]
struct Cli {
    #[arg(long, default_value = "false")]
    dry_run: bool,
    #[arg(long, default_value = "16")]
    worker_num: usize,
    #[arg(long, default_value = "http://127.0.0.1:10000/nekoimg/NekoImage")]
    url_prefix: String,
    #[arg(long, default_value = "qdrant_point_reset_errors")]
    save_result_prefix: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let stdout = tracing_subscriber::fmt::layer().with_filter(EnvFilter::new("info"));
    let file_appender = RollingFileAppender::new(Rotation::HOURLY, "logs", "stage11.log");
    let file = tracing_subscriber::fmt::layer()
        .with_writer(file_appender)
        .with_filter(EnvFilter::new("info"));
    tracing_subscriber::registry()
        .with(stdout)
        .with(file)
        .init();
    let file = fs::read("final_classification.json")?;
    let res: Vec<FinalClassification> = serde_json::from_slice(&*file)?;
    let points_metadata = fs::read(r"points_map.bin")?;
    let points_metadata_ex: HashMap<Uuid, NekoPoint> =
        bincode::serde::decode_from_slice(&points_metadata, bincode::config::standard())?.0;
    let all_tasks: Vec<ReSetPointTask<'_>> = res
        .iter()
        .map(|item| {
            let mut keep_point_list = Vec::new();
            let mut discard_point_list = Vec::new();
            let mut keep_point_tags_set_list = Vec::new();
            let mut discard_point_tags_set = HashSet::new();
            item.kept_text_anomalies_group.as_ref().map(|uuids| {
                keep_point_list.extend(uuids);
                uuids.iter().for_each(|uuid| {
                    into_keep_tags(uuid, &mut keep_point_tags_set_list, &points_metadata_ex)
                })
            });
            item.triaged_gif_and_invalid_group.as_ref().map(|uuids| {
                discard_point_list.extend(uuids.0.iter());
                uuids.0.iter().for_each(|uuid| {
                    into_duplicate_tags(uuid, &mut discard_point_tags_set, &points_metadata_ex);
                });
            });
            item.triaged_gif_and_discard_same_frame_group
                .as_ref()
                .map(|uuids| {
                    discard_point_list.extend(uuids.iter());
                    uuids.iter().for_each(|uuid| {
                        into_duplicate_tags(uuid, &mut discard_point_tags_set, &points_metadata_ex);
                    });
                });
            item.triaged_gif_and_then_will_keep_group
                .as_ref()
                .map(|uuids| {
                    keep_point_list.extend(uuids.iter());
                    uuids.iter().for_each(|uuid| {
                        into_keep_tags(uuid, &mut keep_point_tags_set_list, &points_metadata_ex);
                    });
                });
            item.triaged_gif_and_then_will_delete_group
                .as_ref()
                .map(|uuids| {
                    discard_point_list.extend(uuids.iter());
                    uuids.iter().for_each(|uuid| {
                        into_duplicate_tags(uuid, &mut discard_point_tags_set, &points_metadata_ex);
                    });
                });
            item.kept_non_gif.as_ref().map(|uuid| {
                keep_point_list.push(uuid);
                into_keep_tags(uuid, &mut keep_point_tags_set_list, &points_metadata_ex);
            });
            item.other_need_delete_group.as_ref().map(|uuids| {
                discard_point_list.extend(uuids.iter());
                uuids.iter().for_each(|uuid| {
                    into_duplicate_tags(uuid, &mut discard_point_tags_set, &points_metadata_ex);
                });
            });
            let transfer_tag_list: Vec<Vec<&str>> = keep_point_tags_set_list
                .into_iter()
                .map(|mut km| {
                    km.extend(discard_point_tags_set.iter());
                    km.into_iter().collect::<Vec<&str>>()
                })
                .collect::<Vec<Vec<&str>>>();
            assert_eq!(transfer_tag_list.len(), keep_point_list.len());
            ReSetPointTask {
                keep_point_list,
                discard_point_list,
                transfer_tag_list,
            }
        })
        .collect();
    let collection_name = env::var("QDRANT_COLLECTION_NAME")?;
    let client = Arc::new(Stage11GenshinQdrantClient::new(
        &collection_name,
        cli.dry_run,
        cli.worker_num,
        &cli.url_prefix,
    )?);
    let res = client.set_reset_point_task(&all_tasks).await?;
    if let Some(failed_tasks) = res {
        let filename = format!(
            "{}_{}.json",
            cli.save_result_prefix,
            chrono::Local::now().format("%Y%m%d_%H%M%S")
        );
        let failed_file = File::create(&filename)?;
        serde_json::to_writer_pretty(failed_file, &failed_tasks)?;
        tracing::error!(
            "Some tasks failed, details saved to {}. Total failed tasks: {}",
            &filename,
            failed_tasks.len()
        );
    } else {
        tracing::info!("All tasks completed successfully.");
    }
    Ok(())
}
