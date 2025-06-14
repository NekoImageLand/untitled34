use candle_core::{D, DType, Device, Error as CandleError, Result, Tensor, WithDType};
use candle_nn::VarBuilder;
use candle_transformers::models::clip::{ClipConfig, ClipModel};
use image::{ImageReader, imageops};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use shared::cosine_sim::{Cosine, cosine_sim};
use shared::structure::{
    IMAGE_SIM_THRESHOLD, TriageGif, TriageGifClip, TriageGifGroupsClipStagePair,
    TriageGifGroupsClipStageReq, TriageGifGroupsClipStageRes,
};
use std::collections::HashMap;
use std::fmt::Debug;

pub trait ClipWorkerInput: Sync + Sized {
    fn to_raw(&self, size: usize) -> anyhow::Result<Vec<u8>>;
}

impl ClipWorkerInput for &str {
    fn to_raw(&self, size: usize) -> anyhow::Result<Vec<u8>> {
        let img = ImageReader::open(self)?
            .decode()
            .map_err(|e| CandleError::Msg(format!("Failed to decode image: {}", e).into()).bt())?;
        let (height, width) = (size, size);
        let img = img.resize_to_fill(width as u32, height as u32, imageops::FilterType::Triangle);
        let img = img.to_rgb8();
        let img = img.into_raw();
        Ok(img)
    }
}

impl<'a> ClipWorkerInput for &'a [u8] {
    fn to_raw(&self, _: usize) -> anyhow::Result<Vec<u8>> {
        Ok((*self).to_vec())
    }
}

impl<'a, U> ClipWorkerInput for &'a U
where
    U: ClipWorkerInput + Sync,
{
    fn to_raw(&self, size: usize) -> anyhow::Result<Vec<u8>> {
        (*self).to_raw(size)
    }
}

pub struct ClipWorker {
    config: ClipConfig,
    device: Device,
    model: ClipModel,
    tensor_type: DType,
}

impl ClipWorker {
    pub fn new(
        model_filepath: &str,
        clip_config: ClipConfig,
        tensor_type: DType,
        use_gpu: bool,
    ) -> anyhow::Result<Self> {
        let device = if use_gpu && cfg!(feature = "cuda") {
            Device::new_cuda(0)?
        } else if use_gpu && !cfg!(feature = "cuda") {
            panic!("CUDA feature is not enabled. Please enable it in Cargo.toml.");
        } else {
            Device::Cpu
        };
        let var_builder = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_filepath.to_string()],
                tensor_type,
                &device,
            )?
        };
        let model = ClipModel::new(var_builder, &clip_config)?;
        Ok(Self {
            device,
            model,
            tensor_type,
            config: clip_config,
        })
    }

    fn div_l2_norm(&self, v: &Tensor) -> Result<Tensor> {
        let l2_norm = v.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        v.broadcast_div(&l2_norm)
    }

    fn load_image<T>(&self, image: T, image_size: usize) -> Result<Tensor>
    where
        T: ClipWorkerInput,
    {
        let img = image
            .to_raw(image_size)
            .map_err(|e| CandleError::Msg(e.to_string()))?;
        let img = Tensor::from_vec(img, (image_size, image_size, 3), &self.device)?
            .permute((2, 0, 1))?
            .to_dtype(DType::F32)?
            .affine(2. / 255., -1.)?
            .to_dtype(self.tensor_type)?;
        Ok(img)
    }

    fn load_images<T>(&self, images: &[T], image_size: usize) -> Result<Tensor>
    where
        T: ClipWorkerInput,
    {
        let raws: Vec<Result<Tensor>> = images
            .par_iter()
            .map(|path| self.load_image(path, image_size))
            .collect();
        let imgs: Vec<Tensor> = raws.into_iter().collect::<Result<Vec<_>>>()?;
        Tensor::stack(&imgs, 0)
    }

    #[allow(dead_code)]
    fn get_images_embedding<T>(&self, images: &[T]) -> Result<Tensor>
    where
        T: ClipWorkerInput,
    {
        let image = self
            .load_images(images, self.config.image_size)
            .map_err(|e| CandleError::Msg(format!("Failed to load image: {}", e).into()).bt())?;
        let image_features = self.model.get_image_features(&image)?;
        self.div_l2_norm(&image_features)
    }

    pub fn get_images_embedding_batched<T>(&self, images: &[T]) -> Result<Tensor>
    where
        T: ClipWorkerInput,
    {
        const BATCH_SIZE: usize = 32;
        let batches: Vec<Tensor> = images
            .chunks(BATCH_SIZE)
            .map(|chunk| {
                let imgs = self
                    .load_images(chunk, self.config.image_size)
                    .map_err(|e| {
                        CandleError::Msg(format!("Failed to load image batch: {}", e).into()).bt()
                    })?;
                self.model.get_image_features(&imgs)
            })
            .collect::<Result<_>>()?;
        let features = match batches.as_slice() {
            [single] => single.clone(),
            _ => Tensor::cat(&batches, 0)?,
        };
        self.div_l2_norm(&features)
    }

    fn find_gif_embedding_clusters<'a, 'b, T>(
        &self,
        items: &'b [(TriageGifClip<'a>, Vec<T>)],
    ) -> Vec<Vec<&'b TriageGifClip<'a>>>
    where
        T: WithDType + Cosine + Debug,
    {
        let mut id_map = HashMap::with_capacity(items.len());
        for it in items {
            id_map.insert(it.0.id, it);
        }
        let mut clusters: Vec<Vec<&TriageGifClip<'a>>> = Vec::new();
        for (it, vec_i) in items {
            let mut placed = false;
            for cl in clusters.iter_mut() {
                let ok = cl.iter().all(|c| {
                    let vec_j = &id_map.get(&c.id).unwrap().1;
                    cosine_sim(vec_i, vec_j) > IMAGE_SIM_THRESHOLD
                });
                if ok {
                    cl.push(&it);
                    placed = true;
                    break; // TODO: no break for edge case? (/cc @jj)
                }
            }
            if !placed {
                clusters.push(vec![&it]);
            }
        }
        clusters
    }

    pub fn get_images_embedding_adapted<'a, T>(
        &self,
        req: TriageGifGroupsClipStageReq<'a>,
    ) -> Result<TriageGifGroupsClipStageRes<'a>>
    where
        T: WithDType + Cosine + Debug,
    {
        let mut final_res: TriageGifGroupsClipStageRes<'a> = Vec::with_capacity(req.len());
        let pb = ProgressBar::new(req.len() as u64);
        let style = ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .map_err(|_| CandleError::Msg("Building ProgressStyle failed!".to_string()))?;
        pb.set_style(style);
        pb.set_message("Generating image embeddings...");
        for group_outer in req {
            match group_outer {
                Some(Some(grp)) => {
                    let mut kept: Option<Vec<TriageGif<'a>>> = None;
                    let mut discarded: Option<Vec<TriageGif<'a>>> = None;
                    let frame_lens: Vec<usize> = grp.iter().map(|clip| clip.frame.len()).collect();
                    let flatted_slices: Vec<&[u8]> = grp
                        .iter()
                        .flat_map(|clip| clip.frame.iter().map(|f| f.as_slice()))
                        .collect();
                    let flatted_embeddings = self.get_images_embedding_batched(&flatted_slices)?;
                    let items: Vec<(TriageGifClip<'a>, Vec<T>)> = frame_lens
                        .into_iter()
                        .scan(0usize, |state, count| {
                            let start = *state;
                            *state += count;
                            Some((start, count))
                        })
                        .zip(grp.into_iter())
                        .map(|((start, count), clip)| -> Result<_> {
                            let tensor = flatted_embeddings.narrow(0, start, count)?.mean(0)?;
                            let tensor = self.div_l2_norm(&tensor)?;
                            Ok((clip, tensor.to_vec1::<T>()?))
                        })
                        .collect::<Result<_>>()?;
                    tracing::debug!("Items: {}", items.len());
                    // FIXME:
                    let clusters: Vec<Vec<&TriageGifClip<'a>>> =
                        self.find_gif_embedding_clusters(&items);
                    tracing::debug!("Clusters: {}", clusters.len());
                    let mut max_clips = Vec::with_capacity(clusters.len());
                    let mut other_clips = Vec::with_capacity(items.len() - clusters.len());
                    for cluster in clusters.iter() {
                        let (max_idx, &tgc) = cluster
                            .iter()
                            .enumerate()
                            .max_by_key(|&(_, clip)| clip.size)
                            .unwrap();
                        max_clips.push(TriageGif {
                            uuid: tgc.id,
                            path: tgc.path,
                            size: tgc.size,
                        });
                        other_clips.extend(
                            cluster
                                .iter()
                                .take(max_idx)
                                .chain(cluster.iter().skip(max_idx + 1))
                                .map(|&clip| TriageGif {
                                    uuid: clip.id,
                                    path: clip.path,
                                    size: clip.size,
                                }),
                        );
                    }
                    match kept {
                        Some(ref mut v) => v.extend(max_clips),
                        None => kept = Some(max_clips),
                    }
                    match discarded {
                        Some(ref mut v) => v.extend(other_clips),
                        None => discarded = Some(other_clips),
                    }
                    // Edge case
                    if kept.as_ref().is_none() && discarded.as_ref().is_some() {
                        tracing::debug!("Edge case: kept = {:?} discarded = {:?}", kept, discarded);
                        // TODO: do we need this ???
                        // if let Some(mut dis) = discarded.take() {
                        //     if let Some(max_idx) = dis
                        //         .iter()
                        //         .enumerate()
                        //         .max_by_key(|&(_, item)| item.size)
                        //         .map(|(idx, _)| idx)
                        //     {
                        //         let tg = dis.remove(max_idx);
                        //         kept = Some(vec![tg]);
                        //     }
                        //     if !dis.is_empty() {
                        //         discarded = Some(dis);
                        //     }
                        // }
                    }
                    let res = TriageGifGroupsClipStagePair {
                        kept_gifs: kept,
                        discard_duplicate_gifs: discarded,
                    };
                    final_res.push(Some(Some(res)));
                }
                Some(None) => final_res.push(Some(None)),
                None => final_res.push(None),
            }
            pb.inc(1);
        }
        pb.finish_with_message("All images processed");
        Ok(final_res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gif_worker::GifWorker;
    use anyhow::Result;
    use half::bf16;
    use shared::cosine_sim::cosine_sim;
    use std::env;
    use std::path::PathBuf;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    use tracing_subscriber::{EnvFilter, Layer};
    use uuid::Uuid;

    #[test]
    fn test_clip_worker() -> Result<()> {
        // TODO: auto download it!
        let model_path = PathBuf::from(env::var("CLIP_MODEL_PATH")?);
        let worker = ClipWorker::new(
            model_path.to_str().unwrap(),
            ClipConfig::baai_bge_vl_large(),
            DType::F32,
            false,
        )?;
        let pics = vec![
            "../assets/test_images/bsn_0.jpg",
            "../assets/test_images/bsn_1.jpg",
        ];
        let bsn = worker.get_images_embedding_batched(&pics)?;
        println!("{:?}", bsn);
        let bsn_1 = bsn.get(0)?.to_vec1::<f32>()?;
        let bsn_2 = bsn.get(1)?.to_vec1::<f32>()?;
        // compare cosine similarity
        let cosine_similarity = cosine_sim(&bsn_1, &bsn_2);
        println!("Cosine similarity between images: {}", cosine_similarity);
        Ok(())
    }

    #[test]
    fn test_adapted_worker() -> Result<()> {
        let stdout = tracing_subscriber::fmt::layer().with_filter(EnvFilter::new("debug"));
        tracing_subscriber::registry().with(stdout).init();
        tracing::info!("Starting adapted worker test...");
        let clip_config = ClipConfig::baai_bge_vl_large();
        let gif_worker = GifWorker::new(clip_config.image_size as u32);
        let model_path = PathBuf::from(env::var("CLIP_MODEL_PATH")?);
        let clip_worker = ClipWorker::new(
            model_path.to_str().unwrap(),
            ClipConfig::baai_bge_vl_large(),
            DType::F32,
            false,
        )?;
        let uuids: [Uuid; 4] = std::array::from_fn(|_| Uuid::new_v4());
        let paths = [
            "../assets/test_images/mcat_0.gif",
            "../assets/test_images/mcat_1.gif",
            "../assets/test_images/bq_0.gif",
            "../assets/test_images/bq_1.gif",
        ];
        let gifs = vec![
            None,
            Some(vec![
                TriageGif {
                    uuid: uuids.get(0).unwrap(),
                    path: paths.get(0).unwrap(),
                    size: PathBuf::from(paths.get(0).unwrap()).metadata()?.len() as usize,
                },
                TriageGif {
                    uuid: uuids.get(1).unwrap(),
                    path: paths.get(1).unwrap(),
                    size: PathBuf::from(paths.get(1).unwrap()).metadata()?.len() as usize,
                },
            ]),
            None,
            Some(vec![
                TriageGif {
                    uuid: uuids.get(2).unwrap(),
                    path: paths.get(2).unwrap(),
                    size: PathBuf::from(paths.get(2).unwrap()).metadata()?.len() as usize,
                },
                TriageGif {
                    uuid: uuids.get(3).unwrap(),
                    path: paths.get(3).unwrap(),
                    size: PathBuf::from(paths.get(3).unwrap()).metadata()?.len() as usize,
                },
            ]),
            None,
            None,
        ];
        let res = gif_worker.process(&gifs)?;
        let clip_req: TriageGifGroupsClipStageReq = res
            .into_iter()
            .map(|pair| pair.map(|p| p.prepare_clip_gif_pair))
            .collect();
        let clip_res = clip_worker.get_images_embedding_adapted::<f32>(clip_req)?;
        println!("{:?}", clip_res);
        Ok(())
    }
}
