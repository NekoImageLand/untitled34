use crate::structure::{
    IMAGE_SIM_THRESHOLD, TriageGif, TriageGifClip, TriageGifGroupsClipStagePair,
    TriageGifGroupsClipStageReq, TriageGifGroupsClipStageRes,
};
use candle_core::{D, DType, Device, Error as CandleError, Result, Tensor, WithDType};
use candle_nn::VarBuilder;
use candle_transformers::models::clip::{ClipConfig, ClipModel};
use image::{ImageReader, imageops};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use shared::cosine_sim::{Cosine, cosine_sim};
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
        let device = match use_gpu {
            true => Device::new_cuda(0)?,
            false => Device::Cpu,
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
        for group in req {
            let mut kept: Option<Vec<TriageGif<'a>>> = None;
            let mut discarded: Option<Vec<TriageGif<'a>>> = None;
            let frame_lens: Vec<usize> = group.iter().map(|clip| clip.frame.len()).collect();
            let flatted_slices: Vec<&[u8]> = group
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
                .zip(group.into_iter())
                .map(|((start, count), clip)| -> Result<_> {
                    let tensor = flatted_embeddings.narrow(0, start, count)?.mean(0)?;
                    let tensor = self.div_l2_norm(&tensor)?;
                    Ok((clip, tensor.to_vec1::<T>()?))
                })
                .collect::<Result<_>>()?;
            tracing::debug!("Items: {}", items.len());
            for (clip, vec_i) in items.iter() {
                let is_similar = items
                    .iter()
                    .as_ref()
                    .into_iter()
                    .filter(|(other_clip, _)| other_clip.id != clip.id)
                    .all(|(c, vec_j)| {
                        let sim = cosine_sim(vec_i, vec_j);
                        tracing::debug!("Similar clip = {}, {:?} vs {:?}", sim, clip.path, c.path);
                        sim > IMAGE_SIM_THRESHOLD
                    });
                let tg = TriageGif {
                    id: clip.id,
                    path: clip.path,
                    size: clip.size,
                };
                match is_similar {
                    true => match discarded {
                        Some(ref mut v) => v.push(tg),
                        None => discarded = Some(vec![tg]),
                    },
                    false => match kept {
                        Some(ref mut v) => v.push(tg),
                        None => kept = Some(vec![tg]),
                    },
                }
            }
            // Edge case
            if kept.as_ref().is_none() && discarded.as_ref().is_some() {
                if let Some(mut dis) = discarded.take() {
                    if let Some(max_idx) = dis
                        .iter()
                        .enumerate()
                        .max_by_key(|&(_, item)| item.size)
                        .map(|(idx, _)| idx)
                    {
                        let tg = dis.remove(max_idx);
                        kept = Some(vec![tg]);
                    }
                    if !dis.is_empty() {
                        discarded = Some(dis);
                    }
                }
            }
            pb.inc(1);
            let res = TriageGifGroupsClipStagePair {
                kept_gifs: kept,
                discard_duplicate_gifs: discarded,
            };
            final_res.push(res);
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
            vec![
                TriageGif {
                    id: uuids.get(0).unwrap(),
                    path: paths.get(0).unwrap(),
                    size: PathBuf::from(paths.get(0).unwrap()).metadata()?.len() as usize,
                },
                TriageGif {
                    id: uuids.get(1).unwrap(),
                    path: paths.get(1).unwrap(),
                    size: PathBuf::from(paths.get(1).unwrap()).metadata()?.len() as usize,
                },
            ],
            vec![
                TriageGif {
                    id: uuids.get(2).unwrap(),
                    path: paths.get(2).unwrap(),
                    size: PathBuf::from(paths.get(2).unwrap()).metadata()?.len() as usize,
                },
                TriageGif {
                    id: uuids.get(3).unwrap(),
                    path: paths.get(3).unwrap(),
                    size: PathBuf::from(paths.get(3).unwrap()).metadata()?.len() as usize,
                },
            ],
        ];
        let res = gif_worker.process(&gifs)?;
        let clip_req: TriageGifGroupsClipStageReq = res
            .into_iter()
            .filter_map(|pair| pair.prepare_clip_gif_pair)
            .collect();
        let clip_res = clip_worker.get_images_embedding_adapted::<f32>(clip_req)?;
        println!("{:?}", clip_res);
        Ok(())
    }
}
