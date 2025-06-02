use crate::structure::{
    GifFrames, TriageGif, TriageGifClip, TriageGifGroupsGifStagePair, TriageGifGroupsGifStageReq,
    TriageGifGroupsGifStageRes, TriageGifPair,
};
use anyhow::Result;
use image::codecs::gif::GifDecoder;
use image::error::{ParameterError, ParameterErrorKind};
use image::imageops::FilterType;
use image::{AnimationDecoder, DynamicImage, ImageBuffer, ImageDecoder, ImageError, Rgba};
use image_hasher::{Hasher, HasherConfig, ImageHash};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io::BufReader;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
enum GifWorkerError {
    #[error("Gif frames are too poor: {0}, expected at least 5 frames")]
    PoorFrames(usize),
    #[error(transparent)]
    InternalImageError(#[from] ImageError),
    #[error(transparent)]
    InternalIOError(#[from] std::io::Error),
    #[error(transparent)]
    InternalHashError(#[from] anyhow::Error),
}

pub struct GifWorker {
    hasher: Hasher,
    extract_hw: u32,
}

impl GifWorker {
    pub fn new(extract_hw: u32) -> Self {
        let hasher = HasherConfig::new()
            .hash_alg(image_hasher::HashAlg::Gradient)
            .resize_filter(FilterType::Lanczos3)
            .hash_size(32, 32)
            .to_hasher();
        Self { extract_hw, hasher }
    }

    pub fn process<'a>(
        &self,
        gifs: &'a TriageGifGroupsGifStageReq,
    ) -> Result<TriageGifGroupsGifStageRes<'a>> {
        let pb = ProgressBar::new(gifs.len() as u64);
        let style = ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")?;
        pb.set_style(style);
        pb.set_message("Extracting GIF frames...");
        let results: Vec<Option<TriageGifGroupsGifStagePair<'a>>> = gifs
            .par_iter()
            .map(|gif_pair| {
                pb.inc(1);
                // Encapsulate the internal errors of GIFs within process_pair
                gif_pair.as_ref().map(|p| self.process_pair(p))
            })
            .collect();
        pb.finish_with_message("All GIFs processed");
        Ok(results)
    }

    /// Determining whether all frames of a GIF image are identical
    fn judge_gif_frame(&self, path: &str) -> Result<bool, GifWorkerError> {
        tracing::debug!("Judging GIF frame: {}", path);
        let file = File::open(path)?;
        let reader = GifDecoder::new(BufReader::new(file))?;
        let (width, height) = reader.dimensions();
        let frames = reader.into_frames().collect_frames()?;
        if frames.len() <= 1 {
            return Ok(true);
        }
        let hashes: Vec<ImageHash> = frames
            .into_iter()
            .map(|frame| -> Result<ImageHash, GifWorkerError> {
                let raw: Vec<u8> = frame.buffer().to_vec();
                let img_buf: ImageBuffer<Rgba<u8>, Vec<u8>> =
                    ImageBuffer::from_raw(width, height, raw).ok_or_else(|| {
                        ImageError::Parameter(ParameterError::from_kind(
                            ParameterErrorKind::DimensionMismatch,
                        ))
                    })?;
                let dyn_img = DynamicImage::ImageRgba8(img_buf);
                let hash = self.hasher.hash_image(&dyn_img);
                Ok(hash)
            })
            .collect::<Result<Vec<_>, GifWorkerError>>()?;
        match hashes.split_first() {
            None => panic!("Cannot happen at all!"),
            Some((first_hash, rest_hashes)) => {
                let is_all_same = rest_hashes.iter().enumerate().all(|(i, h)| {
                    let original_idx = i + 1;
                    let score = first_hash.dist(h);
                    tracing::debug!(
                        "Comparing image {}'s idx=0 vs idx={}, score = {}",
                        path,
                        original_idx,
                        score
                    );
                    score < 5
                });
                if is_all_same {
                    tracing::debug!("All frames in GIF {} are identical", path);
                }
                Ok(is_all_same)
            }
        }
    }

    fn process_pair<'a>(&self, gifs: &'a TriageGifPair<'a>) -> TriageGifGroupsGifStagePair<'a> {
        type InvalidGifIdT<'a> = Option<Vec<(&'a Uuid, &'a str, usize, String)>>;
        /// id, path, size, frame_len
        type DiscardFrameGifT<'a> = Option<Vec<(&'a Uuid, &'a str, usize, Option<usize>)>>;
        type PrepareClipGifT<'a> = Option<Vec<(&'a Uuid, &'a str, usize, GifFrames)>>;

        let mut invalid_gif_id: InvalidGifIdT<'a> = None;
        let mut discard_same_frame_gif_id: DiscardFrameGifT<'a> = None;
        let mut prepare_clip_gif_id: PrepareClipGifT<'a> = None;

        let try_add_invalid = |opt: &mut InvalidGifIdT<'a>,
                               id: &'a Uuid,
                               path: &'a str,
                               size: usize,
                               reason: &str| {
            match opt {
                Some(vec) => vec.push((id, path, size, reason.to_string())),
                None => *opt = Some(vec![(id, path, size, reason.to_string())]),
            }
        };
        let try_add_prepare_clip = |opt: &mut PrepareClipGifT<'a>,
                                    id: &'a Uuid,
                                    path: &'a str,
                                    size: usize,
                                    frame: Vec<Vec<u8>>| {
            match opt {
                Some(vec) => vec.push((id, path, size, frame)),
                None => *opt = Some(vec![(id, path, size, frame)]),
            }
        };

        // preprocess
        let gifs = gifs
            .iter()
            .filter(|gif| {
                let res = self.judge_gif_frame(gif.path).unwrap_or(false);
                if res {
                    match discard_same_frame_gif_id {
                        Some(ref mut vec) => vec.push((gif.uuid, gif.path, gif.size, None)),
                        None => {
                            discard_same_frame_gif_id =
                                Some(vec![(gif.uuid, gif.path, gif.size, None)])
                        }
                    }
                }
                !res
            })
            .collect::<Vec<_>>();

        for &TriageGif {
            uuid: id,
            path,
            size,
        } in gifs
        {
            match self.process_single(path, true) {
                Ok(frames) => {
                    try_add_prepare_clip(&mut prepare_clip_gif_id, id, path, size, frames)
                }
                Err(
                    e @ GifWorkerError::InternalImageError(_)
                    | e @ GifWorkerError::InternalIOError(_),
                ) => {
                    tracing::error!("Error processing GIF {}: {}", id, e);
                    try_add_invalid(&mut invalid_gif_id, id, path, size, &e.to_string());
                }
                _ => {} // cannot exist
            }
        }

        let invalid_group = invalid_gif_id.map(|entries| {
            let (ids, reasons): (Vec<&Uuid>, Vec<String>) = entries
                .into_iter()
                .map(|(id, _, _, reason)| (id, reason))
                .unzip();
            (ids, reasons)
        });

        let discard_same_frame_group = discard_same_frame_gif_id
            .map(|entries| entries.into_iter().map(|(id, _, _, _)| id).collect());

        let prepare_group = prepare_clip_gif_id.map(|entries| {
            entries
                .into_iter()
                .map(|(id, path, size, frame)| TriageGifClip {
                    id,
                    path,
                    size,
                    frame,
                })
                .collect()
        });

        TriageGifGroupsGifStagePair {
            invalid_gif_id: invalid_group,
            discard_same_frame_gif_id: discard_same_frame_group,
            prepare_clip_gif_pair: prepare_group,
        }
    }

    fn process_single(
        &self,
        gif_path: &str,
        allow_poor_frame: bool,
    ) -> Result<GifFrames, GifWorkerError> {
        let file = File::open(gif_path).map_err(GifWorkerError::InternalIOError)?;
        let reader =
            GifDecoder::new(BufReader::new(file)).map_err(GifWorkerError::InternalImageError)?;
        let (w, h) = reader.dimensions();
        let frames = reader
            .into_frames()
            .collect_frames()
            .map_err(GifWorkerError::InternalImageError)?;
        let total = frames.len();
        // TODO: d63f2ed8-a3ed-54ba-8624-34d1a049735b vs 42fdd210-3755-5613-a922-5a8d10622024 (?)
        let selected_idxs = match total {
            n if n < 5 && !allow_poor_frame => Err(GifWorkerError::PoorFrames(n)),
            n if n < 5 && allow_poor_frame => Ok((0..n).collect::<Vec<_>>()),
            _ => Ok(Vec::from([
                0,
                total / 4,
                total / 2,
                total * 3 / 4,
                total - 1,
            ])),
        }?;
        let picked = frames
            .into_iter()
            .enumerate()
            .filter_map(|(i, frame)| {
                if selected_idxs.contains(&i) {
                    Some(frame)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let frames_bytes = picked
            .iter()
            .map(|frame| {
                let buf: Vec<u8> = frame.buffer().to_vec();
                let img: ImageBuffer<Rgba<u8>, _> =
                    ImageBuffer::from_raw(w, h, buf).ok_or_else(|| {
                        ImageError::Parameter(ParameterError::from_kind(
                            ParameterErrorKind::DimensionMismatch,
                        ))
                    })?;
                let img = DynamicImage::ImageRgba8(img);
                Ok::<Vec<u8>, ImageError>(
                    img.resize_to_fill(self.extract_hw, self.extract_hw, FilterType::Triangle)
                        .to_rgb8()
                        .into_raw(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(frames_bytes)
    }
}
