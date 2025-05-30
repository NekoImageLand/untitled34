use crate::structure::{
    GifFrames, TriageGif, TriageGifClip, TriageGifGroupsGifStagePair, TriageGifGroupsGifStageReq,
    TriageGifGroupsGifStageRes, TriageGifPair,
};
use anyhow::Result;
use image::codecs::gif::GifDecoder;
use image::error::{ParameterError, ParameterErrorKind};
use image::imageops::FilterType;
use image::{AnimationDecoder, DynamicImage, ImageBuffer, ImageDecoder, ImageError, Rgba};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::fs::File;
use std::io::BufReader;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
enum GifWorkerError {
    #[error("Gif frames are too small: {0}, expected at least 2 frames")]
    PoorFrames(usize),
    #[error(transparent)]
    InternalImageError(#[from] ImageError),
    #[error(transparent)]
    InternalIOError(#[from] std::io::Error),
}

pub struct GifWorker {
    extract_hw: u32,
}

impl GifWorker {
    pub fn new(extract_hw: u32) -> Self {
        Self { extract_hw }
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
        let results: Vec<TriageGifGroupsGifStagePair<'a>> = gifs
            .par_iter()
            .map(|gif_pair| {
                pb.inc(1);
                self.process_pair(gif_pair) // Encapsulate the internal errors of GIFs within process_pair
            })
            .collect::<Vec<TriageGifGroupsGifStagePair<'a>>>();
        pb.finish_with_message("All GIFs processed");
        Ok(results)
    }

    fn process_pair<'a>(&self, gifs: &'a TriageGifPair<'a>) -> TriageGifGroupsGifStagePair<'a> {
        type InvalidGifIdT<'a> = Option<Vec<(&'a Uuid, &'a str, usize, String)>>;
        type DiscardSingleFrameGifT<'a> = Option<Vec<(&'a Uuid, &'a str, usize)>>;
        type PrepareClipGifT<'a> = Option<Vec<(&'a Uuid, &'a str, usize, GifFrames)>>;

        let mut invalid_gif_id: InvalidGifIdT<'a> = None;
        let mut discard_single_frame_gif_id: DiscardSingleFrameGifT<'a> = None;
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
        let try_add_discard_single_frame =
            |opt: &mut DiscardSingleFrameGifT<'a>, id: &'a Uuid, path: &'a str, size: usize| {
                match opt {
                    Some(vec) => vec.push((id, path, size)),
                    None => *opt = Some(vec![(id, path, size)]),
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

        for &TriageGif { id, path, size } in gifs {
            match self.process_single(path, false) {
                Ok(frames) => {
                    try_add_prepare_clip(&mut prepare_clip_gif_id, id, path, size, frames)
                }
                Err(GifWorkerError::PoorFrames(_)) => {
                    tracing::warn!("GIF {} has too few frames, discarding: {}", id, path);
                    try_add_discard_single_frame(&mut discard_single_frame_gif_id, id, path, size)
                }
                Err(
                    e @ GifWorkerError::InternalImageError(_)
                    | e @ GifWorkerError::InternalIOError(_),
                ) => {
                    tracing::error!("Error processing GIF {}: {}", id, e);
                    try_add_invalid(&mut invalid_gif_id, id, path, size, &e.to_string());
                }
            }
        }
        // Check the edge case: are all frame durations in the GIF < 5?
        // Control reversal
        if prepare_clip_gif_id.as_ref().is_none() && discard_single_frame_gif_id.as_ref().is_some()
        {
            if let Some(mut entries) = discard_single_frame_gif_id.take() {
                if !entries.is_empty() {
                    let min_idx = entries
                        .iter()
                        .enumerate()
                        .min_by_key(|&(_, &(_, _, size))| size)
                        .map(|(idx, _)| idx)
                        .unwrap();
                    let (discard_id, discard_path, discard_size) = entries.remove(min_idx);
                    discard_single_frame_gif_id =
                        Some(vec![(discard_id, discard_path.clone(), discard_size)]);

                    for (id, path, size) in entries {
                        match self.process_single(path.clone(), true) {
                            Ok(frames) => {
                                try_add_prepare_clip(
                                    &mut prepare_clip_gif_id,
                                    id,
                                    path,
                                    size,
                                    frames,
                                );
                            }
                            Err(e) => {
                                tracing::error!(
                                    "GIF {} has too few frames even after allowing poor frames,\
                                     error = {}, discarding: {}",
                                    id,
                                    e.to_string(),
                                    path
                                );
                                try_add_discard_single_frame(
                                    &mut discard_single_frame_gif_id,
                                    id,
                                    path,
                                    size,
                                );
                            }
                        }
                    }
                } else {
                    discard_single_frame_gif_id = Some(entries);
                }
            }
        }

        let invalid_group = invalid_gif_id.map(|entries| {
            let (ids, reasons): (Vec<&Uuid>, Vec<String>) = entries
                .into_iter()
                .map(|(id, _, _, reason)| (id, reason))
                .unzip();
            (ids, reasons)
        });

        let discard_group = discard_single_frame_gif_id
            .map(|entries| entries.into_iter().map(|(id, _, _)| id).collect());

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
            discard_single_frame_gif_id: discard_group,
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
            n if allow_poor_frame => Ok((0..n).collect::<Vec<_>>()),
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
