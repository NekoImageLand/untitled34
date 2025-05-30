use serde::Serialize;
use serde::ser::SerializeStruct;
use uuid::Uuid;

pub const TEXT_SIM_THRESHOLD: f32 = 0.9;
pub const IMAGE_SIM_THRESHOLD: f32 = 0.985; // TODO: ?

#[derive(Debug)]
pub struct TriageGif<'a> {
    pub id: &'a Uuid,
    pub path: &'a str,
    pub size: usize,
}

impl Serialize for TriageGif<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("TriageGif", 3)?;
        state.serialize_field("id", self.id)?;
        state.serialize_field("path", self.path)?;
        state.serialize_field("size", &self.size)?;
        state.end()
    }
}

pub type TriageGifPair<'a> = Vec<TriageGif<'a>>;

pub type TriageGifGroupsGifStageReq<'a> = Vec<TriageGifPair<'a>>;

pub type GifFrame = Vec<u8>; // TODO: make it into really "new type" ?
pub type GifFrames = Vec<GifFrame>;

#[derive(Debug)]
pub struct TriageGifClip<'a> {
    pub id: &'a Uuid,
    pub path: &'a str,
    pub size: usize,
    pub frame: GifFrames,
}

impl Serialize for TriageGifClip<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("TriageGifClip", 4)?;
        state.serialize_field("id", self.id)?;
        state.serialize_field("path", self.path)?;
        state.serialize_field("size", &self.size)?;
        state.serialize_field("frame", &format!("[Frame] len={}", &self.frame.len()))?;
        state.end()
    }
}

pub type TriageGifClipPair<'a> = Vec<TriageGifClip<'a>>;

#[derive(Debug, Serialize)]
pub struct TriageGifGroupsGifStagePair<'a> {
    pub invalid_gif_id: Option<(Vec<&'a Uuid>, Vec<String>)>, // (uuid, FailedReason)
    pub discard_single_frame_gif_id: Option<Vec<&'a Uuid>>,
    pub prepare_clip_gif_pair: Option<TriageGifClipPair<'a>>,
}

pub type TriageGifGroupsGifStageRes<'a> = Vec<TriageGifGroupsGifStagePair<'a>>;

pub type TriageGifGroupsClipStageReq<'a> = Vec<TriageGifClipPair<'a>>;

#[derive(Debug, Serialize)]
pub struct TriageGifGroupsClipStagePair<'a> {
    pub kept_gifs: Option<Vec<TriageGif<'a>>>,
    pub discard_duplicate_gifs: Option<Vec<TriageGif<'a>>>,
}

pub type TriageGifGroupsClipStageRes<'a> = Vec<TriageGifGroupsClipStagePair<'a>>;
