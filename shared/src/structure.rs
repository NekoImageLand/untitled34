use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// P1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NekoPoint {
    pub id: Uuid,
    pub height: usize,
    pub weight: usize,
    pub size: Option<usize>, // FIXME: always None in stage2
    pub categories: Option<Vec<String>>,
    pub text_info: Option<NekoPointText>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NekoPointText {
    pub text: String,
    pub text_vector: Vec<f32>, // 768 Dimension
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NekoPointExt {
    pub file_path: String,
    pub source: Option<NekoPointExtResource>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NekoPointExtResource {
    LocalPath(String),
    Blob(Vec<u8>),
}

impl NekoPointExt {
    #[inline]
    pub fn ext(&self) -> &str {
        self.file_path.rsplit('.').next().unwrap()
    }
}

/// P2
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WrongExtFile {
    pub path: String,
    pub expected_ext: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FailedExtFile {
    pub path: String,
    pub error: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum TriageFile {
    Wrong(WrongExtFile),
    Failed(FailedExtFile),
}

/// P3
pub const TEXT_SIM_THRESHOLD: f32 = 0.9;
pub const IMAGE_SIM_THRESHOLD: f32 = 0.985; // TODO: ?

#[derive(Debug, Serialize)]
pub struct TriageGif<'a> {
    pub uuid: &'a Uuid,
    pub path: &'a str,
    pub size: usize,
}

pub type TriageGifPair<'a> = Vec<TriageGif<'a>>;

pub type TriageGifGroupsGifStageReq<'a> = Vec<Option<TriageGifPair<'a>>>;

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
    pub discard_same_frame_gif_id: Option<Vec<&'a Uuid>>,
    // pub discard_poor_frame_gif_id: Option<Vec<&'a Uuid>>,
    pub prepare_clip_gif_pair: Option<TriageGifClipPair<'a>>,
}

pub type TriageGifGroupsGifStageRes<'a> = Vec<Option<TriageGifGroupsGifStagePair<'a>>>;

pub type TriageGifGroupsClipStageReq<'a> = Vec<Option<Option<TriageGifClipPair<'a>>>>;

#[derive(Debug, Serialize)]
pub struct TriageGifGroupsClipStagePair<'a> {
    pub kept_gifs: Option<Vec<TriageGif<'a>>>,
    pub discard_duplicate_gifs: Option<Vec<TriageGif<'a>>>,
}

pub type TriageGifGroupsClipStageRes<'a> = Vec<Option<Option<TriageGifGroupsClipStagePair<'a>>>>;

#[derive(Debug, Serialize, Deserialize)]
pub struct FinalClassification {
    /// KeptTextAnomaliesPic region
    pub kept_text_anomalies_group: Option<Vec<Uuid>>,
    /// NeedTriageGifs region
    pub triaged_gif_and_invalid_group: Option<(Vec<Uuid>, Vec<String>)>,
    pub triaged_gif_and_discard_same_frame_group: Option<Vec<Uuid>>,
    pub triaged_gif_and_then_will_keep_group: Option<Vec<Uuid>>,
    pub triaged_gif_and_then_will_delete_group: Option<Vec<Uuid>>,
    /// KeptNonGif region
    pub kept_non_gif: Option<Uuid>,
    /// OtherNeedDeletePics region
    pub other_need_delete_group: Option<Vec<Uuid>>,
}
