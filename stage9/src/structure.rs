use serde::Serialize;
use serde::ser::SerializeStruct;
use uuid::Uuid;

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
    pub discard_poor_frame_gif_id: Option<Vec<&'a Uuid>>,
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

#[derive(Debug, Serialize)]
pub struct FinalClassification<'a> {
    /// KeptTextAnomaliesPic region
    pub kept_text_anomalies_group: &'a Option<Vec<&'a Uuid>>,
    /// NeedTriageGifs region
    pub triaged_gif_and_invalid_group: &'a Option<(Vec<&'a Uuid>, Vec<String>)>,
    pub triaged_gif_and_discard_same_frame_group: &'a Option<Vec<&'a Uuid>>,
    pub triaged_gif_and_discard_poor_frame_group: &'a Option<Vec<&'a Uuid>>,
    pub triaged_gif_and_then_will_keep_group: Option<Vec<&'a Uuid>>,
    pub triaged_gif_and_then_will_delete_group: Option<Vec<&'a Uuid>>,
    /// KeptNonGif region
    pub kept_non_gif: &'a Option<&'a Uuid>,
    /// OtherNeedDeletePics region
    pub other_need_delete_group: &'a Option<Vec<&'a Uuid>>,
}
