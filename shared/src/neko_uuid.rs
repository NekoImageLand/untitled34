use sha1::{Digest, Sha1};
use uuid::Uuid;

pub struct NekoUuid {
    namespace: Uuid,
}

impl NekoUuid {
    pub fn new() -> Self {
        NekoUuid {
            namespace: Uuid::new_v5(
                &Uuid::NAMESPACE_DNS,
                "github.com/hv0905/NekoImageGallery".as_ref(),
            ),
        }
    }

    pub fn generate(&self, data: &[u8]) -> Uuid {
        let digest = Sha1::digest(data);
        self.generate_from_sha1(&digest.into())
    }

    #[inline]
    pub fn generate_from_sha1(&self, digest: &[u8; 20]) -> Uuid {
        let hex_str = hex::encode(digest);
        Uuid::new_v5(&self.namespace, hex_str.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neko_uuid_generate() {
        let neko_uuid = NekoUuid::new();
        let data = b"qwq";
        let uuid = neko_uuid.generate(data);
        assert_eq!(uuid.to_string(), "6c439572-44ed-5ba9-a6fb-627b06406c73");
    }

    #[test]
    fn test_neko_uuid_generate_from_sha1() {
        let neko_uuid = NekoUuid::new();
        let qwq = Sha1::digest(b"qwq");
        let uuid = neko_uuid.generate_from_sha1(&qwq.into());
        assert_eq!(uuid.to_string(), "6c439572-44ed-5ba9-a6fb-627b06406c73");
    }
}
