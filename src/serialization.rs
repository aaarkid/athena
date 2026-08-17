//! On-disk format for saved networks and agents.
//!
//! Every file written by `save` starts with four magic bytes and a format version, so a
//! file from another program, or from an older version of this crate, is reported rather
//! than decoded into nonsense. The payload itself is bincode.
//!
//! # Format
//!
//! ```text
//! offset 0  4 bytes   "ATHN"
//! offset 4  4 bytes   format version, little-endian u32
//! offset 8  ...       bincode payload
//! ```
//!
//! Version 1 is the first versioned format. Files written before it have no header and
//! are not readable; they also carried the forward-pass caches, which are now skipped.

use std::fs;
use std::io::{Read, Write};

use bincode::Options;
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::error::{AthenaError, Result};

/// First four bytes of every file this crate writes.
pub const MAGIC: [u8; 4] = *b"ATHN";

/// Current on-disk format version.
pub const FORMAT_VERSION: u32 = 1;

/// Largest payload `decode` will allocate for, as a guard against a corrupt length field
/// asking for a terabyte.
pub const MAX_PAYLOAD_BYTES: u64 = 1 << 30;

const HEADER_LEN: usize = 8;

fn payload_options() -> impl Options {
    // Matches what bincode::serialize writes, with a ceiling on what a length field can
    // ask the allocator for
    bincode::options()
        .with_fixint_encoding()
        .with_limit(MAX_PAYLOAD_BYTES)
}

/// Serialize `value` behind the magic bytes and the format version.
pub fn encode<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let payload = bincode::serialize(value)?;

    let mut out = Vec::with_capacity(HEADER_LEN + payload.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Read back what `encode` wrote, refusing anything else.
pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    if bytes.len() < HEADER_LEN {
        return Err(AthenaError::SerializationError(format!(
            "file is {} bytes, too short to hold the {}-byte header",
            bytes.len(),
            HEADER_LEN
        )));
    }

    if bytes[..4] != MAGIC {
        return Err(AthenaError::SerializationError(
            "not an athena file: the first four bytes are not ATHN".to_string(),
        ));
    }

    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if version != FORMAT_VERSION {
        return Err(AthenaError::SerializationError(format!(
            "file format version {}, this build reads version {}",
            version, FORMAT_VERSION
        )));
    }

    payload_options()
        .deserialize(&bytes[HEADER_LEN..])
        .map_err(AthenaError::from)
}

/// `encode` straight to a file.
pub fn save_to_file<T: Serialize>(value: &T, path: &str) -> Result<()> {
    let encoded = encode(value)?;
    let mut file = fs::File::create(path)?;
    file.write_all(&encoded)?;
    Ok(())
}

/// `decode` straight from a file.
pub fn load_from_file<T: DeserializeOwned>(path: &str) -> Result<T> {
    let mut file = fs::File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    decode(&buffer)
}
