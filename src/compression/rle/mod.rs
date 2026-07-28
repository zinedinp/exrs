use std::convert::TryInto;

use super::{optimize_bytes::*, Error, Result, *};

// inspired by  https://github.com/openexr/openexr/blob/master/OpenEXR/IlmImf/ImfRle.cpp

const MIN_RUN_LENGTH: usize = 3;
const MAX_RUN_LENGTH: usize = 127;

pub fn decompress_bytes(
    channels: &ChannelList,
    compressed_le: &[u8],
    rectangle: IntegerBounds,
    expected_byte_size: usize,
    pedantic: bool,
) -> Result<ByteVec> {
    let mut decompressed_le = unpack_rle_tokens(compressed_le, expected_byte_size, pedantic)?;
    differences_to_samples(&mut decompressed_le);
    interleave_byte_blocks(&mut decompressed_le);
    super::convert_little_endian_to_current(decompressed_le, channels, rectangle)
    // TODO no alloc
}

/// Shared by this compression method and DWA's RLE scheme (both port OpenEXR's
/// `internal_rle_decompress`, see `compression::dwa`) - kept separate from
/// `decompress_bytes` because DWA does not apply the delta prediction /
/// byte-block interleaving done there.
pub(super) fn unpack_rle_tokens(
    compressed_le: &[u8],
    expected_byte_size: usize,
    pedantic: bool,
) -> Result<ByteVec> {
    let mut remaining_le = compressed_le;
    let mut decompressed_le = Vec::with_capacity(expected_byte_size.min(8 * 2048));

    while !remaining_le.is_empty() && decompressed_le.len() != expected_byte_size {
        let count = take_1(&mut remaining_le)? as i8 as i32;

        if count < 0 {
            // take the next '-count' bytes as-is
            let values = take_n(&mut remaining_le, -count as usize)?;
            decompressed_le.extend_from_slice(values);
        } else {
            // repeat the next value 'count + 1' times
            let value = take_1(&mut remaining_le)?;
            decompressed_le.resize(decompressed_le.len() + (count as usize) + 1, value);
        }
    }

    if pedantic && !remaining_le.is_empty() {
        return Err(Error::invalid("data amount"));
    }

    Ok(decompressed_le)
}

/// The same token format as `unpack_rle_tokens`, but expanding into a
/// caller-owned buffer the way OpenEXR's `internal_rle_decompress` does, so
/// the buffer can be reused across chunks instead of being allocated (and
/// page-faulted in) once per chunk. Returns the number of bytes written;
/// unlike `unpack_rle_tokens` a token stream that would run past the end of
/// the buffer is rejected rather than expanded and then ignored.
pub(super) fn unpack_rle_tokens_into(compressed_le: &[u8], out: &mut [u8]) -> Result<usize> {
    let mut remaining_le = compressed_le;
    let mut written = 0usize;

    while !remaining_le.is_empty() && written != out.len() {
        let count = take_1(&mut remaining_le)? as i8 as i32;

        if count < 0 {
            let length = -count as usize;
            let values = take_n(&mut remaining_le, length)?;
            let end = written.checked_add(length).ok_or_else(|| Error::invalid("compressed data"))?;
            if end > out.len() {
                return Err(Error::invalid("compressed data"));
            }
            out[written..end].copy_from_slice(values);
            written = end;
        } else {
            let length = count as usize + 1;
            let value = take_1(&mut remaining_le)?;
            let end = written.checked_add(length).ok_or_else(|| Error::invalid("compressed data"))?;
            if end > out.len() {
                return Err(Error::invalid("compressed data"));
            }
            out[written..end].fill(value);
            written = end;
        }
    }

    Ok(written)
}

pub fn compress_bytes(
    channels: &ChannelList,
    uncompressed_ne: ByteVec,
    rectangle: IntegerBounds,
) -> Result<ByteVec> {
    // see https://github.com/AcademySoftwareFoundation/openexr/blob/3bd93f85bcb74c77255f28cdbb913fdbfbb39dfe/OpenEXR/IlmImf/ImfTiledOutputFile.cpp#L750-L842
    let mut data_le =
        super::convert_current_to_little_endian(uncompressed_ne, channels, rectangle)?; // TODO no alloc

    separate_bytes_fragments(&mut data_le);
    samples_to_differences(&mut data_le);

    Ok(pack_rle_tokens(&data_le))
}

/// Shared by this compression method and DWA's RLE section. This only emits
/// the byte-oriented RLE token stream; callers are responsible for any byte
/// prediction, byte interleaving, or zlib wrapping required by their format.
pub(super) fn pack_rle_tokens(data_le: &[u8]) -> ByteVec {
    let mut compressed_le = crate::block::pool::take_with_capacity(data_le.len());
    let mut run_start = 0;

    while run_start < data_le.len() {
        let mut run_end = run_start + run_length_at(data_le, run_start);

        if run_end - run_start >= MIN_RUN_LENGTH {
            compressed_le.push((((run_end - run_start) as i32) - 1) as u8);
            compressed_le.push(data_le[run_start]);
            run_start = run_end;
        } else {
            while run_end < data_le.len()
                && (run_end + 1 >= data_le.len()
                    || data_le[run_end] != data_le[run_end + 1]
                    || run_end + 2 >= data_le.len()
                    || data_le[run_end + 1] != data_le[run_end + 2])
                && run_end - run_start < MAX_RUN_LENGTH
            {
                run_end += 1;
            }

            compressed_le.push(((run_start as i32) - (run_end as i32)) as u8);
            compressed_le.extend_from_slice(&data_le[run_start..run_end]);

            run_start = run_end;
        }
    }

    compressed_le
}

/// How many consecutive bytes starting at `start` equal `data[start]`,
/// capped at `MAX_RUN_LENGTH + 1` (the actual achievable repeat-run length --
/// OpenEXR caps a single repeat token at 128 bytes; `MAX_RUN_LENGTH` itself
/// is 127, one less, because the original loop's `(run_end - run_start) - 1
/// < MAX_RUN_LENGTH` bound lets one extra byte through). Always >= 1, since
/// `data[start]` trivially matches itself.
///
/// Wordwise (SWAR): compares up to 8 bytes at once against a broadcast
/// target via XOR + `trailing_zeros`, instead of one byte per iteration.
/// Real content is a mix of short literal runs
/// and long flat runs
/// (constant-color regions: mattes, alpha, skies), where cutting iterations
/// up to 8x is a real, measured win. See `rle_encode_experiment` for the
/// isolated A/B this was benchmark-gated on before shipping here.
pub(super) fn run_length_at(data: &[u8], start: usize) -> usize {
    let target = data[start];
    let limit = (data.len() - start).min(MAX_RUN_LENGTH + 1);
    let region = &data[start..start + limit];
    let pattern = u64::from_le_bytes([target; 8]);

    let mut count = 0usize;
    let mut chunks = region.chunks_exact(8);
    for chunk in &mut chunks {
        let word = u64::from_le_bytes(chunk.try_into().unwrap());
        let diff = word ^ pattern;
        if diff == 0 {
            count += 8;
        } else {
            return count + (diff.trailing_zeros() / 8) as usize;
        }
    }
    for &byte in chunks.remainder() {
        if byte != target {
            return count;
        }
        count += 1;
    }
    count
}

fn take_1(slice: &mut &[u8]) -> Result<u8> {
    if !slice.is_empty() {
        let result = slice[0];
        *slice = &slice[1..];
        Ok(result)
    } else {
        Err(Error::invalid("compressed data"))
    }
}

fn take_n<'s>(slice: &mut &'s [u8], n: usize) -> Result<&'s [u8]> {
    if n <= slice.len() {
        let (front, back) = slice.split_at(n);
        *slice = back;
        Ok(front)
    } else {
        Err(Error::invalid("compressed data"))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn assert_roundtrips(data: &[u8]) {
        let packed = pack_rle_tokens(data);
        let unpacked = unpack_rle_tokens(&packed, data.len(), true).unwrap();
        assert_eq!(data, &unpacked[..], "roundtrip failed for {} bytes", data.len());
    }

    /// Exercises the exact boundary the widened `run_length_at` has to get
    /// right: OpenEXR's repeat token caps a run at 128 bytes even though
    /// `MAX_RUN_LENGTH` is 127 (see `run_length_at`'s doc comment) > lengths
    /// just below/at/above that boundary are where an off-by-one would hide.
    #[test]
    fn run_length_boundary_lengths_roundtrip() {
        for &len in &[1usize, 2, 3, 4, 63, 64, 65, 126, 127, 128, 129, 130, 255, 256, 257, 1000] {
            assert_roundtrips(&vec![7u8; len]);
        }
    }

    #[test]
    fn empty_roundtrips() {
        assert_roundtrips(&[]);
    }

    #[test]
    fn no_repeats_roundtrips() {
        let data: Vec<u8> = (0..500).map(|i| (i * 37) as u8).collect();
        assert_roundtrips(&data);
    }

    #[test]
    fn mixed_runs_and_literals_roundtrips() {
        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        for _ in 0..200 {
            let mut data = Vec::new();
            while data.len() < 4000 {
                if rng.random_range(0.0..1.0) < 0.3 {
                    let run_len = rng.random_range(1..=140);
                    let value = rng.random::<u8>();
                    data.extend(std::iter::repeat(value).take(run_len));
                } else {
                    data.push(rng.random::<u8>());
                }
            }
            assert_roundtrips(&data);
        }
    }

    /// `run_length_at` itself must never report a length that isn't actually
    /// backed by matching bytes, and must never exceed the 128-byte cap.
    #[test]
    fn run_length_at_matches_naive_scan() {
        fn naive(data: &[u8], start: usize) -> usize {
            let target = data[start];
            let limit = (data.len() - start).min(MAX_RUN_LENGTH + 1);
            let mut len = 1;
            while len < limit && data[start + len] == target {
                len += 1;
            }
            len
        }

        use rand::{RngExt, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut data = Vec::new();
        while data.len() < 20_000 {
            if rng.random_range(0.0..1.0) < 0.4 {
                let run_len = rng.random_range(1..=140);
                let value = rng.random::<u8>();
                data.extend(std::iter::repeat(value).take(run_len));
            } else {
                data.push(rng.random::<u8>());
            }
        }

        for start in 0..data.len() {
            let got = run_length_at(&data, start);
            assert!(got >= 1 && got <= MAX_RUN_LENGTH + 1);
            assert_eq!(got, naive(&data, start), "start={start}");
        }
    }
}
