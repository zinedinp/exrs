// Moving channel samples between the layouts DWA needs: the interleaved
// scanline buffer the rest of the crate uses, the per-channel byte runs, the
// planar byte planes of the UNKNOWN/RLE sections, and back again.

use super::{ChannelInfo, CompressorScheme};
use crate::{
    error::{Error, Result},
    meta::attribute::{ChannelList, IntegerBounds},
};
#[cfg(test)]
use crate::meta::attribute::SampleType;

pub(super) fn split_scanline_channels(
    data: &[u8],
    channels: &ChannelList,
    infos: &[ChannelInfo],
    rectangle: IntegerBounds,
) -> Result<Vec<Vec<u8>>> {
    // The scanline buffer is already in channel order, but the encoder needs
    // each channel sliced back out so the per-scheme packing matches the C
    // reference's running cursors.
    let mut per_channel: Vec<Vec<u8>> = infos
        .iter()
        .map(|info| Vec::with_capacity(info.width * info.height * info.bytes_per_sample))
        .collect();
    let mut input = data;

    for y in rectangle.position.y()..rectangle.end().y() {
        for (index, channel) in channels.list.iter().enumerate() {
            let sampling_y = channel.sampling.y().max(1) as i32;
            if y % sampling_y != 0 {
                continue;
            }

            let row_length = infos[index].width * infos[index].bytes_per_sample;
            if row_length > input.len() {
                return Err(Error::invalid("DWA input data truncated"));
            }
            let (row, rest) = input.split_at(row_length);
            per_channel[index].extend_from_slice(row);
            input = rest;
        }
    }

    if !input.is_empty() {
        return Err(Error::invalid("DWA input data size mismatch"));
    }

    Ok(per_channel)
}

pub(super) fn pack_unknown_channels(
    infos: &[ChannelInfo],
    channel_bytes: &[Vec<u8>],
    scheme: CompressorScheme,
) -> Vec<u8> {
    // UNKNOWN channels stay planar and are concatenated in channel order
    // before the zlib step.
    let total_len = infos
        .iter()
        .zip(channel_bytes)
        .filter(|(info, _)| info.scheme == scheme)
        .map(|(_, bytes)| bytes.len())
        .sum();
    let mut out = Vec::with_capacity(total_len);

    for (info, bytes) in infos.iter().zip(channel_bytes) {
        if info.scheme == scheme {
            out.extend_from_slice(bytes);
        }
    }
    out
}

pub(super) fn pack_rle_channels(infos: &[ChannelInfo], channel_bytes: &[Vec<u8>]) -> Vec<u8> {
    // RLE channels are repacked into byte planes first, then byte-RLE'd and
    // zlib-compressed by the caller.
    let total_len = infos
        .iter()
        .zip(channel_bytes)
        .filter(|(info, _)| info.scheme == CompressorScheme::Rle)
        .map(|(_, bytes)| bytes.len())
        .sum();
    let mut out = Vec::with_capacity(total_len);

    for (info, bytes) in infos.iter().zip(channel_bytes) {
        if info.scheme == CompressorScheme::Rle {
            out.extend_from_slice(&separate_byte_planes(bytes, info.bytes_per_sample));
        }
    }
    out
}

fn separate_byte_planes(interleaved: &[u8], bytes_per_sample: usize) -> Vec<u8> {
    let sample_count = interleaved.len() / bytes_per_sample;
    let mut planar = vec![0u8; interleaved.len()];

    // bytes_per_sample is 2 (F16) or 4 (U32/F32) for every RLE channel the
    // encoder ever produces (see channel_rules.rs). Iterating via
    // `.chunks_exact()`/`.iter_mut()` over equal-length slices instead of
    // indexing by `sample` lets the compiler prove every access is in-bounds
    // and drop the per-element bounds check; mirrors the fix
    // for the decode-side interleave direction in
    // `write_scanlines_fused`.
    match bytes_per_sample {
        2 => {
            let (plane0, plane1) = planar.split_at_mut(sample_count);
            for (chunk, (p0, p1)) in
                interleaved.chunks_exact(2).zip(plane0.iter_mut().zip(plane1.iter_mut()))
            {
                *p0 = chunk[0];
                *p1 = chunk[1];
            }
        }
        4 => {
            let (plane0, rest) = planar.split_at_mut(sample_count);
            let (plane1, rest) = rest.split_at_mut(sample_count);
            let (plane2, plane3) = rest.split_at_mut(sample_count);
            for (chunk, (((p0, p1), p2), p3)) in interleaved.chunks_exact(4).zip(
                plane0
                    .iter_mut()
                    .zip(plane1.iter_mut())
                    .zip(plane2.iter_mut())
                    .zip(plane3.iter_mut()),
            ) {
                *p0 = chunk[0];
                *p1 = chunk[1];
                *p2 = chunk[2];
                *p3 = chunk[3];
            }
        }
        _ => {
            for byte in 0..bytes_per_sample {
                for sample in 0..sample_count {
                    planar[byte * sample_count + sample] =
                        interleaved[sample * bytes_per_sample + byte];
                }
            }
        }
    }

    planar
}

pub(super) fn u16s_to_le_bytes(values: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// Split a planar buffer into one byte run per channel of the given scheme,
/// in channel order (mirrors "DwaCompressor_setupChannelData"s running
/// per-scheme cursor). Other schemes get an empty vec.
#[cfg(test)]
pub(super) fn split_planar_channels(
    infos: &[ChannelInfo],
    scheme: CompressorScheme,
    planar: &[u8],
) -> Result<Vec<Vec<u8>>> {
    let mut per_channel = vec![vec![]; infos.len()];
    let mut cursor = 0;

    for (channel_bytes, info) in per_channel.iter_mut().zip(infos) {
        if info.scheme != scheme {
            continue;
        }
        let length = info.width * info.height * info.bytes_per_sample;
        *channel_bytes = planar
            .get(cursor..cursor + length)
            .ok_or_else(|| Error::invalid("truncated DWA channel data"))?
            .to_vec();
        cursor += length;
    }
    Ok(per_channel)
}

/// Restore per-sample byte order from byte planes
#[cfg(test)]
pub(super) fn interleave_byte_planes(planar: &[u8], bytes_per_sample: usize) -> Vec<u8> {
    let sample_count = planar.len() / bytes_per_sample;
    let mut interleaved = vec![0u8; planar.len()];
    for sample in 0..sample_count {
        for byte in 0..bytes_per_sample {
            interleaved[sample * bytes_per_sample + byte] = planar[byte * sample_count + sample];
        }
    }
    interleaved
}

/// Byte offset, for each channel and each of its local (subsampling-adjusted)
/// rows, into the final scanline-interleaved output buffer: rows of "y"
/// ascending, channels in list order within each row, samples little-endian.
/// Shared by the lossy DCT decode path (which writes its output directly at
/// these offsets, avoiding an intermediate per-channel buffer + copy) and
/// `write_scanlines_fused` below (which still copies UNKNOWN/RLE planar data,
/// since those need a layout transform decode_lossy_dct_group doesn't).
pub(super) fn compute_row_offsets(
    channels: &ChannelList,
    infos: &[ChannelInfo],
    rectangle: IntegerBounds,
) -> Vec<Vec<usize>> {
    let mut offsets: Vec<Vec<usize>> =
        infos.iter().map(|info| vec![0usize; info.height]).collect();
    let mut cursor = 0usize;

    for y in rectangle.position.y()..rectangle.end().y() {
        for (index, channel) in channels.list.iter().enumerate() {
            let sampling_y = channel.sampling.y().max(1) as i32;
            if y % sampling_y != 0 {
                continue;
            }

            let info = &infos[index];
            let row = ((y - rectangle.position.y()) / sampling_y) as usize;
            offsets[index][row] = cursor;
            cursor += info.width * info.bytes_per_sample;
        }
    }

    offsets
}

/// Copy the UNKNOWN/RLE planar decode results into the scanline layout the
/// rest of exrs expects, at the offsets `compute_row_offsets` assigned them.
/// LossyDct channels are skipped: the lossy DCT decode already wrote them
/// directly into `out` at the same offsets. Reads straight from the section
/// planar buffers into `out` in one pass -- no intermediate per-channel
/// allocation. Mirrors OpenEXR C++'s `LOSSY_DCT`-sibling `RLE`/`UNKNOWN`
/// cases in `internal_dwa_compressor.h` (t6/t7), which read directly from
/// their planar-decode cursors into the final per-channel output rows.
pub(super) fn write_scanlines_fused(
    channels: &ChannelList,
    infos: &[ChannelInfo],
    rectangle: IntegerBounds,
    row_offsets: &[Vec<usize>],
    unknown_planar: &[u8],
    rle_planar: &[u8],
    out: &mut [u8],
) -> Result<()> {
    // Each channel's starting cursor into its scheme's planar buffer, in
    // channel-list order -- mirrors split_planar_channels' cursor advance,
    // computed once so the per-row loop below can index directly.
    let mut unknown_cursor = vec![0usize; infos.len()];
    let mut cursor = 0usize;
    for (info, slot) in infos.iter().zip(unknown_cursor.iter_mut()) {
        if info.scheme == CompressorScheme::Unknown {
            *slot = cursor;
            cursor += info.width * info.height * info.bytes_per_sample;
        }
    }
    if cursor > unknown_planar.len() {
        return Err(Error::invalid("truncated DWA channel data"));
    }

    let mut rle_cursor = vec![0usize; infos.len()];
    let mut cursor = 0usize;
    for (info, slot) in infos.iter().zip(rle_cursor.iter_mut()) {
        if info.scheme == CompressorScheme::Rle {
            *slot = cursor;
            cursor += info.width * info.height * info.bytes_per_sample;
        }
    }
    if cursor > rle_planar.len() {
        return Err(Error::invalid("truncated DWA channel data"));
    }

    for y in rectangle.position.y()..rectangle.end().y() {
        for (index, channel) in channels.list.iter().enumerate() {
            let sampling_y = channel.sampling.y().max(1) as i32;
            if y % sampling_y != 0 {
                continue;
            }

            let info = &infos[index];
            if info.scheme != CompressorScheme::Unknown && info.scheme != CompressorScheme::Rle {
                continue;
            }

            let row = ((y - rectangle.position.y()) / sampling_y) as usize;
            let offset = row_offsets[index][row];
            let width = info.width;
            let bytes_per_sample = info.bytes_per_sample;
            let row_length = width * bytes_per_sample;
            let out_row = &mut out[offset..offset + row_length];

            if info.scheme == CompressorScheme::Unknown {
                let base = unknown_cursor[index] + row * row_length;
                out_row.copy_from_slice(&unknown_planar[base..base + row_length]);
            } else {
                // RLE-decoded channels stay byte-plane separated (see
                // `separate_byte_planes`): plane `byte` of this channel spans
                // `sample_count` bytes starting at `channel_base + byte *
                // sample_count`, samples in row-major (y then x) order.
                let sample_count = width * info.height;
                let channel_base = rle_cursor[index];
                let row_base = channel_base + row * width;
                // bytes_per_sample is 2 (F16) or 4 (U32/F32) for every RLE
                // channel the encoder ever produces (see channel_rules.rs).
                // Unrolling those two cases turns the `byte in
                // 0..bytes_per_sample` loop; 2 iterations of real work
                // dominated by its own branch/counter overhead, per a perf
                // profile -- into straight-line code the compiler can pipeline.
                // Iterate via `.zip()` over equal-length slices rather than
                // indexing by `x`; lets the compiler prove every access is
                // in-bounds and drop the per-element bounds check, which a
                // perf profile showed dominating this loop's cost (the loop
                // body is only a byte load + store, so a compare+branch pair
                // per element roughly doubled the work).
                match bytes_per_sample {
                    2 => {
                        let plane0 = &rle_planar[row_base..row_base + width];
                        let plane1 = &rle_planar
                            [row_base + sample_count..row_base + sample_count + width];
                        for (out_pair, (&b0, &b1)) in
                            out_row.chunks_exact_mut(2).zip(plane0.iter().zip(plane1))
                        {
                            out_pair[0] = b0;
                            out_pair[1] = b1;
                        }
                    }
                    4 => {
                        let plane0 = &rle_planar[row_base..row_base + width];
                        let plane1 = &rle_planar
                            [row_base + sample_count..row_base + sample_count + width];
                        let plane2 = &rle_planar[row_base + 2 * sample_count
                            ..row_base + 2 * sample_count + width];
                        let plane3 = &rle_planar[row_base + 3 * sample_count
                            ..row_base + 3 * sample_count + width];
                        for (out_quad, (((&b0, &b1), &b2), &b3)) in out_row
                            .chunks_exact_mut(4)
                            .zip(plane0.iter().zip(plane1).zip(plane2).zip(plane3))
                        {
                            out_quad[0] = b0;
                            out_quad[1] = b1;
                            out_quad[2] = b2;
                            out_quad[3] = b3;
                        }
                    }
                    _ => {
                        for x in 0..width {
                            for byte in 0..bytes_per_sample {
                                out_row[x * bytes_per_sample + byte] =
                                    rle_planar[row_base + byte * sample_count + x];
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use rand::{Rng, RngExt, SeedableRng};

    use super::*;

    const SEED: [u8; 32] = [
        3, 128, 9, 44, 201, 17, 88, 6, 255, 61, 30, 11, 2, 121, 99, 1, 250, 77, 33, 7, 42, 13, 200,
        176, 22, 5, 66, 100, 19, 240, 8, 91,
    ];

    fn random_bytes(random: &mut impl Rng, count: usize) -> Vec<u8> {
        (0..count).map(|_| random.random()).collect()
    }

    fn channel_info(scheme: CompressorScheme, width: usize, height: usize) -> ChannelInfo {
        // Use F16 (2 bytes) as a representative sample type; the layout code
        // only cares about the byte counts, not the semantics.
        ChannelInfo {
            scheme,
            width,
            height,
            bytes_per_sample: SampleType::F16.bytes_per_sample(),
            sample_type: SampleType::F16,
            quantize_linearly: false,
        }
    }

    /// Splitting interleaved samples into byte planes and interleaving them
    /// back must be the identity, for several samples-per-byte widths.
    #[test]
    fn byte_planes_roundtrip() {
        let mut random = rand::rngs::StdRng::from_seed(SEED);

        for bytes_per_sample in [2usize, 4] {
            for sample_count in [0usize, 1, 5, 37] {
                let original = random_bytes(&mut random, sample_count * bytes_per_sample);
                let planar = separate_byte_planes(&original, bytes_per_sample);
                let interleaved = interleave_byte_planes(&planar, bytes_per_sample);
                assert_eq!(interleaved, original);
            }
        }
    }

    /// Packing UNKNOWN channels into one planar buffer and splitting it back
    /// out must reproduce every channel's bytes exactly.
    #[test]
    fn pack_split_unknown_roundtrip() {
        let mut random = rand::rngs::StdRng::from_seed(SEED);

        let infos = vec![
            channel_info(CompressorScheme::Unknown, 4, 3),
            channel_info(CompressorScheme::Unknown, 5, 2),
        ];
        let channel_bytes: Vec<Vec<u8>> = infos
            .iter()
            .map(|info| random_bytes(&mut random, info.width * info.height * info.bytes_per_sample))
            .collect();

        let packed = pack_unknown_channels(&infos, &channel_bytes, CompressorScheme::Unknown);
        let split = split_planar_channels(&infos, CompressorScheme::Unknown, &packed).unwrap();

        assert_eq!(split, channel_bytes);
    }

    /// RLE channels are packed via byte-plane separation; splitting the planar
    /// buffer and interleaving each plane back must recover the input bytes.
    #[test]
    fn pack_split_rle_roundtrip() {
        let mut random = rand::rngs::StdRng::from_seed(SEED);

        let infos = vec![
            channel_info(CompressorScheme::Rle, 4, 3),
            channel_info(CompressorScheme::Rle, 6, 2),
        ];
        let channel_bytes: Vec<Vec<u8>> = infos
            .iter()
            .map(|info| random_bytes(&mut random, info.width * info.height * info.bytes_per_sample))
            .collect();

        let packed = pack_rle_channels(&infos, &channel_bytes);
        // Mirror `mod.rs::decompress`: split the planar buffer per channel,
        // then interleave each channel's byte planes back to sample order.
        let planar_per_channel =
            split_planar_channels(&infos, CompressorScheme::Rle, &packed).unwrap();
        let decoded: Vec<Vec<u8>> = infos
            .iter()
            .zip(&planar_per_channel)
            .map(|(info, planar)| interleave_byte_planes(planar, info.bytes_per_sample))
            .collect();

        assert_eq!(decoded, channel_bytes);
    }
}
