// The four on-disk DWA payload sections (UNKNOWN, AC, DC, RLE): splitting the
// chunk body into them, the zlib inflate wrapper they share, and the
// differencing transform applied to the DC stream.

use super::chunk_header::{AcCompression, DwaHeader};
use crate::error::{Error, Result};

/// Split the data after header + rules into the four sections, in on-disk
/// order. Errors on truncation like the C parser.
pub(super) fn split_sections<'d>(data: &'d [u8], header: &DwaHeader) -> Result<[&'d [u8]; 4]> {
    let mut rest = data;
    let mut take = |length: usize| -> Result<&'d [u8]> {
        if length > rest.len() {
            return Err(Error::invalid("truncated DWA section"));
        }
        let (section, remaining) = rest.split_at(length);
        rest = remaining;
        Ok(section)
    };

    Ok([
        take(header.unknown_compressed_size)?,
        take(header.ac_compressed_size)?,
        take(header.dc_compressed_size)?,
        take(header.rle_compressed_size)?,
    ])
}

fn inflate(compressed: &[u8], expected_size: usize) -> Result<Vec<u8>> {
    let options = zune_inflate::DeflateOptions::default()
        .set_limit(expected_size)
        .set_size_hint(expected_size);

    let inflated = zune_inflate::DeflateDecoder::new_with_options(compressed, options)
        .decode_zlib()
        .map_err(|_| Error::invalid("DWA zlib data malformed"))?;

    if inflated.len() != expected_size {
        return Err(Error::invalid("DWA zlib data size mismatch"));
    }
    Ok(inflated)
}

/// UNKNOWN section: raw (non-DCT-compressible) channel data,
/// zlib-compressed, planar in channel order.
pub(super) fn decode_unknown_section(section: &[u8], header: &DwaHeader) -> Result<Vec<u8>> {
    if header.unknown_uncompressed_size == 0 {
        return Ok(vec![]);
    }
    inflate(section, header.unknown_uncompressed_size)
}

/// AC section: RLE DCT coefficients as u16, entropy coded with either the
/// PIZ static Huffman coder or zlib.
/// Expands into `out`, reusing whatever capacity it already has (see
/// `huffman::decompress_into`), and returns how many entries were written.
/// always `header.ac_count`. `words` is scratch space forwarded to the
/// Huffman decoder, reused the same way.
pub(super) fn decode_ac_section_into(
    section: &[u8],
    header: &DwaHeader,
    out: &mut Vec<u16>,
    words: &mut Vec<u64>,
) -> Result<usize> {
    if header.ac_count == 0 {
        return Ok(0);
    }

    match header.ac_compression {
        AcCompression::StaticHuffman => {
            crate::compression::huffman::decompress_into(section, header.ac_count, out, words)?;
        }
        AcCompression::Deflate => {
            let bytes = inflate(section, header.ac_count * 2)?;
            if out.len() < header.ac_count {
                out.resize(header.ac_count, 0);
            }
            for (slot, pair) in out[..header.ac_count].iter_mut().zip(bytes.chunks_exact(2)) {
                *slot = u16::from_le_bytes([pair[0], pair[1]]);
            }
        }
    }

    Ok(header.ac_count)
}

/// DC section: one u16 (half bits) per 8x8 block, zlib-compressed after
/// the "zip reconstruct" transform (differencing + byte deinterleave).
pub(super) fn decode_dc_section(section: &[u8], header: &DwaHeader) -> Result<Vec<u16>> {
    if header.dc_count == 0 {
        return Ok(vec![]);
    }

    let bytes = inflate(section, header.dc_count * 2)?;
    let bytes = undo_zip_reconstruct(&bytes);
    Ok(bytes.chunks_exact(2).map(|pair| u16::from_le_bytes([pair[0], pair[1]])).collect())
}

/// RLE section: zlib, then classic byte-oriented RLE. Result is planar per
/// channel, each channel further split into byte planes.
/// Expands into `buffer`, reusing whatever capacity it already has, and
/// returns how many bytes were written -- OpenEXR keeps one `_planarUncBuffer`
/// alive across every chunk it decodes, and allocating a fresh output buffer
/// per chunk instead costs more (in page faults on the never-touched pages)
/// than the entire RLE expansion does.
pub(super) fn decode_rle_section_into(
    section: &[u8],
    header: &DwaHeader,
    max_raw_size: usize,
    buffer: &mut Vec<u8>,
) -> Result<usize> {
    if header.rle_raw_size == 0 {
        return Ok(0);
    }
    // The buffer is sized from the header up front, so a corrupt chunk must
    // not be able to ask for an arbitrarily large allocation: the RLE
    // channels can never read more than `max_raw_size` bytes back out.
    if header.rle_raw_size > max_raw_size {
        return Err(Error::invalid("DWA RLE data size"));
    }
    #[cfg(feature = "dwa-profile")]
    let t = super::profile::start();
    let inflated = inflate(section, header.rle_uncompressed_size)?;
    #[cfg(feature = "dwa-profile")]
    t.stop(&super::profile::RLE_INFLATE_NS);

    // Only ever grown, never re-zeroed per chunk: the caller is handed
    // `[..written]`, and the expansion below writes every one of those bytes,
    // so bytes left over from a previous chunk are never observable.
    #[cfg(feature = "dwa-profile")]
    let t = super::profile::start();
    if buffer.len() < header.rle_raw_size {
        buffer.resize(header.rle_raw_size, 0);
    }
    #[cfg(feature = "dwa-profile")]
    t.stop(&super::profile::RLE_ALLOC_NS);

    #[cfg(feature = "dwa-profile")]
    let t = super::profile::start();
    let written = crate::compression::rle::unpack_rle_tokens_into(
        &inflated,
        &mut buffer[..header.rle_raw_size],
    )?;
    #[cfg(feature = "dwa-profile")]
    t.stop(&super::profile::RLE_UNPACK_NS);

    Ok(written)
}

/// Ports "internal_zip_reconstruct_bytes": undo differencing, then
/// interleave the two buffer halves.
fn undo_zip_reconstruct(source: &[u8]) -> Vec<u8> {
    if source.len() < 2 {
        return source.to_vec();
    }

    let mut deltas = source.to_vec();
    for index in 1..deltas.len() {
        deltas[index] = ((deltas[index - 1] as i32) + (deltas[index] as i32) - 128) as u8;
    }

    let (first_half, second_half) = deltas.split_at((deltas.len() + 1) / 2);
    let mut out = vec![0u8; deltas.len()];
    for (index, slot) in out.iter_mut().enumerate() {
        *slot = if index % 2 == 0 {
            first_half[index / 2]
        } else {
            second_half[index / 2]
        };
    }
    out
}

/// Encoder-side companion applied to the DC byte stream before zlib:
/// byte-fragment separation followed by successive differencing.
pub(super) fn zip_deconstruct_bytes(bytes: &mut [u8]) {
    crate::compression::optimize_bytes::separate_bytes_fragments(bytes);
    crate::compression::optimize_bytes::samples_to_differences(bytes);
}

#[cfg(test)]
mod test {
    use rand::{RngExt, SeedableRng};

    use super::*;

    const SEED: [u8; 32] = [
        19, 240, 8, 91, 3, 128, 9, 44, 201, 17, 88, 6, 255, 61, 30, 11, 2, 121, 99, 1, 250, 77, 33,
        7, 42, 13, 200, 176, 22, 5, 66, 100,
    ];

    /// The encoder-side DC transform (`zip_deconstruct_bytes`) and the
    /// decoder-side reconstruction (`undo_zip_reconstruct`) must be inverses,
    /// including for odd lengths and the < 2 byte short-circuit.
    #[test]
    fn zip_deconstruct_reconstruct_roundtrip() {
        let mut random = rand::rngs::StdRng::from_seed(SEED);

        for length in [0usize, 1, 2, 3, 4, 5, 17, 64, 129] {
            let original: Vec<u8> = (0..length).map(|_| random.random()).collect();

            let mut deconstructed = original.clone();
            zip_deconstruct_bytes(&mut deconstructed);
            let reconstructed = undo_zip_reconstruct(&deconstructed);

            assert_eq!(reconstructed, original, "failed at length {length}");
        }
    }
}
