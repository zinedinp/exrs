//! 32-bit ARM write-row acceleration for `decode_lossy_dct_group`, mirrors
//! `aarch64`

#[cfg(feature = "arm-neon")]
pub mod neon;

#[cfg(feature = "arm-neon")]
pub(super) fn try_write_row_f16(
    row: &[f32],
    to_linear: Option<&[u16; 65536]>,
    out_row: &mut [u8],
) -> bool {
    match pulp::aarch32::Neon::try_new() {
        Some(simd) => neon::write_row_f16(simd, row, to_linear, out_row),
        None => false,
    }
}

#[cfg(not(feature = "arm-neon"))]
pub(super) fn try_write_row_f16(
    _row: &[f32],
    _to_linear: Option<&[u16; 65536]>,
    _out_row: &mut [u8],
) -> bool {
    false
}

#[cfg(feature = "arm-neon")]
pub(super) fn try_write_row_f32(
    row: &[f32],
    to_linear: Option<&[u16; 65536]>,
    out_row: &mut [u8],
) -> bool {
    match pulp::aarch32::Neon::try_new() {
        Some(simd) => neon::write_row_f32(simd, row, to_linear, out_row),
        None => false,
    }
}

#[cfg(not(feature = "arm-neon"))]
pub(super) fn try_write_row_f32(
    _row: &[f32],
    _to_linear: Option<&[u16; 65536]>,
    _out_row: &mut [u8],
) -> bool {
    false
}
