//! aarch64 write-row acceleration for `decode_lossy_dct_group`.

pub mod neon;

pub(super) fn try_write_row_f16(
    row: &[f32],
    to_linear: Option<&[u16; 65536]>,
    out_row: &mut [u8],
) -> bool {
    match pulp::aarch64::Neon::try_new() {
        Some(simd) => neon::write_row_f16(simd, row, to_linear, out_row),
        None => false,
    }
}

pub(super) fn try_write_row_f32(
    row: &[f32],
    to_linear: Option<&[u16; 65536]>,
    out_row: &mut [u8],
) -> bool {
    match pulp::aarch64::Neon::try_new() {
        Some(simd) => neon::write_row_f32(simd, row, to_linear, out_row),
        None => false,
    }
}
