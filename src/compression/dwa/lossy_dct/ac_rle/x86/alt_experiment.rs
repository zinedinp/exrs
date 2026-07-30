//! EXPERIMENTAL alternatives to token-level AVX-512 unRLE (see
//! `v4_experiment.rs`, whose scatter/compress kernels both lost to scalar).
//! Not wired into any real decode path.
//!
//! This file originally explored three angles against `un_rle_ac`'s
//! branch-misprediction-bound cost (see `dwa-un-rle-ac-cache-vs-branch-
//! findings`). Angle 1, branchless advance, won decisively and has since
//! shipped into production `un_rle_ac` itself (`ac_rle::decode_token_branchless`)
//! -- see `dwa-un-rle-ac-branchless-findings`. What is left here builds on
//! top of that now-branchless baseline:
//!
//!  2. **Fused** (`un_rle_ac_fused_dct`): decode straight into DCT-order
//!     `f32`, skipping the intermediate `[u16; 64]` zig-zag buffer and the
//!     separate `from_half_zigzag` pass. Comes in a branching flavor and a
//!     branchless one (`un_rle_ac_fused_dct_branchless`, reusing the shipped
//!     `decode_token_branchless`) to isolate fusion's own contribution from
//!     the branchless win it already has for free via the shared helper.
//!  3. **Pipelined** (`pipeline_overlapped`): decode block i+1's
//!     coefficients while block i's iDCT runs, for ILP across pipeline
//!     stages instead of across tokens. Comes in a plain and a fused flavor
//!     (`pipeline_overlapped` / `pipeline_overlapped_fused`).
//!
//! Bench: `cargo test --release --features avx2-tests -- --ignored --nocapture
//! bench_rle_alt`.
//
// Only reachable from this module's own `#[cfg(test)]` bench/tests right
// now, so a plain `cargo build`/`cargo clippy --lib` (no `--tests`) sees
// every item here as dead code without this.
#![allow(dead_code)]

use half::f16;

use super::super::{decode_token_branchless, un_rle_ac, MAX_TOKENS_PER_BLOCK};
use super::super::super::quantization::{from_half_zigzag_scalar, ZIGZAG_ORDER};
use super::super::super::PackedStream;
use crate::compression::dwa::discrete_cosine_transform;

// ---------------------------------------------------------------------------
// Shared building blocks
// ---------------------------------------------------------------------------

/// Inverse of `ZIGZAG_ORDER`: `INV_ZIGZAG[zig_pos]` is the DCT-order index
/// that reads that zig-zag slot in `from_half_zigzag`.
const INV_ZIGZAG: [usize; 64] = {
	let mut inv = [0usize; 64];
	let mut i = 0;
	while i < 64 {
		inv[ZIGZAG_ORDER[i]] = i;
		i += 1;
	}
	inv
};

// ---------------------------------------------------------------------------
// Angle 2: fused DCT-order write, with or without the shipped branchless
// advance (`decode_token_branchless`, `rle/mod.rs`)
// ---------------------------------------------------------------------------

/// Un-RLE AC tokens straight into DCT-order `f32` coefficients (plus DC),
/// skipping the intermediate `[u16; 64]` zig-zag buffer and separate
/// `from_half_zigzag` pass. Still branches on run-vs-literal (see
/// `un_rle_ac_fused_dct_branchless` for both angles combined).
///
/// Returns the zig-zag index of the last non-zero AC (same as `un_rle_ac`),
/// so callers can still take the DC-only iDCT fast path when it is 0. `dst`
/// is fully overwritten (zeros for missing AC); matches `un_rle_ac` +
/// `from_half_zigzag` bit-for-bit when both use scalar half->f32.
#[inline(always)]
pub(crate) fn un_rle_ac_fused_dct(
	ac: &mut PackedStream<'_>,
	dc: u16,
	dst: &mut [f32; 64],
) -> usize {
	*dst = [0.0f32; 64];
	// ZIGZAG_ORDER[0] == 0, so DC lives at DCT index 0.
	dst[0] = f16::from_bits(dc).to_f32();

	let mut last_non_zero = 0usize;
	let mut position = 1usize;

	if ac.remaining() >= MAX_TOKENS_PER_BLOCK {
		let tokens = ac.peek_slice(MAX_TOKENS_PER_BLOCK);
		let mut consumed = 0usize;

		while position < 64 {
			let value = tokens[consumed];
			consumed += 1;

			if (value & 0xff00) == 0xff00 {
				let count = (value & 0xff) as usize;
				position += if count == 0 { 64 } else { count };
			} else {
				last_non_zero = position;
				dst[INV_ZIGZAG[position]] = f16::from_bits(value).to_f32();
				position += 1;
			}
		}

		ac.advance(consumed);
	} else {
		while position < 64 {
			let value = ac.next().expect("un_rle_ac_fused_dct: truncated AC");

			if (value & 0xff00) == 0xff00 {
				let count = (value & 0xff) as usize;
				position += if count == 0 { 64 } else { count };
			} else {
				last_non_zero = position;
				dst[INV_ZIGZAG[position]] = f16::from_bits(value).to_f32();
				position += 1;
			}
		}
	}

	last_non_zero
}

/// Fused DCT-order write (angle 2) on top of the shipped branchless advance
/// (`decode_token_branchless`) -- both angles combined.
#[inline(always)]
pub(crate) fn un_rle_ac_fused_dct_branchless(
	ac: &mut PackedStream<'_>,
	dc: u16,
	dst: &mut [f32; 64],
) -> usize {
	*dst = [0.0f32; 64];
	dst[0] = f16::from_bits(dc).to_f32();

	if ac.remaining() < MAX_TOKENS_PER_BLOCK {
		return un_rle_ac_fused_dct(ac, dc, dst);
	}

	let mut last_non_zero = 0usize;
	let mut position = 1usize;
	let tokens = ac.peek_slice(MAX_TOKENS_PER_BLOCK);
	let mut consumed = 0usize;

	while position < 64 {
		let value = tokens[consumed];
		consumed += 1;

		let write_pos = position;
		let is_run_mask = decode_token_branchless(value, &mut position, &mut last_non_zero);

		// Always convert, then mask the bit pattern to 0.0 on a run token --
		// avoids a second branch (see `decode_token_branchless`'s doc
		// comment in `rle/mod.rs`).
		let literal_bits = !(is_run_mask as u32);
		let f_bits = f16::from_bits(value).to_f32().to_bits() & literal_bits;
		dst[INV_ZIGZAG[write_pos]] = f32::from_bits(f_bits);
	}

	ac.advance(consumed);
	last_non_zero
}

// ---------------------------------------------------------------------------
// Angle 3: software pipeline (overlap unRLE of block i+1 with iDCT of i)
// ---------------------------------------------------------------------------

/// Baseline sequential stage order: for each block, unRLE -> zigzag -> iDCT.
#[inline(never)]
pub(crate) fn pipeline_sequential(
	tokens: &[u16],
	dcs: &[u16],
	block_count: usize,
	out_checksum: &mut u64,
) {
	let mut ac = PackedStream::new(tokens);
	let mut sum = 0u64;
	for b in 0..block_count {
		let mut zig = [0u16; 64];
		zig[0] = dcs[b];
		let lnz = un_rle_ac(&mut ac, &mut zig).unwrap();
		let mut dct = [0.0f32; 64];
		if lnz == 0 {
			dct[0] = f16::from_bits(zig[0]).to_f32();
			discrete_cosine_transform::dct_inverse_8x8_dc_only(&mut dct);
		} else {
			from_half_zigzag_scalar(&zig, &mut dct);
			discrete_cosine_transform::dct_inverse_8x8_autovectorized(&mut dct);
		}
		sum = sum.wrapping_add(dct[0].to_bits() as u64);
		sum = sum.wrapping_add(dct[7].to_bits() as u64);
		sum = sum.wrapping_add(dct[63].to_bits() as u64);
	}
	*out_checksum = sum;
}

/// Software-pipelined: keep block i+1's coefficients decoding while iDCT of
/// block i runs. Two coefficient buffers; AC stream still serial.
#[inline(never)]
pub(crate) fn pipeline_overlapped(
	tokens: &[u16],
	dcs: &[u16],
	block_count: usize,
	out_checksum: &mut u64,
) {
	if block_count == 0 {
		*out_checksum = 0;
		return;
	}

	let mut ac = PackedStream::new(tokens);
	let mut buf = [[0.0f32; 64]; 2];
	let mut lnz = [0usize; 2];

	// Decode block 0.
	{
		let mut zig = [0u16; 64];
		zig[0] = dcs[0];
		lnz[0] = un_rle_ac(&mut ac, &mut zig).unwrap();
		if lnz[0] == 0 {
			buf[0][0] = f16::from_bits(zig[0]).to_f32();
		} else {
			from_half_zigzag_scalar(&zig, &mut buf[0]);
		}
	}

	let mut sum = 0u64;
	for b in 0..block_count {
		let cur = b & 1;
		let next = (b + 1) & 1;

		// Overlap: pull next block's coeffs while finishing current iDCT.
		if b + 1 < block_count {
			let mut zig = [0u16; 64];
			zig[0] = dcs[b + 1];
			lnz[next] = un_rle_ac(&mut ac, &mut zig).unwrap();
			if lnz[next] == 0 {
				buf[next] = [0.0f32; 64];
				buf[next][0] = f16::from_bits(zig[0]).to_f32();
			} else {
				from_half_zigzag_scalar(&zig, &mut buf[next]);
			}
		}

		if lnz[cur] == 0 {
			discrete_cosine_transform::dct_inverse_8x8_dc_only(&mut buf[cur]);
		} else {
			discrete_cosine_transform::dct_inverse_8x8_autovectorized(&mut buf[cur]);
		}
		sum = sum.wrapping_add(buf[cur][0].to_bits() as u64);
		sum = sum.wrapping_add(buf[cur][7].to_bits() as u64);
		sum = sum.wrapping_add(buf[cur][63].to_bits() as u64);
	}
	*out_checksum = sum;
}

/// Same pipeline shape as `pipeline_overlapped`, but coefficient decode uses
/// the fused unRLE->f32 path (no zig-zag staging buffer).
#[inline(never)]
pub(crate) fn pipeline_overlapped_fused(
	tokens: &[u16],
	dcs: &[u16],
	block_count: usize,
	out_checksum: &mut u64,
) {
	if block_count == 0 {
		*out_checksum = 0;
		return;
	}

	let mut ac = PackedStream::new(tokens);
	let mut buf = [[0.0f32; 64]; 2];
	let mut lnz = [0usize; 2];

	lnz[0] = un_rle_ac_fused_dct(&mut ac, dcs[0], &mut buf[0]);

	let mut sum = 0u64;
	for b in 0..block_count {
		let cur = b & 1;
		let next = (b + 1) & 1;

		if b + 1 < block_count {
			lnz[next] = un_rle_ac_fused_dct(&mut ac, dcs[b + 1], &mut buf[next]);
		}

		if lnz[cur] == 0 {
			discrete_cosine_transform::dct_inverse_8x8_dc_only(&mut buf[cur]);
		} else {
			discrete_cosine_transform::dct_inverse_8x8_autovectorized(&mut buf[cur]);
		}
		sum = sum.wrapping_add(buf[cur][0].to_bits() as u64);
		sum = sum.wrapping_add(buf[cur][7].to_bits() as u64);
		sum = sum.wrapping_add(buf[cur][63].to_bits() as u64);
	}
	*out_checksum = sum;
}

// ---------------------------------------------------------------------------
// Tests + isolated A/B benchmark
// ---------------------------------------------------------------------------

#[cfg(test)]
mod test {
	use half::f16;
	use rand::RngExt;

	use super::from_half_zigzag_scalar;
	use super::{
		pipeline_overlapped, pipeline_overlapped_fused, pipeline_sequential,
		un_rle_ac_fused_dct, un_rle_ac_fused_dct_branchless, INV_ZIGZAG,
	};
	use crate::compression::dwa::lossy_dct::{
		quantization::ZIGZAG_ORDER,
		ac_rle::{rle_ac, un_rle_ac},
		PackedStream,
	};

	fn random_block(rng: &mut impl rand::Rng, nonzero_probability: f64) -> [u16; 64] {
		let mut block = [0u16; 64];
		for slot in block.iter_mut().skip(1) {
			if rng.random_bool(nonzero_probability) {
				let mut v: u16 = rng.random();
				while (v & 0xff00) == 0xff00 {
					v = rng.random();
				}
				*slot = v.max(1);
			}
		}
		block
	}

	fn encode_with_pad(block: &[u16; 64], trailing: usize) -> Vec<u16> {
		let mut ac_tokens = Vec::new();
		rle_ac(block, &mut ac_tokens);
		for _ in 0..trailing {
			ac_tokens.push(0xff00);
		}
		ac_tokens
	}

	#[test]
	fn inv_zigzag_is_inverse() {
		for i in 0..64 {
			assert_eq!(INV_ZIGZAG[ZIGZAG_ORDER[i]], i);
			assert_eq!(ZIGZAG_ORDER[INV_ZIGZAG[i]], i);
		}
	}

	#[test]
	fn fused_matches_unrle_plus_zigzag() {
		let mut rng = rand::rng();
		for &p in &[0.0, 0.05, 0.3, 0.7, 1.0] {
			for _ in 0..500 {
				let want = random_block(&mut rng, p);
				let dc: u16 = rng.random();
				let tokens = encode_with_pad(&want, 64);

				let mut zig = [0u16; 64];
				zig[0] = dc;
				let mut s_stream = PackedStream::new(&tokens);
				let s_lnz = un_rle_ac(&mut s_stream, &mut zig).unwrap();
				let mut s_dct = [0.0f32; 64];
				from_half_zigzag_scalar(&zig, &mut s_dct);

				let mut f_dct = [0.0f32; 64];
				let mut f_stream = PackedStream::new(&tokens);
				let f_lnz = un_rle_ac_fused_dct(&mut f_stream, dc, &mut f_dct);

				assert_eq!(f_lnz, s_lnz, "lnz mismatch p={p}");
				assert_eq!(f_stream.remaining(), s_stream.remaining());
				for i in 0..64 {
					assert_eq!(
						f_dct[i].to_bits(),
						s_dct[i].to_bits(),
						"dct[{i}] mismatch p={p} (f={} s={})",
						f_dct[i],
						s_dct[i]
					);
				}

				let mut fb_dct = [0.0f32; 64];
				let mut fb_stream = PackedStream::new(&tokens);
				let fb_lnz = un_rle_ac_fused_dct_branchless(&mut fb_stream, dc, &mut fb_dct);
				assert_eq!(fb_lnz, s_lnz);
				for i in 0..64 {
					assert_eq!(fb_dct[i].to_bits(), s_dct[i].to_bits());
				}
			}
		}
	}

	#[test]
	fn pipelines_agree_checksum() {
		let mut rng = rand::rng();
		const N: usize = 256;
		let mut tokens = Vec::new();
		let mut dcs = Vec::with_capacity(N);
		for _ in 0..N {
			let block = random_block(&mut rng, 0.3);
			dcs.push(rng.random::<u16>());
			rle_ac(&block, &mut tokens);
		}
		for _ in 0..64 {
			tokens.push(0xff00);
		}

		let mut c0 = 0u64;
		let mut c1 = 0u64;
		let mut c2 = 0u64;
		pipeline_sequential(&tokens, &dcs, N, &mut c0);
		pipeline_overlapped(&tokens, &dcs, N, &mut c1);
		pipeline_overlapped_fused(&tokens, &dcs, N, &mut c2);
		assert_eq!(c0, c1, "overlapped checksum");
		assert_eq!(c0, c2, "fused overlapped checksum");
	}

	/// Isolated A/B of the three attack angles vs baseline scalar unRLE (+
	/// zigzag / pipeline). Run with:
	/// `cargo test --release -p exr --features avx2-tests -- --ignored --nocapture bench_rle_alt`
	#[test]
	#[ignore]
	fn bench_rle_alt() {
		use crate::compression::dwa::lossy_dct::quantization::from_half_zigzag;

		let mut rng = rand::rng();
		const BLOCK_COUNT: usize = 200_000;
		const ROUNDS: usize = 6;

		// --- token-only corpus (sparse-ish, realistic DWA density) ---
		let mut tokens = Vec::new();
		let mut dcs = Vec::with_capacity(BLOCK_COUNT);
		for _ in 0..BLOCK_COUNT {
			let block = random_block(&mut rng, 0.3);
			dcs.push(rng.random::<u16>());
			rle_ac(&block, &mut tokens);
		}
		for _ in 0..64 {
			tokens.push(0xff00);
		}

		let mut base_scalar_zz_ms = Vec::new();
		let mut base_prod_zz_ms = Vec::new();
		let mut fused_ms = Vec::new();
		let mut fused_bl_ms = Vec::new();
		let mut seq_ms = Vec::new();
		let mut ov_ms = Vec::new();
		let mut ov_fused_ms = Vec::new();

		for round in 0..ROUNDS {
			// Baseline is the shipped `un_rle_ac` (already branchless as of
			// `decode_token_branchless`) -- what's timed here is what angle 2
			// (fusion) has left to win, not angle 1 (already banked).

			// a) shipped unRLE + scalar zigzag (fair vs fused's scalar half->f32)
			let start = std::time::Instant::now();
			let mut stream = PackedStream::new(&tokens);
			let mut dct = [0.0f32; 64];
			for b in 0..BLOCK_COUNT {
				let mut zig = [0u16; 64];
				zig[0] = dcs[b];
				un_rle_ac(&mut stream, &mut zig).unwrap();
				from_half_zigzag_scalar(&zig, &mut dct);
			}
			let t_base_scalar_zz = start.elapsed();
			std::hint::black_box(dct);

			// b) shipped unRLE + production zigzag (F16C shuffle when
			// available) -- this is what today's real decode path runs.
			let start = std::time::Instant::now();
			let mut stream = PackedStream::new(&tokens);
			let mut dct = [0.0f32; 64];
			for b in 0..BLOCK_COUNT {
				let mut zig = [0u16; 64];
				zig[0] = dcs[b];
				un_rle_ac(&mut stream, &mut zig).unwrap();
				from_half_zigzag(&zig, &mut dct);
			}
			let t_base_prod_zz = start.elapsed();
			std::hint::black_box(dct);

			let start = std::time::Instant::now();
			let mut stream = PackedStream::new(&tokens);
			let mut dct = [0.0f32; 64];
			for b in 0..BLOCK_COUNT {
				un_rle_ac_fused_dct(&mut stream, dcs[b], &mut dct);
			}
			let t_fused = start.elapsed();
			std::hint::black_box(dct);

			let start = std::time::Instant::now();
			let mut stream = PackedStream::new(&tokens);
			let mut dct = [0.0f32; 64];
			for b in 0..BLOCK_COUNT {
				un_rle_ac_fused_dct_branchless(&mut stream, dcs[b], &mut dct);
			}
			let t_fused_bl = start.elapsed();
			std::hint::black_box(dct);

			// 3) software pipeline (includes iDCT -- end-to-end-ish)
			let mut c = 0u64;
			let start = std::time::Instant::now();
			pipeline_sequential(&tokens, &dcs, BLOCK_COUNT, &mut c);
			let t_seq = start.elapsed();
			std::hint::black_box(c);

			let start = std::time::Instant::now();
			pipeline_overlapped(&tokens, &dcs, BLOCK_COUNT, &mut c);
			let t_ov = start.elapsed();
			std::hint::black_box(c);

			let start = std::time::Instant::now();
			pipeline_overlapped_fused(&tokens, &dcs, BLOCK_COUNT, &mut c);
			let t_ov_f = start.elapsed();
			std::hint::black_box(c);

			eprintln!(
				"round {round}:\n  +zz scalar   base   {:>8.3} ms | fused      {:>8.3} ms ({:+.2}%) | fused+bl {:>8.3} ms ({:+.2}%)\n  +zz prod     base   {:>8.3} ms | fused vs prod {:+.2}% | fused+bl vs prod {:+.2}%\n  +iDCT pipe   seq    {:>8.3} ms | overlap    {:>8.3} ms ({:+.2}%) | ov+fused {:>8.3} ms ({:+.2}%)",
				t_base_scalar_zz.as_secs_f64() * 1e3,
				t_fused.as_secs_f64() * 1e3,
				(t_fused.as_secs_f64() / t_base_scalar_zz.as_secs_f64() - 1.0) * 100.0,
				t_fused_bl.as_secs_f64() * 1e3,
				(t_fused_bl.as_secs_f64() / t_base_scalar_zz.as_secs_f64() - 1.0) * 100.0,
				t_base_prod_zz.as_secs_f64() * 1e3,
				(t_fused.as_secs_f64() / t_base_prod_zz.as_secs_f64() - 1.0) * 100.0,
				(t_fused_bl.as_secs_f64() / t_base_prod_zz.as_secs_f64() - 1.0) * 100.0,
				t_seq.as_secs_f64() * 1e3,
				t_ov.as_secs_f64() * 1e3,
				(t_ov.as_secs_f64() / t_seq.as_secs_f64() - 1.0) * 100.0,
				t_ov_f.as_secs_f64() * 1e3,
				(t_ov_f.as_secs_f64() / t_seq.as_secs_f64() - 1.0) * 100.0,
			);

			base_scalar_zz_ms.push(t_base_scalar_zz.as_secs_f64());
			base_prod_zz_ms.push(t_base_prod_zz.as_secs_f64());
			fused_ms.push(t_fused.as_secs_f64());
			fused_bl_ms.push(t_fused_bl.as_secs_f64());
			seq_ms.push(t_seq.as_secs_f64());
			ov_ms.push(t_ov.as_secs_f64());
			ov_fused_ms.push(t_ov_f.as_secs_f64());
		}

		let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
		let bsz = avg(&base_scalar_zz_ms);
		let bpz = avg(&base_prod_zz_ms);
		let f = avg(&fused_ms);
		let fbl = avg(&fused_bl_ms);
		let seq = avg(&seq_ms);
		let ov = avg(&ov_ms);
		let ovf = avg(&ov_fused_ms);

		eprintln!(
			"avg ({BLOCK_COUNT} blocks x {ROUNDS} rounds):\n  +zz scalar base   {:.3} ms | fused      {:.3} ms ({:+.2}%) | fused+bl {:.3} ms ({:+.2}%)\n  +zz prod   base   {:.3} ms | fused vs prod {:+.2}% | fused+bl vs prod {:+.2}%\n  +iDCT pipe seq    {:.3} ms | overlap    {:.3} ms ({:+.2}%) | ov+fused {:.3} ms ({:+.2}%)",
			bsz * 1e3,
			f * 1e3,
			(f / bsz - 1.0) * 100.0,
			fbl * 1e3,
			(fbl / bsz - 1.0) * 100.0,
			bpz * 1e3,
			(f / bpz - 1.0) * 100.0,
			(fbl / bpz - 1.0) * 100.0,
			seq * 1e3,
			ov * 1e3,
			(ov / seq - 1.0) * 100.0,
			ovf * 1e3,
			(ovf / seq - 1.0) * 100.0,
		);

		let _ = f16::from_bits;
	}
}
