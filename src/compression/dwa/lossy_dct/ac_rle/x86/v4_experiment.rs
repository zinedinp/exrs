//! EXPERIMENTAL, Stage 1 only (not wired into any real decode path): a
//! vectorized alternative to `ac_rle::un_rle_ac`'s AC-RLE decode loop.
//
// `pulp::v4_fn!` may leave the free helpers unused under some feature matrices
// (bodies live inside the macro expansion); this module is experiment-only.
#![allow(dead_code)]

// EXPERIMENTAL, Stage 1 only (not wired into any real decode path): a
// vectorized alternative to `ac_rle::un_rle_ac`'s AC-RLE decode loop.


use pulp::x86::{V4, V4Vbmi2};
use pulp::{b16, b32, cast, u32x16, Simd};

use super::super::MAX_TOKENS_PER_BLOCK;
use super::super::super::PackedStream;

const LANES: usize = 16;

#[inline(always)]
fn u32x16_from_array(a: [u32; LANES]) -> u32x16 {
	u32x16 {
		0: a[0], 1: a[1], 2: a[2], 3: a[3], 4: a[4], 5: a[5], 6: a[6], 7: a[7],
		8: a[8], 9: a[9], 10: a[10], 11: a[11], 12: a[12], 13: a[13], 14: a[14], 15: a[15],
	}
}

#[inline(always)]
fn u32x16_to_array(v: u32x16) -> [u32; LANES] {
	[
		v.0, v.1, v.2, v.3, v.4, v.5, v.6, v.7, v.8, v.9, v.10, v.11, v.12, v.13, v.14, v.15,
	]
}

// Vectorized alternative to `ac_rle::un_rle_ac`'s fast path (used when
// `ac.remaining() >= MAX_TOKENS_PER_BLOCK`). Same contract: `block` must
// already be zeroed, returns the zig-zag index of the last non-zero value
// written (0 if none).
pulp::v4_fn! {
	pub(crate) fn un_rle_ac_v4(v4: V4, ac: &mut PackedStream<'_>, block: &mut [u16; 64]) -> usize {
		let fast = ac.peek_slice(MAX_TOKENS_PER_BLOCK);

		let splat_0 = v4.splat_u32x16(0);
		let splat_1 = v4.splat_u32x16(1);
		let splat_64 = v4.splat_u32x16(64);
		let splat_0xff = v4.splat_u32x16(0xff);
		let splat_0xff00 = v4.splat_u32x16(0xff00);

		// One `u32` slot per zig-zag position, scattered into directly so a
		// literal token's position never has to survive a round trip through
		// scalar code. Narrowed back into the real `[u16; 64]` block (which
		// AVX-512F has no native 16-bit scatter to write directly -- see
		// `V4::scatter_u32x16`'s doc comment) once every chunk is done.
		let mut block_u32 = [0u32; 64];

		let mut chunk_base = 1u32;
		let mut last_non_zero = 0u32;
		let mut total_consumed = 0usize;

		// 63 tokens split into ceil(63/16) = 4 chunks of 16 (last one has
		// only 15 real tokens; lane 15 is forced out-of-bounds below).
		let mut chunk_start = 0usize;
		while chunk_base < 64 {
			assert!(
				chunk_start < MAX_TOKENS_PER_BLOCK,
				"un_rle_ac_v4: read past its bounded fast slice"
			);
			let real_lanes = (MAX_TOKENS_PER_BLOCK - chunk_start).min(LANES);

			let mut token_arr = [0u32; LANES];
			for lane in 0..real_lanes {
				token_arr[lane] = fast[chunk_start + lane] as u32;
			}
			let tokens = u32x16_from_array(token_arr);

			let is_run = v4.cmp_eq_u32x16(v4.and_u32x16(tokens, splat_0xff00), splat_0xff00);
			let raw_count = v4.and_u32x16(tokens, splat_0xff);
			let count_is_zero = v4.cmp_eq_u32x16(raw_count, splat_0);
			let run_advance = v4.select_u32x16(count_is_zero, splat_64, raw_count);
			let advance = v4.select_u32x16(is_run, run_advance, splat_1);

			// Hillis-Steele inclusive prefix sum over 16 lanes (4 steps):
			// `rotate_right_u32s(x, k)` moves lane `i`'s value to lane
			// `i + k` (mod 16); the low `k` lanes of that rotation wrapped
			// around from the top, so they're replaced with 0 via `mask_k`
			// before adding -- turning a rotate into a zero-filled shift.
			let step = |x: u32x16, k: u32, mask_bits: u16| {
				let shifted = v4.select_u32x16(
					b16(mask_bits),
					v4.rotate_right_u32s(x, k as usize),
					splat_0,
				);
				v4.wrapping_add_u32x16(x, shifted)
			};
			let x1 = step(advance, 1, 0xFFFE);
			let x2 = step(x1, 2, 0xFFFC);
			let x4 = step(x2, 4, 0xFFF0);
			let inclusive = step(x4, 8, 0xFF00);

			let exclusive = v4.wrapping_sub_u32x16(inclusive, advance);
			let prefix = v4.wrapping_add_u32x16(exclusive, v4.splat_u32x16(chunk_base));

			let in_bounds_bits: u16 = if real_lanes == LANES {
				0xFFFF
			} else {
				(1u16 << real_lanes) - 1
			};
			let content_valid = v4.cmp_lt_u32x16(prefix, v4.splat_u32x16(64));
			let valid_bits = in_bounds_bits & content_valid.0;
			let literal_bits = valid_bits & !is_run.0;

			if literal_bits != 0 {
				v4.scatter_u32x16(&mut block_u32, b16(literal_bits), prefix, tokens);
			}

			let consumed_count = valid_bits.trailing_ones() as usize;

			if literal_bits != 0 {
				let hi = 15 - literal_bits.leading_zeros();
				last_non_zero = u32x16_to_array(prefix)[hi as usize];
			}

			total_consumed += consumed_count;

			if consumed_count == 0 {
				// `chunk_base < 64` (loop guard) guarantees lane 0's prefix
				// is always < 64, so it's always valid -- this can't happen
				// on well-formed data.
				unreachable!("un_rle_ac_v4: lane 0 must always be valid while chunk_base < 64");
			}

			chunk_base += u32x16_to_array(inclusive)[consumed_count - 1];

			if consumed_count < real_lanes {
				// Block finished mid-chunk (`chunk_base` is now >= 64).
				break;
			}
			chunk_start += LANES;
		}

		ac.advance(total_consumed);

		for (i, &v) in block_u32.iter().enumerate().skip(1) {
			block[i] = v as u16;
		}

		last_non_zero as usize
	}
}

// Same classify + prefix-sum front end as `un_rle_ac_v4`, but instead of a
// masked `scatter_u32x16` into a `u32` shadow buffer (measured ~2x slower
// than the scalar loop -> AVX-512 scatter is a microcoded, per-active-lane
// instruction, and an average sparse block only has a handful of literals
// to place), this compacts each chunk's literal positions and values to a
// contiguous prefix with VBMI2 `mask_compress_u16x32`,
// then writes them with a plain unconditional loop over
// exactly `popcount(literal_bits)` elements (the
// loop's trip count is a simple counter, not content-dependent
// classification) and no shadow buffer, straight into `block`.
pulp::v4_vbmi2_fn! {
	pub(crate) fn un_rle_ac_v4_compress(v4: V4Vbmi2, ac: &mut PackedStream<'_>, block: &mut [u16; 64]) -> usize {
		let fast = ac.peek_slice(MAX_TOKENS_PER_BLOCK);

		let splat_0 = v4.splat_u32x16(0);
		let splat_1 = v4.splat_u32x16(1);
		let splat_64 = v4.splat_u32x16(64);
		let splat_0xff = v4.splat_u32x16(0xff);
		let splat_0xff00 = v4.splat_u32x16(0xff00);

		let mut chunk_base = 1u32;
		let mut last_non_zero = 0u32;
		let mut total_consumed = 0usize;

		let mut chunk_start = 0usize;
		while chunk_base < 64 {
			assert!(
				chunk_start < MAX_TOKENS_PER_BLOCK,
				"un_rle_ac_v4_compress: read past its bounded fast slice"
			);
			let real_lanes = (MAX_TOKENS_PER_BLOCK - chunk_start).min(LANES);

			let mut token_arr = [0u32; LANES];
			for lane in 0..real_lanes {
				token_arr[lane] = fast[chunk_start + lane] as u32;
			}
			let tokens = u32x16_from_array(token_arr);

			let is_run = v4.cmp_eq_u32x16(v4.and_u32x16(tokens, splat_0xff00), splat_0xff00);
			let raw_count = v4.and_u32x16(tokens, splat_0xff);
			let count_is_zero = v4.cmp_eq_u32x16(raw_count, splat_0);
			let run_advance = v4.select_u32x16(count_is_zero, splat_64, raw_count);
			let advance = v4.select_u32x16(is_run, run_advance, splat_1);

			let step = |x: u32x16, k: u32, mask_bits: u16| {
				let shifted = v4.select_u32x16(
					b16(mask_bits),
					v4.rotate_right_u32s(x, k as usize),
					splat_0,
				);
				v4.wrapping_add_u32x16(x, shifted)
			};
			let x1 = step(advance, 1, 0xFFFE);
			let x2 = step(x1, 2, 0xFFFC);
			let x4 = step(x2, 4, 0xFFF0);
			let inclusive = step(x4, 8, 0xFF00);

			let exclusive = v4.wrapping_sub_u32x16(inclusive, advance);
			let prefix = v4.wrapping_add_u32x16(exclusive, v4.splat_u32x16(chunk_base));

			let in_bounds_bits: u16 = if real_lanes == LANES {
				0xFFFF
			} else {
				(1u16 << real_lanes) - 1
			};
			let content_valid = v4.cmp_lt_u32x16(prefix, v4.splat_u32x16(64));
			let valid_bits = in_bounds_bits & content_valid.0;
			let literal_bits = valid_bits & !is_run.0;

			if literal_bits != 0 {
				// Pad lanes 16..32 with zeros and a clear mask -- compress
				// only rearranges what the mask selects, so the padding is
				// never read into the output.
				let mut pos_pad = [0u16; 32];
				let mut val_pad = [0u16; 32];
				let prefix_arr = u32x16_to_array(prefix);
				for lane in 0..16 {
					pos_pad[lane] = prefix_arr[lane] as u16;
					val_pad[lane] = token_arr[lane] as u16;
				}
				let mask = b32(literal_bits as u32);
				let compressed_pos: [u16; 32] = cast!(v4.mask_compress_u16x32(mask, cast!(pos_pad)));
				let compressed_val: [u16; 32] = cast!(v4.mask_compress_u16x32(mask, cast!(val_pad)));

				let count = literal_bits.count_ones() as usize;
				for k in 0..count {
					block[compressed_pos[k] as usize] = compressed_val[k];
				}

				let hi = 15 - literal_bits.leading_zeros();
				last_non_zero = u32x16_to_array(prefix)[hi as usize];
			}

			let valid_bits_for_consumed = valid_bits;
			let consumed_count = valid_bits_for_consumed.trailing_ones() as usize;
			total_consumed += consumed_count;

			if consumed_count == 0 {
				unreachable!(
					"un_rle_ac_v4_compress: lane 0 must always be valid while chunk_base < 64"
				);
			}

			chunk_base += u32x16_to_array(inclusive)[consumed_count - 1];

			if consumed_count < real_lanes {
				break;
			}
			chunk_start += LANES;
		}

		ac.advance(total_consumed);

		last_non_zero as usize
	}
}

#[cfg(all(test, feature = "avx512-tests"))]
mod test {
	use pulp::x86::{V4, V4Vbmi2};
	use rand::RngExt;

	use super::{un_rle_ac_v4, un_rle_ac_v4_compress};
	use crate::compression::dwa::lossy_dct::{ac_rle::{rle_ac, un_rle_ac}, PackedStream};

	fn expect_avx512() -> V4 {
		V4::try_new().expect("test host must support AVX-512F/BW/CD/DQ/VL")
	}

	fn expect_avx512_vbmi2() -> V4Vbmi2 {
		V4Vbmi2::try_new().expect("test host must support AVX-512F/BW/CD/DQ/VL + VBMI2")
	}

	/// Builds a random 8x8 zig-zag block with roughly `nonzero_probability`
	/// of each AC slot (1..64) being nonzero, matching realistic DWA content
	/// far better than an all-nonzero or fixed-pattern block would.
	fn random_block(rng: &mut impl rand::Rng, nonzero_probability: f64) -> [u16; 64] {
		let mut block = [0u16; 64];
		for slot in block.iter_mut().skip(1) {
			if rng.random_bool(nonzero_probability) {
				// Avoid generating a literal that collides with the
				// 0xffxx run-token encoding, matching what the real
				// quantizer's output range guarantees.
				let mut v: u16 = rng.random();
				while (v & 0xff00) == 0xff00 {
					v = rng.random();
				}
				*slot = v.max(1);
			}
		}
		block
	}

	fn round_trip_matches_scalar(nonzero_probability: f64, trailing_tokens: usize) {
		let v4 = expect_avx512();
		let v4vbmi2 = expect_avx512_vbmi2();
		let mut rng = rand::rng();

		for _ in 0..2000 {
			let want_block = random_block(&mut rng, nonzero_probability);

			let mut ac_tokens = Vec::new();
			rle_ac(&want_block, &mut ac_tokens);
			// Real streams have more blocks' tokens after this one; pad so
			// both the scalar fast path and this kernel take their
			// `remaining() >= 63` branch and neither reads past the end.
			for _ in 0..trailing_tokens {
				ac_tokens.push(0xff00);
			}

			let mut scalar_block = [0u16; 64];
			let mut scalar_stream = PackedStream::new(&ac_tokens);
			let scalar_lnz = un_rle_ac(&mut scalar_stream, &mut scalar_block).unwrap();

			let mut scatter_block = [0u16; 64];
			let mut scatter_stream = PackedStream::new(&ac_tokens);
			let scatter_lnz = un_rle_ac_v4(v4, &mut scatter_stream, &mut scatter_block);

			assert_eq!(scatter_block, scalar_block, "block mismatch (scatter), p={nonzero_probability}");
			assert_eq!(scatter_lnz, scalar_lnz, "last_non_zero mismatch (scatter), p={nonzero_probability}");
			assert_eq!(
				scatter_stream.remaining(), scalar_stream.remaining(),
				"consumed-token-count mismatch (scatter), p={nonzero_probability}"
			);

			let mut compress_block = [0u16; 64];
			let mut compress_stream = PackedStream::new(&ac_tokens);
			let compress_lnz = un_rle_ac_v4_compress(v4vbmi2, &mut compress_stream, &mut compress_block);

			assert_eq!(compress_block, scalar_block, "block mismatch (compress), p={nonzero_probability}");
			assert_eq!(compress_lnz, scalar_lnz, "last_non_zero mismatch (compress), p={nonzero_probability}");
			assert_eq!(
				compress_stream.remaining(), scalar_stream.remaining(),
				"consumed-token-count mismatch (compress), p={nonzero_probability}"
			);
		}
	}

	#[test]
	fn matches_scalar_sparse() {
		round_trip_matches_scalar(0.05, 64);
	}

	#[test]
	fn matches_scalar_medium() {
		round_trip_matches_scalar(0.3, 64);
	}

	#[test]
	fn matches_scalar_dense() {
		round_trip_matches_scalar(0.7, 64);
	}

	#[test]
	fn matches_scalar_all_literal() {
		// Every AC slot nonzero: exercises the exact-63-token boundary
		// (63 literals fill positions 1..64 with zero run tokens).
		round_trip_matches_scalar(1.0, 64);
	}

	#[test]
	fn matches_scalar_all_zero() {
		// A single "rest of block" run token: exercises the
		// count-== 0-means-64 special case at chunk 0, lane 0.
		round_trip_matches_scalar(0.0, 64);
	}

	/// Isolated A/B: builds one big stream of `block_count` *distinct*
	/// random blocks (not one block decoded in a repeated hot loop -- the
	/// earlier VBMI2 probe's synthetic benchmark did that, and its
	/// near-zero branch-miss rate looked suspiciously low next to this
	/// function's real 8K-file measurement; varied content per block is the
	/// point here) and times both implementations decoding the whole thing,
	/// several interleaved rounds. `cargo test --release --features
	/// avx512-tests -- --ignored --nocapture bench_v4_vs_scalar`.
	#[test]
	#[ignore]
	fn bench_v4_vs_scalar() {
		let v4 = expect_avx512();
		let v4vbmi2 = expect_avx512_vbmi2();
		let mut rng = rand::rng();

		const BLOCK_COUNT: usize = 200_000;
		let mut tokens = Vec::new();
		for _ in 0..BLOCK_COUNT {
			let block = random_block(&mut rng, 0.3);
			rle_ac(&block, &mut tokens);
		}
		for _ in 0..64 {
			tokens.push(0xff00);
		}

		let rounds = 6;
		let mut scalar_totals = Vec::with_capacity(rounds);
		let mut scatter_totals = Vec::with_capacity(rounds);
		let mut compress_totals = Vec::with_capacity(rounds);

		for round in 0..rounds {
			let start = std::time::Instant::now();
			let mut stream = PackedStream::new(&tokens);
			let mut block = [0u16; 64];
			for _ in 0..BLOCK_COUNT {
				block = [0u16; 64];
				un_rle_ac(&mut stream, &mut block).unwrap();
			}
			let scalar_elapsed = start.elapsed();
			std::hint::black_box(block);

			let start = std::time::Instant::now();
			let mut stream = PackedStream::new(&tokens);
			let mut block = [0u16; 64];
			for _ in 0..BLOCK_COUNT {
				block = [0u16; 64];
				un_rle_ac_v4(v4, &mut stream, &mut block);
			}
			let scatter_elapsed = start.elapsed();
			std::hint::black_box(block);

			let start = std::time::Instant::now();
			let mut stream = PackedStream::new(&tokens);
			let mut block = [0u16; 64];
			for _ in 0..BLOCK_COUNT {
				block = [0u16; 64];
				un_rle_ac_v4_compress(v4vbmi2, &mut stream, &mut block);
			}
			let compress_elapsed = start.elapsed();
			std::hint::black_box(block);

			eprintln!(
				"round {round}: scalar {:>8.3} ms, scatter {:>8.3} ms ({:.3}x), compress {:>8.3} ms ({:.3}x)",
				scalar_elapsed.as_secs_f64() * 1e3,
				scatter_elapsed.as_secs_f64() * 1e3,
				scatter_elapsed.as_secs_f64() / scalar_elapsed.as_secs_f64(),
				compress_elapsed.as_secs_f64() * 1e3,
				compress_elapsed.as_secs_f64() / scalar_elapsed.as_secs_f64(),
			);
			scalar_totals.push(scalar_elapsed.as_secs_f64());
			scatter_totals.push(scatter_elapsed.as_secs_f64());
			compress_totals.push(compress_elapsed.as_secs_f64());
		}

		let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
		let scalar_avg = avg(&scalar_totals);
		let scatter_avg = avg(&scatter_totals);
		let compress_avg = avg(&compress_totals);
		eprintln!(
			"avg: scalar {:.3} ms, scatter {:.3} ms ({:+.2}%), compress {:.3} ms ({:+.2}%)",
			scalar_avg * 1e3,
			scatter_avg * 1e3,
			(scatter_avg / scalar_avg - 1.0) * 100.0,
			compress_avg * 1e3,
			(compress_avg / scalar_avg - 1.0) * 100.0,
		);
	}
}
