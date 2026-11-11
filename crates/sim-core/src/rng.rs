use crate::math::Vec3;
use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};

pub const STREAM_INIT: u32 = 1;
pub const STREAM_WIENER: u32 = 2;
pub const STREAM_PHASE: u32 = 3;

/// Deterministic counter-keyed PRNG per `(master_seed, stream_id, tick_idx, agent_id)`.
/// Invariant to thread count (`rayon` vs single-threaded `wasm32`) and unrelated subsystem state.
#[inline]
pub fn keyed_rng(master_seed: u64, stream_id: u32, tick_idx: u64, agent_id: u32) -> ChaCha8Rng {
    let mut seed = [0u8; 32];
    seed[0..8].copy_from_slice(&master_seed.to_le_bytes());
    seed[8..12].copy_from_slice(&stream_id.to_le_bytes());
    seed[12..16].copy_from_slice(&agent_id.to_le_bytes());
    seed[16..24].copy_from_slice(&tick_idx.to_le_bytes());
    let domain_tag: u64 = 0x9E37_79B9_7F4A_7C15u64 ^ ((stream_id as u64) << 32) ^ (agent_id as u64);
    seed[24..32].copy_from_slice(&domain_tag.to_le_bytes());
    ChaCha8Rng::from_seed(seed)
}

/// Deterministic uniform float strictly in `(0, 1)` with exact 24-bit mantissa resolution.
#[inline]
pub fn uniform_open01(rng: &mut ChaCha8Rng) -> f32 {
    let u24 = rng.next_u32() >> 8;
    ((u24 as f32) + 0.5) * (1.0 / 16_777_216.0)
}

/// Pure-Rust `libm` Box-Muller standard normal draw `(N(0,1), N(0,1))`.
#[inline]
pub fn normal_pair(rng: &mut ChaCha8Rng) -> (f32, f32) {
    let u1 = uniform_open01(rng);
    let u2 = uniform_open01(rng);
    let r = libm::sqrtf(-2.0 * libm::logf(u1));
    let theta = (2.0 * core::f32::consts::PI) * u2;
    (r * libm::cosf(theta), r * libm::sinf(theta))
}

/// Standard 3D isotropic Gaussian vector `Z ~ N(0, I_3)`.
#[inline]
pub fn normal_vec3(rng: &mut ChaCha8Rng) -> Vec3 {
    let (z0, z1) = normal_pair(rng);
    let (z2, _) = normal_pair(rng);
    Vec3::new(z0, z1, z2)
}

/// Deterministic initial phase offset `k_i^(0) in 0..period_ticks` for asynchronous reaction clocks.
#[inline]
pub fn deterministic_phase_offset(master_seed: u64, agent_id: u32, period_ticks: u32) -> u32 {
    if period_ticks <= 1 {
        return 0;
    }
    let mut rng = keyed_rng(master_seed, STREAM_PHASE, 0, agent_id);
    rng.next_u32() % period_ticks
}
