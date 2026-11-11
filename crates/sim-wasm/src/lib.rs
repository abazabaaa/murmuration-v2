use sim_core::coupled::{CoupledConfig, CoupledFlock};
use sim_core::instance::{
    generate_stress_cloud, BirdInstanceGpu, BirdVisualGaitBank, PoseInterpolator,
};
use sim_core::math::Vec3;
use sim_core::models::{ExternalTurnPulse, SpeedControlLaw};
use sim_core::neighbour::{LbvhIndex, NeighbourRule};
use sim_core::parity::{
    default_coupled_browser_config, run_parity_1000_ticks, run_tost_seed_rollout,
    Parity1000TicksDump,
};

#[inline]
fn pack_seed(seed_lo: u32, seed_hi: u32) -> u64 {
    (seed_lo as u64) | ((seed_hi as u64) << 32)
}

pub struct WasmSimHandle {
    pub cfg: CoupledConfig,
    pub flock: CoupledFlock,
    pub gait: BirdVisualGaitBank,
    pub interp: PoseInterpolator,
    pub telemetry: [f32; 9],
}

impl WasmSimHandle {
    pub fn new(regime_idx: usize, n_birds: usize, seed: u64) -> Self {
        let n = n_birds.max(2);
        let cfg = default_coupled_browser_config(seed);
        let (p0, u0) = generate_stress_cloud(regime_idx, n, seed);
        let v0: Vec<Vec3> = u0.into_iter().map(|u| u * 12.0).collect();
        let flock = CoupledFlock::from_cartesian(p0, v0);
        let gait = BirdVisualGaitBank::new(n, seed);
        let interp = PoseInterpolator::from_coupled(&flock, &gait);
        let mut h = Self {
            cfg,
            flock,
            gait,
            interp,
            telemetry: [0.0; 9],
        };
        h.refresh_telemetry();
        h
    }

    pub fn refresh_telemetry(&mut self) {
        let n = self.flock.pos.len();
        if n == 0 {
            self.telemetry = [0.0; 9];
            return;
        }
        let inv_n = 1.0 / (n as f32);
        let mut cx = 0.0f32;
        let mut cy = 0.0f32;
        let mut cz = 0.0f32;
        let mut sum_s = 0.0f32;
        let mut min_s = f32::INFINITY;
        let mut max_s = 0.0f32;
        let mut sum_u = Vec3::ZERO;

        for i in 0..n {
            let p = self.flock.pos[i];
            let s = self.flock.speed[i];
            cx += p.x;
            cy += p.y;
            cz += p.z;
            sum_s += s;
            if s < min_s {
                min_s = s;
            }
            if s > max_s {
                max_s = s;
            }
            sum_u += self.flock.heading[i];
        }
        cx *= inv_n;
        cy *= inv_n;
        cz *= inv_n;
        let c0 = Vec3::new(cx, cy, cz);
        let mut rms_r2 = 0.0f32;
        for &p in &self.flock.pos {
            rms_r2 += (p - c0).norm_sq();
        }
        let rms_r = libm::sqrtf(rms_r2 * inv_n);

        // Two-pass Cauchy-weighted core centroid so stragglers or split packets never pull the camera into empty air
        let mut c_core = c0;
        for &scale_m in &[45.0f32, 24.0f32] {
            let inv_s2 = 1.0 / (scale_m * scale_m);
            let mut w_sum = 0.0f32;
            let mut wp_sum = Vec3::ZERO;
            for &p in &self.flock.pos {
                let d2 = (p - c_core).norm_sq();
                let w = 1.0 / (1.0 + d2 * inv_s2);
                w_sum += w;
                wp_sum += p * w;
            }
            if w_sum > 1e-6 {
                c_core = wp_sum * (1.0 / w_sum);
            }
        }
        let mut core_w_sum = 0.0f32;
        let mut core_r2_sum = 0.0f32;
        let inv_core_s2 = 1.0 / (36.0f32 * 36.0f32);
        for &p in &self.flock.pos {
            let d2 = (p - c_core).norm_sq();
            let w = 1.0 / (1.0 + d2 * inv_core_s2);
            core_w_sum += w;
            core_r2_sum += w * d2;
        }
        let core_rms_r = if core_w_sum > 1e-6 {
            libm::sqrtf(core_r2_sum / core_w_sum)
        } else {
            rms_r
        };

        let phi = (sum_u * inv_n).norm();

        // [0] mean_speed, [1] min_speed, [2] max_speed, [3..6] core_centroid, [6] rms_radius, [7] phi, [8] core_rms_radius
        self.telemetry = [
            sum_s * inv_n,
            min_s,
            max_s,
            c_core.x,
            c_core.y,
            c_core.z,
            rms_r,
            phi,
            core_rms_r,
        ];
    }
}

#[no_mangle]
pub extern "C" fn sim_create(
    regime_idx: u32,
    n_birds: u32,
    seed_lo: u32,
    seed_hi: u32,
) -> *mut WasmSimHandle {
    let seed = pack_seed(seed_lo, seed_hi);
    Box::into_raw(Box::new(WasmSimHandle::new(
        regime_idx as usize,
        n_birds as usize,
        seed,
    )))
}

/// # Safety
/// `handle` must be a valid pointer returned by `sim_create` and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn sim_destroy(handle: *mut WasmSimHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// # Safety
/// `handle` must be a valid pointer returned by `sim_create`.
#[no_mangle]
pub unsafe extern "C" fn sim_set_params(
    handle: *mut WasmSimHandle,
    j_spp: f32,
    tau_omega: f32,
    lambda: f32,
    v0: f32,
    d_v: f32,
    k_coh: f32,
    k_rep: f32,
    vertical_coh_gain: f32,
) {
    if handle.is_null() {
        return;
    }
    let h = &mut *handle;
    h.cfg.j_spp = j_spp.max(0.0);
    h.cfg.tau_omega = tau_omega.max(0.0);
    h.cfg.d_v = d_v.max(0.0);
    h.cfg.d_omega = d_v.max(0.0);
    h.cfg.speed_control = SpeedControlLaw::MarginalOctic {
        lambda: lambda.max(0.001),
        v0: v0.max(1.0),
    };
    h.cfg.positional.k_coh = k_coh.max(0.0);
    h.cfg.positional.k_rep = k_rep.max(0.0);
    h.cfg.positional.vertical_coh_gain = vertical_coh_gain.max(0.2);
}

/// # Safety
/// `handle` must be a valid pointer returned by `sim_create`.
#[no_mangle]
pub unsafe extern "C" fn sim_set_roost_params(
    handle: *mut WasmSimHandle,
    roost_radius_m: f32,
    roost_pull_mps2: f32,
) {
    if handle.is_null() {
        return;
    }
    let h = &mut *handle;
    h.cfg.roost.roost_radius_m = roost_radius_m.max(0.0);
    h.cfg.roost.roost_pull_mps2 = roost_pull_mps2.max(0.0);
}

/// # Safety
/// `handle` must be a valid pointer returned by `sim_create`.
#[no_mangle]
pub unsafe extern "C" fn sim_trigger_turn_pulse(
    handle: *mut WasmSimHandle,
    initiator_count: u32,
    accel_x: f32,
    accel_y: f32,
    accel_z: f32,
    duration_s: f32,
) {
    if handle.is_null() {
        return;
    }
    let h = &mut *handle;
    let t_now = (h.flock.tick as f32) * h.cfg.dt;
    h.cfg.turn_pulse = Some(ExternalTurnPulse {
        initiator_ids: (initiator_count as usize).max(1).min(h.flock.pos.len()),
        start_time_s: t_now,
        end_time_s: t_now + duration_s.max(h.cfg.dt),
        accel_vec: Vec3::new(accel_x, accel_y, accel_z),
    });
}

/// Advances `n_ticks` fixed `5 ms` (`200 Hz`) simulation steps and updates the double-buffered
/// pose interpolator (`prev`, `curr`).
///
/// # Safety
/// `handle` must be a valid pointer returned by `sim_create`.
#[no_mangle]
pub unsafe extern "C" fn sim_step_ticks(handle: *mut WasmSimHandle, n_ticks: u32) -> u32 {
    if handle.is_null() {
        return 0;
    }
    let h = &mut *handle;
    for _ in 0..n_ticks {
        h.flock.step(&h.cfg, None);
        h.gait
            .step_gait(h.cfg.dt, &h.flock.speed, &h.flock.bank_angle_rad);
        h.interp.push_coupled_tick(&h.flock, &h.gait);
    }
    h.flock.tick as u32
}

/// Packs `48 * N` bytes of `BirdInstanceGpu` interpolated at `alpha in [0, 1]` directly in Wasm
/// linear memory and returns its pointer for zero-copy `Float32Array` view -> `queue.writeBuffer`.
///
/// # Safety
/// `handle` must be a valid pointer returned by `sim_create`.
#[no_mangle]
pub unsafe extern "C" fn sim_pack_instances(
    handle: *mut WasmSimHandle,
    alpha: f32,
) -> *const BirdInstanceGpu {
    if handle.is_null() {
        return core::ptr::null();
    }
    let h = &mut *handle;
    h.interp.pack_interpolated(alpha).as_ptr()
}

/// # Safety
/// `handle` must be a valid pointer returned by `sim_create`.
#[no_mangle]
pub unsafe extern "C" fn sim_instances_byte_len(handle: *const WasmSimHandle) -> usize {
    if handle.is_null() {
        return 0;
    }
    let h = &*handle;
    h.interp.out.len() * core::mem::size_of::<BirdInstanceGpu>()
}

/// # Safety
/// `handle` must be a valid pointer returned by `sim_create`.
#[no_mangle]
pub unsafe extern "C" fn sim_bird_count(handle: *const WasmSimHandle) -> u32 {
    if handle.is_null() {
        return 0;
    }
    ((*handle).flock.pos.len()) as u32
}

/// # Safety
/// `handle` must be a valid pointer returned by `sim_create`.
#[no_mangle]
pub unsafe extern "C" fn sim_tick_count(handle: *const WasmSimHandle) -> u32 {
    if handle.is_null() {
        return 0;
    }
    ((*handle).flock.tick) as u32
}

/// Computes `O(N)` lightweight visual HUD telemetry (`8 * f32`) without invoking any `O(N^2)` evaluators.
///
/// # Safety
/// `handle` must be a valid pointer returned by `sim_create`.
#[no_mangle]
pub unsafe extern "C" fn sim_write_telemetry(handle: *mut WasmSimHandle) -> *const f32 {
    if handle.is_null() {
        return core::ptr::null();
    }
    let h = &mut *handle;
    h.refresh_telemetry();
    h.telemetry.as_ptr()
}

// ============================================================================
// Stage 3 Gate 1: Native <-> `wasm32 +simd128` Parity & Benchmark C-ABI Exports
// ============================================================================

#[no_mangle]
pub extern "C" fn wasm_parity_run_1000_ticks(
    model_kind: u32,
    n_birds: u32,
    seed_lo: u32,
    seed_hi: u32,
) -> *mut Parity1000TicksDump {
    let seed = pack_seed(seed_lo, seed_hi);
    Box::into_raw(Box::new(run_parity_1000_ticks(
        model_kind,
        n_birds as usize,
        seed,
    )))
}

/// # Safety
/// `dump` must be a valid pointer returned by `wasm_parity_run_1000_ticks`.
#[no_mangle]
pub unsafe extern "C" fn wasm_parity_tick600_ptr(dump: *const Parity1000TicksDump) -> *const f32 {
    if dump.is_null() {
        return core::ptr::null();
    }
    (*dump).state_tick_600.as_ptr()
}

/// # Safety
/// `dump` must be a valid pointer returned by `wasm_parity_run_1000_ticks`.
#[no_mangle]
pub unsafe extern "C" fn wasm_parity_tick1000_ptr(dump: *const Parity1000TicksDump) -> *const f32 {
    if dump.is_null() {
        return core::ptr::null();
    }
    (*dump).state_tick_1000.as_ptr()
}

/// # Safety
/// `dump` must be a valid pointer returned by `wasm_parity_run_1000_ticks`.
#[no_mangle]
pub unsafe extern "C" fn wasm_parity_float_len(dump: *const Parity1000TicksDump) -> usize {
    if dump.is_null() {
        return 0;
    }
    (*dump).state_tick_1000.len()
}

/// # Safety
/// `dump` must be a valid pointer returned by `wasm_parity_run_1000_ticks` and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn wasm_parity_free(dump: *mut Parity1000TicksDump) {
    if !dump.is_null() {
        drop(Box::from_raw(dump));
    }
}

#[no_mangle]
pub extern "C" fn wasm_tost_run_seed(
    seed_lo: u32,
    seed_hi: u32,
    n_birds: u32,
) -> *mut Vec<f32> {
    let seed = pack_seed(seed_lo, seed_hi);
    let summary = run_tost_seed_rollout(seed, n_birds as usize);
    Box::into_raw(Box::new(summary.to_f32_vec()))
}

/// # Safety
/// `v` must be a valid pointer returned by `wasm_tost_run_seed`.
#[no_mangle]
pub unsafe extern "C" fn wasm_vec_ptr(v: *const Vec<f32>) -> *const f32 {
    if v.is_null() {
        return core::ptr::null();
    }
    (*v).as_ptr()
}

/// # Safety
/// `v` must be a valid pointer returned by `wasm_tost_run_seed`.
#[no_mangle]
pub unsafe extern "C" fn wasm_vec_len(v: *const Vec<f32>) -> usize {
    if v.is_null() {
        return 0;
    }
    (*v).len()
}

/// # Safety
/// `v` must be a valid pointer returned by `wasm_tost_run_seed` and not yet freed.
#[no_mangle]
pub unsafe extern "C" fn wasm_vec_free(v: *mut Vec<f32>) {
    if !v.is_null() {
        drop(Box::from_raw(v));
    }
}

#[no_mangle]
pub extern "C" fn wasm_bench_lbvh_only(
    regime_idx: u32,
    n_birds: u32,
    seed_lo: u32,
    seed_hi: u32,
    iters: u32,
) -> u32 {
    let seed = pack_seed(seed_lo, seed_hi);
    let (pos, u) = generate_stress_cloud(regime_idx as usize, n_birds as usize, seed);
    let rule = NeighbourRule::ExactTopoK {
        k: 7,
        blind_cos: -0.5,
    };
    let mut acc = 0u32;
    for _ in 0..iters.max(1) {
        let lbvh = LbvhIndex::build(&pos);
        let flat = lbvh.query_all_flat(&pos, &u, rule);
        acc = acc.wrapping_add(flat.entries.len() as u32);
    }
    acc
}
