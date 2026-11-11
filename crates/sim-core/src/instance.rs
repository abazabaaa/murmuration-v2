use crate::coupled::CoupledFlock;
use crate::math::Vec3;
use crate::rng::{keyed_rng, normal_vec3, uniform_open01, STREAM_INIT};

pub const STREAM_GAIT: u32 = 5;

/// GPU-ready 48-byte (`12 * f32`) instanced bird record shared across `sim-wasm` (browser WebGPU)
/// and Stage 4 headless `wgpu` rendering (§4.2.1, §4.4).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BirdInstanceGpu {
    /// `[x, y, z, speed_mps]` in world meters and m/s.
    pub pos_speed: [f32; 4],
    /// Unit quaternion `[qx, qy, qz, qw]` rotating local starling mesh axes
    /// `(+X = beak/forward, +Y = left wing, +Z = dorsal back)` into world space.
    pub quat_xyzw: [f32; 4],
    /// `[flap_phase_01, glide_weight_01, bound_weight_01, bank_angle_rad]`.
    pub gait_params: [f32; 4],
}

const _: () = assert!(core::mem::size_of::<BirdInstanceGpu>() == 48);

/// Canonical spatial stress cloud generator shared identically between `sim-cli` and `sim-wasm` (§4.3):
/// - `0`: `"calm_ribbon"` (`~ 1 : 2.8 : 5.6` sheet, NND ~ 1 m)
/// - `1`: `"compacted_core_5x"` (5x local compaction spike)
/// - `2`: `"diluted_cloud_3x"` (3x post-attack diluted cloud)
/// - `3`: `"split_subflocks_stragglers"` (200 m sub-flock gap + 12 stragglers)
pub fn generate_stress_cloud(regime_idx: usize, n: usize, seed: u64) -> (Vec<Vec3>, Vec<Vec3>) {
    let mut pos = Vec::with_capacity(n);
    let mut u = Vec::with_capacity(n);
    for i in 0..n {
        let mut rng = keyed_rng(seed + (regime_idx as u64) * 100, STREAM_INIT, 0, i as u32);
        let p = match regime_idx {
            0 => {
                let g = normal_vec3(&mut rng);
                let scale = libm::cbrtf(n as f32) * 0.55;
                Vec3::new(g.x * 2.8 * scale, g.y * 5.6 * scale, g.z * 1.0 * scale)
            }
            1 => {
                let g = normal_vec3(&mut rng);
                let scale = libm::cbrtf(n as f32) * 0.28;
                g * scale
            }
            2 => {
                let g = normal_vec3(&mut rng);
                let scale = libm::cbrtf(n as f32) * 3.0;
                g * scale
            }
            _ => {
                if i + 12 < n {
                    let offset = if i % 2 == 0 { -100.0 } else { 100.0 };
                    let scale = libm::cbrtf((n / 2) as f32) * 0.7;
                    normal_vec3(&mut rng) * scale + Vec3::new(offset, 0.0, 0.0)
                } else {
                    let r = uniform_open01(&mut rng);
                    Vec3::new(450.0 * r - 225.0, 320.0, -180.0 * r)
                }
            }
        };
        pos.push(p);
        let dir = (Vec3::EX + normal_vec3(&mut rng) * 0.15).normalized_or(Vec3::EX);
        u.push(dir);
    }
    (pos, u)
}

/// Converts a right-handed orthonormal basis `(forward = +X_local, left_wing = +Y_local, dorsal = +Z_local)`
/// into a canonical (`qw >= 0`) unit quaternion `[qx, qy, qz, qw]` via Shepperd's method.
#[inline]
pub fn quat_from_orthonormal_basis(forward: Vec3, left_wing: Vec3, dorsal: Vec3) -> [f32; 4] {
    // Columns of rotation matrix R: col0 = forward, col1 = left_wing, col2 = dorsal.
    let m00 = forward.x;
    let m10 = forward.y;
    let m20 = forward.z;

    let m01 = left_wing.x;
    let m11 = left_wing.y;
    let m21 = left_wing.z;

    let m02 = dorsal.x;
    let m12 = dorsal.y;
    let m22 = dorsal.z;

    let trace = m00 + m11 + m22;
    let (mut qx, mut qy, mut qz, mut qw) = if trace > 0.0 {
        let s = libm::sqrtf(trace + 1.0) * 2.0;
        let inv_s = 1.0 / s;
        (
            (m21 - m12) * inv_s,
            (m02 - m20) * inv_s,
            (m10 - m01) * inv_s,
            0.25 * s,
        )
    } else if m00 > m11 && m00 > m22 {
        let s = libm::sqrtf((1.0 + m00 - m11 - m22).max(1e-24)) * 2.0;
        let inv_s = 1.0 / s;
        (
            0.25 * s,
            (m01 + m10) * inv_s,
            (m02 + m20) * inv_s,
            (m21 - m12) * inv_s,
        )
    } else if m11 > m22 {
        let s = libm::sqrtf((1.0 + m11 - m00 - m22).max(1e-24)) * 2.0;
        let inv_s = 1.0 / s;
        (
            (m01 + m10) * inv_s,
            0.25 * s,
            (m12 + m21) * inv_s,
            (m02 - m20) * inv_s,
        )
    } else {
        let s = libm::sqrtf((1.0 + m22 - m00 - m11).max(1e-24)) * 2.0;
        let inv_s = 1.0 / s;
        (
            (m02 + m20) * inv_s,
            (m12 + m21) * inv_s,
            0.25 * s,
            (m10 - m01) * inv_s,
        )
    };

    let n2 = qx * qx + qy * qy + qz * qz + qw * qw;
    if n2 > 1e-24 {
        let inv = 1.0 / libm::sqrtf(n2);
        qx *= inv;
        qy *= inv;
        qz *= inv;
        qw *= inv;
    } else {
        return [0.0, 0.0, 0.0, 1.0];
    }
    if qw < 0.0 {
        [-qx, -qy, -qz, -qw]
    } else {
        [qx, qy, qz, qw]
    }
}

/// Rotates local vector `v` by unit quaternion `q = [qx, qy, qz, qw]`.
#[inline]
pub fn quat_rotate_vec3(q: [f32; 4], v: Vec3) -> Vec3 {
    let qv = Vec3::new(q[0], q[1], q[2]);
    let qw = q[3];
    let t = qv.cross(v) * 2.0;
    v + t * qw + qv.cross(t)
}

/// Shortest-arc normalized linear/spherical interpolation between unit quaternions `q0` and `q1`.
#[inline]
pub fn interp_quat_shortest(q0: [f32; 4], q1_in: [f32; 4], alpha: f32) -> [f32; 4] {
    let a = alpha.clamp(0.0, 1.0);
    let mut dot = q0[0] * q1_in[0] + q0[1] * q1_in[1] + q0[2] * q1_in[2] + q0[3] * q1_in[3];
    let q1 = if dot < 0.0 {
        dot = -dot;
        [-q1_in[0], -q1_in[1], -q1_in[2], -q1_in[3]]
    } else {
        q1_in
    };

    let (w0, w1) = if dot > 0.9995 {
        (1.0 - a, a)
    } else {
        let theta = libm::acosf(dot.clamp(-1.0, 1.0));
        let inv_sin = 1.0 / libm::sinf(theta);
        (
            libm::sinf((1.0 - a) * theta) * inv_sin,
            libm::sinf(a * theta) * inv_sin,
        )
    };

    let mut qx = q0[0] * w0 + q1[0] * w1;
    let mut qy = q0[1] * w0 + q1[1] * w1;
    let mut qz = q0[2] * w0 + q1[2] * w1;
    let mut qw = q0[3] * w0 + q1[3] * w1;
    let n2 = qx * qx + qy * qy + qz * qz + qw * qw;
    if n2 > 1e-24 {
        let inv = 1.0 / libm::sqrtf(n2);
        qx *= inv;
        qy *= inv;
        qz *= inv;
        qw *= inv;
    } else {
        qw = 1.0;
    }
    if qw < 0.0 {
        [-qx, -qy, -qz, -qw]
    } else {
        [qx, qy, qz, qw]
    }
}

/// Builds the `Mode K` unit quaternion from heading `u_unit`, horizon reference `right_0`, and
/// bank angle `phi_bank` (`phi_bank > 0` = right wing dips downward, dorsal lift tilts toward `+right_0`).
#[inline]
pub fn mode_k_quat(u_unit: Vec3, prev_right: Vec3, phi_bank: f32) -> [f32; 4] {
    let horiz = u_unit.cross(Vec3::EZ);
    let right_0 = if horiz.norm_sq() > 1e-4 {
        horiz.normalized_or(Vec3::new(0.0, -1.0, 0.0))
    } else {
        prev_right
            .project_perp(u_unit)
            .normalized_or(Vec3::new(0.0, -1.0, 0.0))
    };
    let dorsal_0 = right_0.cross(u_unit).normalized_or(Vec3::EZ);
    let c = libm::cosf(phi_bank);
    let s = libm::sinf(phi_bank);
    let right_b = right_0 * c - dorsal_0 * s;
    let dorsal_b = dorsal_0 * c + right_0 * s;
    let left_b = -right_b;
    quat_from_orthonormal_basis(u_unit, left_b, dorsal_b)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GaitMode {
    Flap,
    Glide,
    Bound,
}

/// Rayner et al. (2001, Table 1) & Tobalske (1995) deterministic per-bird intermittent flap/glide/bound
/// gait state + smooth visual roll filter for `Mode K` rendering.
#[derive(Clone, Debug)]
pub struct BirdVisualGaitBank {
    master_seed: u64,
    bout_counter: Vec<u32>,
    mode: Vec<GaitMode>,
    bout_remaining_s: Vec<f32>,
    wing_freq_hz: Vec<f32>,
    pub flap_phase_01: Vec<f32>,
    pub glide_weight_01: Vec<f32>,
    pub bound_weight_01: Vec<f32>,
    /// Rate-limited visual bank angle (`rad`) so stochastic `5 ms` Wiener kicks do not chatter wings.
    pub visual_bank_rad: Vec<f32>,
}

impl BirdVisualGaitBank {
    pub fn new(n: usize, master_seed: u64) -> Self {
        let mut bout_counter = vec![0u32; n];
        let mut mode = Vec::with_capacity(n);
        let mut bout_remaining_s = Vec::with_capacity(n);
        let mut wing_freq_hz = Vec::with_capacity(n);
        let mut flap_phase_01 = Vec::with_capacity(n);
        let mut glide_weight_01 = Vec::with_capacity(n);
        let mut bound_weight_01 = Vec::with_capacity(n);

        for i in 0..n {
            let mut rng = keyed_rng(master_seed, STREAM_GAIT, 0, i as u32);
            let phase0 = uniform_open01(&mut rng);
            let freq_jitter = 1.0 + (uniform_open01(&mut rng) - 0.5) * 0.12;
            let init_roll = uniform_open01(&mut rng);
            let init_rem = 0.25 + uniform_open01(&mut rng) * 1.10;
            let m = if init_roll < 0.68 {
                GaitMode::Flap
            } else {
                GaitMode::Glide
            };
            bout_counter[i] = 1;
            mode.push(m);
            bout_remaining_s.push(init_rem);
            wing_freq_hz.push(11.1 * freq_jitter);
            flap_phase_01.push(phase0);
            glide_weight_01.push(if m == GaitMode::Glide { 1.0 } else { 0.0 });
            bound_weight_01.push(0.0);
        }

        Self {
            master_seed,
            bout_counter,
            mode,
            bout_remaining_s,
            wing_freq_hz,
            flap_phase_01,
            glide_weight_01,
            bound_weight_01,
            visual_bank_rad: vec![0.0; n],
        }
    }

    /// Interpolates Rayner et al. (2001) Table 1 mean flap and glide bout durations (`s`) as a function of speed.
    #[inline]
    fn rayner_mean_bouts(speed_mps: f32) -> (f32, f32) {
        if speed_mps <= 8.5 {
            (1.14, 0.64)
        } else if speed_mps <= 10.2 {
            let t = (speed_mps - 8.5) / (10.2 - 8.5);
            (1.14 + t * (1.05 - 1.14), 0.64 + t * (1.01 - 0.64))
        } else if speed_mps <= 12.4 {
            let t = (speed_mps - 10.2) / (12.4 - 10.2);
            (1.05 + t * (1.35 - 1.05), 1.01 + t * (0.58 - 1.01))
        } else if speed_mps <= 14.1 {
            let t = (speed_mps - 12.4) / (14.1 - 12.4);
            (1.35 + t * (2.06 - 1.35), 0.58 + t * (0.23 - 0.58))
        } else {
            (2.06, 0.23)
        }
    }

    pub fn step_gait(&mut self, dt: f32, speeds: &[f32], target_banks: &[f32]) {
        let n = speeds.len().min(self.mode.len());
        // Smooth visual roll first-order time constant `tau_roll = 85 ms` + bounded visual roll rate `12 rad/s`
        // (Hemelrijk & Hildenbrandt 2015starling roll time ~0.1-0.2 s).
        let roll_alpha = (dt / (0.085 + dt)).clamp(0.0, 1.0);
        let max_dphi = 12.0 * dt;
        let blend_rate = (dt / 0.075).clamp(0.0, 1.0);

        for i in 0..n {
            let spd = speeds[i];
            let target_phi = target_banks[i].clamp(-1.35, 1.35);
            let cur_phi = self.visual_bank_rad[i];
            let dphi = ((target_phi - cur_phi) * roll_alpha).clamp(-max_dphi, max_dphi);
            let next_phi = cur_phi + dphi;
            self.visual_bank_rad[i] = next_phi;

            self.bout_remaining_s[i] -= dt;
            // Below 7.5 m/s or under heavy banking (> 38 deg), return to powered flapping
            let force_flap = spd < 7.5 || next_phi.abs() > 0.66;
            if force_flap && self.mode[i] != GaitMode::Flap {
                self.mode[i] = GaitMode::Flap;
                self.bout_remaining_s[i] = 0.65;
            } else if self.bout_remaining_s[i] <= 0.0 {
                let c = self.bout_counter[i];
                self.bout_counter[i] = c.wrapping_add(1);
                let mut rng = keyed_rng(self.master_seed, STREAM_GAIT, c as u64, i as u32);
                let jitter = 0.70 + 0.60 * uniform_open01(&mut rng);
                let (mean_flap, mean_glide) = Self::rayner_mean_bouts(spd);
                match self.mode[i] {
                    GaitMode::Flap => {
                        if force_flap {
                            self.mode[i] = GaitMode::Flap;
                            self.bout_remaining_s[i] = mean_flap * jitter;
                        } else {
                            let u_mode = uniform_open01(&mut rng);
                            let p_bound = if spd > 12.8 {
                                ((spd - 12.8) * 0.06).clamp(0.0, 0.22)
                            } else {
                                0.0
                            };
                            if u_mode < p_bound {
                                self.mode[i] = GaitMode::Bound;
                                self.bout_remaining_s[i] = 0.14 * jitter;
                            } else {
                                self.mode[i] = GaitMode::Glide;
                                self.bout_remaining_s[i] = mean_glide * jitter;
                            }
                        }
                    }
                    GaitMode::Glide | GaitMode::Bound => {
                        self.mode[i] = GaitMode::Flap;
                        self.bout_remaining_s[i] = mean_flap * jitter;
                    }
                }
            }

            let (tgt_glide, tgt_bound) = match self.mode[i] {
                GaitMode::Flap => (0.0f32, 0.0f32),
                GaitMode::Glide => (1.0f32, 0.0f32),
                GaitMode::Bound => (0.0f32, 1.0f32),
            };
            self.glide_weight_01[i] += (tgt_glide - self.glide_weight_01[i]) * blend_rate;
            self.bound_weight_01[i] += (tgt_bound - self.bound_weight_01[i]) * blend_rate;

            // Advance wingbeat phase during flapping (slowing smoothly as glide/bound approaches 1)
            let active_flap = (1.0 - self.glide_weight_01[i] - self.bound_weight_01[i]).clamp(0.0, 1.0);
            let dphase = self.wing_freq_hz[i] * dt * (0.15 + 0.85 * active_flap);
            let mut ph = self.flap_phase_01[i] + dphase;
            ph -= libm::floorf(ph);
            self.flap_phase_01[i] = ph;
        }
    }
}

/// Double-buffered `(prev, curr)` pose snapshot bank supporting exact sub-tick interpolation
/// (`alpha = accumulator / dt_sim in [0, 1)`) into a contiguous `Vec<BirdInstanceGpu>` (§4.2.1).
#[derive(Clone, Debug)]
pub struct PoseInterpolator {
    pub prev: Vec<BirdInstanceGpu>,
    pub curr: Vec<BirdInstanceGpu>,
    pub out: Vec<BirdInstanceGpu>,
}

impl PoseInterpolator {
    pub fn from_coupled(flock: &CoupledFlock, gait: &BirdVisualGaitBank) -> Self {
        let n = flock.pos.len();
        let mut curr = vec![BirdInstanceGpu::default(); n];
        Self::snapshot_coupled_into(flock, gait, &mut curr);
        let prev = curr.clone();
        let out = curr.clone();
        Self { prev, curr, out }
    }

    pub fn snapshot_coupled_into(
        flock: &CoupledFlock,
        gait: &BirdVisualGaitBank,
        dst: &mut [BirdInstanceGpu],
    ) {
        let n = flock.pos.len().min(dst.len());
        for i in 0..n {
            let p = flock.pos[i];
            let spd = flock.speed[i];
            let phi = gait.visual_bank_rad[i];
            let q = mode_k_quat(flock.heading[i], flock.right_ref[i], phi);
            dst[i] = BirdInstanceGpu {
                pos_speed: [p.x, p.y, p.z, spd],
                quat_xyzw: q,
                gait_params: [
                    gait.flap_phase_01[i],
                    gait.glide_weight_01[i],
                    gait.bound_weight_01[i],
                    phi,
                ],
            };
        }
    }

    pub fn push_coupled_tick(&mut self, flock: &CoupledFlock, gait: &BirdVisualGaitBank) {
        self.prev.copy_from_slice(&self.curr);
        Self::snapshot_coupled_into(flock, gait, &mut self.curr);
    }

    /// Interpolates between `prev` and `curr` by `alpha in [0, 1]` directly into `self.out` with zero allocations.
    pub fn pack_interpolated(&mut self, alpha: f32) -> &[BirdInstanceGpu] {
        let a = alpha.clamp(0.0, 1.0);
        if a >= 0.9999 {
            self.out.copy_from_slice(&self.curr);
            return &self.out;
        }
        let oma = 1.0 - a;
        let n = self.curr.len();
        for i in 0..n {
            let p0 = self.prev[i];
            let p1 = self.curr[i];
            let mut dph = p1.gait_params[0] - p0.gait_params[0];
            if dph < -0.5 {
                dph += 1.0;
            } else if dph > 0.5 {
                dph -= 1.0;
            }
            let mut ph = p0.gait_params[0] + a * dph;
            ph -= libm::floorf(ph);

            self.out[i] = BirdInstanceGpu {
                pos_speed: [
                    p0.pos_speed[0] * oma + p1.pos_speed[0] * a,
                    p0.pos_speed[1] * oma + p1.pos_speed[1] * a,
                    p0.pos_speed[2] * oma + p1.pos_speed[2] * a,
                    p0.pos_speed[3] * oma + p1.pos_speed[3] * a,
                ],
                quat_xyzw: interp_quat_shortest(p0.quat_xyzw, p1.quat_xyzw, a),
                gait_params: [
                    ph,
                    p0.gait_params[1] * oma + p1.gait_params[1] * a,
                    p0.gait_params[2] * oma + p1.gait_params[2] * a,
                    p0.gait_params[3] * oma + p1.gait_params[3] * a,
                ],
            };
        }
        &self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_k_quat_rotates_local_starling_axes_to_banked_world_frame() {
        let u = Vec3::EX; // Flying East (+X); right wing is -Y, left wing is +Y, dorsal is +Z
        let q_level = mode_k_quat(u, Vec3::new(0.0, -1.0, 0.0), 0.0);
        let fwd = quat_rotate_vec3(q_level, Vec3::EX);
        let left = quat_rotate_vec3(q_level, Vec3::EY);
        let dor = quat_rotate_vec3(q_level, Vec3::EZ);
        assert!((fwd - Vec3::EX).norm() < 1e-5);
        assert!((left - Vec3::EY).norm() < 1e-5);
        assert!((dor - Vec3::EZ).norm() < 1e-5);

        // +45 deg right bank: dorsal tilts toward -Y (inside of right turn), right wing (-Y_local) dips -Z
        let q_bank = mode_k_quat(u, Vec3::new(0.0, -1.0, 0.0), core::f32::consts::FRAC_PI_4);
        let dor_bank = quat_rotate_vec3(q_bank, Vec3::EZ);
        let s45 = core::f32::consts::FRAC_1_SQRT_2;
        assert!((dor_bank - Vec3::new(0.0, -s45, s45)).norm() < 1e-5);
        let right_wing_tip = quat_rotate_vec3(q_bank, Vec3::new(0.0, -1.0, 0.0));
        assert!(right_wing_tip.z < -0.70, "right wing should dip downward in right turn");
    }
}
