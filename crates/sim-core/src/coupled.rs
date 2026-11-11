use crate::math::{compute_mode_k_bank_angle, rodrigues_rotate, Vec3};
use crate::models::{
    compute_positional_force, ExternalTurnPulse, PositionalInteractionConfig, SpeedControlLaw,
};
use crate::neighbour::{
    FlatNeighbourTable, GraphSymmetry, LbvhIndex, NeighbourEntry, NeighbourRule, WeightedGraph,
};
use crate::rng::{keyed_rng, normal_vec3, STREAM_WIENER};
use serde::{Deserialize, Serialize};

pub const STREAM_WIENER_OU_AUX: u32 = 4;

/// Optional roost-site wheeling attraction and reciprocal saturating cohesion (`Hildenbrandt et al. 2010`,
/// *StarDisplay* Eq. 9) keeping long-running murmurations cohesive above the sleeping site without
/// sub-flock fission or unbounded Hookean straggler slingshotting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RoostBoundaryConfig {
    /// Computes pairwise comfort-zone separation (`d_eff < rep_shell_m`) and saturating cohesion
    /// (`d_eff > coh_shell_m`) over the symmetric weighted graph `graph.row_edges(i)`, preserving
    /// exact pairwise antisymmetry (`sum_i F_i^sym = 0`) with zero rear-cone blind-spot fission or collapse.
    pub symmetric_saturating_cohesion: bool,
    /// Short-range reciprocal separation radius `r_rep` (`m` in anisotropic `d_eff` space).
    pub rep_shell_m: f32,
    /// Topological shell attraction onset `r_coh >= r_rep` (`m` in anisotropic `d_eff` space).
    pub coh_shell_m: f32,
    /// Maximum effective spring displacement `r_sat` (`m`) in `excess / (1 + excess / r_sat)`.
    pub coh_sat_m: f32,
    /// Horizontal roost comfort radius `R_roost` (`m`; `0.0` disables roost return).
    pub roost_radius_m: f32,
    /// Maximum horizontal inward-turning roost acceleration (`m/s^2`).
    pub roost_pull_mps2: f32,
    /// Preferred vertical altitude half-band `H_roost` (`m`) around `z = 0`.
    pub roost_alt_band_m: f32,
    /// Maximum vertical altitude-restoring acceleration (`m/s^2`) toward `z = 0`.
    pub roost_vertical_pull_mps2: f32,
}

/// Stage 2 candidate coupled speed-heading SDE configuration (§2.2).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoupledConfig {
    pub dt: f32,
    pub master_seed: u64,
    /// Alignment coupling `J_SPP` (`s^-1`) in `F_i^align = J_SPP * sum_j w_ij (v_j - v_i)`.
    pub j_spp: f32,
    /// Longitudinal speed diffusion `D_v` (`m^2/s^3`).
    pub d_v: f32,
    /// Transverse angular diffusion `D_omega` (`m^2/s^3`; equals `D_v` when matching isotropic Cartesian noise).
    pub d_omega: f32,
    /// Angular relaxation timescale `tau_omega = chi / eta_ISM >= 0.0` (`s`).
    /// `tau_omega == 0.0` executes the exact polar overdamped limit; `tau_omega > 0.0` uses exact bivariate OU.
    pub tau_omega: f32,
    /// Regularization floor `v_eps` (`m/s`, default `0.25`) for `1 / max(v_i, v_eps)` divisions; all hits are logged.
    pub v_eps: f32,
    /// Includes the mandatory Ito spherical Jacobian drift `+2 * D_v / v_i^* * dt` in `dv_i` (§2.2).
    /// Exposed as a boolean strictly for Stage 2 ablation tests proving its necessity.
    pub include_ito_speed_drift: bool,
    pub gravity_mps2: f32,
    pub speed_control: SpeedControlLaw,
    pub neighbour_rule: NeighbourRule,
    pub graph_symmetry: GraphSymmetry,
    pub positional: PositionalInteractionConfig,
    pub turn_pulse: Option<ExternalTurnPulse>,
    #[serde(default)]
    pub roost: RoostBoundaryConfig,
}

/// Stage 2 coupled flock state `(x_i, v_i, u_i, omega_i)` with `Mode K` curvature+gravity bank visualization.
#[derive(Clone, Debug)]
pub struct CoupledFlock {
    pub tick: u64,
    pub pos: Vec<Vec3>,
    pub speed: Vec<f32>,
    pub heading: Vec<Vec3>,
    pub omega: Vec<Vec3>,
    /// Instantaneous transverse trajectory acceleration `a_{i,perp} = v_i * (omega_i x u_i)` (`200 Hz` `E7`).
    pub accel_perp: Vec<Vec3>,
    /// `Mode K` inferred horizon bank angle `phi_bank` (`rad`).
    pub bank_angle_rad: Vec<f32>,
    pub right_ref: Vec<Vec3>,
    /// Telemetry: number of bird-ticks where `speed[i] <= v_eps` regularization floor was active.
    pub v_eps_floor_events: u64,
    /// Telemetry: artificial hard speed clamp count — strictly `0` in Stage 2 (§2.3 Rule 3).
    pub speed_clamp_events: u64,
}

impl CoupledFlock {
    pub fn from_cartesian(pos: Vec<Vec3>, vel: Vec<Vec3>) -> Self {
        let n = pos.len();
        assert_eq!(n, vel.len());
        let mut speed = Vec::with_capacity(n);
        let mut heading = Vec::with_capacity(n);
        let mut right_ref = Vec::with_capacity(n);

        for &v in &vel {
            let s = v.norm();
            let u = if s > 1e-12 { v / s } else { Vec3::EX };
            let r0 = u.cross(Vec3::EZ).normalized_or(Vec3::new(0.0, -1.0, 0.0));
            speed.push(s);
            heading.push(u);
            right_ref.push(r0);
        }

        Self {
            tick: 0,
            pos,
            speed,
            heading,
            omega: vec![Vec3::ZERO; n],
            accel_perp: vec![Vec3::ZERO; n],
            bank_angle_rad: vec![0.0; n],
            right_ref,
            v_eps_floor_events: 0,
            speed_clamp_events: 0,
        }
    }

    pub fn velocities(&self) -> Vec<Vec3> {
        self.heading
            .iter()
            .zip(self.speed.iter())
            .map(|(&u, &s)| u * s)
            .collect()
    }

    pub fn step(
        &mut self,
        cfg: &CoupledConfig,
        fixed_neighbours: Option<&[Vec<NeighbourEntry>]>,
    ) -> WeightedGraph {
        let n = self.pos.len();
        let dt = cfg.dt;
        let t_now = (self.tick as f32) * dt;
        let vel_curr = self.velocities();

        let flat_nb: FlatNeighbourTable = match fixed_neighbours {
            Some(nb) => FlatNeighbourTable::from_vecs(nb),
            None => {
                let lbvh = LbvhIndex::build(&self.pos);
                lbvh.query_all_flat(&self.pos, &self.heading, cfg.neighbour_rule)
            }
        };
        let graph = WeightedGraph::from_flat_neighbours(&flat_nb, cfg.graph_symmetry);

        let sqrt_2dv_dt = libm::sqrtf(2.0 * cfg.d_v * dt);

        let mut next_pos = Vec::with_capacity(n);
        let mut next_speed = Vec::with_capacity(n);
        let mut next_heading = Vec::with_capacity(n);
        let mut next_omega = Vec::with_capacity(n);
        let mut next_accel_perp = Vec::with_capacity(n);
        let mut next_bank = Vec::with_capacity(n);
        let mut next_right = Vec::with_capacity(n);

        for i in 0..n {
            let xi = self.pos[i];
            let vi_scalar = self.speed[i];
            let ui = self.heading[i];
            let vi_vec = vel_curr[i];
            let v_star = if vi_scalar <= cfg.v_eps {
                self.v_eps_floor_events += 1;
                cfg.v_eps.max(1e-4)
            } else {
                vi_scalar
            };

            let (cols, ws) = graph.row_edges(i);
            let mut f_align = Vec3::ZERO;
            let mut w_row_sum = 0.0f32;
            let mut align_v_dot_u = 0.0f32;
            for (&j, &w) in cols.iter().zip(ws.iter()) {
                let vj = vel_curr[j as usize];
                f_align += (vj - vi_vec) * (cfg.j_spp * w);
                w_row_sum += w;
                align_v_dot_u += w * vj.dot(ui);
            }
            let mut f_pos_ext = if cfg.roost.symmetric_saturating_cohesion {
                let mut rep_only = cfg.positional;
                rep_only.k_coh = 0.0;
                // Keep StarEscape n_rep=1 nearest-neighbour directional kick + reciprocal comfort-zone shell
                let mut f_p = compute_positional_force(i, &self.pos, flat_nb.row(i), &rep_only) * 0.5;
                let gz = cfg.positional.vertical_coh_gain;
                let aniso = (gz - 1.0).abs() > 1e-6;
                let r_rep = cfg.roost.rep_shell_m.max(cfg.positional.r0_m);
                let r_coh = cfg.roost.coh_shell_m.max(r_rep);
                let r_sat = cfg.roost.coh_sat_m;
                let mut coh_sum = Vec3::ZERO;
                let mut rep_sum = Vec3::ZERO;
                for (&j, &w) in cols.iter().zip(ws.iter()) {
                    let diff = self.pos[j as usize] - xi;
                    let de = if aniso {
                        Vec3::new(diff.x, diff.y, diff.z * gz)
                    } else {
                        diff
                    };
                    let d_eff = de.norm();
                    if d_eff > r_coh && cfg.positional.k_coh > 0.0 {
                        let excess = d_eff - r_coh;
                        let eff_excess = if r_sat > 0.0 {
                            excess / (1.0 + excess / r_sat)
                        } else {
                            excess
                        };
                        let mut spring = de * (eff_excess / d_eff);
                        if aniso {
                            spring.z *= gz;
                        }
                        coh_sum += spring * w;
                    } else if d_eff < r_rep && cfg.positional.k_rep > 0.0 {
                        let frac = (r_rep - d_eff) / r_rep.max(1e-4);
                        let core_boost = 1.0 + r_rep / d_eff.max(0.40);
                        let mag = frac * core_boost;
                        let mut push = if d_eff > 1e-5 {
                            de * (-mag / d_eff)
                        } else {
                            Vec3::EX * mag
                        };
                        if aniso {
                            push.z *= gz;
                        }
                        rep_sum += push * w;
                    }
                }
                f_p += coh_sum * cfg.positional.k_coh + rep_sum * cfg.positional.k_rep;
                f_p
            } else {
                compute_positional_force(i, &self.pos, flat_nb.row(i), &cfg.positional)
            };

            if cfg.roost.roost_radius_m > 0.0 && cfg.roost.roost_pull_mps2 > 0.0 {
                let r_h = libm::sqrtf(xi.x * xi.x + xi.y * xi.y);
                let r_free = 0.25 * cfg.roost.roost_radius_m;
                if r_h > r_free && r_h > 1e-4 {
                    let inv_rh = 1.0 / r_h;
                    // Outward radial unit vector r_hat_h and inward unit vector d_in = -r_hat_h
                    let r_out = Vec3::new(xi.x * inv_rh, xi.y * inv_rh, 0.0);
                    let d_in = -r_out;
                    let q = (r_h - r_free) / (0.75 * cfg.roost.roost_radius_m).max(1.0);
                    let gain = (q * q) / (1.0 + q * q);
                    // StarDisplay (Hildenbrandt et al. 2010, Eq. 9): turn urgency grows as sin(theta/2) =
                    // sqrt((1 - u . d_in)/2), so it is 0 when already flying inward across the roost and reaches
                    // full lateral bank (1.0, zero 180-deg dead-spot!) when heading straight outward.
                    let cos_in = ui.dot(d_in).clamp(-1.0, 1.0);
                    let turn_urgency = libm::sqrtf(0.5 * (1.0 - cos_in));
                    let lat_ccw = Vec3::new(-ui.y, ui.x, 0.0).normalized_or(Vec3::EY);
                    let sin_in = lat_ccw.dot(d_in);
                    let turn_dir = if sin_in < -0.15 { -lat_ccw } else { lat_ccw };
                    f_pos_ext += (turn_dir * 0.90 + d_in * 0.10)
                        * (cfg.roost.roost_pull_mps2 * gain * turn_urgency);
                }
            }
            if cfg.roost.roost_alt_band_m > 0.0 && cfg.roost.roost_vertical_pull_mps2 > 0.0 {
                let qz = xi.z / cfg.roost.roost_alt_band_m;
                f_pos_ext.z += -cfg.roost.roost_vertical_pull_mps2 * (qz / (1.0 + qz.abs()));
            }
            if let Some(pulse) = cfg.turn_pulse {
                if i < pulse.initiator_ids && t_now >= pulse.start_time_s && t_now <= pulse.end_time_s {
                    f_pos_ext += pulse.accel_vec;
                }
            }
            let f_tot = f_align + f_pos_ext;
            let f_par_pos_ext = f_pos_ext.dot(ui);
            let f_perp = f_tot - ui * f_tot.dot(ui);

            let ito_drift = if cfg.include_ito_speed_drift {
                (2.0 * cfg.d_v) / v_star
            } else {
                0.0
            };

            // Primary isotropic 3D Wiener increment `Z_1` shared with `Ref-MarginalSPP` stream (`STREAM_WIENER`)
            let mut rng1 = keyed_rng(cfg.master_seed, STREAM_WIENER, self.tick, i as u32);
            let z1 = normal_vec3(&mut rng1);
            let z_par = z1.dot(ui);
            let z_rot1 = ui.cross(z1); // Orthogonal to `ui`, unit variance per tangent axis

            // 1. Longitudinal speed update (`dv_i`) via drift-implicit Euler-Maruyama on the 1D scalar speed SDE:
            //    unconditionally L-stable for superlinear octic (`-v^7`) and stiff `lambda -> inf` / `J_SPP` regimes
            //    with ZERO artificial speed clamps (`Hutzenthaler, Jentzen & Kloeden 2011`).
            let rhs_v = (vi_scalar as f64)
                + ((f_par_pos_ext + ito_drift) as f64) * (dt as f64)
                + ((sqrt_2dv_dt * z_par) as f64);
            let v_new_raw = solve_1d_speed_step(
                vi_scalar as f64,
                rhs_v,
                dt as f64,
                cfg.j_spp as f64,
                w_row_sum as f64,
                align_v_dot_u as f64,
                cfg.speed_control,
            ) as f32;
            // Reflect speed if discrete Euler step ever crosses 0 (polar radial reflection `|-v|`), logging floor hit.
            // Note: `d(-u)/dt = omega x (-u)`, so axial angular velocity `omega` retains its sign under `u -> -u`.
            let (v_new, u_base, omega_prev) = if v_new_raw >= 0.0 {
                (v_new_raw, ui, self.omega[i])
            } else {
                self.v_eps_floor_events += 1;
                (-v_new_raw, -ui, self.omega[i])
            };

            // 2. Transverse heading & angular-velocity update via unconditional Exact Bivariate OU
            let omega_inf = u_base.cross(f_perp) * (1.0 / v_star);
            let (dtheta, omega_end, a_perp_inst) = if cfg.tau_omega <= 1e-9 {
                // Exact polar overdamped limit (`tau_omega == 0`)
                let dtheta_det = omega_inf * dt;
                let dtheta_stoch = z_rot1 * (libm::sqrtf(2.0 * cfg.d_omega * dt) / v_star);
                let dtheta_tot = dtheta_det + dtheta_stoch;
                let omega_eff = if dt > 1e-12 { dtheta_tot / dt } else { omega_inf };
                let a_perp = omega_inf.cross(u_base) * vi_scalar;
                (dtheta_tot, omega_eff, a_perp)
            } else {
                let tau = cfg.tau_omega;
                let x = dt / tau;
                // Avoid catastrophic `f32` cancellation in `1 - exp(-x)` for `x < 0.05`
                let (alpha, one_minus_alpha) = if x < 0.05 {
                    let x2 = x * x;
                    let oma = x * (1.0 - 0.5 * x + (1.0 / 6.0) * x2 - (1.0 / 24.0) * x2 * x);
                    (1.0 - oma, oma)
                } else if x > 80.0 {
                    (0.0, 1.0)
                } else {
                    let a = libm::expf(-x);
                    (a, 1.0 - a)
                };

                // Symplectic exponential-Euler update: advance `omega_det` FIRST (`det = alpha = exp(-dt/tau) <= 1`),
                // then rotate heading by `omega_det * dt`. Preserves exact wave energy as `tau -> inf` (`Ref-ISM`)
                // AND reduces continuously to `omega_inf * dt` as `tau -> 0^+` (`Ref-MarginalSPP`).
                let omega_det = omega_prev * alpha + omega_inf * one_minus_alpha;
                let dtheta_det = omega_det * dt;

                let (dtheta_stoch, omega_stoch) = if cfg.d_omega > 0.0 {
                    let mut rng2 = keyed_rng(cfg.master_seed, STREAM_WIENER_OU_AUX, self.tick, i as u32);
                    let z2 = u_base.cross(normal_vec3(&mut rng2));
                    sample_bivariate_ou_tangent(
                        dt,
                        tau,
                        x,
                        alpha,
                        one_minus_alpha,
                        cfg.d_omega,
                        v_star,
                        z_rot1,
                        z2,
                    )
                } else {
                    (Vec3::ZERO, Vec3::ZERO)
                };

                let omega_tot = omega_det + omega_stoch;
                let a_perp = omega_tot.cross(u_base) * vi_scalar;
                (dtheta_det + dtheta_stoch, omega_tot, a_perp)
            };

            let u_new = rodrigues_rotate(u_base, dtheta).normalized_or(Vec3::EX);
            // Parallel-transport / tangent-project `omega_end` onto `T_{u_new} S^2`
            let omega_new = rodrigues_rotate(omega_end, dtheta).project_perp(u_new);

            let v_vec_new = u_new * v_new;
            let x_new = xi + (vi_vec + v_vec_new) * (0.5 * dt);

            let (r_ref, _, phi_bank) =
                compute_mode_k_bank_angle(u_new, a_perp_inst, cfg.gravity_mps2, self.right_ref[i]);

            next_pos.push(x_new);
            next_speed.push(v_new);
            next_heading.push(u_new);
            next_omega.push(omega_new);
            next_accel_perp.push(a_perp_inst);
            next_bank.push(phi_bank);
            next_right.push(r_ref);
        }

        self.pos = next_pos;
        self.speed = next_speed;
        self.heading = next_heading;
        self.omega = next_omega;
        self.accel_perp = next_accel_perp;
        self.bank_angle_rad = next_bank;
        self.right_ref = next_right;
        self.tick += 1;
        graph
    }
}

/// Exact Cholesky sampling of `(A, B) = (int_0^dt omega_stoch(s) ds, omega_stoch(dt))` on `T_u S^2`.
/// Ordering `A` first guarantees `A -> sqrt(2 D_omega dt) / v^* * z_rot1` smoothly as `tau_omega -> 0^+`.
#[allow(clippy::too_many_arguments)]
fn sample_bivariate_ou_tangent(
    dt: f32,
    tau: f32,
    x: f32,
    alpha: f32,
    one_minus_alpha: f32,
    d_omega: f32,
    v_star: f32,
    z_rot1: Vec3,
    z_rot2: Vec3,
) -> (Vec3, Vec3) {
    let inv_v2 = 1.0 / (v_star * v_star);

    // `int_0^x (1 - e^-r)^2 dr`: Taylor series for `x < 0.02` avoids catastrophic cancellation
    let int_disp = if x < 0.02 {
        let x2 = x * x;
        let x3 = x2 * x;
        x3 * ((1.0 / 3.0) - 0.25 * x + (7.0 / 60.0) * x2)
    } else {
        (x - 0.5 * one_minus_alpha * (3.0 - alpha)).max(0.0)
    };
    let var_a = (2.0 * d_omega * tau * inv_v2) * int_disp;

    let one_minus_alpha_sq = if x < 0.01 {
        let two_x = 2.0 * x;
        two_x * (1.0 - x + (1.0 / 3.0) * two_x * x)
    } else {
        (1.0 - alpha * alpha).max(0.0)
    };
    let var_b = (d_omega * inv_v2 / tau) * one_minus_alpha_sq;
    let cov_ab = (d_omega * inv_v2) * (one_minus_alpha * one_minus_alpha);

    let sigma_a = libm::sqrtf(var_a.max(0.0));
    if sigma_a <= 1e-16 {
        let _ = dt;
        return (Vec3::ZERO, z_rot2 * libm::sqrtf(var_b.max(0.0)));
    }
    let c_b1 = cov_ab / sigma_a;
    let rem_var_b = (var_b - c_b1 * c_b1).max(0.0);
    let c_b2 = libm::sqrtf(rem_var_b);

    let dtheta_stoch = z_rot1 * sigma_a;
    let omega_stoch = z_rot1 * c_b1 + z_rot2 * c_b2;
    (dtheta_stoch, omega_stoch)
}

/// Solves `v - theta * dt * g(v) = rhs + (1 - theta) * dt * g(v_init)` in `f64`
/// (`theta = 0.5` second-order trapezoidal in smooth regimes; smoothly increasing toward `1.0`
/// L-stable backward Euler when `dt * |g'(v_init)| > 1`), with zero artificial bounds on `v`.
fn solve_1d_speed_step(
    v_init: f64,
    rhs: f64,
    dt: f64,
    j_spp: f64,
    w_sum: f64,
    align_v_dot_u: f64,
    law: SpeedControlLaw,
) -> f64 {
    let lin_damp = j_spp * w_sum;

    match law {
        SpeedControlLaw::LinearGaussian { g_lin, v0 } => {
            let g = g_lin as f64;
            let v0_d = v0 as f64;
            let k_tot = lin_damp + g;
            let s = dt * k_tot;
            let theta = if s <= 0.5 { 0.0 } else { (s - 0.5) / s };
            let g0 = j_spp * (align_v_dot_u - w_sum * v_init) - g * (v_init - v0_d);
            if theta == 0.0 {
                return rhs + dt * g0;
            }
            let num = rhs + (1.0 - theta) * dt * g0 + theta * dt * (j_spp * align_v_dot_u + g * v0_d);
            num / (1.0 + theta * dt * k_tot)
        }
        SpeedControlLaw::MarginalOctic { lambda, v0 } => {
            let lam = lambda as f64;
            let v0_d = v0 as f64;
            let v0_sq = v0_d * v0_d;
            let c8 = if lam > 0.0 {
                (8.0 * lam) / (v0_sq * v0_sq * v0_sq)
            } else {
                0.0
            };
            let g_eval = |v: f64| -> f64 {
                let d = v0_sq - v * v;
                j_spp * (align_v_dot_u - w_sum * v) + c8 * v * (d * d * d)
            };
            let neg_dg = |v: f64| -> f64 {
                let v2 = v * v;
                let d = v0_sq - v2;
                lin_damp - c8 * (d * d) * (v0_sq - 7.0 * v2)
            };

            let s = (dt * neg_dg(v_init).abs()).max(dt * lin_damp);
            if s <= 0.5 {
                let v_exp = rhs + dt * g_eval(v_init);
                // Guard against a large stochastic jump landing far in the stiff `v^7` tail:
                if dt * neg_dg(v_exp).abs() <= 0.5 {
                    return v_exp;
                }
            }
            let s_eff = s.max(1.0);
            let theta = 1.0 - 0.5 / s_eff;
            if lam <= 0.0 {
                let g0 = j_spp * (align_v_dot_u - w_sum * v_init);
                return (rhs + (1.0 - theta) * dt * g0 + theta * dt * j_spp * align_v_dot_u)
                    / (1.0 + theta * dt * lin_damp);
            }

            let rhs_tot = rhs + (1.0 - theta) * dt * g_eval(v_init);
            let dt_th = theta * dt;

            let phi = |v: f64| -> f64 { v - dt_th * g_eval(v) - rhs_tot };
            let dphi = |v: f64| -> f64 { 1.0 + dt_th * neg_dg(v) };

            let scale = v0_d.abs().max(v_init.abs()).max(rhs_tot.abs()).max(1.0);
            let mut span = scale * 2.0;
            while phi(-span) > 0.0 || phi(span) < 0.0 {
                span *= 2.0;
                if span > 1e12 {
                    break;
                }
            }
            let mut lo = -span;
            let mut hi = span;
            let mut v = v_init.clamp(lo, hi);

            for _ in 0..12 {
                let fv = phi(v);
                if fv.abs() <= 1e-11 {
                    return v;
                }
                if fv < 0.0 {
                    lo = v;
                } else {
                    hi = v;
                }
                let dfv = dphi(v);
                let cand = if dfv > 1e-6 { v - fv / dfv } else { lo - 1.0 };
                if cand > lo && cand < hi {
                    v = cand;
                } else {
                    v = 0.5 * (lo + hi);
                }
            }
            v
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{MarginalSppConfig, MarginalSppFlock};
    use crate::rng::STREAM_INIT;

    #[test]
    fn coupled_deterministic_overdamped_limit_matches_ref_marginal_spp() {
        let n = 48usize;
        let mut p0 = Vec::with_capacity(n);
        let mut v0 = Vec::with_capacity(n);
        for i in 0..n {
            let mut rng = keyed_rng(1234, STREAM_INIT, 0, i as u32);
            p0.push(normal_vec3(&mut rng) * 3.0);
            v0.push(Vec3::EX * 11.5 + normal_vec3(&mut rng) * 1.8);
        }

        let spp_cfg = MarginalSppConfig {
            dt: 0.002,
            master_seed: 7,
            j_spp: 3.5,
            d_v: 0.0,
            speed_control: SpeedControlLaw::MarginalOctic {
                lambda: 0.9,
                v0: 12.0,
            },
            neighbour_rule: NeighbourRule::ExactTopoK {
                k: 7,
                blind_cos: -0.5,
            },
            graph_symmetry: GraphSymmetry::SymmetricMean,
            positional: PositionalInteractionConfig {
                k_coh: 0.6,
                r0_m: 1.1,
                k_rep: 1.0,
                n_rep: 1,
                vertical_coh_gain: 1.0,
            },
        };
        let mut ref_flock = MarginalSppFlock::new(p0.clone(), v0.clone());
        for _ in 0..500 {
            ref_flock.step(&spp_cfg, None);
        }

        let mut prev_err = f32::INFINITY;
        for &tau in &[0.08f32, 0.02f32, 0.002f32, 0.0f32] {
            let c_cfg = CoupledConfig {
                dt: 0.002,
                master_seed: 7,
                j_spp: 3.5,
                d_v: 0.0,
                d_omega: 0.0,
                tau_omega: tau,
                v_eps: 0.25,
                include_ito_speed_drift: true,
                gravity_mps2: 9.81,
                speed_control: spp_cfg.speed_control,
                neighbour_rule: spp_cfg.neighbour_rule,
                graph_symmetry: spp_cfg.graph_symmetry,
                positional: spp_cfg.positional,
                turn_pulse: None,
                roost: RoostBoundaryConfig::default(),
            };
            let mut c_flock = CoupledFlock::from_cartesian(p0.clone(), v0.clone());
            for _ in 0..500 {
                c_flock.step(&c_cfg, None);
            }
            let max_err = c_flock
                .pos
                .iter()
                .zip(ref_flock.pos.iter())
                .map(|(&a, &b)| (a - b).norm())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < prev_err,
                "Expected monotonic convergence as tau_omega -> 0; tau={tau}, err={max_err}, prev={prev_err}"
            );
            prev_err = max_err;
            if tau == 0.0 {
                assert!(
                    max_err < 3e-3,
                    "At tau_omega = 0, coupled polar integrator must match Cartesian Ref-MarginalSPP to O(dt) truncation; got {max_err}"
                );
            }
        }
    }

    #[test]
    fn coupled_ito_speed_drift_matches_cartesian_marginal_spp_and_omission_biases_low() {
        use crate::eval::{wasserstein_1d, SpeedAccumulator};

        let n = 12usize;
        let dt = 0.005f32;
        let burn_steps = 400usize;
        let meas_steps = 600usize;
        let pos_cfg = PositionalInteractionConfig {
            k_coh: 1.0,
            r0_m: 1.0,
            k_rep: 1.4,
            n_rep: 1,
            vertical_coh_gain: 1.0,
        };
        let speed_law = SpeedControlLaw::LinearGaussian {
            g_lin: 0.25,
            v0: 12.0,
        };
        let nb_rule = NeighbourRule::ExactTopoK {
            k: 7,
            blind_cos: -0.5,
        };

        let mut ref_acc = SpeedAccumulator::new(n);
        let mut ito_zero_acc = SpeedAccumulator::new(n);
        let mut ito_tiny_acc = SpeedAccumulator::new(n);
        let mut no_ito_acc = SpeedAccumulator::new(n);

        for seed in 0..8u64 {
            let mut p0 = Vec::with_capacity(n);
            let mut v0 = Vec::with_capacity(n);
            for i in 0..n {
                let mut rng = keyed_rng(9100 + seed, STREAM_INIT, 0, i as u32);
                p0.push(normal_vec3(&mut rng) * 1.5);
                v0.push(Vec3::EX * 12.0 + normal_vec3(&mut rng) * 0.6);
            }

            let m_cfg = MarginalSppConfig {
                dt,
                master_seed: 33_000 + seed,
                j_spp: 4.0,
                d_v: 3.5,
                speed_control: speed_law,
                neighbour_rule: nb_rule,
                graph_symmetry: GraphSymmetry::SymmetricMean,
                positional: pos_cfg,
            };
            let mut m_flock = MarginalSppFlock::new(p0.clone(), v0.clone());
            for _ in 0..burn_steps {
                m_flock.step(&m_cfg, None);
            }
            for k in 0..meas_steps {
                m_flock.step(&m_cfg, None);
                if k % 10 == 0 {
                    ref_acc.record(&m_flock.vel);
                }
            }

            let make_c_cfg = |tau_omega: f32, include_ito: bool| CoupledConfig {
                dt,
                master_seed: 33_000 + seed,
                j_spp: 4.0,
                d_v: 3.5,
                d_omega: 3.5,
                tau_omega,
                v_eps: 0.25,
                include_ito_speed_drift: include_ito,
                gravity_mps2: 9.81,
                speed_control: speed_law,
                neighbour_rule: nb_rule,
                graph_symmetry: GraphSymmetry::SymmetricMean,
                positional: pos_cfg,
                turn_pulse: None,
                roost: RoostBoundaryConfig::default(),
            };

            for (acc, cfg) in [
                (&mut ito_zero_acc, make_c_cfg(0.0, true)),
                (&mut ito_tiny_acc, make_c_cfg(1e-4, true)),
                (&mut no_ito_acc, make_c_cfg(0.0, false)),
            ] {
                let mut c_flock = CoupledFlock::from_cartesian(p0.clone(), v0.clone());
                for _ in 0..burn_steps {
                    c_flock.step(&cfg, None);
                }
                for k in 0..meas_steps {
                    c_flock.step(&cfg, None);
                    if k % 10 == 0 {
                        acc.record(&c_flock.velocities());
                    }
                }
            }
        }

        let w1_zero = wasserstein_1d(ref_acc.samples(), ito_zero_acc.samples());
        let w1_tiny = wasserstein_1d(ref_acc.samples(), ito_tiny_acc.samples());
        let w1_no_ito = wasserstein_1d(ref_acc.samples(), no_ito_acc.samples());

        let mean_ref = ref_acc.summarize(0).mean_individual_mps;
        let mean_no_ito = no_ito_acc.summarize(0).mean_individual_mps;

        assert!(
            w1_zero < 0.08 && w1_tiny < 0.12,
            "With Ito +2D_v/v drift, polar CoupledFlock must match Cartesian Ref-MarginalSPP: w1_zero={w1_zero}, w1_tiny={w1_tiny}"
        );
        assert!(
            mean_ref - mean_no_ito > 0.45 && w1_no_ito > 0.45,
            "Omitting Ito +2D_v/v drift must bias speed systematically low: mean_ref={mean_ref}, mean_no_ito={mean_no_ito}, w1_no_ito={w1_no_ito}"
        );
    }

    #[test]
    fn coupled_stiff_speed_reproduces_ref_ism_spin_wave_dispersion() {
        use crate::eval::TurnWaveRecorder;

        let mut ribbon_pos = Vec::with_capacity(360);
        for ix in 0..40 {
            for iy in 0..3 {
                for iz in 0..3 {
                    ribbon_pos.push(Vec3::new(ix as f32, (iy as f32) - 1.0, (iz as f32) - 1.0));
                }
            }
        }
        let n_rib = ribbon_pos.len();
        let v0 = 12.0f32;
        let dt = 0.005f32;
        let init_u = vec![Vec3::EY; n_rib];
        let init_vel: Vec<Vec3> = init_u.iter().map(|&u| u * v0).collect();
        let rule = NeighbourRule::ExactTopoK {
            k: 7,
            blind_cos: -1.0,
        };
        let frozen_nb = LbvhIndex::build(&ribbon_pos).query_all(&ribbon_pos, &init_u, rule);
        let sym_graph = WeightedGraph::from_neighbours(&frozen_nb, GraphSymmetry::SymmetricMean);
        let m2_ex = sym_graph.second_spatial_moment(&ribbon_pos, Vec3::EX);

        let j_spp = 6000.0f32;
        let tau_omega = 5.0f32;
        // Section 2.2 Eq.: v0 cancels in F_align / v0, so J_ISM / chi = J_SPP / tau_omega = 1200 s^-2
        // and c_s = sqrt((J_SPP * M_2) / (2 * tau_omega))
        let pred_cs = libm::sqrtf((0.5 * j_spp * m2_ex) / tau_omega);

        let cfg = CoupledConfig {
            dt,
            master_seed: 99,
            j_spp,
            d_v: 0.0,
            d_omega: 0.0,
            tau_omega,
            v_eps: 0.25,
            include_ito_speed_drift: true,
            gravity_mps2: 9.81,
            speed_control: SpeedControlLaw::MarginalOctic {
                lambda: 20.0,
                v0,
            },
            neighbour_rule: rule,
            graph_symmetry: GraphSymmetry::SymmetricMean,
            positional: PositionalInteractionConfig::default(),
            turn_pulse: Some(ExternalTurnPulse {
                start_time_s: 0.02,
                end_time_s: 0.06,
                accel_vec: Vec3::new(-12.0, 0.0, 0.0),
                initiator_ids: 9,
            }),
            roost: RoostBoundaryConfig::default(),
        };

        let mut flock = CoupledFlock::from_cartesian(ribbon_pos.clone(), init_vel);
        let mut rec = TurnWaveRecorder::new(ribbon_pos, 9, Vec3::EX, dt);
        for _ in 0..400 {
            flock.step(&cfg, Some(&frozen_nb));
            rec.record_tick(&flock.accel_perp);
        }
        let summary = rec.analyze(pred_cs);
        let rel_err = (summary.measured_cs_mps - pred_cs).abs() / pred_cs;
        assert!(
            summary.linear_r_squared > 0.99 && rel_err < 0.08,
            "Coupled stiff-speed wave dispersion must match Ref-ISM formula: measured={}, pred={}, R^2={}",
            summary.measured_cs_mps,
            pred_cs,
            summary.linear_r_squared
        );
    }

    /// Regression test preventing long-horizon sub-flock fission (`4.44 km` dispersion) and density collapse.
    /// Full `N = 1,000`, `120,000`-tick (`600 s`) verification settled at `rms_r = 14.17 m`, `|c| = 82.32 m`,
    /// `speed = [11.73..13.01] m/s`, `clamps = 0`; default suite runs `N = 300` for `24,000` ticks (`120 s`) in ~4s.
    #[test]
    fn coupled_roost_murmuration_stays_cohesive_over_long_horizon() {
        use crate::instance::generate_stress_cloud;
        use crate::parity::default_coupled_browser_config;

        let n = 300usize;
        let seed = 20_261_110u64;
        let cfg = default_coupled_browser_config(seed);
        let (p0, u0) = generate_stress_cloud(0, n, seed);
        let v0: Vec<Vec3> = u0.into_iter().map(|u| u * 12.0).collect();
        let mut flock = CoupledFlock::from_cartesian(p0, v0);

        for step_idx in 1..=24_000usize {
            flock.step(&cfg, None);
            if step_idx % 8_000 == 0 {
                let inv_n = 1.0 / (n as f32);
                let c = flock.pos.iter().copied().fold(Vec3::ZERO, |a, b| a + b) * inv_n;
                let rms_r = libm::sqrtf(
                    flock.pos.iter().map(|&p| (p - c).norm_sq()).sum::<f32>() * inv_n,
                );
                let max_spd = flock.speed.iter().copied().fold(0.0f32, f32::max);
                let min_spd = flock.speed.iter().copied().fold(f32::INFINITY, f32::min);
                assert!(
                    rms_r > 7.5 && rms_r < 85.0,
                    "Flock out of realistic volume envelope at tick {step_idx}: rms_r = {rms_r:.2} m (expected 7.5..85 m)"
                );
                assert!(
                    c.norm() < 105.0,
                    "Flock drifted off roost at tick {step_idx}: |c| = {:.2} m",
                    c.norm()
                );
                assert!(
                    max_spd < 19.5 && min_spd > 5.0,
                    "Unphysical speed slingshot at tick {step_idx}: [{min_spd:.2}..{max_spd:.2}] m/s"
                );
                assert_eq!(flock.speed_clamp_events, 0);
            }
        }
    }
}
