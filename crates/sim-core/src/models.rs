use crate::math::{rodrigues_rotate_fixed_speed, Vec3};
use crate::neighbour::{GraphSymmetry, LbvhIndex, NeighbourEntry, NeighbourRule, WeightedGraph};
use crate::rng::{keyed_rng, normal_vec3, STREAM_WIENER};
use serde::{Deserialize, Serialize};

/// Optional soft positional cohesion/repulsion used in cohesive flock scenarios (zeroed in pure spin/speed tests).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PositionalInteractionConfig {
    /// Attraction weight toward topological shell (`0.0` disables cohesion).
    pub k_coh: f32,
    /// Equilibrium spacing `r_0` (m) where pair spring crosses zero (`F = k * (r - r_0) * r_hat`).
    pub r0_m: f32,
    /// Repulsion strength (`0.0` disables repulsion).
    pub k_rep: f32,
    /// Repulsion neighbour count (`1` for StarEscape nearest-neighbour avoidance; `7` for full shell).
    pub n_rep: usize,
    /// Vertical (`z`) cohesion multiplier (`1.0` = isotropic; `> 1.0` = dorsoventral slab flattening in Exp 3).
    #[serde(default = "default_vertical_coh_gain")]
    pub vertical_coh_gain: f32,
}

fn default_vertical_coh_gain() -> f32 {
    1.0
}

impl Default for PositionalInteractionConfig {
    fn default() -> Self {
        Self {
            k_coh: 0.0,
            r0_m: 1.0,
            k_rep: 0.0,
            n_rep: 1,
            vertical_coh_gain: 1.0,
        }
    }
}

/// Optional deterministic external turn ramp applied to a subset of initiator birds (`Experiment 1`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExternalTurnPulse {
    pub start_time_s: f32,
    pub end_time_s: f32,
    /// Transverse acceleration vector (`m/s^2`) requested on initiator birds (`id < n_initiators` or spatial slab).
    pub accel_vec: Vec3,
    pub initiator_ids: usize,
}

/// Constant-speed turning family (`Experiment 1`: `Ref-ISM` vs `OverdampedVicsek`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "turning_law", rename_all = "snake_case")]
pub enum ConstantSpeedTurningLaw {
    /// Exact Cavagna et al. (2015) Inertial Spin Model (`arxiv:1403.1202`, §2.1A).
    RefIsm {
        j_ism: f32,
        chi: f32,
        eta_ism: f32,
        temp_ism: f32,
    },
    /// First-order constant-speed continuous-time Vicsek model (matched comparator for `Exp 1`).
    OverdampedVicsek {
        /// Alignment frequency (`J_ISM / eta_ISM` in the overdamped limit of ISM).
        j_vicsek: f32,
        /// Rotational diffusion coefficient (`T_ISM / eta_ISM` in the overdamped limit of ISM).
        d_rot: f32,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstantSpeedConfig {
    pub v0: f32,
    pub dt: f32,
    pub master_seed: u64,
    pub neighbour_rule: NeighbourRule,
    pub graph_symmetry: GraphSymmetry,
    pub turning: ConstantSpeedTurningLaw,
    pub positional: PositionalInteractionConfig,
    pub turn_pulse: Option<ExternalTurnPulse>,
}

#[derive(Clone, Debug)]
pub struct ConstantSpeedFlock {
    pub tick: u64,
    pub pos: Vec<Vec3>,
    pub vel: Vec<Vec3>,
    pub spin: Vec<Vec3>,
    /// Instantaneous transverse acceleration `a_{i,perp}(t)` (`m/s^2`), recorded every `200 Hz` tick for `E7`.
    pub accel_perp: Vec<Vec3>,
}

impl ConstantSpeedFlock {
    pub fn new(pos: Vec<Vec3>, headings: Vec<Vec3>, v0: f32) -> Self {
        let n = pos.len();
        assert_eq!(n, headings.len());
        let vel: Vec<Vec3> = headings
            .into_iter()
            .map(|u| u.normalized_or(Vec3::EX) * v0)
            .collect();
        Self {
            tick: 0,
            pos,
            vel,
            spin: vec![Vec3::ZERO; n],
            accel_perp: vec![Vec3::ZERO; n],
        }
    }

    #[inline]
    pub fn total_spin(&self) -> Vec3 {
        let mut sum = Vec3::ZERO;
        for &s in &self.spin {
            sum += s;
        }
        sum
    }

    pub fn headings(&self, v0: f32) -> Vec<Vec3> {
        let inv_v0 = if v0 > 1e-12 { 1.0 / v0 } else { 1.0 };
        self.vel
            .iter()
            .map(|&v| (v * inv_v0).normalized_or(Vec3::EX))
            .collect()
    }

    /// Advance one fixed simulation tick (`dt`), optionally keeping topology frozen (`fixed_neighbours`)
    /// for pure lattice/graph dispersion tests or rebuilding via `LbvhIndex` on moving birds.
    pub fn step(
        &mut self,
        cfg: &ConstantSpeedConfig,
        fixed_neighbours: Option<&[Vec<NeighbourEntry>]>,
    ) -> WeightedGraph {
        let n = self.pos.len();
        let v0 = cfg.v0;
        let dt = cfg.dt;
        let t_now = (self.tick as f32) * dt;
        let u_list = self.headings(v0);

        let owned_neighbours;
        let neighbours: &[Vec<NeighbourEntry>] = match fixed_neighbours {
            Some(nb) => nb,
            None => {
                let lbvh = LbvhIndex::build(&self.pos);
                owned_neighbours = lbvh.query_all(&self.pos, &u_list, cfg.neighbour_rule);
                &owned_neighbours
            }
        };
        let graph = WeightedGraph::from_neighbours(neighbours, cfg.graph_symmetry);

        let mut next_pos = Vec::with_capacity(n);
        let mut next_vel = Vec::with_capacity(n);
        let mut next_spin = Vec::with_capacity(n);
        let mut next_accel_perp = Vec::with_capacity(n);

        let inv_v0_sq = 1.0 / (v0 * v0);

        for i in 0..n {
            let xi = self.pos[i];
            let vi = self.vel[i];
            let ui = u_list[i];
            let (cols, ws) = graph.row_edges(i);

            let mut align_v_sum = Vec3::ZERO;
            for (&j, &w) in cols.iter().zip(ws.iter()) {
                align_v_sum += self.vel[j as usize] * w;
            }

            let mut f_pos_ext = compute_positional_force(i, &self.pos, &neighbours[i], &cfg.positional);
            if let Some(pulse) = cfg.turn_pulse {
                if i < pulse.initiator_ids && t_now >= pulse.start_time_s && t_now <= pulse.end_time_s {
                    f_pos_ext += pulse.accel_vec;
                }
            }

            let mut rng = keyed_rng(cfg.master_seed, STREAM_WIENER, self.tick, i as u32);
            let z_noise = normal_vec3(&mut rng);

            match cfg.turning {
                ConstantSpeedTurningLaw::RefIsm {
                    j_ism,
                    chi,
                    eta_ism,
                    temp_ism,
                } => {
                    // Internal alignment torque `v_i x ((J_ISM / v0^2) sum_j w_ij v_j)` + positional/external torque.
                    // Updating `s_i^*` in `T_{v_i(t)} S^2` FIRST and then rotating `v_i(t)` around `s_i^*`
                    // simultaneously preserves:
                    //   (1) `||v_i|| == v0`,
                    //   (2) `s_i . v_i == 0`, and
                    //   (3) exact `sum_i s_i` conservation to machine precision when `eta_ism == 0`, `f_pos_ext == 0`,
                    //       and `w_ij == w_ji` (`SymmetricMean` or `SymmetricUnnormalized`).
                    let tau_align = vi.cross(align_v_sum) * (j_ism * inv_v0_sq);
                    let tau_ext = ui.cross(f_pos_ext) * (chi / v0);
                    let noise_amp = libm::sqrtf(2.0 * eta_ism * temp_ism * dt);
                    let ds_noise = ui.cross(z_noise) * noise_amp;

                    let damping = (1.0 - (eta_ism / chi) * dt).max(0.0);
                    let s_star = self.spin[i] * damping + (tau_align + tau_ext) * dt + ds_noise;
                    let omega = s_star * (1.0 / chi);
                    let v_new = rodrigues_rotate_fixed_speed(vi, omega * dt, v0);
                    let a_perp = omega.cross(vi);
                    let x_new = xi + (vi + v_new) * (0.5 * dt);

                    next_spin.push(s_star);
                    next_vel.push(v_new);
                    next_pos.push(x_new);
                    next_accel_perp.push(a_perp);
                }
                ConstantSpeedTurningLaw::OverdampedVicsek { j_vicsek, d_rot } => {
                    // Overdamped counterpart: `omega_det = (u_i x ((j_vicsek / v0) sum_j w_ij v_j + f_pos_ext / v0))`.
                    let omega_det = ui.cross(align_v_sum * (j_vicsek / v0) + f_pos_ext * (1.0 / v0));
                    let dtheta_noise = ui.cross(z_noise) * libm::sqrtf(2.0 * d_rot * dt);
                    let dtheta = omega_det * dt + dtheta_noise;
                    let v_new = rodrigues_rotate_fixed_speed(vi, dtheta, v0);
                    let a_perp = omega_det.cross(vi);
                    let x_new = xi + (vi + v_new) * (0.5 * dt);

                    next_spin.push(Vec3::ZERO);
                    next_vel.push(v_new);
                    next_pos.push(x_new);
                    next_accel_perp.push(a_perp);
                }
            }
        }

        self.pos = next_pos;
        self.vel = next_vel;
        self.spin = next_spin;
        self.accel_perp = next_accel_perp;
        self.tick += 1;
        graph
    }
}

/// Speed-confining potential law in `Ref-MarginalSPP` (`Cavagna et al. 2022`, `PMC9090766`, §2.1B).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "speed_law", rename_all = "snake_case")]
pub enum SpeedControlLaw {
    /// Marginal octic potential `V(v) = (lambda / v0^6) * (v^2 - v0^2)^4` => `F_sc = (8*lambda / v0^6) * v * (v0^2 - v^2)^3`.
    MarginalOctic { lambda: f32, v0: f32 },
    /// Linear Gaussian speed restoring `V(v) = 0.5 * g_lin * (||v|| - v0)^2` => `F_sc = -g_lin * (||v|| - v0) * u_hat`.
    LinearGaussian { g_lin: f32, v0: f32 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MarginalSppConfig {
    pub dt: f32,
    pub master_seed: u64,
    pub j_spp: f32,
    /// Isotropic Cartesian velocity diffusion `D_v` (`<eta_{i,a}(t) eta_{j,b}(t')> = 2 D_v delta_ij delta_ab delta(t-t')`).
    pub d_v: f32,
    pub speed_control: SpeedControlLaw,
    pub neighbour_rule: NeighbourRule,
    pub graph_symmetry: GraphSymmetry,
    pub positional: PositionalInteractionConfig,
}

#[derive(Clone, Debug)]
pub struct MarginalSppFlock {
    pub tick: u64,
    pub pos: Vec<Vec3>,
    pub vel: Vec<Vec3>,
    /// Explicitly tracked clamp counter — guaranteed `0` in `Ref-MarginalSPP` (§2.3 Rule 3).
    pub speed_clamp_events: u64,
}

impl MarginalSppFlock {
    pub fn new(pos: Vec<Vec3>, vel: Vec<Vec3>) -> Self {
        assert_eq!(pos.len(), vel.len());
        Self {
            tick: 0,
            pos,
            vel,
            speed_clamp_events: 0,
        }
    }

    pub fn headings(&self) -> Vec<Vec3> {
        self.vel
            .iter()
            .map(|&v| v.normalized_or(Vec3::EX))
            .collect()
    }

    /// Advance one fixed Cartesian Euler-Maruyama tick (`dt`) with ZERO speed clamps (§2.1B).
    pub fn step(
        &mut self,
        cfg: &MarginalSppConfig,
        fixed_neighbours: Option<&[Vec<NeighbourEntry>]>,
    ) {
        let n = self.pos.len();
        let dt = cfg.dt;
        let headings = self.headings();

        let owned_neighbours;
        let neighbours: &[Vec<NeighbourEntry>] = match fixed_neighbours {
            Some(nb) => nb,
            None => {
                let lbvh = LbvhIndex::build(&self.pos);
                owned_neighbours = lbvh.query_all(&self.pos, &headings, cfg.neighbour_rule);
                &owned_neighbours
            }
        };
        let graph = WeightedGraph::from_neighbours(neighbours, cfg.graph_symmetry);

        let noise_scale = libm::sqrtf(2.0 * cfg.d_v * dt);
        let mut next_pos = Vec::with_capacity(n);
        let mut next_vel = Vec::with_capacity(n);

        for i in 0..n {
            let xi = self.pos[i];
            let vi = self.vel[i];
            let (cols, ws) = graph.row_edges(i);

            let mut f_align = Vec3::ZERO;
            for (&j, &w) in cols.iter().zip(ws.iter()) {
                f_align += (self.vel[j as usize] - vi) * (cfg.j_spp * w);
            }

            let f_sc = match cfg.speed_control {
                SpeedControlLaw::MarginalOctic { lambda, v0 } => {
                    let v0_sq = v0 * v0;
                    let v0_6 = v0_sq * v0_sq * v0_sq;
                    let diff = v0_sq - vi.norm_sq();
                    let diff3 = diff * diff * diff;
                    vi * ((8.0 * lambda / v0_6) * diff3)
                }
                SpeedControlLaw::LinearGaussian { g_lin, v0 } => {
                    let speed = vi.norm();
                    if speed > 1e-12 {
                        vi * ((-g_lin * (speed - v0)) / speed)
                    } else {
                        Vec3::EX * (g_lin * v0)
                    }
                }
            };

            let f_pos = compute_positional_force(i, &self.pos, &neighbours[i], &cfg.positional);

            let mut rng = keyed_rng(cfg.master_seed, STREAM_WIENER, self.tick, i as u32);
            let dw = normal_vec3(&mut rng) * noise_scale;

            let v_new = vi + (f_align + f_sc + f_pos) * dt + dw;
            let x_new = xi + (vi + v_new) * (0.5 * dt);

            next_vel.push(v_new);
            next_pos.push(x_new);
        }

        self.pos = next_pos;
        self.vel = next_vel;
        self.tick += 1;
    }
}

pub fn compute_positional_force(
    i: usize,
    positions: &[Vec3],
    row: &[NeighbourEntry],
    pos_cfg: &PositionalInteractionConfig,
) -> Vec3 {
    if (pos_cfg.k_coh == 0.0 && pos_cfg.k_rep == 0.0) || row.is_empty() {
        return Vec3::ZERO;
    }
    let xi = positions[i];
    let gz = pos_cfg.vertical_coh_gain;
    let aniso = (gz - 1.0).abs() > 1e-6;
    let mut f = Vec3::ZERO;

    if pos_cfg.k_coh > 0.0 {
        let inv_m = 1.0 / (row.len() as f32);
        let mut coh_sum = Vec3::ZERO;
        for e in row {
            let diff = positions[e.id as usize] - xi;
            let (diff_eff, d_eff) = if aniso {
                let de = Vec3::new(diff.x, diff.y, diff.z * gz);
                (de, de.norm())
            } else {
                (diff, libm::sqrtf(e.dist_sq))
            };
            if d_eff > pos_cfg.r0_m {
                let mut spring = diff_eff * ((d_eff - pos_cfg.r0_m) / d_eff);
                if aniso {
                    spring.z *= gz;
                }
                coh_sum += spring;
            }
        }
        f += coh_sum * (pos_cfg.k_coh * inv_m);
    }

    if pos_cfg.k_rep > 0.0 && pos_cfg.n_rep > 0 {
        let m_rep = row.len().min(pos_cfg.n_rep);
        let inv_rep = 1.0 / (m_rep as f32);
        let mut rep_sum = Vec3::ZERO;
        for e in &row[..m_rep] {
            let diff = positions[e.id as usize] - xi;
            let (diff_eff, d_eff) = if aniso {
                let de = Vec3::new(diff.x, diff.y, diff.z * gz);
                (de, de.norm())
            } else {
                (diff, libm::sqrtf(e.dist_sq))
            };
            if d_eff < pos_cfg.r0_m {
                let mut push = if d_eff > 1e-6 {
                    diff_eff * (-(pos_cfg.r0_m - d_eff) / d_eff)
                } else {
                    Vec3::EX * (pos_cfg.r0_m - d_eff)
                };
                if aniso {
                    push.z *= gz;
                }
                rep_sum += push;
            }
        }
        f += rep_sum * (pos_cfg.k_rep * inv_rep);
    }

    f
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::STREAM_INIT;

    #[test]
    fn ref_ism_strictly_conserves_total_spin_on_symmetric_graph_and_drifts_on_directed_graph() {
        let n = 64usize;
        let v0 = 12.0f32;
        let mut init_pos = Vec::with_capacity(n);
        let mut init_u = Vec::with_capacity(n);
        for i in 0..n {
            let mut rng = keyed_rng(77, STREAM_INIT, 0, i as u32);
            init_pos.push(normal_vec3(&mut rng) * 3.0);
            // Slightly perturbed headings around +EX so alignment torques `v_i x v_j` are active and non-zero!
            let u = (Vec3::EX + normal_vec3(&mut rng) * 0.25).normalized_or(Vec3::EX);
            init_u.push(u);
        }

        // 1. SymmetricMean graph (`eta_ism = 0`, `temp_ism = 0`): total spin MUST remain 0 to `~1e-5`!
        let cfg_sym = ConstantSpeedConfig {
            v0,
            dt: 0.005,
            master_seed: 1,
            neighbour_rule: NeighbourRule::ExactTopoK {
                k: 7,
                blind_cos: -0.5,
            },
            graph_symmetry: GraphSymmetry::SymmetricMean,
            turning: ConstantSpeedTurningLaw::RefIsm {
                j_ism: 40.0,
                chi: 1.2,
                eta_ism: 0.0,
                temp_ism: 0.0,
            },
            positional: PositionalInteractionConfig::default(),
            turn_pulse: None,
        };
        let mut flock_sym = ConstantSpeedFlock::new(init_pos.clone(), init_u.clone(), v0);
        for _ in 0..400 {
            flock_sym.step(&cfg_sym, None);
        }
        let sym_drift = flock_sym.total_spin().norm();
        assert!(
            sym_drift < 1e-4,
            "Symmetric graph must conserve total internal spin; got drift={sym_drift}"
        );
        for (&v, &s) in flock_sym.vel.iter().zip(flock_sym.spin.iter()) {
            assert!((v.norm() - v0).abs() < 1e-4);
            assert!(s.dot(v).abs() < 1e-3);
        }

        // 2. DirectedNormalized graph (`w_ij != w_ji`): non-reciprocity breaks exact `sum_i s_i` conservation!
        let mut cfg_dir = cfg_sym.clone();
        cfg_dir.graph_symmetry = GraphSymmetry::DirectedNormalized;
        let mut flock_dir = ConstantSpeedFlock::new(init_pos, init_u, v0);
        for _ in 0..400 {
            flock_dir.step(&cfg_dir, None);
        }
        let dir_drift = flock_dir.total_spin().norm();
        assert!(
            dir_drift > 1e-2,
            "Directed k-NN graph should exhibit measurable non-reciprocal spin drift; got {dir_drift}"
        );
    }
}
