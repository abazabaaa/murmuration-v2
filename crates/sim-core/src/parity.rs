use crate::coupled::{CoupledConfig, CoupledFlock, RoostBoundaryConfig};
use crate::eval::{
    compute_polarization, compute_spatial_correlations, SpeedAccumulator, TurnWaveRecorder,
};
use crate::instance::generate_stress_cloud;
use crate::math::{canonical_f32_cmp, Vec3};
use crate::models::{
    ConstantSpeedConfig, ConstantSpeedFlock, ConstantSpeedTurningLaw, ExternalTurnPulse,
    MarginalSppConfig, MarginalSppFlock, PositionalInteractionConfig, SpeedControlLaw,
};
use crate::neighbour::{GraphSymmetry, NeighbourRule};
use serde::{Deserialize, Serialize};

/// Canonical Stage 3 `200 Hz` coupled configuration used by the interactive browser WebGPU viewer,
/// stress benchmarks, and Native <-> `wasm32` parity gates (§2.2, §4.1–§4.2).
pub fn default_coupled_browser_config(master_seed: u64) -> CoupledConfig {
    CoupledConfig {
        dt: 0.005,
        master_seed,
        j_spp: 18.0,
        d_v: 0.28,
        d_omega: 0.28,
        tau_omega: 0.12,
        v_eps: 0.25,
        include_ito_speed_drift: true,
        gravity_mps2: 9.81,
        speed_control: SpeedControlLaw::MarginalOctic {
            lambda: 0.35,
            v0: 12.0,
        },
        neighbour_rule: NeighbourRule::ExactTopoK {
            k: 7,
            blind_cos: -0.5,
        },
        graph_symmetry: GraphSymmetry::SymmetricMean,
        positional: PositionalInteractionConfig {
            k_coh: 0.55,
            r0_m: 1.35,
            k_rep: 3.40,
            n_rep: 1,
            vertical_coh_gain: 2.20,
        },
        turn_pulse: None,
        roost: RoostBoundaryConfig {
            symmetric_saturating_cohesion: true,
            rep_shell_m: 5.2,
            coh_shell_m: 6.8,
            coh_sat_m: 7.0,
            roost_radius_m: 65.0,
            roost_pull_mps2: 7.5,
            roost_alt_band_m: 24.0,
            roost_vertical_pull_mps2: 3.2,
        },
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Parity1000TicksDump {
    pub model_kind: u32,
    pub n_birds: usize,
    pub seed: u64,
    /// Packed `[px, py, pz, vx, vy, vz]` (`6 * n_birds`) at tick 600 (`3.0 s`).
    pub state_tick_600: Vec<f32>,
    /// Packed `[px, py, pz, vx, vy, vz]` (`6 * n_birds`) at tick 1000 (`5.0 s`).
    pub state_tick_1000: Vec<f32>,
}

fn pack_pos_vel(pos: &[Vec3], vel: &[Vec3]) -> Vec<f32> {
    let n = pos.len().min(vel.len());
    let mut out = Vec::with_capacity(n * 6);
    for i in 0..n {
        out.push(pos[i].x);
        out.push(pos[i].y);
        out.push(pos[i].z);
        out.push(vel[i].x);
        out.push(vel[i].y);
        out.push(vel[i].z);
    }
    out
}

/// Executes 1,000 fixed `5 ms` ticks (`5.0 s`) for `model_kind`:
/// - `0`: `Ref-ISM` (`ConstantSpeedFlock`)
/// - `1`: `Ref-MarginalSPP` (`MarginalSppFlock`)
/// - `2`: `CoupledFlock` (`Stage 2–3 coupled speed-heading SDE`)
pub fn run_parity_1000_ticks(model_kind: u32, n_birds: usize, seed: u64) -> Parity1000TicksDump {
    let (p0, u0) = generate_stress_cloud(0, n_birds, seed);
    let mut s600 = Vec::new();
    let s1000;

    match model_kind {
        0 => {
            let cfg = ConstantSpeedConfig {
                v0: 12.0,
                dt: 0.005,
                master_seed: seed,
                neighbour_rule: NeighbourRule::ExactTopoK {
                    k: 7,
                    blind_cos: -0.5,
                },
                graph_symmetry: GraphSymmetry::SymmetricMean,
                turning: ConstantSpeedTurningLaw::RefIsm {
                    j_ism: 36.0,
                    chi: 1.2,
                    eta_ism: 0.35,
                    temp_ism: 0.08,
                },
                positional: PositionalInteractionConfig {
                    k_coh: 1.0,
                    r0_m: 1.15,
                    k_rep: 2.0,
                    n_rep: 1,
                    vertical_coh_gain: 2.0,
                },
                turn_pulse: None,
            };
            let mut flock = ConstantSpeedFlock::new(p0, u0, cfg.v0);
            for k in 1..=1000 {
                flock.step(&cfg, None);
                if k == 600 {
                    s600 = pack_pos_vel(&flock.pos, &flock.vel);
                }
            }
            s1000 = pack_pos_vel(&flock.pos, &flock.vel);
        }
        1 => {
            let cfg = MarginalSppConfig {
                dt: 0.005,
                master_seed: seed,
                j_spp: 18.0,
                d_v: 0.28,
                speed_control: SpeedControlLaw::MarginalOctic {
                    lambda: 0.35,
                    v0: 12.0,
                },
                neighbour_rule: NeighbourRule::ExactTopoK {
                    k: 7,
                    blind_cos: -0.5,
                },
                graph_symmetry: GraphSymmetry::SymmetricMean,
                positional: PositionalInteractionConfig {
                    k_coh: 1.0,
                    r0_m: 1.15,
                    k_rep: 2.0,
                    n_rep: 1,
                    vertical_coh_gain: 2.0,
                },
            };
            let v0: Vec<Vec3> = u0.into_iter().map(|u| u * 12.0).collect();
            let mut flock = MarginalSppFlock::new(p0, v0);
            for k in 1..=1000 {
                flock.step(&cfg, None);
                if k == 600 {
                    s600 = pack_pos_vel(&flock.pos, &flock.vel);
                }
            }
            s1000 = pack_pos_vel(&flock.pos, &flock.vel);
        }
        _ => {
            let cfg = default_coupled_browser_config(seed);
            let v0: Vec<Vec3> = u0.into_iter().map(|u| u * 12.0).collect();
            let mut flock = CoupledFlock::from_cartesian(p0, v0);
            for k in 1..=1000 {
                flock.step(&cfg, None);
                if k == 600 {
                    let vel = flock.velocities();
                    s600 = pack_pos_vel(&flock.pos, &vel);
                }
            }
            let vel = flock.velocities();
            s1000 = pack_pos_vel(&flock.pos, &vel);
        }
    }

    Parity1000TicksDump {
        model_kind,
        n_birds,
        seed,
        state_tick_600: s600,
        state_tick_1000: s1000,
    }
}

/// Summary metrics for one 30.0 s (`6,000`-tick) paired TOST seed rollout (§4.1.3).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TostSeedSummary {
    pub seed: u64,
    pub mean_phi: f32,
    pub mean_speed_mps: f32,
    pub xi_orient_over_l: f32,
    pub xi_speed_over_l: f32,
    pub c_s_mps: f32,
    /// 64 empirical speed quantiles `p in (0.5/64 .. 63.5/64)` for exact 1D Wasserstein `W_1`.
    pub speed_quantiles_64: Vec<f32>,
}

impl TostSeedSummary {
    /// Packs into 69 `f32` values `[mean_phi, mean_speed_mps, xi_orient_over_l, xi_speed_over_l, c_s_mps, q0..q63]`.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(5 + self.speed_quantiles_64.len());
        v.push(self.mean_phi);
        v.push(self.mean_speed_mps);
        v.push(self.xi_orient_over_l);
        v.push(self.xi_speed_over_l);
        v.push(self.c_s_mps);
        v.extend_from_slice(&self.speed_quantiles_64);
        v
    }
}

/// Runs one `30.0 s` (`6,000`-tick at `200 Hz`, `5 s` burn-in + `25 s` measurement) coupled rollout
/// plus a seeded `200 Hz` turn-wave rollout to measure `Phi`, `V`, `xi/L`, `c_s`, and `P(v_i)` (§4.1.3).
pub fn run_tost_seed_rollout(seed: u64, n_birds: usize) -> TostSeedSummary {
    let cfg = default_coupled_browser_config(seed);
    let (p0, u0) = generate_stress_cloud(0, n_birds, seed);
    let v0: Vec<Vec3> = u0.into_iter().map(|u| u * 12.0).collect();
    let mut flock = CoupledFlock::from_cartesian(p0, v0);

    // 5.0 s burn-in (1,000 ticks)
    for _ in 0..1000 {
        flock.step(&cfg, None);
    }

    // 25.0 s measurement (5,000 ticks, sampled at 10 Hz = every 20 ticks -> 250 snapshots)
    let mut phi_sum = 0.0f32;
    let mut xi_o_sum = 0.0f32;
    let mut xi_s_sum = 0.0f32;
    let mut snaps = 0usize;
    let mut speed_acc = SpeedAccumulator::new(n_birds);

    for k in 1..=5000 {
        flock.step(&cfg, None);
        if k % 20 == 0 {
            let vel = flock.velocities();
            phi_sum += compute_polarization(&vel);
            speed_acc.record(&vel);
            let corr = compute_spatial_correlations(&flock.pos, &vel, 14);
            xi_o_sum += corr.xi_orient_over_l;
            xi_s_sum += corr.xi_speed_over_l;
            snaps += 1;
        }
    }

    let inv_snaps = 1.0 / (snaps.max(1) as f32);
    let spd_summary = speed_acc.summarize(flock.speed_clamp_events);
    let mut sorted_speeds = speed_acc.samples().to_vec();
    sorted_speeds.sort_by(|a, b| canonical_f32_cmp(*a, *b));
    let mut q64 = Vec::with_capacity(64);
    if sorted_speeds.is_empty() {
        q64.resize(64, 12.0);
    } else {
        let len_f = sorted_speeds.len() as f32;
        for b in 0..64 {
            let u = ((b as f32) + 0.5) / 64.0;
            let idx = ((u * len_f) as usize).min(sorted_speeds.len() - 1);
            q64.push(sorted_speeds[idx]);
        }
    }

    // Controlled 200 Hz turn-wave speed measurement (`E7`) seeded by `seed`
    let n_wave = 144usize;
    let (mut pw, _) = generate_stress_cloud(0, n_wave, seed ^ 0x55AA_1234);
    pw.sort_by(|a, b| canonical_f32_cmp(a.y, b.y));
    let vw: Vec<Vec3> = vec![Vec3::EX * 12.0; n_wave];
    let mut wave_cfg = CoupledConfig {
        dt: 0.005,
        master_seed: seed ^ 0x55AA_1234,
        j_spp: 160.0,
        d_v: 0.01,
        d_omega: 0.01,
        tau_omega: 0.25,
        v_eps: 0.25,
        include_ito_speed_drift: true,
        gravity_mps2: 9.81,
        speed_control: SpeedControlLaw::MarginalOctic {
            lambda: 2.0,
            v0: 12.0,
        },
        neighbour_rule: NeighbourRule::ExactTopoK {
            k: 7,
            blind_cos: -1.0,
        },
        graph_symmetry: GraphSymmetry::SymmetricMean,
        positional: PositionalInteractionConfig::default(),
        turn_pulse: Some(ExternalTurnPulse {
            initiator_ids: 8,
            start_time_s: 0.02,
            end_time_s: 0.14,
            accel_vec: Vec3::new(0.0, 14.0, 0.0),
        }),
        roost: RoostBoundaryConfig::default(),
    };
    let mut wave_flock = CoupledFlock::from_cartesian(pw.clone(), vw);
    let frozen_nb = crate::neighbour::LbvhIndex::build(&pw).query_all(
        &pw,
        &wave_flock.heading,
        wave_cfg.neighbour_rule,
    );
    let mut rec = TurnWaveRecorder::new(pw, 8, Vec3::EY, wave_cfg.dt);
    for _ in 0..320 {
        wave_flock.step(&wave_cfg, Some(&frozen_nb));
        rec.record_tick(&wave_flock.accel_perp);
    }
    wave_cfg.turn_pulse = None;
    let tw = rec.analyze(0.0);

    TostSeedSummary {
        seed,
        mean_phi: phi_sum * inv_snaps,
        mean_speed_mps: spd_summary.mean_individual_mps,
        xi_orient_over_l: xi_o_sum * inv_snaps,
        xi_speed_over_l: xi_s_sum * inv_snaps,
        c_s_mps: tw.measured_cs_mps,
        speed_quantiles_64: q64,
    }
}
