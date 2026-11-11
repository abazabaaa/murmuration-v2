mod stage2;
mod stage3;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sim_core::coupled::{CoupledConfig, CoupledFlock};
use sim_core::eval::{
    compute_polarization, compute_spatial_aspect_ratios, compute_spatial_correlations,
    compute_topological_anisotropy_gamma, wasserstein_1d, AspectRatioSummary,
    InternalDiffusionSummary, InternalDiffusionTracker, SpeedAccumulator,
    SpeedDistributionSummary, TurnWaveRecorder, TurnWaveSummary,
};
use sim_core::math::Vec3;
use sim_core::models::{
    ConstantSpeedConfig, ConstantSpeedFlock, ConstantSpeedTurningLaw, ExternalTurnPulse,
    MarginalSppConfig, MarginalSppFlock, PositionalInteractionConfig, SpeedControlLaw,
};
use sim_core::neighbour::{
    query_neighbours_brute_force, FlatNeighbourTable, GraphSymmetry, LbvhIndex, NeighbourRule,
    WeightedGraph,
};
use sim_core::rng::{keyed_rng, normal_vec3, uniform_open01, STREAM_INIT};
use std::env;
use std::fs;
use std::time::Instant;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ScenarioFile {
    name: String,
    n_birds: usize,
    duration_s: f32,
    burn_in_s: f32,
    #[serde(flatten)]
    runner: ScenarioKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "model_family", rename_all = "snake_case")]
enum ScenarioKind {
    ConstantSpeed { config: ConstantSpeedConfig },
    MarginalSpp { config: MarginalSppConfig },
    Coupled { config: CoupledConfig },
}

#[derive(Clone, Debug, Serialize)]
struct SpatialStressBenchRow {
    regime: &'static str,
    n_birds: usize,
    lbvh_build_ms: f64,
    lbvh_query_ms: f64,
    lbvh_total_ms: f64,
    oracle_exact_match: Option<bool>,
}

#[derive(Clone, Debug, Serialize)]
struct Gate1OracleReport {
    edge_cases_exact_match: bool,
    stress_regimes_oracle_1500_exact_match: bool,
    benchmarks: Vec<SpatialStressBenchRow>,
}

#[derive(Clone, Debug, Serialize)]
struct DtErrorPoint {
    dt_s: f32,
    fixed_graph_max_err_ism_m: f32,
    fixed_graph_max_err_spp_m: f32,
    dynamic_graph_rms_err_ism_m: f32,
    dynamic_graph_rms_err_spp_m: f32,
    dynamic_graph_max_err_spp_m: f32,
}

#[derive(Clone, Debug, Serialize)]
struct Gate2ConvergenceReport {
    deterministic_vs_dt_1ms: Vec<DtErrorPoint>,
    monotonic_fixed_graph_ism: bool,
    monotonic_fixed_graph_spp: bool,
    monotonic_dynamic_rms_spp: bool,
    stochastic_5ms_vs_1ms_delta_phi: f32,
    stochastic_5ms_vs_1ms_delta_mean_v_mps: f32,
    stochastic_5ms_vs_1ms_speed_w1_mps: f32,
    passed: bool,
}

#[derive(Clone, Debug, Serialize)]
struct TurnScalingRow {
    label: String,
    j_over_chi: f32,
    summary: TurnWaveSummary,
}

#[derive(Clone, Debug, Serialize)]
struct SpeedScalingRow {
    law: &'static str,
    n_birds: usize,
    mean_phi: f32,
    mean_xi_orient_over_l: f32,
    mean_xi_speed_over_l: f32,
    speed_stats: SpeedDistributionSummary,
}

#[derive(Clone, Debug, Serialize)]
struct Gate3ReferencePhysicsReport {
    symmetric_spin_conservation_drift: f32,
    directed_nonreciprocal_spin_drift: f32,
    ism_wave_scaling: Vec<TurnScalingRow>,
    overdamped_vicsek_comparator: TurnWaveSummary,
    marginal_vs_linear_speed_across_n: Vec<SpeedScalingRow>,
    calm_cohesive_observables: CalmCohesiveSummary,
    passed: bool,
}

#[derive(Clone, Debug, Serialize)]
struct CalmCohesiveSummary {
    n_birds: usize,
    mean_phi: f32,
    gamma_ranks_1_to_8: Vec<f32>,
    aspect_ratio: AspectRatioSummary,
    diffusion: InternalDiffusionSummary,
}

#[derive(Clone, Debug, Serialize)]
struct FullStage1Report {
    gate1_oracle: Gate1OracleReport,
    gate2_convergence: Gate2ConvergenceReport,
    gate3_reference_physics: Gate3ReferencePhysicsReport,
    overall_stage1_passed: bool,
}

pub(crate) fn generate_stress_cloud(regime_idx: usize, n: usize, seed: u64) -> (Vec<Vec3>, Vec<Vec3>) {
    let mut pos = Vec::with_capacity(n);
    let mut u = Vec::with_capacity(n);
    for i in 0..n {
        let mut rng = keyed_rng(seed + (regime_idx as u64) * 100, STREAM_INIT, 0, i as u32);
        let p = match regime_idx {
            // 0: Calm ribbon (`~ 1 : 2.8 : 5.6`, NND ~ 1 m)
            0 => {
                let g = normal_vec3(&mut rng);
                let scale = libm::cbrtf(n as f32) * 0.55;
                Vec3::new(g.x * 2.8 * scale, g.y * 5.6 * scale, g.z * 1.0 * scale)
            }
            // 1: 5x local density compaction spike
            1 => {
                let g = normal_vec3(&mut rng);
                let scale = libm::cbrtf(n as f32) * 0.28;
                g * scale
            }
            // 2: 3x post-attack diluted cloud
            2 => {
                let g = normal_vec3(&mut rng);
                let scale = libm::cbrtf(n as f32) * 3.0;
                g * scale
            }
            // 3: Split sub-flocks (200 m gap) + isolated stragglers
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

fn run_gate1_oracle() -> Gate1OracleReport {
    let rule = NeighbourRule::ExactTopoK {
        k: 7,
        blind_cos: -0.5, // 240 deg FoV (120 deg rear blind cone)
    };

    // 1. Edge cases: exact 6-way tie, coincident point `x_0 == x_1`, 0 in-FoV straggler, and `<7` sparse flock
    let mut edge_pos = vec![
        Vec3::ZERO,
        Vec3::ZERO,
        Vec3::new(3.0, 0.0, 0.0),
        Vec3::new(-3.0, 0.0, 0.0),
        Vec3::new(0.0, 3.0, 0.0),
        Vec3::new(0.0, -3.0, 0.0),
        Vec3::new(0.0, 0.0, 3.0),
        Vec3::new(0.0, 0.0, -3.0),
        Vec3::new(900.0, 0.0, 0.0), // Facing +EX with narrow FoV => 0 neighbours
    ];
    for k in 9..96u32 {
        let mut rng = keyed_rng(11, STREAM_INIT, 0, k);
        edge_pos.push(normal_vec3(&mut rng) * 10.0 - Vec3::new(50.0, 0.0, 0.0));
    }
    let edge_u: Vec<Vec3> = (0..edge_pos.len())
        .map(|i| if i == 0 || i == 8 { Vec3::EX } else { Vec3::EY })
        .collect();

    let narrow_rule = NeighbourRule::ExactTopoK {
        k: 7,
        blind_cos: 0.5,
    };
    let edge_ok_1 = query_neighbours_brute_force(&edge_pos, &edge_u, rule)
        == LbvhIndex::build(&edge_pos).query_all(&edge_pos, &edge_u, rule);
    let nb_narrow_oracle = query_neighbours_brute_force(&edge_pos, &edge_u, narrow_rule);
    let nb_narrow_lbvh = LbvhIndex::build(&edge_pos).query_all(&edge_pos, &edge_u, narrow_rule);
    let edge_ok_2 = nb_narrow_oracle == nb_narrow_lbvh && nb_narrow_lbvh[8].is_empty();
    let sparse_ok = query_neighbours_brute_force(&edge_pos[..5], &edge_u[..5], rule)
        == LbvhIndex::build(&edge_pos[..5]).query_all(&edge_pos[..5], &edge_u[..5], rule);
    let edge_cases_exact_match = edge_ok_1 && edge_ok_2 && sparse_ok;

    // 2. Oracle check at N = 1,500 + timing across N in {1000, 5000, 10000}
    let regime_names = [
        "calm_ribbon",
        "compacted_core_5x",
        "diluted_cloud_3x",
        "split_subflocks_stragglers",
    ];

    let mut stress_1500_ok = true;
    for r in 0..4 {
        let (pos, u) = generate_stress_cloud(r, 1500, 700);
        let oracle = query_neighbours_brute_force(&pos, &u, rule);
        let accel = LbvhIndex::build(&pos).query_all(&pos, &u, rule);
        if oracle != accel {
            stress_1500_ok = false;
        }
    }

    let mut benchmarks = Vec::new();
    for &n in &[1000usize, 5000usize, 10000usize] {
        for (r, &r_name) in regime_names.iter().enumerate() {
            let (pos, u) = generate_stress_cloud(r, n, 900);
            let t0 = Instant::now();
            let lbvh = LbvhIndex::build(&pos);
            let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let t1 = Instant::now();
            let flat_res = lbvh.query_all_flat(&pos, &u, rule);
            let query_ms = t1.elapsed().as_secs_f64() * 1000.0;

            let oracle_match = if n == 1000 {
                let oracle = query_neighbours_brute_force(&pos, &u, rule);
                let res_vecs = lbvh.query_all(&pos, &u, rule);
                let ok = oracle == res_vecs && FlatNeighbourTable::from_vecs(&oracle) == flat_res;
                if !ok {
                    stress_1500_ok = false;
                }
                Some(ok)
            } else {
                None
            };

            benchmarks.push(SpatialStressBenchRow {
                regime: r_name,
                n_birds: n,
                lbvh_build_ms: (build_ms * 1000.0).round() / 1000.0,
                lbvh_query_ms: (query_ms * 1000.0).round() / 1000.0,
                lbvh_total_ms: ((build_ms + query_ms) * 1000.0).round() / 1000.0,
                oracle_exact_match: oracle_match,
            });
        }
    }

    Gate1OracleReport {
        edge_cases_exact_match,
        stress_regimes_oracle_1500_exact_match: stress_1500_ok,
        benchmarks,
    }
}

fn run_gate2_convergence() -> Gate2ConvergenceReport {
    // 1. Deterministic self-convergence (`t = 2.0 s`, `N = 96`) across `dt in {10, 5, 2, 1} ms`
    let n = 96usize;
    let v0 = 12.0f32;
    let mut init_pos = Vec::with_capacity(n);
    let mut init_u = Vec::with_capacity(n);
    let mut init_vel = Vec::with_capacity(n);
    for i in 0..n {
        let mut rng = keyed_rng(31415, STREAM_INIT, 0, i as u32);
        let p = normal_vec3(&mut rng) * 4.0;
        let u = (Vec3::EX + normal_vec3(&mut rng) * 0.22).normalized_or(Vec3::EX);
        let s = v0 + (uniform_open01(&mut rng) - 0.5) * 3.0;
        init_pos.push(p);
        init_u.push(u);
        init_vel.push(u * s);
    }

    let frozen_det_nb = LbvhIndex::build(&init_pos).query_all(
        &init_pos,
        &init_u,
        NeighbourRule::ExactTopoK {
            k: 7,
            blind_cos: -0.5,
        },
    );

    let run_det_ism = |dt: f32, freeze_graph: bool| -> Vec<Vec3> {
        let steps = ((2.0f32 / dt).round()) as usize;
        let cfg = ConstantSpeedConfig {
            v0,
            dt,
            master_seed: 1,
            neighbour_rule: NeighbourRule::ExactTopoK {
                k: 7,
                blind_cos: -0.5,
            },
            graph_symmetry: GraphSymmetry::SymmetricMean,
            turning: ConstantSpeedTurningLaw::RefIsm {
                j_ism: 30.0,
                chi: 1.2,
                eta_ism: 0.4,
                temp_ism: 0.0,
            },
            positional: PositionalInteractionConfig {
                k_coh: 0.5,
                r0_m: 1.2,
                k_rep: 1.0,
                n_rep: 1,
                vertical_coh_gain: 1.0,
            },
            turn_pulse: None,
        };
        let mut f = ConstantSpeedFlock::new(init_pos.clone(), init_u.clone(), v0);
        let nb_opt = if freeze_graph {
            Some(frozen_det_nb.as_slice())
        } else {
            None
        };
        for _ in 0..steps {
            f.step(&cfg, nb_opt);
        }
        f.pos
    };

    let run_det_spp = |dt: f32, freeze_graph: bool| -> Vec<Vec3> {
        let steps = ((2.0f32 / dt).round()) as usize;
        let cfg = MarginalSppConfig {
            dt,
            master_seed: 1,
            j_spp: 2.5,
            d_v: 0.0,
            speed_control: SpeedControlLaw::MarginalOctic {
                lambda: 0.6,
                v0: 12.0,
            },
            neighbour_rule: NeighbourRule::ExactTopoK {
                k: 7,
                blind_cos: -0.5,
            },
            graph_symmetry: GraphSymmetry::SymmetricMean,
            positional: PositionalInteractionConfig {
                k_coh: 0.5,
                r0_m: 1.2,
                k_rep: 1.0,
                n_rep: 1,
                vertical_coh_gain: 1.0,
            },
        };
        let mut f = MarginalSppFlock::new(init_pos.clone(), init_vel.clone());
        let nb_opt = if freeze_graph {
            Some(frozen_det_nb.as_slice())
        } else {
            None
        };
        for _ in 0..steps {
            f.step(&cfg, nb_opt);
        }
        f.pos
    };

    let ref_ism_fixed_1ms = run_det_ism(0.001, true);
    let ref_spp_fixed_1ms = run_det_spp(0.001, true);
    let ref_ism_dyn_1ms = run_det_ism(0.001, false);
    let ref_spp_dyn_1ms = run_det_spp(0.001, false);

    let max_diff = |a: &[Vec3], b: &[Vec3]| -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&p, &q)| (p - q).norm())
            .fold(0.0f32, f32::max)
    };
    let rms_diff = |a: &[Vec3], b: &[Vec3]| -> f32 {
        let m2 = a
            .iter()
            .zip(b.iter())
            .map(|(&p, &q)| (p - q).norm_sq())
            .sum::<f32>()
            / (a.len() as f32);
        libm::sqrtf(m2)
    };

    let mut det_points = Vec::new();
    for &dt in &[0.010f32, 0.005f32, 0.002f32] {
        let p_ism_fix = run_det_ism(dt, true);
        let p_spp_fix = run_det_spp(dt, true);
        let p_ism_dyn = run_det_ism(dt, false);
        let p_spp_dyn = run_det_spp(dt, false);
        det_points.push(DtErrorPoint {
            dt_s: dt,
            fixed_graph_max_err_ism_m: max_diff(&p_ism_fix, &ref_ism_fixed_1ms),
            fixed_graph_max_err_spp_m: max_diff(&p_spp_fix, &ref_spp_fixed_1ms),
            dynamic_graph_rms_err_ism_m: rms_diff(&p_ism_dyn, &ref_ism_dyn_1ms),
            dynamic_graph_rms_err_spp_m: rms_diff(&p_spp_dyn, &ref_spp_dyn_1ms),
            dynamic_graph_max_err_spp_m: max_diff(&p_spp_dyn, &ref_spp_dyn_1ms),
        });
    }

    let monotonic_fixed_ism =
        det_points[0].fixed_graph_max_err_ism_m > det_points[1].fixed_graph_max_err_ism_m
            && det_points[1].fixed_graph_max_err_ism_m > det_points[2].fixed_graph_max_err_ism_m
            && det_points[1].fixed_graph_max_err_ism_m < 0.05;
    let monotonic_fixed_spp =
        det_points[0].fixed_graph_max_err_spp_m > det_points[1].fixed_graph_max_err_spp_m
            && det_points[1].fixed_graph_max_err_spp_m > det_points[2].fixed_graph_max_err_spp_m
            && det_points[1].fixed_graph_max_err_spp_m < 0.05;
    let monotonic_dyn_rms =
        det_points[0].dynamic_graph_rms_err_spp_m > det_points[1].dynamic_graph_rms_err_spp_m
            && det_points[1].dynamic_graph_rms_err_spp_m > det_points[2].dynamic_graph_rms_err_spp_m
            && det_points[1].dynamic_graph_rms_err_spp_m < 0.03;

    // 2. Stochastic ensemble convergence (`dt = 5 ms` vs `dt = 1 ms`, 16 seeds in parallel, `N = 64`, `8 s`)
    let run_stoch_ensemble = |dt: f32| -> (f32, f32, Vec<f32>) {
        let steps_burn = ((3.0f32 / dt).round()) as usize;
        let steps_meas = ((5.0f32 / dt).round()) as usize;
        let sample_every = ((0.1f32 / dt).round()).max(1.0) as usize;

        let seed_results: Vec<(f32, Vec<f32>)> = (0..16u64)
            .into_par_iter()
            .map(|s| {
                let mut p0 = Vec::with_capacity(64);
                let mut v0_vec = Vec::with_capacity(64);
                for i in 0..64u32 {
                    let mut rng = keyed_rng(5000 + s, STREAM_INIT, 0, i);
                    p0.push(normal_vec3(&mut rng) * 2.5);
                    v0_vec.push(Vec3::EX * 12.0 + normal_vec3(&mut rng) * 0.5);
                }
                let cfg = MarginalSppConfig {
                    dt,
                    master_seed: 8000 + s,
                    j_spp: 3.0,
                    d_v: 0.6,
                    speed_control: SpeedControlLaw::MarginalOctic {
                        lambda: 0.8,
                        v0: 12.0,
                    },
                    neighbour_rule: NeighbourRule::ExactTopoK {
                        k: 7,
                        blind_cos: -0.5,
                    },
                    graph_symmetry: GraphSymmetry::SymmetricMean,
                    positional: PositionalInteractionConfig {
                        k_coh: 0.8,
                        r0_m: 1.0,
                        k_rep: 1.2,
                        n_rep: 1,
                        vertical_coh_gain: 1.0,
                    },
                };
                let mut f = MarginalSppFlock::new(p0, v0_vec);
                for _ in 0..steps_burn {
                    f.step(&cfg, None);
                }
                let mut phi_acc = 0.0f32;
                let mut phi_cnt = 0usize;
                let mut sp_acc = SpeedAccumulator::new(64);
                for step_k in 0..steps_meas {
                    f.step(&cfg, None);
                    if step_k % sample_every == 0 {
                        phi_acc += compute_polarization(&f.vel);
                        phi_cnt += 1;
                        sp_acc.record(&f.vel);
                    }
                }
                (phi_acc / (phi_cnt as f32), sp_acc.samples().to_vec())
            })
            .collect();

        let mean_phi = seed_results.iter().map(|r| r.0).sum::<f32>() / (seed_results.len() as f32);
        let mut all_speeds = Vec::new();
        for (_, sp) in seed_results {
            all_speeds.extend(sp);
        }
        let mean_v = all_speeds.iter().sum::<f32>() / (all_speeds.len() as f32);
        (mean_phi, mean_v, all_speeds)
    };

    let (phi_5ms, v_5ms, sp_5ms) = run_stoch_ensemble(0.005);
    let (phi_1ms, v_1ms, sp_1ms) = run_stoch_ensemble(0.001);
    let d_phi = (phi_5ms - phi_1ms).abs();
    let d_v = (v_5ms - v_1ms).abs();
    let w1 = wasserstein_1d(&sp_5ms, &sp_1ms);

    let passed = monotonic_fixed_ism
        && monotonic_fixed_spp
        && monotonic_dyn_rms
        && d_phi < 0.02
        && d_v < 0.15
        && w1 < 0.15;

    Gate2ConvergenceReport {
        deterministic_vs_dt_1ms: det_points,
        monotonic_fixed_graph_ism: monotonic_fixed_ism,
        monotonic_fixed_graph_spp: monotonic_fixed_spp,
        monotonic_dynamic_rms_spp: monotonic_dyn_rms,
        stochastic_5ms_vs_1ms_delta_phi: d_phi,
        stochastic_5ms_vs_1ms_delta_mean_v_mps: d_v,
        stochastic_5ms_vs_1ms_speed_w1_mps: w1,
        passed,
    }
}

fn run_gate3_reference_physics() -> Gate3ReferencePhysicsReport {
    let v0 = 12.0f32;
    let dt = 0.005f32; // 200 Hz (`5 ms`)

    // 1. Spin conservation check: SymmetricMean (`w_ij == w_ji`) vs DirectedNormalized (`w_ij != w_ji`)
    let n_spin = 120usize;
    let mut spin_pos = Vec::with_capacity(n_spin);
    let mut spin_u = Vec::with_capacity(n_spin);
    for i in 0..n_spin {
        let mut rng = keyed_rng(4242, STREAM_INIT, 0, i as u32);
        spin_pos.push(normal_vec3(&mut rng) * 4.0);
        spin_u.push((Vec3::EX + normal_vec3(&mut rng) * 0.25).normalized_or(Vec3::EX));
    }
    let run_spin_drift = |sym: GraphSymmetry| -> f32 {
        let cfg = ConstantSpeedConfig {
            v0,
            dt,
            master_seed: 1,
            neighbour_rule: NeighbourRule::ExactTopoK {
                k: 7,
                blind_cos: -0.5,
            },
            graph_symmetry: sym,
            turning: ConstantSpeedTurningLaw::RefIsm {
                j_ism: 45.0,
                chi: 1.2,
                eta_ism: 0.0,
                temp_ism: 0.0,
            },
            positional: PositionalInteractionConfig::default(),
            turn_pulse: None,
        };
        let mut flock = ConstantSpeedFlock::new(spin_pos.clone(), spin_u.clone(), v0);
        for _ in 0..400 {
            flock.step(&cfg, None);
        }
        flock.total_spin().norm()
    };
    let sym_drift = run_spin_drift(GraphSymmetry::SymmetricMean);
    let dir_drift = run_spin_drift(GraphSymmetry::DirectedNormalized);

    // 2. Experiment 1: Constant-speed turning wave dispersion (`Ref-ISM` across `J_ISM/chi` vs `OverdampedVicsek`)
    // Construct a 3D ordered ribbon along `x in [0, 39] m` (`40 x 3 x 3 = 360` birds, spacing `a = 1.0 m`),
    // sorted by `x` so birds `0..9` (`x = 0`) form the initiator slab.
    let mut ribbon_pos = Vec::with_capacity(360);
    for ix in 0..40 {
        for iy in 0..3 {
            for iz in 0..3 {
                ribbon_pos.push(Vec3::new(ix as f32, (iy as f32) - 1.0, (iz as f32) - 1.0));
            }
        }
    }
    let n_rib = ribbon_pos.len();
    let init_u_rib = vec![Vec3::EY; n_rib]; // Cruising along +EY; wave propagates laterally along +EX!
    let rule_rib = NeighbourRule::ExactTopoK {
        k: 7,
        blind_cos: -1.0,
    };
    let frozen_nb = LbvhIndex::build(&ribbon_pos).query_all(&ribbon_pos, &init_u_rib, rule_rib);
    let sym_graph = WeightedGraph::from_neighbours(&frozen_nb, GraphSymmetry::SymmetricMean);
    let m2_ex = sym_graph.second_spatial_moment(&ribbon_pos, Vec3::EX);

    let run_turn_experiment = |turning: ConstantSpeedTurningLaw, predicted_cs: f32| -> TurnWaveSummary {
        let cfg = ConstantSpeedConfig {
            v0,
            dt,
            master_seed: 99,
            neighbour_rule: rule_rib,
            graph_symmetry: GraphSymmetry::SymmetricMean,
            turning,
            positional: PositionalInteractionConfig::default(),
            turn_pulse: Some(ExternalTurnPulse {
                start_time_s: 0.02,
                end_time_s: 0.06,
                accel_vec: Vec3::new(-12.0, 0.0, 0.0), // Sharp 40 ms (`8-tick`) lateral turn impulse
                initiator_ids: 9,
            }),
        };
        let mut flock = ConstantSpeedFlock::new(ribbon_pos.clone(), init_u_rib.clone(), v0);
        let mut recorder = TurnWaveRecorder::new(ribbon_pos.clone(), 9, Vec3::EX, dt);
        // Simulate 2.5 s = 500 ticks at `200 Hz`, recording `accel_perp` EVERY SINGLE `5 ms` TICK (`E7`)!
        for _ in 0..500 {
            flock.step(&cfg, Some(&frozen_nb));
            recorder.record_tick(&flock.accel_perp);
        }
        recorder.analyze(predicted_cs)
    };

    let mut ism_wave_scaling = Vec::new();
    for &(label, j_ism, chi) in &[
        ("ism_1x", 150.0f32, 1.0f32),
        ("ism_2x", 600.0f32, 1.0f32),
        ("ism_3x", 1350.0f32, 1.0f32),
        ("ism_4x_starling_band", 2400.0f32, 1.0f32),
    ] {
        // Exact continuum spin-wave speed `c_s = sqrt((J_ISM * M_2(k_hat)) / (2 * chi))` (Taylor 1/2 factor!)
        let pred_cs = libm::sqrtf((0.5 * j_ism * m2_ex) / chi);
        let summary = run_turn_experiment(
            ConstantSpeedTurningLaw::RefIsm {
                j_ism,
                chi,
                eta_ism: 0.05,
                temp_ism: 0.0,
            },
            pred_cs,
        );
        ism_wave_scaling.push(TurnScalingRow {
            label: label.to_string(),
            j_over_chi: j_ism / chi,
            summary,
        });
    }

    let vicsek_comp = run_turn_experiment(
        ConstantSpeedTurningLaw::OverdampedVicsek {
            j_vicsek: 40.0,
            d_rot: 0.0,
        },
        0.0,
    );

    // 3. `Ref-MarginalSPP`: Small-N (`N in {10, 20, 50, 200, 1000}`) speed statistics (`E5`) & `xi_sp/L` (`E4`)
    // Compare weak `LinearGaussian` (which allows speed correlations at large N, but suffers entropic speed push
    // at small N due to the `v^2 dv` 3D Cartesian phase-space volume!) vs `MarginalOctic` (`Cavagna et al. 2022`).
    let flock_sizes = [10usize, 20, 50, 200, 1000];
    let mut speed_rows = Vec::new();

    for &(law_name, speed_law) in &[
        (
            "linear_gaussian_weak",
            SpeedControlLaw::LinearGaussian {
                g_lin: 0.12,
                v0: 12.0,
            },
        ),
        (
            "marginal_octic",
            SpeedControlLaw::MarginalOctic {
                lambda: 1.2,
                v0: 12.0,
            },
        ),
    ] {
        for &n_b in &flock_sizes {
            let n_seeds: u64 = if n_b <= 200 { 12 } else { 4 };
            let burn_steps = 800usize; // 4.0 s
            let meas_steps = 800usize; // 4.0 s @ 200 Hz (`10 Hz` = every 20 steps)

            let per_seed: Vec<(f32, f32, f32, SpeedAccumulator, u64)> = (0..n_seeds)
                .into_par_iter()
                .map(|s| {
                    let mut p0 = Vec::with_capacity(n_b);
                    let mut v0_init = Vec::with_capacity(n_b);
                    let rad = libm::cbrtf(n_b as f32) * 0.95;
                    for i in 0..n_b {
                        let mut rng = keyed_rng(20_000 + s * 31 + (n_b as u64), STREAM_INIT, 0, i as u32);
                        // Bounded uniform ball (`r = rad * u^(1/3)`) avoids unbounded 3.5-sigma Gaussian outlier stragglers distorting `L`
                        let dir = normal_vec3(&mut rng).normalized_or(Vec3::EX);
                        let r = rad * libm::cbrtf(uniform_open01(&mut rng));
                        p0.push(dir * r);
                        v0_init.push(Vec3::EX * 12.0 + normal_vec3(&mut rng) * 0.8);
                    }
                    let cfg = MarginalSppConfig {
                        dt,
                        master_seed: 90_000 + s * 97 + (n_b as u64),
                        j_spp: 4.5,
                        d_v: 3.2,
                        speed_control: speed_law,
                        neighbour_rule: NeighbourRule::ExactTopoK {
                            k: 7,
                            blind_cos: -0.5,
                        },
                        graph_symmetry: GraphSymmetry::SymmetricMean,
                        positional: PositionalInteractionConfig {
                            k_coh: 1.2,
                            r0_m: 1.0,
                            k_rep: 1.6,
                            n_rep: 1,
                            vertical_coh_gain: 1.0,
                        },
                    };
                    let mut flock = MarginalSppFlock::new(p0, v0_init);
                    for _ in 0..burn_steps {
                        flock.step(&cfg, None);
                    }
                    let mut acc = SpeedAccumulator::new(n_b);
                    let mut phi_sum = 0.0f32;
                    let mut xio_sum = 0.0f32;
                    let mut xis_sum = 0.0f32;
                    let mut snaps = 0usize;
                    for k in 0..meas_steps {
                        flock.step(&cfg, None);
                        if k % 20 == 0 {
                            acc.record(&flock.vel);
                            phi_sum += compute_polarization(&flock.vel);
                            let corr = compute_spatial_correlations(&flock.pos, &flock.vel, 16);
                            xio_sum += corr.xi_orient_over_l;
                            xis_sum += corr.xi_speed_over_l;
                            snaps += 1;
                        }
                    }
                    let inv = 1.0 / (snaps as f32);
                    (
                        phi_sum * inv,
                        xio_sum * inv,
                        xis_sum * inv,
                        acc,
                        flock.speed_clamp_events,
                    )
                })
                .collect();

            let mut merged_acc = SpeedAccumulator::new(n_b);
            let mut mean_phi = 0.0f32;
            let mut mean_xio = 0.0f32;
            let mut mean_xis = 0.0f32;
            let mut total_clamps = 0u64;
            for (phi, xio, xis, acc, cl) in &per_seed {
                mean_phi += *phi;
                mean_xio += *xio;
                mean_xis += *xis;
                total_clamps += *cl;
                // Re-populate merged individual samples via 1-bird chunks to preserve quantiles cleanly
                for &sv in acc.samples() {
                    merged_acc.record(&[Vec3::new(sv, 0.0, 0.0)]);
                }
            }
            let inv_s = 1.0 / (per_seed.len() as f32);
            let mut summary = merged_acc.summarize(total_clamps);
            summary.n_birds = n_b;
            // Compute exact group scalar speed stats across seeds from per-seed summaries
            let seed_grp_means: Vec<f32> = per_seed
                .iter()
                .map(|(_, _, _, a, c)| a.summarize(*c).mean_group_scalar_speed_mps)
                .collect();
            let grp_mean = seed_grp_means.iter().sum::<f32>() * inv_s;
            let grp_var = seed_grp_means
                .iter()
                .map(|&x| (x - grp_mean) * (x - grp_mean))
                .sum::<f32>()
                * inv_s;
            summary.mean_group_scalar_speed_mps = grp_mean;
            summary.sd_group_scalar_speed_mps = libm::sqrtf(grp_var);

            speed_rows.push(SpeedScalingRow {
                law: law_name,
                n_birds: n_b,
                mean_phi: mean_phi * inv_s,
                mean_xi_orient_over_l: mean_xio * inv_s,
                mean_xi_speed_over_l: mean_xis * inv_s,
                speed_stats: summary,
            });
        }
    }

    // 4. Multi-rate calm cohesive observables (`E1`, `E2`, `E3`, `E6a` @ 40 Hz, `E6b` @ 40 Hz, `E6c` @ 10 Hz)
    let n_calm = 250usize;
    let (mut p_calm, u_calm) = generate_stress_cloud(0, n_calm, 555);
    for (p, u) in p_calm.iter_mut().zip(u_calm.iter()) {
        *p = *p * 0.8;
        let _ = u;
    }
    let v_calm: Vec<Vec3> = u_calm.iter().map(|&u| u * 12.0).collect();
    let calm_cfg = MarginalSppConfig {
        dt,
        master_seed: 7777,
        j_spp: 4.5,
        d_v: 0.8,
        speed_control: SpeedControlLaw::MarginalOctic {
            lambda: 1.2,
            v0: 12.0,
        },
        neighbour_rule: NeighbourRule::ExactTopoK {
            k: 7,
            blind_cos: -0.5,
        },
        graph_symmetry: GraphSymmetry::SymmetricMean,
        positional: PositionalInteractionConfig {
            k_coh: 1.0,
            r0_m: 1.1,
            k_rep: 1.4,
            n_rep: 1,
            vertical_coh_gain: 1.0,
        },
    };
    let mut calm_flock = MarginalSppFlock::new(p_calm, v_calm);
    for _ in 0..400 {
        calm_flock.step(&calm_cfg, None);
    }
    let t0_diff = (calm_flock.tick as f32) * dt;
    let mut diff_tracker = InternalDiffusionTracker::initialize(&calm_flock.pos, 7, t0_diff);
    let mut calm_phi_sum = 0.0f32;
    let mut calm_snaps = 0usize;
    for k in 0..800 {
        calm_flock.step(&calm_cfg, None);
        let t_now = (calm_flock.tick as f32) * dt;
        if k % 5 == 0 {
            // 40 Hz (`25 ms`)
            diff_tracker.record_40hz(&calm_flock.pos, t_now);
        }
        if k % 20 == 0 {
            // 10 Hz (`100 ms`)
            diff_tracker.record_qm_10hz(&calm_flock.pos);
            calm_phi_sum += compute_polarization(&calm_flock.vel);
            calm_snaps += 1;
        }
    }
    let calm_summary = CalmCohesiveSummary {
        n_birds: n_calm,
        mean_phi: calm_phi_sum / (calm_snaps as f32),
        gamma_ranks_1_to_8: compute_topological_anisotropy_gamma(&calm_flock.pos, 8),
        aspect_ratio: compute_spatial_aspect_ratios(&calm_flock.pos),
        diffusion: diff_tracker.summarize(),
    };

    // Acceptance checks:
    // (a) Symmetric spin conservation < 1e-3 and strictly smaller than directed spin drift
    let spin_ok = sym_drift < 1e-3 && dir_drift > 10.0 * sym_drift.max(1e-5);
    // (b) ISM wave speed scales monotonically `1x < 2x < 3x` with high linear R^2 (> 0.95) and preserves far-field amplitude better than overdamped Vicsek
    let cs_1 = ism_wave_scaling[0].summary.measured_cs_mps;
    let cs_2 = ism_wave_scaling[1].summary.measured_cs_mps;
    let cs_3 = ism_wave_scaling[2].summary.measured_cs_mps;
    let ratio_2_to_1 = if cs_1 > 1e-6 { cs_2 / cs_1 } else { 0.0 };
    let ratio_3_to_1 = if cs_1 > 1e-6 { cs_3 / cs_1 } else { 0.0 };
    let ism_ok = ism_wave_scaling[1].summary.linear_r_squared > 0.95
        && (ratio_2_to_1 - 2.0).abs() < 0.35
        && (ratio_3_to_1 - 3.0).abs() < 0.55
        && ism_wave_scaling[1].summary.far_to_near_amplitude_ratio
            > vicsek_comp.far_to_near_amplitude_ratio;

    // (c) Marginal octic speed stays bounded near `v0 = 12 m/s` from `N = 10` to `N = 1000` with ZERO clamps,
    //     while weak linear Gaussian exhibits significantly larger small-N entropic speed shift `|V(10) - V(1000)|`!
    let lin_n10 = speed_rows
        .iter()
        .find(|r| r.law == "linear_gaussian_weak" && r.n_birds == 10)
        .unwrap();
    let lin_n1000 = speed_rows
        .iter()
        .find(|r| r.law == "linear_gaussian_weak" && r.n_birds == 1000)
        .unwrap();
    let oct_n10 = speed_rows
        .iter()
        .find(|r| r.law == "marginal_octic" && r.n_birds == 10)
        .unwrap();
    let oct_n1000 = speed_rows
        .iter()
        .find(|r| r.law == "marginal_octic" && r.n_birds == 1000)
        .unwrap();

    let lin_shift =
        (lin_n10.speed_stats.mean_group_scalar_speed_mps - lin_n1000.speed_stats.mean_group_scalar_speed_mps).abs();
    let oct_shift =
        (oct_n10.speed_stats.mean_group_scalar_speed_mps - oct_n1000.speed_stats.mean_group_scalar_speed_mps).abs();
    let zero_clamps = speed_rows
        .iter()
        .all(|r| r.speed_stats.speed_clamp_events == 0);
    let marginal_ok = zero_clamps
        && oct_shift < 0.6
        && lin_shift > oct_shift
        && (oct_n1000.speed_stats.mean_group_scalar_speed_mps - 12.0).abs() < 1.5
        && oct_n1000.mean_xi_speed_over_l > 0.20;

    Gate3ReferencePhysicsReport {
        symmetric_spin_conservation_drift: sym_drift,
        directed_nonreciprocal_spin_drift: dir_drift,
        ism_wave_scaling,
        overdamped_vicsek_comparator: vicsek_comp,
        marginal_vs_linear_speed_across_n: speed_rows,
        calm_cohesive_observables: calm_summary,
        passed: spin_ok && ism_ok && marginal_ok,
    }
}

fn run_scenario_file(path: &str) {
    let raw = fs::read_to_string(path).expect("Failed to read scenario TOML file");
    let scen: ScenarioFile = toml::from_str(&raw).expect("Invalid scenario TOML");
    match scen.runner {
        ScenarioKind::MarginalSpp { config } => {
            let dt = config.dt;
            let burn_steps = ((scen.burn_in_s / dt).round()) as usize;
            let meas_steps = (((scen.duration_s - scen.burn_in_s).max(dt) / dt).round()) as usize;
            let (p0, u0) = generate_stress_cloud(0, scen.n_birds, config.master_seed);
            let v0: Vec<Vec3> = u0.into_iter().map(|u| u * 12.0).collect();
            let mut flock = MarginalSppFlock::new(p0, v0);
            for _ in 0..burn_steps {
                flock.step(&config, None);
            }
            let mut acc = SpeedAccumulator::new(scen.n_birds);
            let mut phi_sum = 0.0f32;
            let mut corr_last = compute_spatial_correlations(&flock.pos, &flock.vel, 16);
            let mut snaps = 0usize;
            for k in 0..meas_steps {
                flock.step(&config, None);
                if k % 20 == 0 {
                    acc.record(&flock.vel);
                    phi_sum += compute_polarization(&flock.vel);
                    corr_last = compute_spatial_correlations(&flock.pos, &flock.vel, 16);
                    snaps += 1;
                }
            }
            let out = serde_json::json!({
                "scenario": scen.name,
                "mean_phi": phi_sum / (snaps.max(1) as f32),
                "aspect_ratio": compute_spatial_aspect_ratios(&flock.pos),
                "correlations": corr_last,
                "speed": acc.summarize(flock.speed_clamp_events),
            });
            println!("{}", serde_json::to_string_pretty(&out).unwrap());
        }
        ScenarioKind::ConstantSpeed { config } => {
            let dt = config.dt;
            let total_steps = ((scen.duration_s / dt).round()) as usize;
            let (mut p0, mut u0) = generate_stress_cloud(0, scen.n_birds, config.master_seed);
            if config.turn_pulse.is_some() {
                // Sort along longest ribbon axis (`EY`) and start from aligned cruising flight (`+EX`)
                // so unthermalized t=0 heading noise does not mask the controlled initiator wave.
                p0.sort_by(|a, b| a.y.total_cmp(&b.y));
                u0.fill(Vec3::EX);
            }
            let init_count = config
                .turn_pulse
                .map(|tp| tp.initiator_ids)
                .unwrap_or(8);
            let mut flock = ConstantSpeedFlock::new(p0.clone(), u0, config.v0);
            let mut rec = TurnWaveRecorder::new(p0.clone(), init_count, Vec3::EY, dt);
            let mut pred_cs = 0.0f32;
            for step_idx in 0..total_steps {
                let g = flock.step(&config, None);
                if step_idx == 0 {
                    if let ConstantSpeedTurningLaw::RefIsm { j_ism, chi, .. } = config.turning {
                        let m2 = g.second_spatial_moment(&p0, Vec3::EY);
                        pred_cs = libm::sqrtf((0.5 * j_ism * m2) / chi);
                    }
                }
                rec.record_tick(&flock.accel_perp);
            }
            let out = serde_json::json!({
                "scenario": scen.name,
                "final_phi": compute_polarization(&flock.vel),
                "total_spin_norm": flock.total_spin().norm(),
                "turn_wave": rec.analyze(pred_cs),
            });
            println!("{}", serde_json::to_string_pretty(&out).unwrap());
        }
        ScenarioKind::Coupled { config } => {
            let dt = config.dt;
            let burn_steps = ((scen.burn_in_s / dt).round()) as usize;
            let meas_steps = (((scen.duration_s - scen.burn_in_s).max(dt) / dt).round()) as usize;
            let (p0, u0) = generate_stress_cloud(0, scen.n_birds, config.master_seed);
            let v0: Vec<Vec3> = u0.into_iter().map(|u| u * 12.0).collect();
            let mut flock = CoupledFlock::from_cartesian(p0, v0);
            for _ in 0..burn_steps {
                flock.step(&config, None);
            }
            let t0 = (flock.tick as f32) * dt;
            let mut diff = InternalDiffusionTracker::initialize(&flock.pos, 7, t0);
            let mut acc = SpeedAccumulator::new(scen.n_birds);
            let mut phi_sum = 0.0f32;
            let mut corr_last =
                compute_spatial_correlations(&flock.pos, &flock.velocities(), 16);
            let mut snaps = 0usize;
            for k in 0..meas_steps {
                flock.step(&config, None);
                let t_now = (flock.tick as f32) * dt;
                if k % 5 == 0 {
                    diff.record_40hz(&flock.pos, t_now);
                }
                if k % 20 == 0 {
                    diff.record_qm_10hz(&flock.pos);
                    let vel = flock.velocities();
                    acc.record(&vel);
                    phi_sum += compute_polarization(&vel);
                    corr_last = compute_spatial_correlations(&flock.pos, &vel, 16);
                    snaps += 1;
                }
            }
            let out = serde_json::json!({
                "scenario": scen.name,
                "mean_phi": phi_sum / (snaps.max(1) as f32),
                "aspect_ratio": compute_spatial_aspect_ratios(&flock.pos),
                "correlations": corr_last,
                "diffusion": diff.summarize(),
                "speed": acc.summarize(flock.speed_clamp_events),
                "v_eps_floor_events": flock.v_eps_floor_events,
            });
            println!("{}", serde_json::to_string_pretty(&out).unwrap());
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("verify-stage1");

    match cmd {
        "gate1-oracle" => {
            let r = run_gate1_oracle();
            println!("{}", serde_json::to_string_pretty(&r).unwrap());
            if !(r.edge_cases_exact_match && r.stress_regimes_oracle_1500_exact_match) {
                std::process::exit(1);
            }
        }
        "gate2-convergence" => {
            let r = run_gate2_convergence();
            println!("{}", serde_json::to_string_pretty(&r).unwrap());
            if !r.passed {
                std::process::exit(1);
            }
        }
        "gate3-physics" => {
            let r = run_gate3_reference_physics();
            println!("{}", serde_json::to_string_pretty(&r).unwrap());
            if !r.passed {
                std::process::exit(1);
            }
        }
        "verify-stage1" => {
            let g1 = run_gate1_oracle();
            let g2 = run_gate2_convergence();
            let g3 = run_gate3_reference_physics();
            let passed = g1.edge_cases_exact_match
                && g1.stress_regimes_oracle_1500_exact_match
                && g2.passed
                && g3.passed;
            let full = FullStage1Report {
                gate1_oracle: g1,
                gate2_convergence: g2,
                gate3_reference_physics: g3,
                overall_stage1_passed: passed,
            };
            println!("{}", serde_json::to_string_pretty(&full).unwrap());
            if !passed {
                std::process::exit(1);
            }
        }
        "verify-stage2" => {
            let full = stage2::run_verify_stage2();
            println!("{}", serde_json::to_string_pretty(&full).unwrap());
            if !full.overall_stage2_passed {
                std::process::exit(1);
            }
        }
        "stage3-native-parity" => {
            let bundle = stage3::run_native_stage3_bundle();
            println!("{}", serde_json::to_string_pretty(&bundle).unwrap());
        }
        "run" => {
            let path = args
                .get(2)
                .expect("Usage: sim-cli run <path/to/scenario.toml>");
            run_scenario_file(path);
        }
        other => {
            eprintln!(
                "Unknown subcommand '{other}'. Expected: verify-stage1 | verify-stage2 | gate1-oracle | gate2-convergence | gate3-physics | run <scenario.toml>"
            );
            std::process::exit(2);
        }
    }
}
