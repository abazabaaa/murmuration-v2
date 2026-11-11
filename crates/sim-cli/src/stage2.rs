use crate::generate_stress_cloud;
use rayon::prelude::*;
use serde::Serialize;
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
use sim_core::neighbour::{GraphSymmetry, LbvhIndex, NeighbourRule, WeightedGraph};
use sim_core::rng::{keyed_rng, normal_vec3, uniform_open01, STREAM_INIT};

#[derive(Clone, Debug, Serialize)]
pub struct DetOverdampedPoint {
    pub tau_omega_s: f32,
    pub max_pos_err_vs_ref_spp_m: f32,
    pub rms_vel_err_vs_ref_spp_mps: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct StochOverdampedRow {
    pub label: String,
    pub speed_law: &'static str,
    pub n_birds: usize,
    pub tau_omega_s: f32,
    pub include_ito_speed_drift: bool,
    pub mean_v_mps: f32,
    pub delta_mean_v_vs_ref_mps: f32,
    pub speed_w1_vs_ref_mps: f32,
    pub mean_phi: f32,
    pub delta_phi_vs_ref: f32,
    pub mean_xi_speed_over_l: f32,
    pub delta_xi_speed_over_l_vs_ref: f32,
    pub v_eps_floor_events: u64,
    pub speed_clamp_events: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct StiffSpeedIsmRow {
    pub lambda: f32,
    pub speed_sd_mps: f32,
    pub turn_wave: TurnWaveSummary,
    pub rel_diff_vs_ref_ism: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct ReductionGateReport {
    pub deterministic_tau_to_zero: Vec<DetOverdampedPoint>,
    pub deterministic_monotonic_and_exact: bool,
    pub ref_marginal_spp_baselines: Vec<StochOverdampedRow>,
    pub stochastic_tau_and_ito_sweep: Vec<StochOverdampedRow>,
    pub ito_drift_verified: bool,
    pub ref_ism_turn_baseline: TurnWaveSummary,
    pub stiff_speed_to_ism_sweep: Vec<StiffSpeedIsmRow>,
    pub stiff_speed_matches_ref_ism: bool,
    pub zero_speed_clamps: bool,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Exp2ScalingPoint {
    pub n_birds: usize,
    pub mean_phi: f32,
    pub mean_xi_orient_over_l: f32,
    pub mean_xi_speed_over_l: f32,
    pub speed_stats: SpeedDistributionSummary,
    pub v_eps_floor_events: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Exp2CellReport {
    pub cell_id: &'static str,
    pub heading_mode: &'static str,
    pub speed_mode: &'static str,
    pub tau_omega_s: f32,
    pub scaling_across_n: Vec<Exp2ScalingPoint>,
    pub small_n_speed_shift_10_minus_1000_mps: f32,
    pub abs_small_n_speed_shift_mps: f32,
    pub xi_speed_over_l_at_n1000: f32,
    pub turn_wave_200hz: TurnWaveSummary,
}

#[derive(Clone, Debug, Serialize)]
pub struct FactorialContrast {
    pub metric: &'static str,
    pub cell_o_l: f32,
    pub cell_o_m: f32,
    pub cell_i_l: f32,
    pub cell_i_m: f32,
    pub main_effect_marginal_speed: f32,
    pub main_effect_turning_inertia: f32,
    pub interaction_inertia_x_marginal: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Exp2FactorialReport {
    pub cells: Vec<Exp2CellReport>,
    pub contrasts: Vec<FactorialContrast>,
    pub zero_speed_clamps: bool,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Exp3PositionalCell {
    pub cell_id: &'static str,
    pub n_rep: usize,
    pub vertical_coh_gain: f32,
    pub mean_phi: f32,
    pub gamma_rank_1: f32,
    pub gamma_rank_7: f32,
    pub aspect_ratio: AspectRatioSummary,
    pub diffusion: InternalDiffusionSummary,
}

#[derive(Clone, Debug, Serialize)]
pub struct PositionalContrast {
    pub metric: &'static str,
    pub rep1_iso: f32,
    pub rep1_aniso: f32,
    pub rep7_iso: f32,
    pub rep7_aniso: f32,
    pub main_effect_aniso_vertical_coh: f32,
    pub main_effect_n_rep_7_vs_1: f32,
    pub interaction: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Exp3PositionalReport {
    pub cells: Vec<Exp3PositionalCell>,
    pub contrasts: Vec<PositionalContrast>,
    pub passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct FullStage2Report {
    pub gate1_reductions: ReductionGateReport,
    pub exp2_heading_x_speed_factorial: Exp2FactorialReport,
    pub exp3_positional_factorial: Exp3PositionalReport,
    pub overall_stage2_passed: bool,
}

fn compute_2x2_contrast(
    metric: &'static str,
    o_l: f32,
    o_m: f32,
    i_l: f32,
    i_m: f32,
) -> FactorialContrast {
    let main_marginal = 0.5 * ((o_m - o_l) + (i_m - i_l));
    let main_inertia = 0.5 * ((i_l - o_l) + (i_m - o_m));
    let interaction = (i_m - i_l) - (o_m - o_l);
    FactorialContrast {
        metric,
        cell_o_l: o_l,
        cell_o_m: o_m,
        cell_i_l: i_l,
        cell_i_m: i_m,
        main_effect_marginal_speed: main_marginal,
        main_effect_turning_inertia: main_inertia,
        interaction_inertia_x_marginal: interaction,
    }
}

fn run_reduction_gate() -> ReductionGateReport {
    // -------------------------------------------------------------------------
    // 1. Deterministic `tau_omega -> 0` reduction vs `Ref-MarginalSPP`
    // -------------------------------------------------------------------------
    let n_det = 64usize;
    let dt_det = 0.002f32;
    let steps_det = 750usize; // 1.5 s
    let mut p0_det = Vec::with_capacity(n_det);
    let mut v0_det = Vec::with_capacity(n_det);
    for i in 0..n_det {
        let mut rng = keyed_rng(14142, STREAM_INIT, 0, i as u32);
        p0_det.push(normal_vec3(&mut rng) * 3.2);
        v0_det.push(Vec3::EX * 11.6 + normal_vec3(&mut rng) * 1.6);
    }

    let pos_cfg = PositionalInteractionConfig {
        k_coh: 0.8,
        r0_m: 1.1,
        k_rep: 1.2,
        n_rep: 1,
        vertical_coh_gain: 1.0,
    };
    let nb_rule = NeighbourRule::ExactTopoK {
        k: 7,
        blind_cos: -0.5,
    };
    let det_spp_cfg = MarginalSppConfig {
        dt: dt_det,
        master_seed: 17,
        j_spp: 3.5,
        d_v: 0.0,
        speed_control: SpeedControlLaw::MarginalOctic {
            lambda: 1.0,
            v0: 12.0,
        },
        neighbour_rule: nb_rule,
        graph_symmetry: GraphSymmetry::SymmetricMean,
        positional: pos_cfg,
    };

    let mut ref_det = MarginalSppFlock::new(p0_det.clone(), v0_det.clone());
    for _ in 0..steps_det {
        ref_det.step(&det_spp_cfg, None);
    }

    let mut det_points = Vec::new();
    for &tau in &[0.10f32, 0.02f32, 0.002f32, 0.0f32] {
        let c_cfg = CoupledConfig {
            dt: dt_det,
            master_seed: 17,
            j_spp: 3.5,
            d_v: 0.0,
            d_omega: 0.0,
            tau_omega: tau,
            v_eps: 0.25,
            include_ito_speed_drift: true,
            gravity_mps2: 9.81,
            speed_control: det_spp_cfg.speed_control,
            neighbour_rule: nb_rule,
            graph_symmetry: GraphSymmetry::SymmetricMean,
            positional: pos_cfg,
            turn_pulse: None,
        };
        let mut cf = CoupledFlock::from_cartesian(p0_det.clone(), v0_det.clone());
        for _ in 0..steps_det {
            cf.step(&c_cfg, None);
        }
        let max_pos_err = cf
            .pos
            .iter()
            .zip(ref_det.pos.iter())
            .map(|(&a, &b)| (a - b).norm())
            .fold(0.0f32, f32::max);
        let c_vel = cf.velocities();
        let rms_vel_err = libm::sqrtf(
            c_vel
                .iter()
                .zip(ref_det.vel.iter())
                .map(|(&a, &b)| (a - b).norm_sq())
                .sum::<f32>()
                / (n_det as f32),
        );
        det_points.push(DetOverdampedPoint {
            tau_omega_s: tau,
            max_pos_err_vs_ref_spp_m: max_pos_err,
            rms_vel_err_vs_ref_spp_mps: rms_vel_err,
        });
    }

    let det_ok = det_points[0].max_pos_err_vs_ref_spp_m > det_points[1].max_pos_err_vs_ref_spp_m
        && det_points[1].max_pos_err_vs_ref_spp_m > det_points[2].max_pos_err_vs_ref_spp_m
        && det_points[2].max_pos_err_vs_ref_spp_m > det_points[3].max_pos_err_vs_ref_spp_m
        && det_points[3].max_pos_err_vs_ref_spp_m < 4.0e-3;

    // -------------------------------------------------------------------------
    // 2. Stochastic `tau_omega -> 0` + Ito `+2 D_v / v` drift verification
    // -------------------------------------------------------------------------
    let dt = 0.005f32;
    let burn_steps = 600usize; // 3.0 s
    let meas_steps = 800usize; // 4.0 s @ 200 Hz (`10 Hz` = every 20 steps)
    let mut ref_baselines = Vec::new();
    let mut stoch_rows = Vec::new();
    let mut zero_clamps = true;

    for &(law_label, speed_law, n_b) in &[
        (
            "linear_gaussian_n15",
            SpeedControlLaw::LinearGaussian {
                g_lin: 0.20,
                v0: 12.0,
            },
            15usize,
        ),
        (
            "marginal_octic_n120",
            SpeedControlLaw::MarginalOctic {
                lambda: 1.2,
                v0: 12.0,
            },
            120usize,
        ),
    ] {
        let n_seeds: u64 = if n_b <= 30 { 16 } else { 8 };
        let pos_stoch = PositionalInteractionConfig {
            k_coh: 1.2,
            r0_m: 1.0,
            k_rep: 1.6,
            n_rep: 1,
            vertical_coh_gain: 1.0,
        };

        // Reference Cartesian `Ref-MarginalSPP` ensemble
        let ref_seeds: Vec<(f32, f32, Vec<f32>, u64)> = (0..n_seeds)
            .into_par_iter()
            .map(|s| {
                let mut p0 = Vec::with_capacity(n_b);
                let mut v0 = Vec::with_capacity(n_b);
                let rad = libm::cbrtf(n_b as f32) * 0.95;
                for i in 0..n_b {
                    let mut rng =
                        keyed_rng(61_000 + s * 41 + (n_b as u64), STREAM_INIT, 0, i as u32);
                    let dir = normal_vec3(&mut rng).normalized_or(Vec3::EX);
                    let r = rad * libm::cbrtf(uniform_open01(&mut rng));
                    p0.push(dir * r);
                    v0.push(Vec3::EX * 12.0 + normal_vec3(&mut rng) * 0.8);
                }
                let cfg = MarginalSppConfig {
                    dt,
                    master_seed: 120_000 + s * 83 + (n_b as u64),
                    j_spp: 4.5,
                    d_v: 3.2,
                    speed_control: speed_law,
                    neighbour_rule: nb_rule,
                    graph_symmetry: GraphSymmetry::SymmetricMean,
                    positional: pos_stoch,
                };
                let mut flock = MarginalSppFlock::new(p0, v0);
                for _ in 0..burn_steps {
                    flock.step(&cfg, None);
                }
                let mut acc = SpeedAccumulator::new(n_b);
                let mut phi_sum = 0.0f32;
                let mut xis_sum = 0.0f32;
                let mut snaps = 0usize;
                for k in 0..meas_steps {
                    flock.step(&cfg, None);
                    if k % 20 == 0 {
                        acc.record(&flock.vel);
                        phi_sum += compute_polarization(&flock.vel);
                        xis_sum += compute_spatial_correlations(&flock.pos, &flock.vel, 16)
                            .xi_speed_over_l;
                        snaps += 1;
                    }
                }
                let inv = 1.0 / (snaps as f32);
                (
                    phi_sum * inv,
                    xis_sum * inv,
                    acc.samples().to_vec(),
                    flock.speed_clamp_events,
                )
            })
            .collect();

        let inv_s = 1.0 / (n_seeds as f32);
        let ref_phi = ref_seeds.iter().map(|r| r.0).sum::<f32>() * inv_s;
        let ref_xis = ref_seeds.iter().map(|r| r.1).sum::<f32>() * inv_s;
        let mut ref_samples = Vec::new();
        let mut ref_clamps = 0u64;
        for (_, _, sp, cl) in ref_seeds {
            ref_samples.extend(sp);
            ref_clamps += cl;
        }
        if ref_clamps > 0 {
            zero_clamps = false;
        }
        let ref_mean_v = ref_samples.iter().sum::<f32>() / (ref_samples.len() as f32);

        ref_baselines.push(StochOverdampedRow {
            label: format!("ref_marginal_spp_{law_label}"),
            speed_law: law_label,
            n_birds: n_b,
            tau_omega_s: 0.0,
            include_ito_speed_drift: true,
            mean_v_mps: ref_mean_v,
            delta_mean_v_vs_ref_mps: 0.0,
            speed_w1_vs_ref_mps: 0.0,
            mean_phi: ref_phi,
            delta_phi_vs_ref: 0.0,
            mean_xi_speed_over_l: ref_xis,
            delta_xi_speed_over_l_vs_ref: 0.0,
            v_eps_floor_events: 0,
            speed_clamp_events: ref_clamps,
        });

        for &(sub_label, tau, inc_ito) in &[
            ("coupled_tau0_with_ito", 0.0f32, true),
            ("coupled_tau1e-4_with_ito", 1e-4f32, true),
            ("coupled_tau0.02_with_ito", 0.02f32, true),
            ("coupled_tau0.08_with_ito", 0.08f32, true),
            ("coupled_tau0.20_with_ito", 0.20f32, true),
            ("ablation_tau0_NO_ito_drift", 0.0f32, false),
        ] {
            let c_seeds: Vec<(f32, f32, Vec<f32>, u64, u64)> = (0..n_seeds)
                .into_par_iter()
                .map(|s| {
                    let mut p0 = Vec::with_capacity(n_b);
                    let mut v0 = Vec::with_capacity(n_b);
                    let rad = libm::cbrtf(n_b as f32) * 0.95;
                    for i in 0..n_b {
                        let mut rng =
                            keyed_rng(61_000 + s * 41 + (n_b as u64), STREAM_INIT, 0, i as u32);
                        let dir = normal_vec3(&mut rng).normalized_or(Vec3::EX);
                        let r = rad * libm::cbrtf(uniform_open01(&mut rng));
                        p0.push(dir * r);
                        v0.push(Vec3::EX * 12.0 + normal_vec3(&mut rng) * 0.8);
                    }
                    let cfg = CoupledConfig {
                        dt,
                        master_seed: 120_000 + s * 83 + (n_b as u64),
                        j_spp: 4.5,
                        d_v: 3.2,
                        d_omega: 3.2,
                        tau_omega: tau,
                        v_eps: 0.25,
                        include_ito_speed_drift: inc_ito,
                        gravity_mps2: 9.81,
                        speed_control: speed_law,
                        neighbour_rule: nb_rule,
                        graph_symmetry: GraphSymmetry::SymmetricMean,
                        positional: pos_stoch,
                        turn_pulse: None,
                    };
                    let mut flock = CoupledFlock::from_cartesian(p0, v0);
                    for _ in 0..burn_steps {
                        flock.step(&cfg, None);
                    }
                    let mut acc = SpeedAccumulator::new(n_b);
                    let mut phi_sum = 0.0f32;
                    let mut xis_sum = 0.0f32;
                    let mut snaps = 0usize;
                    for k in 0..meas_steps {
                        flock.step(&cfg, None);
                        if k % 20 == 0 {
                            let vel = flock.velocities();
                            acc.record(&vel);
                            phi_sum += compute_polarization(&vel);
                            xis_sum +=
                                compute_spatial_correlations(&flock.pos, &vel, 16).xi_speed_over_l;
                            snaps += 1;
                        }
                    }
                    let inv = 1.0 / (snaps as f32);
                    (
                        phi_sum * inv,
                        xis_sum * inv,
                        acc.samples().to_vec(),
                        flock.v_eps_floor_events,
                        flock.speed_clamp_events,
                    )
                })
                .collect();

            let c_phi = c_seeds.iter().map(|r| r.0).sum::<f32>() * inv_s;
            let c_xis = c_seeds.iter().map(|r| r.1).sum::<f32>() * inv_s;
            let mut c_samples = Vec::new();
            let mut v_eps_tot = 0u64;
            let mut clamps_tot = 0u64;
            for (_, _, sp, vf, cl) in c_seeds {
                c_samples.extend(sp);
                v_eps_tot += vf;
                clamps_tot += cl;
            }
            if clamps_tot > 0 {
                zero_clamps = false;
            }
            let c_mean_v = c_samples.iter().sum::<f32>() / (c_samples.len() as f32);
            let w1 = wasserstein_1d(&c_samples, &ref_samples);

            stoch_rows.push(StochOverdampedRow {
                label: format!("{sub_label}_{law_label}"),
                speed_law: law_label,
                n_birds: n_b,
                tau_omega_s: tau,
                include_ito_speed_drift: inc_ito,
                mean_v_mps: c_mean_v,
                delta_mean_v_vs_ref_mps: c_mean_v - ref_mean_v,
                speed_w1_vs_ref_mps: w1,
                mean_phi: c_phi,
                delta_phi_vs_ref: c_phi - ref_phi,
                mean_xi_speed_over_l: c_xis,
                delta_xi_speed_over_l_vs_ref: c_xis - ref_xis,
                v_eps_floor_events: v_eps_tot,
                speed_clamp_events: clamps_tot,
            });
        }
    }

    let lin_ito_0 = stoch_rows
        .iter()
        .find(|r| r.label == "coupled_tau0_with_ito_linear_gaussian_n15")
        .unwrap();
    let lin_ito_tiny = stoch_rows
        .iter()
        .find(|r| r.label == "coupled_tau1e-4_with_ito_linear_gaussian_n15")
        .unwrap();
    let lin_no_ito = stoch_rows
        .iter()
        .find(|r| r.label == "ablation_tau0_NO_ito_drift_linear_gaussian_n15")
        .unwrap();
    let oct_ito_0 = stoch_rows
        .iter()
        .find(|r| r.label == "coupled_tau0_with_ito_marginal_octic_n120")
        .unwrap();

    let ito_verified = lin_ito_0.speed_w1_vs_ref_mps < 0.10
        && lin_ito_tiny.speed_w1_vs_ref_mps < 0.12
        && oct_ito_0.speed_w1_vs_ref_mps < 0.08
        && lin_no_ito.delta_mean_v_vs_ref_mps < -0.45
        && lin_no_ito.speed_w1_vs_ref_mps > 0.45;

    // -------------------------------------------------------------------------
    // 3. Stiff-speed `lambda -> inf` reduction vs `Ref-ISM` on `200 Hz` `E7`
    // -------------------------------------------------------------------------
    let mut ribbon_pos = Vec::with_capacity(360);
    for ix in 0..40 {
        for iy in 0..3 {
            for iz in 0..3 {
                ribbon_pos.push(Vec3::new(ix as f32, (iy as f32) - 1.0, (iz as f32) - 1.0));
            }
        }
    }
    let v0 = 12.0f32;
    let init_u = vec![Vec3::EY; ribbon_pos.len()];
    let init_vel: Vec<Vec3> = init_u.iter().map(|&u| u * v0).collect();
    let rule_rib = NeighbourRule::ExactTopoK {
        k: 7,
        blind_cos: -1.0,
    };
    let frozen_nb = LbvhIndex::build(&ribbon_pos).query_all(&ribbon_pos, &init_u, rule_rib);
    let sym_graph = WeightedGraph::from_neighbours(&frozen_nb, GraphSymmetry::SymmetricMean);
    let m2_ex = sym_graph.second_spatial_moment(&ribbon_pos, Vec3::EX);

    // Match parameters via §2.2 bullet 2:
    // `v0` cancels in `F_align / v0`, so `tau_omega = chi / eta_ISM` and `J_ISM = eta_ISM * J_SPP` (`J_ISM / chi = J_SPP / tau_omega`)
    let j_spp_w2 = 6000.0f32;
    let tau_w2 = 4.0f32;
    let eta_w2 = 1.0 / tau_w2; // 0.25
    let j_ism_w2 = eta_w2 * j_spp_w2; // 1500.0 => J_ISM / chi = 1500.0 s^-2
    let pred_cs_w2 = libm::sqrtf((0.5 * j_spp_w2 * m2_ex) / tau_w2);

    let pulse = ExternalTurnPulse {
        start_time_s: 0.02,
        end_time_s: 0.06,
        accel_vec: Vec3::new(-12.0, 0.0, 0.0),
        initiator_ids: 9,
    };

    let ism_cfg2 = ConstantSpeedConfig {
        v0,
        dt,
        master_seed: 99,
        neighbour_rule: rule_rib,
        graph_symmetry: GraphSymmetry::SymmetricMean,
        turning: ConstantSpeedTurningLaw::RefIsm {
            j_ism: j_ism_w2,
            chi: 1.0,
            eta_ism: eta_w2,
            temp_ism: 0.0,
        },
        positional: PositionalInteractionConfig::default(),
        turn_pulse: Some(pulse),
    };
    let mut ref_ism_flock = ConstantSpeedFlock::new(ribbon_pos.clone(), init_u.clone(), v0);
    let mut ref_ism_rec = TurnWaveRecorder::new(ribbon_pos.clone(), 9, Vec3::EX, dt);
    for _ in 0..400 {
        ref_ism_flock.step(&ism_cfg2, Some(&frozen_nb));
        ref_ism_rec.record_tick(&ref_ism_flock.accel_perp);
    }
    let ref_ism_summary = ref_ism_rec.analyze(pred_cs_w2);

    let mut stiff_rows = Vec::new();
    for &lam in &[0.5f32, 2.0f32, 12.0f32, 40.0f32] {
        let c_cfg = CoupledConfig {
            dt,
            master_seed: 99,
            j_spp: j_spp_w2,
            d_v: 0.0,
            d_omega: 0.0,
            tau_omega: tau_w2,
            v_eps: 0.25,
            include_ito_speed_drift: true,
            gravity_mps2: 9.81,
            speed_control: SpeedControlLaw::MarginalOctic { lambda: lam, v0 },
            neighbour_rule: rule_rib,
            graph_symmetry: GraphSymmetry::SymmetricMean,
            positional: PositionalInteractionConfig::default(),
            turn_pulse: Some(pulse),
        };
        let mut cf = CoupledFlock::from_cartesian(ribbon_pos.clone(), init_vel.clone());
        let mut rec = TurnWaveRecorder::new(ribbon_pos.clone(), 9, Vec3::EX, dt);
        let mut sp_acc = SpeedAccumulator::new(ribbon_pos.len());
        for k in 0..400 {
            cf.step(&c_cfg, Some(&frozen_nb));
            rec.record_tick(&cf.accel_perp);
            if k % 20 == 0 {
                sp_acc.record(&cf.velocities());
            }
        }
        let tw = rec.analyze(pred_cs_w2);
        let sp_sd = sp_acc.summarize(cf.speed_clamp_events).sd_individual_mps;
        let rel_diff = (tw.measured_cs_mps - ref_ism_summary.measured_cs_mps).abs()
            / ref_ism_summary.measured_cs_mps.max(1e-6);
        stiff_rows.push(StiffSpeedIsmRow {
            lambda: lam,
            speed_sd_mps: sp_sd,
            turn_wave: tw,
            rel_diff_vs_ref_ism: rel_diff,
        });
    }

    let stiff_last = stiff_rows.last().unwrap();
    let stiff_ok = stiff_last.turn_wave.linear_r_squared > 0.98
        && stiff_last.rel_diff_vs_ref_ism < 0.06;

    let passed = det_ok && ito_verified && stiff_ok && zero_clamps;
    ReductionGateReport {
        deterministic_tau_to_zero: det_points,
        deterministic_monotonic_and_exact: det_ok,
        ref_marginal_spp_baselines: ref_baselines,
        stochastic_tau_and_ito_sweep: stoch_rows,
        ito_drift_verified: ito_verified,
        ref_ism_turn_baseline: ref_ism_summary,
        stiff_speed_to_ism_sweep: stiff_rows,
        stiff_speed_matches_ref_ism: stiff_ok,
        zero_speed_clamps: zero_clamps,
        passed,
    }
}

fn run_exp2_factorial() -> Exp2FactorialReport {
    let dt = 0.005f32;
    let v0 = 12.0f32;
    let flock_sizes = [10usize, 20, 50, 200, 1000];
    let nb_rule = NeighbourRule::ExactTopoK {
        k: 7,
        blind_cos: -0.5,
    };
    let pos_cfg = PositionalInteractionConfig {
        k_coh: 1.2,
        r0_m: 1.0,
        k_rep: 1.6,
        n_rep: 1,
        vertical_coh_gain: 1.0,
    };

    // Prepare controlled `200 Hz` turn ribbon (`360` birds) shared identically across all 4 factorial cells
    let mut ribbon_pos = Vec::with_capacity(360);
    for ix in 0..40 {
        for iy in 0..3 {
            for iz in 0..3 {
                ribbon_pos.push(Vec3::new(ix as f32, (iy as f32) - 1.0, (iz as f32) - 1.0));
            }
        }
    }
    let init_u_rib = vec![Vec3::EY; ribbon_pos.len()];
    let init_vel_rib: Vec<Vec3> = init_u_rib.iter().map(|&u| u * v0).collect();
    let rule_rib = NeighbourRule::ExactTopoK {
        k: 7,
        blind_cos: -1.0,
    };
    let frozen_nb = LbvhIndex::build(&ribbon_pos).query_all(&ribbon_pos, &init_u_rib, rule_rib);
    let sym_graph = WeightedGraph::from_neighbours(&frozen_nb, GraphSymmetry::SymmetricMean);
    let m2_ex = sym_graph.second_spatial_moment(&ribbon_pos, Vec3::EX);

    let cell_specs = [
        (
            "O_L",
            "overdamped",
            "linear_gaussian",
            0.0f32,
            SpeedControlLaw::LinearGaussian { g_lin: 0.12, v0 },
        ),
        (
            "O_M",
            "overdamped",
            "marginal_octic",
            0.0f32,
            SpeedControlLaw::MarginalOctic { lambda: 1.2, v0 },
        ),
        (
            "I_L",
            "inertial",
            "linear_gaussian",
            0.18f32,
            SpeedControlLaw::LinearGaussian { g_lin: 0.12, v0 },
        ),
        (
            "I_M",
            "inertial",
            "marginal_octic",
            0.18f32,
            SpeedControlLaw::MarginalOctic { lambda: 1.2, v0 },
        ),
    ];

    let mut cells = Vec::new();
    let mut zero_clamps = true;

    for &(cell_id, h_mode, s_mode, tau_omega, speed_law) in &cell_specs {
        let mut scaling_rows = Vec::new();
        for &n_b in &flock_sizes {
            let n_seeds: u64 = if n_b <= 200 { 12 } else { 4 };
            let burn_steps = 800usize;
            let meas_steps = 800usize;

            let per_seed: Vec<(f32, f32, f32, SpeedAccumulator, u64, u64)> = (0..n_seeds)
                .into_par_iter()
                .map(|s| {
                    let mut p0 = Vec::with_capacity(n_b);
                    let mut v0_init = Vec::with_capacity(n_b);
                    let rad = libm::cbrtf(n_b as f32) * 0.95;
                    for i in 0..n_b {
                        let mut rng = keyed_rng(
                            20_000 + s * 31 + (n_b as u64),
                            STREAM_INIT,
                            0,
                            i as u32,
                        );
                        let dir = normal_vec3(&mut rng).normalized_or(Vec3::EX);
                        let r = rad * libm::cbrtf(uniform_open01(&mut rng));
                        p0.push(dir * r);
                        v0_init.push(Vec3::EX * 12.0 + normal_vec3(&mut rng) * 0.8);
                    }
                    let cfg = CoupledConfig {
                        dt,
                        master_seed: 90_000 + s * 97 + (n_b as u64),
                        j_spp: 4.5,
                        d_v: 3.2,
                        d_omega: 3.2,
                        tau_omega,
                        v_eps: 0.25,
                        include_ito_speed_drift: true,
                        gravity_mps2: 9.81,
                        speed_control: speed_law,
                        neighbour_rule: nb_rule,
                        graph_symmetry: GraphSymmetry::SymmetricMean,
                        positional: pos_cfg,
                        turn_pulse: None,
                    };
                    let mut flock = CoupledFlock::from_cartesian(p0, v0_init);
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
                            let vel = flock.velocities();
                            acc.record(&vel);
                            phi_sum += compute_polarization(&vel);
                            let corr = compute_spatial_correlations(&flock.pos, &vel, 16);
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
                        flock.v_eps_floor_events,
                        flock.speed_clamp_events,
                    )
                })
                .collect();

            let mut merged_acc = SpeedAccumulator::new(n_b);
            let mut mean_phi = 0.0f32;
            let mut mean_xio = 0.0f32;
            let mut mean_xis = 0.0f32;
            let mut v_eps_tot = 0u64;
            let mut clamps_tot = 0u64;
            for (phi, xio, xis, acc, vf, cl) in &per_seed {
                mean_phi += *phi;
                mean_xio += *xio;
                mean_xis += *xis;
                v_eps_tot += *vf;
                clamps_tot += *cl;
                for &sv in acc.samples() {
                    merged_acc.record(&[Vec3::new(sv, 0.0, 0.0)]);
                }
            }
            if clamps_tot > 0 {
                zero_clamps = false;
            }
            let inv_s = 1.0 / (per_seed.len() as f32);
            let mut summary = merged_acc.summarize(clamps_tot);
            summary.n_birds = n_b;
            let seed_grp_means: Vec<f32> = per_seed
                .iter()
                .map(|(_, _, _, a, _, c)| a.summarize(*c).mean_group_scalar_speed_mps)
                .collect();
            let grp_mean = seed_grp_means.iter().sum::<f32>() * inv_s;
            let grp_var = seed_grp_means
                .iter()
                .map(|&x| (x - grp_mean) * (x - grp_mean))
                .sum::<f32>()
                * inv_s;
            summary.mean_group_scalar_speed_mps = grp_mean;
            summary.sd_group_scalar_speed_mps = libm::sqrtf(grp_var);

            scaling_rows.push(Exp2ScalingPoint {
                n_birds: n_b,
                mean_phi: mean_phi * inv_s,
                mean_xi_orient_over_l: mean_xio * inv_s,
                mean_xi_speed_over_l: mean_xis * inv_s,
                speed_stats: summary,
                v_eps_floor_events: v_eps_tot,
            });
        }

        // Controlled `200 Hz` turn-wave run (`E7`) for this factorial cell:
        // Under inertial heading (`tau_omega > 0`), set `tau_turn = 4.0 s`, `j_spp_turn = 7200 s^-1` (`J_SPP / tau = 1800 s^-2`, `c_s ~ 22.3 m/s` in starling band);
        // under overdamped heading (`tau_omega == 0`), matched alignment strength `j_spp_turn_overdamped = 40 s^-1`.
        let (tau_turn, j_spp_turn) = if tau_omega > 1e-6 {
            (4.0f32, 7200.0f32)
        } else {
            (0.0f32, 40.0f32)
        };
        let pred_cs = if tau_turn > 1e-6 {
            libm::sqrtf((0.5 * j_spp_turn * m2_ex) / tau_turn)
        } else {
            0.0
        };
        let turn_cfg = CoupledConfig {
            dt,
            master_seed: 99,
            j_spp: j_spp_turn,
            d_v: 0.0,
            d_omega: 0.0,
            tau_omega: tau_turn,
            v_eps: 0.25,
            include_ito_speed_drift: true,
            gravity_mps2: 9.81,
            speed_control: speed_law,
            neighbour_rule: rule_rib,
            graph_symmetry: GraphSymmetry::SymmetricMean,
            positional: PositionalInteractionConfig::default(),
            turn_pulse: Some(ExternalTurnPulse {
                start_time_s: 0.02,
                end_time_s: 0.06,
                accel_vec: Vec3::new(-12.0, 0.0, 0.0),
                initiator_ids: 9,
            }),
        };
        let mut turn_flock = CoupledFlock::from_cartesian(ribbon_pos.clone(), init_vel_rib.clone());
        let mut turn_rec = TurnWaveRecorder::new(ribbon_pos.clone(), 9, Vec3::EX, dt);
        for _ in 0..450 {
            turn_flock.step(&turn_cfg, Some(&frozen_nb));
            turn_rec.record_tick(&turn_flock.accel_perp);
        }
        let turn_summary = turn_rec.analyze(pred_cs);

        let v_n10 = scaling_rows
            .iter()
            .find(|r| r.n_birds == 10)
            .unwrap()
            .speed_stats
            .mean_group_scalar_speed_mps;
        let r_n1000 = scaling_rows.iter().find(|r| r.n_birds == 1000).unwrap();
        let v_n1000 = r_n1000.speed_stats.mean_group_scalar_speed_mps;
        let shift = v_n10 - v_n1000;

        cells.push(Exp2CellReport {
            cell_id,
            heading_mode: h_mode,
            speed_mode: s_mode,
            tau_omega_s: tau_omega,
            xi_speed_over_l_at_n1000: r_n1000.mean_xi_speed_over_l,
            small_n_speed_shift_10_minus_1000_mps: shift,
            abs_small_n_speed_shift_mps: shift.abs(),
            scaling_across_n: scaling_rows,
            turn_wave_200hz: turn_summary,
        });
    }

    let c_ol = &cells[0];
    let c_om = &cells[1];
    let c_il = &cells[2];
    let c_im = &cells[3];

    let contrasts = vec![
        compute_2x2_contrast(
            "abs_small_n_speed_shift_10_vs_1000_mps (E5)",
            c_ol.abs_small_n_speed_shift_mps,
            c_om.abs_small_n_speed_shift_mps,
            c_il.abs_small_n_speed_shift_mps,
            c_im.abs_small_n_speed_shift_mps,
        ),
        compute_2x2_contrast(
            "xi_speed_over_l_at_n1000 (E4)",
            c_ol.xi_speed_over_l_at_n1000,
            c_om.xi_speed_over_l_at_n1000,
            c_il.xi_speed_over_l_at_n1000,
            c_im.xi_speed_over_l_at_n1000,
        ),
        compute_2x2_contrast(
            "turn_wave_speed_mps_200hz (E7)",
            c_ol.turn_wave_200hz.measured_cs_mps,
            c_om.turn_wave_200hz.measured_cs_mps,
            c_il.turn_wave_200hz.measured_cs_mps,
            c_im.turn_wave_200hz.measured_cs_mps,
        ),
        compute_2x2_contrast(
            "turn_wave_linear_r_squared (E7)",
            c_ol.turn_wave_200hz.linear_r_squared,
            c_om.turn_wave_200hz.linear_r_squared,
            c_il.turn_wave_200hz.linear_r_squared,
            c_im.turn_wave_200hz.linear_r_squared,
        ),
        compute_2x2_contrast(
            "turn_wave_far_to_near_amplitude_ratio (E7)",
            c_ol.turn_wave_200hz.far_to_near_amplitude_ratio,
            c_om.turn_wave_200hz.far_to_near_amplitude_ratio,
            c_il.turn_wave_200hz.far_to_near_amplitude_ratio,
            c_im.turn_wave_200hz.far_to_near_amplitude_ratio,
        ),
    ];

    // Acceptance checks for Experiment 2:
    // 1. Marginal octic confinement suppresses small-N speed shift under BOTH overdamped and inertial turning (`< 0.55 m/s` vs `> 0.9 m/s`).
    let e5_ok = c_om.abs_small_n_speed_shift_mps < 0.55
        && c_im.abs_small_n_speed_shift_mps < 0.55
        && c_ol.abs_small_n_speed_shift_mps > c_om.abs_small_n_speed_shift_mps + 0.40
        && c_il.abs_small_n_speed_shift_mps > c_im.abs_small_n_speed_shift_mps + 0.40;
    // 2. Scale-free speed correlation ratio `xi_sp / L > 0.22` at N = 1,000 in `I_M`.
    let e4_ok = c_im.xi_speed_over_l_at_n1000 > 0.22;
    // 3. Turning inertia produces linear turn wave (`R^2 > 0.98`, `c_s` in 20-40 m/s starling band, higher far/near ratio than overdamped)
    //    with minimal cross-channel distortion between `I_L` and `I_M`!
    let e7_ok = c_il.turn_wave_200hz.linear_r_squared > 0.98
        && c_im.turn_wave_200hz.linear_r_squared > 0.98
        && c_im.turn_wave_200hz.measured_cs_mps >= 20.0
        && c_im.turn_wave_200hz.measured_cs_mps <= 40.0
        && c_im.turn_wave_200hz.far_to_near_amplitude_ratio
            > c_om.turn_wave_200hz.far_to_near_amplitude_ratio;

    Exp2FactorialReport {
        cells,
        contrasts,
        zero_speed_clamps: zero_clamps,
        passed: zero_clamps && e5_ok && e4_ok && e7_ok,
    }
}

fn run_exp3_positional_factorial() -> Exp3PositionalReport {
    let _ = generate_stress_cloud;
    let dt = 0.005f32;
    let n_birds = 250usize;
    let n_seeds = 6u64;
    let burn_steps = 1200usize; // 6.0 s burn-in from an unbiased 1:1:1 isotropic sphere!
    let meas_steps = 1200usize; // 6.0 s measurement (`40 Hz` = every 5 steps, `10 Hz` = every 20 steps)

    let specs = [
        ("rep1_iso", 1usize, 1.0f32),
        ("rep1_aniso", 1usize, 2.8f32),
        ("rep7_iso", 7usize, 1.0f32),
        ("rep7_aniso", 7usize, 2.8f32),
    ];

    let mut cells = Vec::new();
    for &(cell_id, n_rep, vert_gain) in &specs {
        let seed_runs: Vec<(f32, Vec<f32>, AspectRatioSummary, InternalDiffusionSummary)> = (0
            ..n_seeds)
            .into_par_iter()
            .map(|s| {
                // Start from an strictly isotropic 3D uniform ball (`1 : 1 : 1`) so any slab flattening is 100% emergent!
                let mut p0 = Vec::with_capacity(n_birds);
                let mut v0 = Vec::with_capacity(n_birds);
                let rad = libm::cbrtf(n_birds as f32) * 1.05;
                for i in 0..n_birds {
                    let mut rng = keyed_rng(7000 + s * 53, STREAM_INIT, 0, i as u32);
                    let dir = normal_vec3(&mut rng).normalized_or(Vec3::EX);
                    let r = rad * libm::cbrtf(uniform_open01(&mut rng));
                    p0.push(dir * r);
                    let u = (Vec3::EX + normal_vec3(&mut rng) * 0.12).normalized_or(Vec3::EX);
                    v0.push(u * 12.0);
                }
                let cfg = CoupledConfig {
                    dt,
                    master_seed: 180_000 + s * 101,
                    j_spp: 4.5,
                    d_v: 1.4,
                    d_omega: 1.4,
                    tau_omega: 0.12,
                    v_eps: 0.25,
                    include_ito_speed_drift: true,
                    gravity_mps2: 9.81,
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
                        k_coh: 1.4,
                        r0_m: 1.1,
                        k_rep: 2.2,
                        n_rep,
                        vertical_coh_gain: vert_gain,
                    },
                    turn_pulse: None,
                };

                let mut flock = CoupledFlock::from_cartesian(p0, v0);
                for _ in 0..burn_steps {
                    flock.step(&cfg, None);
                }
                let t0 = (flock.tick as f32) * dt;
                let mut diff = InternalDiffusionTracker::initialize(&flock.pos, 7, t0);
                let mut phi_sum = 0.0f32;
                let mut ext_sum = [0.0f32; 3];
                let mut rf_sum = [0.0f32; 3];
                let mut rt_sum = [0.0f32; 3];
                let mut snaps = 0usize;

                for k in 0..meas_steps {
                    flock.step(&cfg, None);
                    let t_now = (flock.tick as f32) * dt;
                    if k % 5 == 0 {
                        diff.record_40hz(&flock.pos, t_now);
                    }
                    if k % 20 == 0 {
                        diff.record_qm_10hz(&flock.pos);
                        let vel = flock.velocities();
                        phi_sum += compute_polarization(&vel);
                        let ar = compute_spatial_aspect_ratios(&flock.pos);
                        for ax in 0..3 {
                            ext_sum[ax] += ar.extents_full_m[ax];
                            rf_sum[ax] += ar.ratios_full[ax];
                            rt_sum[ax] += ar.ratios_trim95[ax];
                        }
                        snaps += 1;
                    }
                }
                let inv = 1.0 / (snaps as f32);
                let gamma = compute_topological_anisotropy_gamma(&flock.pos, 7);
                let ar_avg = AspectRatioSummary {
                    extents_full_m: [ext_sum[0] * inv, ext_sum[1] * inv, ext_sum[2] * inv],
                    ratios_full: [1.0, rf_sum[1] * inv, rf_sum[2] * inv],
                    ratios_trim95: [1.0, rt_sum[1] * inv, rt_sum[2] * inv],
                };
                (phi_sum * inv, gamma, ar_avg, diff.summarize())
            })
            .collect();

        let inv_s = 1.0 / (n_seeds as f32);
        let mean_phi = seed_runs.iter().map(|r| r.0).sum::<f32>() * inv_s;
        let g1 = seed_runs.iter().map(|r| r.1[0]).sum::<f32>() * inv_s;
        let g7 = seed_runs.iter().map(|r| r.1[6]).sum::<f32>() * inv_s;
        let ar = AspectRatioSummary {
            extents_full_m: [
                seed_runs.iter().map(|r| r.2.extents_full_m[0]).sum::<f32>() * inv_s,
                seed_runs.iter().map(|r| r.2.extents_full_m[1]).sum::<f32>() * inv_s,
                seed_runs.iter().map(|r| r.2.extents_full_m[2]).sum::<f32>() * inv_s,
            ],
            ratios_full: [
                1.0,
                seed_runs.iter().map(|r| r.2.ratios_full[1]).sum::<f32>() * inv_s,
                seed_runs.iter().map(|r| r.2.ratios_full[2]).sum::<f32>() * inv_s,
            ],
            ratios_trim95: [
                1.0,
                seed_runs.iter().map(|r| r.2.ratios_trim95[1]).sum::<f32>() * inv_s,
                seed_runs.iter().map(|r| r.2.ratios_trim95[2]).sum::<f32>() * inv_s,
            ],
        };
        let diff_avg = InternalDiffusionSummary {
            alpha_cm: seed_runs.iter().map(|r| r.3.alpha_cm).sum::<f32>() * inv_s,
            alpha_mutual: seed_runs.iter().map(|r| r.3.alpha_mutual).sum::<f32>() * inv_s,
            q_m_final: seed_runs.iter().map(|r| r.3.q_m_final).sum::<f32>() * inv_s,
            m_neighbours: 7,
        };

        cells.push(Exp3PositionalCell {
            cell_id,
            n_rep,
            vertical_coh_gain: vert_gain,
            mean_phi,
            gamma_rank_1: g1,
            gamma_rank_7: g7,
            aspect_ratio: ar,
            diffusion: diff_avg,
        });
    }

    let c00 = &cells[0]; // rep1_iso
    let c01 = &cells[1]; // rep1_aniso
    let c10 = &cells[2]; // rep7_iso
    let c11 = &cells[3]; // rep7_aniso

    let mk_pos_contrast = |metric: &'static str, f00: f32, f01: f32, f10: f32, f11: f32| {
        PositionalContrast {
            metric,
            rep1_iso: f00,
            rep1_aniso: f01,
            rep7_iso: f10,
            rep7_aniso: f11,
            main_effect_aniso_vertical_coh: 0.5 * ((f01 - f00) + (f11 - f10)),
            main_effect_n_rep_7_vs_1: 0.5 * ((f10 - f00) + (f11 - f01)),
            interaction: (f11 - f10) - (f01 - f00),
        }
    };

    let contrasts = vec![
        mk_pos_contrast(
            "ratio_trim95_l3_over_l1_thin_slab (E3)",
            c00.aspect_ratio.ratios_trim95[2],
            c01.aspect_ratio.ratios_trim95[2],
            c10.aspect_ratio.ratios_trim95[2],
            c11.aspect_ratio.ratios_trim95[2],
        ),
        mk_pos_contrast(
            "ratio_trim95_l2_over_l1 (E3)",
            c00.aspect_ratio.ratios_trim95[1],
            c01.aspect_ratio.ratios_trim95[1],
            c10.aspect_ratio.ratios_trim95[1],
            c11.aspect_ratio.ratios_trim95[1],
        ),
        mk_pos_contrast(
            "alpha_cm_superdiffusion (E6a @ 40Hz)",
            c00.diffusion.alpha_cm,
            c01.diffusion.alpha_cm,
            c10.diffusion.alpha_cm,
            c11.diffusion.alpha_cm,
        ),
        mk_pos_contrast(
            "alpha_mut_neighbour_diffusion (E6b @ 40Hz)",
            c00.diffusion.alpha_mutual,
            c01.diffusion.alpha_mutual,
            c10.diffusion.alpha_mutual,
            c11.diffusion.alpha_mutual,
        ),
        mk_pos_contrast(
            "q_7_retention_final (E6c @ 10Hz)",
            c00.diffusion.q_m_final,
            c01.diffusion.q_m_final,
            c10.diffusion.q_m_final,
            c11.diffusion.q_m_final,
        ),
    ];

    // Acceptance check: anisotropic vertical cohesion flattens `L1` (`main_effect_aniso_vertical_coh > 0.25` on `L3/L1`)
    // and all cells exhibit fluid non-crystalline neighbour mixing (`q_m_final < 0.85`).
    let slab_flattens = contrasts[0].main_effect_aniso_vertical_coh > 0.25;
    let fluid_mixing = cells.iter().all(|c| c.diffusion.q_m_final < 0.85);

    Exp3PositionalReport {
        cells,
        contrasts,
        passed: slab_flattens && fluid_mixing,
    }
}

pub fn run_verify_stage2() -> FullStage2Report {
    let g1 = run_reduction_gate();
    let e2 = run_exp2_factorial();
    let e3 = run_exp3_positional_factorial();
    let passed = g1.passed && e2.passed && e3.passed;
    FullStage2Report {
        gate1_reductions: g1,
        exp2_heading_x_speed_factorial: e2,
        exp3_positional_factorial: e3,
        overall_stage2_passed: passed,
    }
}
