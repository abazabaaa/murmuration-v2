use rayon::prelude::*;
use serde::Serialize;
use sim_core::coupled::CoupledFlock;
use sim_core::instance::{generate_stress_cloud, BirdVisualGaitBank, PoseInterpolator};
use sim_core::math::Vec3;
use sim_core::parity::{
    default_coupled_browser_config, run_parity_1000_ticks, run_tost_seed_rollout,
};
use std::time::Instant;

#[derive(Clone, Debug, Serialize)]
pub struct NativeModelBitDump {
    pub model_name: &'static str,
    pub model_kind: u32,
    pub n_birds: usize,
    pub seed: u64,
    pub tick_600_u32_bits: Vec<u32>,
    pub tick_1000_u32_bits: Vec<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeTostSeedRow {
    pub seed: u64,
    pub f32_values: Vec<f32>,
    pub u32_bits: Vec<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeStressTickRow {
    pub regime: &'static str,
    pub regime_idx: usize,
    pub n_birds: usize,
    pub t_sim_tick_median_ms: f64,
    pub t_sim_tick_p95_ms: f64,
    pub t_sim_frame_3ticks_ms: f64,
    pub t_sim_frame_4ticks_ms: f64,
    pub t_pack_interp_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeStage3ParityBundle {
    pub models_1000_ticks: Vec<NativeModelBitDump>,
    pub tost_n_birds: usize,
    pub tost_seeds: Vec<NativeTostSeedRow>,
    pub native_single_thread_benchmarks: Vec<NativeStressTickRow>,
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

pub fn run_native_stage3_bundle() -> NativeStage3ParityBundle {
    let seed_parity = 20_261_110u64;
    let n_parity = 128usize;
    let model_specs = [
        (0u32, "ref_ism"),
        (1u32, "ref_marginal_spp"),
        (2u32, "coupled_sde"),
    ];

    let mut models_1000_ticks = Vec::with_capacity(3);
    for (mk, mname) in model_specs {
        let dump = run_parity_1000_ticks(mk, n_parity, seed_parity);
        models_1000_ticks.push(NativeModelBitDump {
            model_name: mname,
            model_kind: mk,
            n_birds: n_parity,
            seed: seed_parity,
            tick_600_u32_bits: dump.state_tick_600.iter().map(|f| f.to_bits()).collect(),
            tick_1000_u32_bits: dump.state_tick_1000.iter().map(|f| f.to_bits()).collect(),
        });
    }

    let tost_n_birds = 96usize;
    let seeds: Vec<u64> = (1000u64..1064u64).collect();
    let tost_seeds: Vec<NativeTostSeedRow> = seeds
        .par_iter()
        .map(|&s| {
            let summary = run_tost_seed_rollout(s, tost_n_birds);
            let fvals = summary.to_f32_vec();
            let ubits = fvals.iter().map(|f| f.to_bits()).collect();
            NativeTostSeedRow {
                seed: s,
                f32_values: fvals,
                u32_bits: ubits,
            }
        })
        .collect();

    // Single-threaded native simulation & instance-packing benchmark across all 4 stress scenarios
    let regimes = [
        "calm_ribbon",
        "compacted_core_5x",
        "diluted_cloud_3x",
        "split_subflocks_stragglers",
    ];
    let mut native_single_thread_benchmarks = Vec::new();

    for &n in &[1000usize, 5000usize, 10000usize] {
        let n_ticks_bench = if n <= 1000 {
            24usize
        } else if n <= 5000 {
            16usize
        } else {
            10usize
        };
        for (r_idx, &r_name) in regimes.iter().enumerate() {
            let seed = 9000u64;
            let cfg = default_coupled_browser_config(seed);
            let (p0, u0) = generate_stress_cloud(r_idx, n, seed);
            let v0: Vec<Vec3> = u0.into_iter().map(|u| u * 12.0).collect();
            let mut flock = CoupledFlock::from_cartesian(p0, v0);
            let mut gait = BirdVisualGaitBank::new(n, seed);
            let mut interp = PoseInterpolator::from_coupled(&flock, &gait);

            // Warm up 2 ticks
            for _ in 0..2 {
                flock.step(&cfg, None);
                gait.step_gait(cfg.dt, &flock.speed, &flock.bank_angle_rad);
                interp.push_coupled_tick(&flock, &gait);
            }

            let mut tick_samples_ms = Vec::with_capacity(n_ticks_bench);
            for _ in 0..n_ticks_bench {
                let t0 = Instant::now();
                flock.step(&cfg, None);
                gait.step_gait(cfg.dt, &flock.speed, &flock.bank_angle_rad);
                interp.push_coupled_tick(&flock, &gait);
                tick_samples_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            tick_samples_ms.sort_by(|a, b| a.total_cmp(b));
            let med = tick_samples_ms[tick_samples_ms.len() / 2];
            let p95_idx = (((tick_samples_ms.len() - 1) as f64) * 0.95).round() as usize;
            let p95 = tick_samples_ms[p95_idx.min(tick_samples_ms.len() - 1)];

            let t_pack0 = Instant::now();
            let iters_pack = 10usize;
            for k in 0..iters_pack {
                let alpha = ((k as f32) + 0.5) / (iters_pack as f32);
                std::hint::black_box(interp.pack_interpolated(alpha));
            }
            let pack_ms = (t_pack0.elapsed().as_secs_f64() * 1000.0) / (iters_pack as f64);

            native_single_thread_benchmarks.push(NativeStressTickRow {
                regime: r_name,
                regime_idx: r_idx,
                n_birds: n,
                t_sim_tick_median_ms: round3(med),
                t_sim_tick_p95_ms: round3(p95),
                t_sim_frame_3ticks_ms: round3(med * 3.0),
                t_sim_frame_4ticks_ms: round3(med * 4.0),
                t_pack_interp_ms: round3(pack_ms),
            });
        }
    }

    NativeStage3ParityBundle {
        models_1000_ticks,
        tost_n_birds,
        tost_seeds,
        native_single_thread_benchmarks,
    }
}
