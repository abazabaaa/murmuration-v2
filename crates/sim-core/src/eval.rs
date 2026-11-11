use crate::math::{canonical_f32_cmp, Vec3};
use crate::neighbour::{LbvhIndex, NeighbourRule};
use serde::{Deserialize, Serialize};

/// `E1`: Global polarization `Phi = || (1/N) sum_i u_i ||`.
pub fn compute_polarization(velocities: &[Vec3]) -> f32 {
    if velocities.is_empty() {
        return 0.0;
    }
    let mut sum_u = Vec3::ZERO;
    for &v in velocities {
        sum_u += v.normalized_or(Vec3::EX);
    }
    (sum_u / (velocities.len() as f32)).norm()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AspectRatioSummary {
    /// Sorted principal bounding-box extents `[L1, L2, L3]` in meters (`L1 <= L2 <= L3`).
    pub extents_full_m: [f32; 3],
    /// Normalized spatial aspect ratios `[1.0, L2/L1, L3/L1]` (`E3`, target `~ 1 : 2.8 : 5.6`).
    pub ratios_full: [f32; 3],
    /// Outlier-robust `2.5%..97.5%` principal-axis spans normalized `[1.0, T2/T1, T3/T1]`.
    pub ratios_trim95: [f32; 3],
}

/// `E3`: Spatial bounding extents along the 3 orthonormal principal axes of the 3D point cloud.
/// Never substitutes covariance eigenvalues (`propto L^2`) or mechanical moments of inertia.
pub fn compute_spatial_aspect_ratios(positions: &[Vec3]) -> AspectRatioSummary {
    let n = positions.len();
    if n < 3 {
        return AspectRatioSummary {
            extents_full_m: [0.0; 3],
            ratios_full: [1.0, 1.0, 1.0],
            ratios_trim95: [1.0, 1.0, 1.0],
        };
    }

    let mut mean = Vec3::ZERO;
    for &p in positions {
        mean += p;
    }
    mean = mean / (n as f32);

    let mut cov = [[0.0f32; 3]; 3];
    for &p in positions {
        let d = p - mean;
        let arr = [d.x, d.y, d.z];
        for r in 0..3 {
            for c in 0..3 {
                cov[r][c] += arr[r] * arr[c];
            }
        }
    }
    let inv_n = 1.0 / (n as f32);
    for r in 0..3 {
        for c in 0..3 {
            cov[r][c] *= inv_n;
        }
    }

    let (_, eigvecs) = jacobi_eigen_3x3(cov);
    let axes = [
        Vec3::new(eigvecs[0][0], eigvecs[1][0], eigvecs[2][0]).normalized_or(Vec3::EX),
        Vec3::new(eigvecs[0][1], eigvecs[1][1], eigvecs[2][1]).normalized_or(Vec3::EY),
        Vec3::new(eigvecs[0][2], eigvecs[1][2], eigvecs[2][2]).normalized_or(Vec3::EZ),
    ];

    let mut full_spans = [0.0f32; 3];
    let mut trim_spans = [0.0f32; 3];
    let mut proj = Vec::with_capacity(n);

    for a in 0..3 {
        proj.clear();
        for &p in positions {
            proj.push((p - mean).dot(axes[a]));
        }
        proj.sort_by(|x, y| canonical_f32_cmp(*x, *y));
        full_spans[a] = (proj[n - 1] - proj[0]).max(1e-6);
        let lo = ((n as f32) * 0.025) as usize;
        let hi = (((n as f32) * 0.975) as usize).min(n - 1);
        trim_spans[a] = (proj[hi] - proj[lo]).max(1e-6);
    }

    full_spans.sort_by(|x, y| canonical_f32_cmp(*x, *y));
    trim_spans.sort_by(|x, y| canonical_f32_cmp(*x, *y));

    AspectRatioSummary {
        extents_full_m: full_spans,
        ratios_full: [
            1.0,
            full_spans[1] / full_spans[0],
            full_spans[2] / full_spans[0],
        ],
        ratios_trim95: [
            1.0,
            trim_spans[1] / trim_spans[0],
            trim_spans[2] / trim_spans[0],
        ],
    }
}

/// `E2`: Topological neighbour anisotropy `gamma(n)` (`n = 1..=max_rank`) for interior birds
/// (`gamma(n) -> 1/3` for isotropic distribution; `gamma(1) << 1/3` under flight-axis avoidance).
pub fn compute_topological_anisotropy_gamma(positions: &[Vec3], max_rank: usize) -> Vec<f32> {
    let n = positions.len();
    if n <= max_rank + 1 || max_rank == 0 {
        return vec![1.0 / 3.0; max_rank];
    }
    let dummy_u = vec![Vec3::EX; n];
    let lbvh = LbvhIndex::build(positions);
    let nb = lbvh.query_all(
        positions,
        &dummy_u,
        NeighbourRule::ExactTopoK {
            k: max_rank.max(12).min(n - 1),
            blind_cos: -1.0,
        },
    );

    // Exclude perimeter birds by requiring neighbours in all 6 signed Cartesian half-spaces (`+-x, +-y, +-z`).
    let mut interior_ids = Vec::with_capacity(n);
    for (i, row) in nb.iter().enumerate() {
        let mut mask = 0u8;
        for e in row {
            let d = positions[e.id as usize] - positions[i];
            if d.x > 0.0 {
                mask |= 1;
            } else if d.x < 0.0 {
                mask |= 2;
            }
            if d.y > 0.0 {
                mask |= 4;
            } else if d.y < 0.0 {
                mask |= 8;
            }
            if d.z > 0.0 {
                mask |= 16;
            } else if d.z < 0.0 {
                mask |= 32;
            }
        }
        if mask == 0x3F {
            interior_ids.push(i);
        }
    }
    let sample_ids: Vec<usize> = if interior_ids.len() >= 8 {
        interior_ids
    } else {
        (0..n).collect()
    };

    let mut gammas = Vec::with_capacity(max_rank);
    for rank_idx in 0..max_rank {
        let mut m = [[0.0f32; 3]; 3];
        let mut count = 0usize;
        for &i in &sample_ids {
            if rank_idx < nb[i].len() {
                let u = (positions[nb[i][rank_idx].id as usize] - positions[i]).normalized_or(Vec3::EX);
                let arr = [u.x, u.y, u.z];
                for r in 0..3 {
                    for c in 0..3 {
                        m[r][c] += arr[r] * arr[c];
                    }
                }
                count += 1;
            }
        }
        if count == 0 {
            gammas.push(1.0 / 3.0);
        } else {
            let inv = 1.0 / (count as f32);
            for r in 0..3 {
                for c in 0..3 {
                    m[r][c] *= inv;
                }
            }
            let (mut evals, _) = jacobi_eigen_3x3(m);
            evals.sort_by(|a, b| canonical_f32_cmp(*a, *b));
            gammas.push(evals[0]);
        }
    }
    gammas
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorrelationSummary {
    pub l_flock_m: f32,
    pub xi_orient_m: f32,
    pub xi_speed_m: f32,
    pub xi_orient_over_l: f32,
    pub xi_speed_over_l: f32,
}

/// `E4`: Connected spatial orientation (`C_{du}(r)`) and scalar speed (`C_{dv}(r)`) correlation lengths (`xi`, `xi_sp`).
pub fn compute_spatial_correlations(
    positions: &[Vec3],
    velocities: &[Vec3],
    n_bins: usize,
) -> CorrelationSummary {
    let n = positions.len();
    assert_eq!(n, velocities.len());
    if n < 4 || n_bins < 2 {
        return CorrelationSummary {
            l_flock_m: 0.0,
            xi_orient_m: 0.0,
            xi_speed_m: 0.0,
            xi_orient_over_l: 0.0,
            xi_speed_over_l: 0.0,
        };
    }

    let mut u_mean = Vec3::ZERO;
    let mut v_mean = 0.0f32;
    let mut u_list = Vec::with_capacity(n);
    let mut s_list = Vec::with_capacity(n);
    for &v in velocities {
        let s = v.norm();
        let u = if s > 1e-12 { v / s } else { Vec3::EX };
        u_mean += u;
        v_mean += s;
        u_list.push(u);
        s_list.push(s);
    }
    u_mean = u_mean / (n as f32);
    v_mean /= n as f32;

    let du: Vec<Vec3> = u_list.into_iter().map(|u| u - u_mean).collect();
    let ds: Vec<f32> = s_list.into_iter().map(|s| s - v_mean).collect();

    let mut c0_orient = 0.0f32;
    let mut c0_speed = 0.0f32;
    for i in 0..n {
        c0_orient += du[i].norm_sq();
        c0_speed += ds[i] * ds[i];
    }
    c0_orient /= n as f32;
    c0_speed /= n as f32;

    let mut max_r_sq = 0.0f32;
    for i in 0..n {
        for j in (i + 1)..n {
            let d2 = (positions[j] - positions[i]).norm_sq();
            if d2 > max_r_sq {
                max_r_sq = d2;
            }
        }
    }
    let l_flock = libm::sqrtf(max_r_sq);
    if l_flock <= 1e-6 {
        return CorrelationSummary {
            l_flock_m: 0.0,
            xi_orient_m: 0.0,
            xi_speed_m: 0.0,
            xi_orient_over_l: 0.0,
            xi_speed_over_l: 0.0,
        };
    }

    let dr = l_flock / (n_bins as f32);
    let mut orient_sum = vec![0.0f32; n_bins];
    let mut speed_sum = vec![0.0f32; n_bins];
    let mut counts = vec![0usize; n_bins];

    for i in 0..n {
        for j in (i + 1)..n {
            let r = (positions[j] - positions[i]).norm();
            let b = ((r / dr) as usize).min(n_bins - 1);
            orient_sum[b] += du[i].dot(du[j]);
            speed_sum[b] += ds[i] * ds[j];
            counts[b] += 1;
        }
    }

    let mut r_centers = Vec::new();
    let mut c_orient = Vec::new();
    let mut c_speed = Vec::new();
    for b in 0..n_bins {
        if counts[b] > 0 {
            r_centers.push(((b as f32) + 0.5) * dr);
            c_orient.push(if c0_orient > 1e-12 {
                orient_sum[b] / ((counts[b] as f32) * c0_orient)
            } else {
                0.0
            });
            c_speed.push(if c0_speed > 1e-12 {
                speed_sum[b] / ((counts[b] as f32) * c0_speed)
            } else {
                0.0
            });
        }
    }

    let xi_orient = first_zero_crossing(&r_centers, &c_orient).unwrap_or(l_flock);
    let xi_speed = first_zero_crossing(&r_centers, &c_speed).unwrap_or(l_flock);

    CorrelationSummary {
        l_flock_m: l_flock,
        xi_orient_m: xi_orient,
        xi_speed_m: xi_speed,
        xi_orient_over_l: xi_orient / l_flock,
        xi_speed_over_l: xi_speed / l_flock,
    }
}

fn first_zero_crossing(r: &[f32], c: &[f32]) -> Option<f32> {
    if r.is_empty() {
        return None;
    }
    if c[0] <= 0.0 {
        return Some(r[0]);
    }
    for k in 0..(c.len() - 1) {
        if c[k] >= 0.0 && c[k + 1] <= 0.0 {
            let denom = c[k] - c[k + 1];
            let frac = if denom > 1e-12 { c[k] / denom } else { 0.5 };
            return Some(r[k] + frac * (r[k + 1] - r[k]));
        }
    }
    None
}

/// `E5`: Individual-bird and group speed distribution summary (zero clamps).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpeedDistributionSummary {
    pub n_birds: usize,
    pub n_snapshots: usize,
    pub mean_individual_mps: f32,
    pub sd_individual_mps: f32,
    /// `[p05, p25, p50, p75, p95]` of individual bird speeds `||v_i||`.
    pub individual_quantiles_mps: [f32; 5],
    pub min_individual_mps: f32,
    pub max_individual_mps: f32,
    /// Mean of flock-averaged scalar speed `V(t) = (1/N) sum_i ||v_i(t)||`.
    pub mean_group_scalar_speed_mps: f32,
    pub sd_group_scalar_speed_mps: f32,
    /// Mean of centre-of-mass velocity norm `|| (1/N) sum_i v_i(t) ||`.
    pub mean_barycenter_speed_mps: f32,
    pub speed_clamp_events: u64,
}

pub struct SpeedAccumulator {
    n_birds: usize,
    indiv_samples: Vec<f32>,
    group_scalar_samples: Vec<f32>,
    bary_speed_samples: Vec<f32>,
}

impl SpeedAccumulator {
    pub fn new(n_birds: usize) -> Self {
        Self {
            n_birds,
            indiv_samples: Vec::new(),
            group_scalar_samples: Vec::new(),
            bary_speed_samples: Vec::new(),
        }
    }

    pub fn record(&mut self, velocities: &[Vec3]) {
        if velocities.is_empty() {
            return;
        }
        let mut sum_s = 0.0f32;
        let mut sum_v = Vec3::ZERO;
        for &v in velocities {
            let s = v.norm();
            self.indiv_samples.push(s);
            sum_s += s;
            sum_v += v;
        }
        let inv_n = 1.0 / (velocities.len() as f32);
        self.group_scalar_samples.push(sum_s * inv_n);
        self.bary_speed_samples.push((sum_v * inv_n).norm());
    }

    pub fn samples(&self) -> &[f32] {
        &self.indiv_samples
    }

    pub fn summarize(&self, speed_clamp_events: u64) -> SpeedDistributionSummary {
        if self.indiv_samples.is_empty() {
            return SpeedDistributionSummary {
                n_birds: self.n_birds,
                n_snapshots: 0,
                mean_individual_mps: 0.0,
                sd_individual_mps: 0.0,
                individual_quantiles_mps: [0.0; 5],
                min_individual_mps: 0.0,
                max_individual_mps: 0.0,
                mean_group_scalar_speed_mps: 0.0,
                sd_group_scalar_speed_mps: 0.0,
                mean_barycenter_speed_mps: 0.0,
                speed_clamp_events,
            };
        }

        let mut sorted = self.indiv_samples.clone();
        sorted.sort_by(|a, b| canonical_f32_cmp(*a, *b));
        let (mean_ind, sd_ind) = mean_and_sd(&sorted);
        let (mean_grp, sd_grp) = mean_and_sd(&self.group_scalar_samples);
        let (mean_bary, _) = mean_and_sd(&self.bary_speed_samples);

        let q = |p: f32| -> f32 {
            let idx = (((sorted.len() - 1) as f32) * p) as usize;
            sorted[idx.min(sorted.len() - 1)]
        };

        SpeedDistributionSummary {
            n_birds: self.n_birds,
            n_snapshots: self.group_scalar_samples.len(),
            mean_individual_mps: mean_ind,
            sd_individual_mps: sd_ind,
            individual_quantiles_mps: [q(0.05), q(0.25), q(0.50), q(0.75), q(0.95)],
            min_individual_mps: sorted[0],
            max_individual_mps: sorted[sorted.len() - 1],
            mean_group_scalar_speed_mps: mean_grp,
            sd_group_scalar_speed_mps: sd_grp,
            mean_barycenter_speed_mps: mean_bary,
            speed_clamp_events,
        }
    }
}

/// Exact 1D Wasserstein-1 (`L1` Earth Mover's) distance between two empirical sample vectors.
pub fn wasserstein_1d(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let mut sa = a.to_vec();
    let mut sb = b.to_vec();
    sa.sort_by(|x, y| canonical_f32_cmp(*x, *y));
    sb.sort_by(|x, y| canonical_f32_cmp(*x, *y));
    let m = sa.len().max(sb.len());
    let mut sum = 0.0f32;
    for k in 0..m {
        let u = ((k as f32) + 0.5) / (m as f32);
        let ia = ((u * (sa.len() as f32)) as usize).min(sa.len() - 1);
        let ib = ((u * (sb.len() as f32)) as usize).min(sb.len() - 1);
        sum += (sa[ia] - sb[ib]).abs();
    }
    sum / (m as f32)
}

/// `E6a` (`alpha_cm`), `E6b` (`alpha_mut`), and `E6c` (`Q_M(t)`): separated internal diffusion observables
/// (`Cavagna et al. 2013`, `PMC3574359`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InternalDiffusionSummary {
    pub alpha_cm: f32,
    pub alpha_mutual: f32,
    pub q_m_final: f32,
    pub m_neighbours: usize,
}

pub struct InternalDiffusionTracker {
    m: usize,
    t0_s: f32,
    rel_cm_t0: Vec<Vec3>,
    init_neighbours: Vec<Vec<u32>>,
    init_pair_dists: Vec<Vec<(u32, f32)>>,
    times_s: Vec<f32>,
    msd_cm: Vec<f32>,
    msd_mut: Vec<f32>,
    last_q_m: f32,
}

impl InternalDiffusionTracker {
    pub fn initialize(positions: &[Vec3], m: usize, t0_s: f32) -> Self {
        let n = positions.len();
        let cm0 = centroid(positions);
        let rel_cm_t0: Vec<Vec3> = positions.iter().map(|&p| p - cm0).collect();
        let dummy_u = vec![Vec3::EX; n];
        let nb = LbvhIndex::build(positions).query_all(
            positions,
            &dummy_u,
            NeighbourRule::ExactTopoK {
                k: m.min(n.saturating_sub(1)),
                blind_cos: -1.0,
            },
        );
        let init_neighbours: Vec<Vec<u32>> = nb
            .iter()
            .map(|row| row.iter().map(|e| e.id).collect())
            .collect();
        let init_pair_dists: Vec<Vec<(u32, f32)>> = nb
            .iter()
            .map(|row| {
                row.iter()
                    .map(|e| (e.id, libm::sqrtf(e.dist_sq)))
                    .collect()
            })
            .collect();

        Self {
            m,
            t0_s,
            rel_cm_t0,
            init_neighbours,
            init_pair_dists,
            times_s: Vec::new(),
            msd_cm: Vec::new(),
            msd_mut: Vec::new(),
            last_q_m: 1.0,
        }
    }

    /// Record `E6a` and `E6b` at `40 Hz` (`every 25 ms`).
    pub fn record_40hz(&mut self, positions: &[Vec3], t_now_s: f32) {
        let dt = t_now_s - self.t0_s;
        if dt <= 1e-5 || positions.is_empty() {
            return;
        }
        let n = positions.len();
        let cm = centroid(positions);

        let mut sum_cm = 0.0f32;
        let mut sum_mut = 0.0f32;
        let mut count_mut = 0usize;

        for i in 0..n {
            let dr_cm = (positions[i] - cm) - self.rel_cm_t0[i];
            sum_cm += dr_cm.norm_sq();
            for &(j, d0) in &self.init_pair_dists[i] {
                let d_now = (positions[j as usize] - positions[i]).norm();
                let diff = d_now - d0;
                sum_mut += diff * diff;
                count_mut += 1;
            }
        }
        self.times_s.push(dt);
        self.msd_cm.push(sum_cm / (n as f32));
        self.msd_mut.push(if count_mut > 0 {
            sum_mut / (count_mut as f32)
        } else {
            0.0
        });
    }

    /// Update `E6c` (`Q_M(t)` neighbour retention fraction) at `10 Hz`.
    pub fn record_qm_10hz(&mut self, positions: &[Vec3]) {
        let n = positions.len();
        if n <= 1 || self.m == 0 {
            return;
        }
        let k_eff = self.m.min(n - 1);
        let dummy_u = vec![Vec3::EX; n];
        let nb_now = LbvhIndex::build(positions).query_all(
            positions,
            &dummy_u,
            NeighbourRule::ExactTopoK {
                k: k_eff,
                blind_cos: -1.0,
            },
        );
        let mut retained = 0usize;
        let mut total = 0usize;
        for (i, row) in nb_now.iter().enumerate() {
            total += self.init_neighbours[i].len();
            for e in row {
                if self.init_neighbours[i].contains(&e.id) {
                    retained += 1;
                }
            }
        }
        if total > 0 {
            self.last_q_m = (retained as f32) / (total as f32);
        }
    }

    pub fn summarize(&self) -> InternalDiffusionSummary {
        InternalDiffusionSummary {
            alpha_cm: fit_log_log_slope(&self.times_s, &self.msd_cm),
            alpha_mutual: fit_log_log_slope(&self.times_s, &self.msd_mut),
            q_m_final: self.last_q_m,
            m_neighbours: self.m,
        }
    }
}

/// `E7`: Full `200 Hz` (`5 ms` tick-rate) directional turn-information propagation estimator
/// (`Attanasi et al. 2014`, `PMC4173114`, target `20..40 m/s` in natural starling flocks).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TurnWaveSummary {
    pub measured_cs_mps: f32,
    pub predicted_cs_mps: f32,
    pub linear_r_squared: f32,
    /// Log-log dispersion exponent `d ~ tau^beta` (`~1.0` for ballistic ISM spin waves; `~0.5` for diffusive Vicsek).
    pub dispersion_exponent_beta: f32,
    /// Ratio of mean peak `||a_perp||` in the far spatial half to the near spatial half.
    pub far_to_near_amplitude_ratio: f32,
    pub participating_birds: usize,
}

pub struct TurnWaveRecorder {
    dt_s: f32,
    pos_t0: Vec<Vec3>,
    initiator_ids: usize,
    k_hat: Vec3,
    /// `[bird_i][tick_k]` transverse acceleration history recorded EVERY `200 Hz` tick.
    traces: Vec<Vec<Vec3>>,
}

impl TurnWaveRecorder {
    pub fn new(pos_t0: Vec<Vec3>, initiator_ids: usize, k_hat: Vec3, dt_s: f32) -> Self {
        let n = pos_t0.len();
        Self {
            dt_s,
            pos_t0,
            initiator_ids: initiator_ids.max(1).min(n),
            k_hat: k_hat.normalized_or(Vec3::EX),
            traces: vec![Vec::new(); n],
        }
    }

    /// Must be called EVERY simulation tick (`200 Hz`, `5 ms`).
    pub fn record_tick(&mut self, accel_perp: &[Vec3]) {
        for (i, &a) in accel_perp.iter().enumerate() {
            self.traces[i].push(a);
        }
    }

    pub fn analyze(&self, predicted_cs_mps: f32) -> TurnWaveSummary {
        let n = self.pos_t0.len();
        let t_len = self.traces.first().map(|v| v.len()).unwrap_or(0);
        if n <= self.initiator_ids || t_len < 8 {
            return TurnWaveSummary {
                measured_cs_mps: 0.0,
                predicted_cs_mps,
                linear_r_squared: 0.0,
                dispersion_exponent_beta: 0.0,
                far_to_near_amplitude_ratio: 0.0,
                participating_birds: 0,
            };
        }

        // Reference initiator waveform = mean `a_perp(t)` over initiator birds `0..initiator_ids`
        let mut x_init = Vec3::ZERO;
        for i in 0..self.initiator_ids {
            x_init += self.pos_t0[i];
        }
        x_init = x_init / (self.initiator_ids as f32);

        let mut ref_trace = vec![Vec3::ZERO; t_len];
        for i in 0..self.initiator_ids {
            for k in 0..t_len {
                ref_trace[k] += self.traces[i][k];
            }
        }
        let inv_init = 1.0 / (self.initiator_ids as f32);
        let mut ref_peak_mag = 0.0f32;
        for v in ref_trace.iter_mut() {
            *v = *v * inv_init;
            let m = v.norm();
            if m > ref_peak_mag {
                ref_peak_mag = m;
            }
        }
        // First local peak of initiator waveform (avoids late boundary reflections)
        let mut ref_peak_tick = 0usize;
        for k in 1..(t_len - 1) {
            let m = ref_trace[k].norm();
            if m >= 0.5 * ref_peak_mag && m >= ref_trace[k - 1].norm() && m >= ref_trace[k + 1].norm() {
                ref_peak_tick = k;
                break;
            }
        }

        let mut max_d = 0.0f32;
        for i in self.initiator_ids..n {
            let d = (self.pos_t0[i] - x_init).dot(self.k_hat);
            if d > max_d {
                max_d = d;
            }
        }

        let mut dists = Vec::new();
        let mut lags_s = Vec::new();
        let mut peaks = Vec::new();

        for i in self.initiator_ids..n {
            let d = (self.pos_t0[i] - x_init).dot(self.k_hat);
            // Exclude open far-end reflecting boundary (`d > 0.85 * max_d`) from dispersion slope
            if d <= 0.05 || d > 0.85 * max_d {
                continue;
            }
            let trace = &self.traces[i];
            let mut global_max = 0.0f32;
            for &a in trace {
                let m = a.norm();
                if m > global_max {
                    global_max = m;
                }
            }
            if global_max < 0.02 * ref_peak_mag {
                continue;
            }

            // Detect first wavefront peak (`m >= 0.4 * global_max` local maximum after `ref_peak_tick`)
            let mut peak_k = 0usize;
            let mut peak_mag = 0.0f32;
            for k in (ref_peak_tick + 1)..(t_len - 1) {
                let m = trace[k].norm();
                if m >= 0.4 * global_max && m >= trace[k - 1].norm() && m >= trace[k + 1].norm() {
                    peak_k = k;
                    peak_mag = m;
                    break;
                }
            }
            if peak_k <= ref_peak_tick {
                continue;
            }

            // Sub-tick parabolic peak refinement on `||a_{i,perp}(t)||` minus reference peak time
            let sub_k = if peak_k > 0 && peak_k + 1 < t_len {
                let y_m = trace[peak_k - 1].norm();
                let y_0 = peak_mag;
                let y_p = trace[peak_k + 1].norm();
                let denom = y_m - 2.0 * y_0 + y_p;
                if denom.abs() > 1e-12 {
                    (0.5 * (y_m - y_p) / denom).clamp(-0.5, 0.5)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let tau_s = (((peak_k as f32) + sub_k) - (ref_peak_tick as f32)) * self.dt_s;
            if tau_s > 0.0 {
                dists.push(d);
                lags_s.push(tau_s);
                peaks.push((d, peak_mag));
            }
        }

        let (slope_cs, r2) = fit_origin_or_linear_slope(&lags_s, &dists);
        let beta = fit_log_log_slope(&lags_s, &dists);

        let amp_ratio = if peaks.len() >= 4 {
            peaks.sort_by(|a, b| canonical_f32_cmp(a.0, b.0));
            let mid = peaks.len() / 2;
            let near_mean: f32 = peaks[..mid].iter().map(|p| p.1).sum::<f32>() / (mid as f32);
            let far_mean: f32 =
                peaks[mid..].iter().map(|p| p.1).sum::<f32>() / ((peaks.len() - mid) as f32);
            if near_mean > 1e-12 {
                far_mean / near_mean
            } else {
                0.0
            }
        } else {
            0.0
        };

        TurnWaveSummary {
            measured_cs_mps: slope_cs,
            predicted_cs_mps,
            linear_r_squared: r2,
            dispersion_exponent_beta: beta,
            far_to_near_amplitude_ratio: amp_ratio,
            participating_birds: dists.len(),
        }
    }
}

fn centroid(positions: &[Vec3]) -> Vec3 {
    if positions.is_empty() {
        return Vec3::ZERO;
    }
    let mut s = Vec3::ZERO;
    for &p in positions {
        s += p;
    }
    s / (positions.len() as f32)
}

fn mean_and_sd(xs: &[f32]) -> (f32, f32) {
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let n = xs.len() as f32;
    let mean = xs.iter().copied().sum::<f32>() / n;
    let var = xs
        .iter()
        .map(|&x| {
            let d = x - mean;
            d * d
        })
        .sum::<f32>()
        / n;
    (mean, libm::sqrtf(var))
}

fn fit_origin_or_linear_slope(xs: &[f32], ys: &[f32]) -> (f32, f32) {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return (0.0, 0.0);
    }
    let mx = xs[..n].iter().sum::<f32>() / (n as f32);
    let my = ys[..n].iter().sum::<f32>() / (n as f32);
    let mut sxx = 0.0f32;
    let mut sxy = 0.0f32;
    let mut syy = 0.0f32;
    for k in 0..n {
        let dx = xs[k] - mx;
        let dy = ys[k] - my;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    if sxx <= 1e-12 || syy <= 1e-12 {
        return (0.0, 0.0);
    }
    let slope = sxy / sxx;
    let r2 = (sxy * sxy) / (sxx * syy);
    (slope, r2)
}

fn fit_log_log_slope(xs: &[f32], ys: &[f32]) -> f32 {
    let mut lx = Vec::new();
    let mut ly = Vec::new();
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        if x > 1e-6 && y > 1e-8 {
            lx.push(libm::logf(x));
            ly.push(libm::logf(y));
        }
    }
    fit_origin_or_linear_slope(&lx, &ly).0
}

/// Pure-Rust symmetric `3x3` Jacobi eigenvalue decomposition (`libm` trig/sqrt).
/// Returns `(eigenvalues, eigenvector_columns)`.
fn jacobi_eigen_3x3(mut a: [[f32; 3]; 3]) -> ([f32; 3], [[f32; 3]; 3]) {
    let mut v = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    for _ in 0..24 {
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max_off = a[0][1].abs();
        if a[0][2].abs() > max_off {
            p = 0;
            q = 2;
            max_off = a[0][2].abs();
        }
        if a[1][2].abs() > max_off {
            p = 1;
            q = 2;
            max_off = a[1][2].abs();
        }
        if max_off < 1e-10 {
            break;
        }

        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + libm::sqrtf(1.0 + tau * tau))
        } else {
            -1.0 / (-tau + libm::sqrtf(1.0 + tau * tau))
        };
        let c = 1.0 / libm::sqrtf(1.0 + t * t);
        let s = t * c;

        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for r in 0..3 {
            if r != p && r != q {
                let arp = a[r][p];
                let arq = a[r][q];
                a[r][p] = c * arp - s * arq;
                a[p][r] = a[r][p];
                a[r][q] = s * arp + c * arq;
                a[q][r] = a[r][q];
            }
        }
        for r in 0..3 {
            let vrp = v[r][p];
            let vrq = v[r][q];
            v[r][p] = c * vrp - s * vrq;
            v[r][q] = s * vrp + c * vrq;
        }
    }

    ([a[0][0], a[1][1], a[2][2]], v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::rodrigues_rotate;
    use crate::rng::{keyed_rng, normal_vec3, uniform_open01, STREAM_INIT};

    #[test]
    fn e3_spatial_aspect_ratio_recovers_linear_dimensions_not_eigenvalue_squares() {
        // Synthetic triaxial ellipsoid with semi-axes `(5.0, 14.0, 28.0)` => true spatial ratio `1 : 2.8 : 5.6`
        // (whereas covariance eigenvalues would give `1 : 7.84 : 31.36`!).
        let mut pts = Vec::with_capacity(1500);
        let rot_axis = Vec3::new(0.4, -0.7, 0.5);
        for i in 0..1500u32 {
            let mut rng = keyed_rng(2026, STREAM_INIT, 0, i);
            let dir = normal_vec3(&mut rng).normalized_or(Vec3::EX);
            let r = libm::cbrtf(uniform_open01(&mut rng));
            let local = Vec3::new(dir.x * r * 5.0, dir.y * r * 14.0, dir.z * r * 28.0);
            pts.push(rodrigues_rotate(local, rot_axis) + Vec3::new(12.0, -8.0, 3.0));
        }
        let summary = compute_spatial_aspect_ratios(&pts);
        assert!(
            (summary.ratios_full[1] - 2.8).abs() < 0.15,
            "Expected L2/L1 ~ 2.8, got {}",
            summary.ratios_full[1]
        );
        assert!(
            (summary.ratios_full[2] - 5.6).abs() < 0.25,
            "Expected L3/L1 ~ 5.6, got {}",
            summary.ratios_full[2]
        );
    }
}
