use crate::math::{canonical_f32_cmp, Vec3};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NeighbourEntry {
    pub id: u32,
    pub dist_sq: f32,
}

impl NeighbourEntry {
    #[inline]
    pub fn cmp_canonical(&self, other: &Self) -> Ordering {
        canonical_f32_cmp(self.dist_sq, other.dist_sq).then_with(|| self.id.cmp(&other.id))
    }
}

/// Biological neighbour interaction rule, defined strictly independently of spatial acceleration (§4.3).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum NeighbourRule {
    /// Exact topological `k`-nearest neighbours in `R^3` within field-of-view (`blind_cos = cos(fov/2)`),
    /// with infinite perception range.
    ExactTopoK { k: usize, blind_cos: f32 },
    /// Explicit biological model variant: topological `k`-NN truncated at finite perception radius `r_perc`.
    BoundedTopoK {
        k: usize,
        blind_cos: f32,
        r_perc: f32,
    },
    /// Explicit biological model variant: up to 1 nearest neighbour per body-frame octant (max 8, capped at `k_max`).
    OctantTopoK { k_max: usize, blind_cos: f32 },
    /// Metric interaction ball `||x_j - x_i|| <= r_metric` within FoV.
    MetricRadius { r_metric: f32, blind_cos: f32 },
}

/// Graph reciprocity / weight convention (§2.1A).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphSymmetry {
    /// Directed topological normalization: `w_ij = 1 / n_i` for `j in N_i` (`w_ij != w_ji` in general).
    DirectedNormalized,
    /// Symmetrized mean weights: `w_ij = 0.5 * (w_ij^dir + w_ji^dir)` (`w_ij == w_ji` strictly).
    SymmetricMean,
    /// Symmetric unnormalized adjacency: `w_ij = 1.0` iff `j in N_i` or `i in N_j` (Cavagna 2015 lattice/metric convention).
    SymmetricUnnormalized,
}

/// Exact algebraic field-of-view predicate (`(x_j - x_i) . u_i >= ||x_j - x_i|| * blind_cos`).
/// Coincident positions (`dist_sq == 0.0`) are defined as eligible (`true`).
#[inline]
pub fn is_in_fov_algebraic(diff: Vec3, dist_sq: f32, u_unit: Vec3, blind_cos: f32) -> bool {
    if blind_cos <= -1.0 || dist_sq == 0.0 {
        return true;
    }
    let dot = diff.dot(u_unit);
    let c2_d2 = (blind_cos * blind_cos) * dist_sq;
    if blind_cos >= 0.0 {
        dot >= 0.0 && (dot * dot) >= c2_d2
    } else {
        dot >= 0.0 || (dot * dot) <= c2_d2
    }
}

/// Exact `O(N^2)` reference oracle (§4.3).
pub fn query_neighbours_brute_force(
    positions: &[Vec3],
    headings: &[Vec3],
    rule: NeighbourRule,
) -> Vec<Vec<NeighbourEntry>> {
    let n = positions.len();
    assert_eq!(n, headings.len());
    let mut all = Vec::with_capacity(n);

    for i in 0..n {
        let xi = positions[i];
        let ui = headings[i];
        let list = match rule {
            NeighbourRule::ExactTopoK { k, blind_cos } => {
                if k == 0 {
                    Vec::new()
                } else {
                    let mut cands = Vec::with_capacity(n.saturating_sub(1));
                    for j in 0..n {
                        if j == i {
                            continue;
                        }
                        let diff = positions[j] - xi;
                        let d2 = diff.norm_sq();
                        if is_in_fov_algebraic(diff, d2, ui, blind_cos) {
                            cands.push(NeighbourEntry {
                                id: j as u32,
                                dist_sq: if d2 == 0.0 { 0.0 } else { d2 },
                            });
                        }
                    }
                    cands.sort_by(|a, b| a.cmp_canonical(b));
                    cands.truncate(k);
                    cands
                }
            }
            NeighbourRule::BoundedTopoK {
                k,
                blind_cos,
                r_perc,
            } => {
                let r2 = r_perc * r_perc;
                let mut cands = Vec::new();
                if k > 0 {
                    for j in 0..n {
                        if j == i {
                            continue;
                        }
                        let diff = positions[j] - xi;
                        let d2 = diff.norm_sq();
                        if d2 <= r2 && is_in_fov_algebraic(diff, d2, ui, blind_cos) {
                            cands.push(NeighbourEntry {
                                id: j as u32,
                                dist_sq: if d2 == 0.0 { 0.0 } else { d2 },
                            });
                        }
                    }
                    cands.sort_by(|a, b| a.cmp_canonical(b));
                    cands.truncate(k);
                }
                cands
            }
            NeighbourRule::OctantTopoK { k_max, blind_cos } => {
                query_octant_single_bird(i, positions, ui, k_max, blind_cos)
            }
            NeighbourRule::MetricRadius {
                r_metric,
                blind_cos,
            } => {
                let r2 = r_metric * r_metric;
                let mut cands = Vec::new();
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let diff = positions[j] - xi;
                    let d2 = diff.norm_sq();
                    if d2 <= r2 && is_in_fov_algebraic(diff, d2, ui, blind_cos) {
                        cands.push(NeighbourEntry {
                            id: j as u32,
                            dist_sq: if d2 == 0.0 { 0.0 } else { d2 },
                        });
                    }
                }
                cands.sort_by(|a, b| a.cmp_canonical(b));
                cands
            }
        };
        all.push(list);
    }
    all
}

fn query_octant_single_bird(
    i: usize,
    positions: &[Vec3],
    ui: Vec3,
    k_max: usize,
    blind_cos: f32,
) -> Vec<NeighbourEntry> {
    if k_max == 0 {
        return Vec::new();
    }
    let horiz = ui.cross(Vec3::EZ);
    let right = if horiz.norm_sq() > 1e-4 {
        horiz.normalized_or(Vec3::EY)
    } else {
        Vec3::EX.project_perp(ui).normalized_or(Vec3::EY)
    };
    let dorsal = right.cross(ui).normalized_or(Vec3::EZ);
    let xi = positions[i];
    let mut best_oct: [Option<NeighbourEntry>; 8] = [None; 8];

    for (j, &xj) in positions.iter().enumerate() {
        if j == i {
            continue;
        }
        let diff = xj - xi;
        let d2 = diff.norm_sq();
        if !is_in_fov_algebraic(diff, d2, ui, blind_cos) {
            continue;
        }
        let bx = if diff.dot(ui) >= 0.0 { 1usize } else { 0usize };
        let by = if diff.dot(right) >= 0.0 { 2usize } else { 0usize };
        let bz = if diff.dot(dorsal) >= 0.0 { 4usize } else { 0usize };
        let oct = bx | by | bz;
        let cand = NeighbourEntry {
            id: j as u32,
            dist_sq: if d2 == 0.0 { 0.0 } else { d2 },
        };
        match best_oct[oct] {
            None => best_oct[oct] = Some(cand),
            Some(prev) => {
                if cand.cmp_canonical(&prev) == Ordering::Less {
                    best_oct[oct] = Some(cand);
                }
            }
        }
    }
    let mut out: Vec<NeighbourEntry> = best_oct.into_iter().flatten().collect();
    out.sort_by(|a, b| a.cmp_canonical(b));
    out.truncate(k_max);
    out
}

/// Contiguous zero-per-bird-allocation neighbour table (`entries[offsets[i]..offsets[i+1]]`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FlatNeighbourTable {
    pub offsets: Vec<u32>,
    pub entries: Vec<NeighbourEntry>,
}

impl FlatNeighbourTable {
    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn row(&self, i: usize) -> &[NeighbourEntry] {
        let s = self.offsets[i] as usize;
        let e = self.offsets[i + 1] as usize;
        &self.entries[s..e]
    }

    pub fn from_vecs(rows: &[Vec<NeighbourEntry>]) -> Self {
        let mut offsets = Vec::with_capacity(rows.len() + 1);
        offsets.push(0);
        let total: usize = rows.iter().map(|r| r.len()).sum();
        let mut entries = Vec::with_capacity(total);
        for r in rows {
            entries.extend_from_slice(r);
            offsets.push(entries.len() as u32);
        }
        Self { offsets, entries }
    }
}

#[derive(Clone, Copy, Debug)]
struct BvhNode {
    aabb_min: Vec3,
    aabb_max: Vec3,
    /// If `count > 0`, leaf covering `ordered_ids[left_or_start .. left_or_start + count]`.
    /// If `count == 0`, internal node with children `left_or_start` and `right_child`.
    left_or_start: u32,
    right_child: u32,
    count: u32,
}

/// Exact Morton-ordered Bounding Volume Hierarchy for accelerated `ExactTopoK`, `BoundedTopoK`,
/// and `MetricRadius` queries with 100% oracle parity (§4.3).
pub struct LbvhIndex {
    nodes: Vec<BvhNode>,
    ordered_ids: Vec<u32>,
    ordered_pos: Vec<Vec3>,
    bird_leaf: Vec<u32>,
}

const LEAF_CAPACITY: usize = 32;

impl LbvhIndex {
    pub fn build(positions: &[Vec3]) -> Self {
        let n = positions.len();
        if n == 0 {
            return Self {
                nodes: Vec::new(),
                ordered_ids: Vec::new(),
                ordered_pos: Vec::new(),
                bird_leaf: Vec::new(),
            };
        }

        let mut items: Vec<(u32, Vec3)> = positions
            .iter()
            .copied()
            .enumerate()
            .map(|(i, p)| (i as u32, p))
            .collect();

        let mut nodes = Vec::with_capacity(2 * (n / LEAF_CAPACITY + 2));
        let mut bird_leaf = vec![0u32; n];
        Self::build_kd_bvh(&mut items, 0, n, &mut nodes, &mut bird_leaf);

        let mut ordered_ids = Vec::with_capacity(n);
        let mut ordered_pos = Vec::with_capacity(n);
        for (id, p) in items {
            ordered_ids.push(id);
            ordered_pos.push(p);
        }

        Self {
            nodes,
            ordered_ids,
            ordered_pos,
            bird_leaf,
        }
    }

    fn build_kd_bvh(
        items: &mut [(u32, Vec3)],
        start: usize,
        end: usize,
        nodes: &mut Vec<BvhNode>,
        bird_leaf: &mut [u32],
    ) -> u32 {
        let idx = nodes.len() as u32;
        let slice = &mut items[start..end];
        let mut aabb_min = slice[0].1;
        let mut aabb_max = slice[0].1;
        let mut mean = Vec3::ZERO;
        let mut m2 = Vec3::ZERO;
        for (step_n, &(_, p)) in slice.iter().enumerate() {
            aabb_min.x = aabb_min.x.min(p.x);
            aabb_min.y = aabb_min.y.min(p.y);
            aabb_min.z = aabb_min.z.min(p.z);
            aabb_max.x = aabb_max.x.max(p.x);
            aabb_max.y = aabb_max.y.max(p.y);
            aabb_max.z = aabb_max.z.max(p.z);
            let inv = 1.0 / ((step_n + 1) as f32);
            let d = p - mean;
            mean += d * inv;
            let d2 = p - mean;
            m2.x += d.x * d2.x;
            m2.y += d.y * d2.y;
            m2.z += d.z * d2.z;
        }

        let count = end - start;
        nodes.push(BvhNode {
            aabb_min,
            aabb_max,
            left_or_start: start as u32,
            right_child: 0,
            count: count as u32,
        });

        if count > LEAF_CAPACITY {
            let axis = if m2.x >= m2.y && m2.x >= m2.z {
                0usize
            } else if m2.y >= m2.z {
                1usize
            } else {
                2usize
            };
            let mid_local = count / 2;
            slice.select_nth_unstable_by(mid_local, |a, b| {
                let ca = match axis {
                    0 => a.1.x,
                    1 => a.1.y,
                    _ => a.1.z,
                };
                let cb = match axis {
                    0 => b.1.x,
                    1 => b.1.y,
                    _ => b.1.z,
                };
                canonical_f32_cmp(ca, cb).then_with(|| a.0.cmp(&b.0))
            });
            let mid = start + mid_local;
            let left = Self::build_kd_bvh(items, start, mid, nodes, bird_leaf);
            let right = Self::build_kd_bvh(items, mid, end, nodes, bird_leaf);
            nodes[idx as usize].left_or_start = left;
            nodes[idx as usize].right_child = right;
            nodes[idx as usize].count = 0;
        } else {
            for &(bird_id, _) in &items[start..end] {
                bird_leaf[bird_id as usize] = idx;
            }
        }
        idx
    }

    pub fn query_all(
        &self,
        positions: &[Vec3],
        headings: &[Vec3],
        rule: NeighbourRule,
    ) -> Vec<Vec<NeighbourEntry>> {
        let flat = self.query_all_flat(positions, headings, rule);
        let n = positions.len();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(flat.row(i).to_vec());
        }
        out
    }

    /// Zero-per-bird-allocation batch query into a single contiguous `FlatNeighbourTable`.
    /// Traverses queries in spatial BVH order (`ordered_ids`) for L1 cache locality when `k <= 16`.
    pub fn query_all_flat(
        &self,
        positions: &[Vec3],
        headings: &[Vec3],
        rule: NeighbourRule,
    ) -> FlatNeighbourTable {
        let n = positions.len();
        assert_eq!(n, headings.len());
        if n == 0 {
            return FlatNeighbourTable {
                offsets: vec![0],
                entries: Vec::new(),
            };
        }

        if let NeighbourRule::ExactTopoK { k, blind_cos } = rule {
            if k > 0 && k <= 16 {
                return self.query_all_knn_spatial_order(positions, headings, k, blind_cos, f32::INFINITY);
            }
        }
        if let NeighbourRule::BoundedTopoK { k, blind_cos, r_perc } = rule {
            if k > 0 && k <= 16 {
                return self.query_all_knn_spatial_order(positions, headings, k, blind_cos, r_perc * r_perc);
            }
        }

        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0);
        let mut entries = Vec::with_capacity(n * 8);
        for i in 0..n {
            self.append_query_single(i, positions, headings[i], rule, &mut entries);
            offsets.push(entries.len() as u32);
        }
        FlatNeighbourTable { offsets, entries }
    }

    fn query_all_knn_spatial_order(
        &self,
        positions: &[Vec3],
        headings: &[Vec3],
        k: usize,
        blind_cos: f32,
        max_dist_sq_cutoff: f32,
    ) -> FlatNeighbourTable {
        let n = positions.len();
        let dummy = NeighbourEntry {
            id: u32::MAX,
            dist_sq: f32::INFINITY,
        };
        let mut slots = vec![dummy; n * k];
        let mut counts = vec![0u8; n];

        // Query in Morton/KD leaf order so adjacent birds share L1-cached BVH nodes!
        for &bird_id in &self.ordered_ids {
            let i = bird_id as usize;
            let dst = &mut slots[i * k..(i + 1) * k];
            let c = self.fill_knn_stack16(
                bird_id,
                positions[i],
                headings[i],
                k,
                blind_cos,
                max_dist_sq_cutoff,
                dst,
            );
            counts[i] = c as u8;
        }

        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0);
        let total: usize = counts.iter().map(|&c| c as usize).sum();
        let mut entries = Vec::with_capacity(total);
        for i in 0..n {
            let c = counts[i] as usize;
            entries.extend_from_slice(&slots[i * k..i * k + c]);
            offsets.push(entries.len() as u32);
        }
        FlatNeighbourTable { offsets, entries }
    }

    pub fn query_single(
        &self,
        i: usize,
        positions: &[Vec3],
        ui: Vec3,
        rule: NeighbourRule,
    ) -> Vec<NeighbourEntry> {
        let mut out = Vec::new();
        self.append_query_single(i, positions, ui, rule, &mut out);
        out
    }

    #[inline]
    fn append_query_single(
        &self,
        i: usize,
        positions: &[Vec3],
        ui: Vec3,
        rule: NeighbourRule,
        out: &mut Vec<NeighbourEntry>,
    ) {
        if self.nodes.is_empty() {
            return;
        }
        match rule {
            NeighbourRule::ExactTopoK { k, blind_cos } => {
                self.append_knn(i as u32, positions[i], ui, k, blind_cos, f32::INFINITY, out);
            }
            NeighbourRule::BoundedTopoK {
                k,
                blind_cos,
                r_perc,
            } => {
                self.append_knn(
                    i as u32,
                    positions[i],
                    ui,
                    k,
                    blind_cos,
                    r_perc * r_perc,
                    out,
                );
            }
            NeighbourRule::OctantTopoK { k_max, blind_cos } => {
                out.extend_from_slice(&query_octant_single_bird(i, positions, ui, k_max, blind_cos));
            }
            NeighbourRule::MetricRadius {
                r_metric,
                blind_cos,
            } => {
                out.extend_from_slice(&self.query_metric(
                    i as u32,
                    positions[i],
                    ui,
                    blind_cos,
                    r_metric * r_metric,
                ));
            }
        }
    }

    fn append_knn(
        &self,
        self_id: u32,
        xi: Vec3,
        ui: Vec3,
        k: usize,
        blind_cos: f32,
        max_dist_sq_cutoff: f32,
        out: &mut Vec<NeighbourEntry>,
    ) {
        if k == 0 || self.nodes.is_empty() {
            return;
        }
        if k <= 16 {
            self.append_knn_stack16(self_id, xi, ui, k, blind_cos, max_dist_sq_cutoff, out);
            return;
        }
        let mut best: Vec<NeighbourEntry> = Vec::with_capacity(k + 1);
        let root_d2 = aabb_min_dist_sq(xi, self.nodes[0].aabb_min, self.nodes[0].aabb_max);
        if root_d2 > max_dist_sq_cutoff {
            return;
        }
        let mut stack = [(0u32, root_d2); 64];
        let mut sp = 1usize;

        while sp > 0 {
            sp -= 1;
            let (node_idx, d_min_sq) = stack[sp];
            if best.len() == k && d_min_sq > best[k - 1].dist_sq {
                continue;
            }
            let node = self.nodes[node_idx as usize];
            if (node.count == 0 || best.len() < k)
                && blind_cos > -1.0
                && d_min_sq > 0.0
                && aabb_strictly_in_blind_cone(
                    xi,
                    ui,
                    blind_cos,
                    node.aabb_min,
                    node.aabb_max,
                    d_min_sq,
                )
            {
                continue;
            }

            if node.count > 0 {
                let start = node.left_or_start as usize;
                let end = start + (node.count as usize);
                for idx in start..end {
                    let cand_id = self.ordered_ids[idx];
                    if cand_id == self_id {
                        continue;
                    }
                    let diff = self.ordered_pos[idx] - xi;
                    let d2_raw = diff.norm_sq();
                    if d2_raw > max_dist_sq_cutoff {
                        continue;
                    }
                    let d2 = if d2_raw == 0.0 { 0.0 } else { d2_raw };
                    let cand = NeighbourEntry {
                        id: cand_id,
                        dist_sq: d2,
                    };
                    if best.len() == k && cand.cmp_canonical(&best[k - 1]) != Ordering::Less {
                        continue;
                    }
                    if !is_in_fov_algebraic(diff, d2_raw, ui, blind_cos) {
                        continue;
                    }
                    let pos_ins = best.partition_point(|e| e.cmp_canonical(&cand) == Ordering::Less);
                    if best.len() < k {
                        best.insert(pos_ins, cand);
                    } else if pos_ins < k {
                        best.pop();
                        best.insert(pos_ins, cand);
                    }
                }
            } else {
                let cutoff = if best.len() == k {
                    best[k - 1].dist_sq.min(max_dist_sq_cutoff)
                } else {
                    max_dist_sq_cutoff
                };
                let left = node.left_or_start;
                let right = node.right_child;
                let nl = &self.nodes[left as usize];
                let nr = &self.nodes[right as usize];
                let dl = aabb_min_dist_sq(xi, nl.aabb_min, nl.aabb_max);
                let dr = aabb_min_dist_sq(xi, nr.aabb_min, nr.aabb_max);
                let l_ok = dl <= cutoff;
                let r_ok = dr <= cutoff;
                if l_ok && r_ok {
                    if dl <= dr {
                        stack[sp] = (right, dr);
                        stack[sp + 1] = (left, dl);
                    } else {
                        stack[sp] = (left, dl);
                        stack[sp + 1] = (right, dr);
                    }
                    sp += 2;
                } else if l_ok {
                    stack[sp] = (left, dl);
                    sp += 1;
                } else if r_ok {
                    stack[sp] = (right, dr);
                    sp += 1;
                }
            }
        }
        out.extend_from_slice(&best);
    }

    #[inline]
    fn append_knn_stack16(
        &self,
        self_id: u32,
        xi: Vec3,
        ui: Vec3,
        k: usize,
        blind_cos: f32,
        max_dist_sq_cutoff: f32,
        out: &mut Vec<NeighbourEntry>,
    ) {
        let mut best = [NeighbourEntry {
            id: u32::MAX,
            dist_sq: f32::INFINITY,
        }; 16];
        let best_len = self.fill_knn_stack16(
            self_id,
            xi,
            ui,
            k,
            blind_cos,
            max_dist_sq_cutoff,
            &mut best[..k],
        );
        out.extend_from_slice(&best[..best_len]);
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn fill_knn_stack16(
        &self,
        self_id: u32,
        xi: Vec3,
        ui: Vec3,
        k: usize,
        blind_cos: f32,
        max_dist_sq_cutoff: f32,
        out_slice: &mut [NeighbourEntry],
    ) -> usize {
        debug_assert!(k > 0 && k <= 16 && out_slice.len() >= k);
        let root_d2 = aabb_min_dist_sq(xi, self.nodes[0].aabb_min, self.nodes[0].aabb_max);
        if root_d2 > max_dist_sq_cutoff {
            return 0;
        }
        let mut best = [NeighbourEntry {
            id: u32::MAX,
            dist_sq: f32::INFINITY,
        }; 16];
        let mut best_len = 0usize;
        let mut cutoff = max_dist_sq_cutoff;
        let check_cone = blind_cos > -1.0;

        // Warm-start `best` from `self_id`'s own spatial leaf (when valid) so `cutoff`
        // tightens immediately in registers before top-down tree traversal!
        let home_leaf = self
            .bird_leaf
            .get(self_id as usize)
            .copied()
            .unwrap_or(u32::MAX);
        if (home_leaf as usize) < self.nodes.len() {
            let hnode = self.nodes[home_leaf as usize];
            if hnode.count > 0 {
                let start = hnode.left_or_start as usize;
                let end = start + (hnode.count as usize);
                for idx in start..end {
                    let cand_id = self.ordered_ids[idx];
                    if cand_id == self_id {
                        continue;
                    }
                    let diff = self.ordered_pos[idx] - xi;
                    let d2_raw = diff.norm_sq();
                    if d2_raw > cutoff {
                        continue;
                    }
                    let d2 = if d2_raw == 0.0 { 0.0 } else { d2_raw };
                    let cand = NeighbourEntry {
                        id: cand_id,
                        dist_sq: d2,
                    };
                    if best_len == k && cand.cmp_canonical(&best[k - 1]) != Ordering::Less {
                        continue;
                    }
                    if check_cone && !is_in_fov_algebraic(diff, d2_raw, ui, blind_cos) {
                        continue;
                    }
                    let mut ins = best_len;
                    while ins > 0 && cand.cmp_canonical(&best[ins - 1]) == Ordering::Less {
                        if ins < 16 {
                            best[ins] = best[ins - 1];
                        }
                        ins -= 1;
                    }
                    best[ins] = cand;
                    if best_len < k {
                        best_len += 1;
                    }
                    if best_len == k {
                        cutoff = best[k - 1].dist_sq.min(max_dist_sq_cutoff);
                    }
                }
            }
        }

        let mut stack = [(0u32, root_d2); 64];
        let mut sp = 1usize;

        while sp > 0 {
            sp -= 1;
            let (node_idx, d_min_sq) = stack[sp];
            if node_idx == home_leaf || d_min_sq > cutoff {
                continue;
            }
            let node = self.nodes[node_idx as usize];
            if (node.count == 0 || best_len < k)
                && check_cone
                && d_min_sq > 0.0
                && aabb_strictly_in_blind_cone(
                    xi,
                    ui,
                    blind_cos,
                    node.aabb_min,
                    node.aabb_max,
                    d_min_sq,
                )
            {
                continue;
            }

            if node.count > 0 {
                let start = node.left_or_start as usize;
                let end = start + (node.count as usize);
                for idx in start..end {
                    let cand_id = self.ordered_ids[idx];
                    if cand_id == self_id {
                        continue;
                    }
                    let diff = self.ordered_pos[idx] - xi;
                    let d2_raw = diff.norm_sq();
                    if d2_raw > cutoff {
                        continue;
                    }
                    let d2 = if d2_raw == 0.0 { 0.0 } else { d2_raw };
                    let cand = NeighbourEntry {
                        id: cand_id,
                        dist_sq: d2,
                    };
                    if best_len == k && cand.cmp_canonical(&best[k - 1]) != Ordering::Less {
                        continue;
                    }
                    if check_cone && !is_in_fov_algebraic(diff, d2_raw, ui, blind_cos) {
                        continue;
                    }
                    let mut ins = best_len;
                    while ins > 0 && cand.cmp_canonical(&best[ins - 1]) == Ordering::Less {
                        if ins < 16 {
                            best[ins] = best[ins - 1];
                        }
                        ins -= 1;
                    }
                    best[ins] = cand;
                    if best_len < k {
                        best_len += 1;
                    }
                    if best_len == k {
                        cutoff = best[k - 1].dist_sq.min(max_dist_sq_cutoff);
                    }
                }
            } else {
                let left = node.left_or_start;
                let right = node.right_child;
                let nl = &self.nodes[left as usize];
                let nr = &self.nodes[right as usize];
                let dl = aabb_min_dist_sq(xi, nl.aabb_min, nl.aabb_max);
                let dr = aabb_min_dist_sq(xi, nr.aabb_min, nr.aabb_max);
                let l_ok = dl <= cutoff && left != home_leaf;
                let r_ok = dr <= cutoff && right != home_leaf;
                if l_ok && r_ok {
                    if dl <= dr {
                        stack[sp] = (right, dr);
                        stack[sp + 1] = (left, dl);
                    } else {
                        stack[sp] = (left, dl);
                        stack[sp + 1] = (right, dr);
                    }
                    sp += 2;
                } else if l_ok {
                    stack[sp] = (left, dl);
                    sp += 1;
                } else if r_ok {
                    stack[sp] = (right, dr);
                    sp += 1;
                }
            }
        }
        out_slice[..best_len].copy_from_slice(&best[..best_len]);
        best_len
    }

    fn query_metric(
        &self,
        self_id: u32,
        xi: Vec3,
        ui: Vec3,
        blind_cos: f32,
        r_sq: f32,
    ) -> Vec<NeighbourEntry> {
        let mut out = Vec::new();
        let mut stack = [0u32; 64];
        let mut sp = 1usize;
        stack[0] = 0;

        while sp > 0 {
            sp -= 1;
            let node = self.nodes[stack[sp] as usize];
            let d_min_sq = aabb_min_dist_sq(xi, node.aabb_min, node.aabb_max);
            if d_min_sq > r_sq {
                continue;
            }
            if blind_cos > -1.0 && d_min_sq > 0.0 && aabb_strictly_in_blind_cone(xi, ui, blind_cos, node.aabb_min, node.aabb_max, d_min_sq) {
                continue;
            }
            if node.count > 0 {
                let start = node.left_or_start as usize;
                let end = start + (node.count as usize);
                for idx in start..end {
                    let cand_id = self.ordered_ids[idx];
                    if cand_id == self_id {
                        continue;
                    }
                    let diff = self.ordered_pos[idx] - xi;
                    let d2_raw = diff.norm_sq();
                    if d2_raw <= r_sq && is_in_fov_algebraic(diff, d2_raw, ui, blind_cos) {
                        out.push(NeighbourEntry {
                            id: cand_id,
                            dist_sq: if d2_raw == 0.0 { 0.0 } else { d2_raw },
                        });
                    }
                }
            } else {
                stack[sp] = node.right_child;
                stack[sp + 1] = node.left_or_start;
                sp += 2;
            }
        }
        out.sort_by(|a, b| a.cmp_canonical(b));
        out
    }
}

/// Monotone IEEE-754 lower bound on squared distance from `p` to AABB `[bmin, bmax]`.
#[inline]
fn aabb_min_dist_sq(p: Vec3, bmin: Vec3, bmax: Vec3) -> f32 {
    let dx = if p.x < bmin.x {
        bmin.x - p.x
    } else if p.x > bmax.x {
        p.x - bmax.x
    } else {
        0.0
    };
    let dy = if p.y < bmin.y {
        bmin.y - p.y
    } else if p.y > bmax.y {
        p.y - bmax.y
    } else {
        0.0
    };
    let dz = if p.z < bmin.z {
        bmin.z - p.z
    } else if p.z > bmax.z {
        p.z - bmax.z
    } else {
        0.0
    };
    (dx * dx) + (dy * dy) + (dz * dz)
}

/// Conservative proof that EVERY point `q` in `[bmin - xi, bmax - xi]` violates `q . ui >= ||q|| * blind_cos`.
/// Includes a `1e-4` relative safety margin so multi-ULP rounding can never prune a boundary point.
#[inline]
fn aabb_strictly_in_blind_cone(
    xi: Vec3,
    ui: Vec3,
    blind_cos: f32,
    bmin: Vec3,
    bmax: Vec3,
    d_min_sq: f32,
) -> bool {
    let ax = bmin.x - xi.x;
    let bx = bmax.x - xi.x;
    let ay = bmin.y - xi.y;
    let by = bmax.y - xi.y;
    let az = bmin.z - xi.z;
    let bz = bmax.z - xi.z;

    let max_dot = (ax * ui.x).max(bx * ui.x) + (ay * ui.y).max(by * ui.y) + (az * ui.z).max(bz * ui.z);

    if blind_cos >= 0.0 {
        if max_dot < -1e-5 {
            return true;
        }
        (1.0 + 1e-4) * (max_dot * max_dot) < (blind_cos * blind_cos) * (1.0 - 1e-4) * d_min_sq
    } else {
        if max_dot >= 0.0 {
            return false;
        }
        let r_max_sq = (ax * ax).max(bx * bx) + (ay * ay).max(by * by) + (az * az).max(bz * bz);
        (1.0 - 1e-4) * (max_dot * max_dot) > (blind_cos * blind_cos) * (1.0 + 1e-4) * r_max_sq
    }
}

/// Weighted interaction graph `w_ij` in compressed-sparse-row (CSR) layout.
pub struct WeightedGraph {
    pub row_offsets: Vec<usize>,
    pub col_indices: Vec<u32>,
    pub weights: Vec<f32>,
}

impl WeightedGraph {
    pub fn from_neighbours(neighbours: &[Vec<NeighbourEntry>], symmetry: GraphSymmetry) -> Self {
        Self::from_flat_neighbours(&FlatNeighbourTable::from_vecs(neighbours), symmetry)
    }

    /// Two-pass contiguous CSR construction with zero per-bird heap `Vec` allocations.
    pub fn from_flat_neighbours(flat: &FlatNeighbourTable, symmetry: GraphSymmetry) -> Self {
        let n = flat.len();
        match symmetry {
            GraphSymmetry::DirectedNormalized => {
                let mut row_offsets = Vec::with_capacity(n + 1);
                row_offsets.push(0);
                let total_edges = flat.entries.len();
                let mut col_indices = Vec::with_capacity(total_edges);
                let mut weights = Vec::with_capacity(total_edges);
                for i in 0..n {
                    let row = flat.row(i);
                    let w = if row.is_empty() {
                        0.0
                    } else {
                        1.0 / (row.len() as f32)
                    };
                    for e in row {
                        col_indices.push(e.id);
                        weights.push(w);
                    }
                    row_offsets.push(col_indices.len());
                }
                Self {
                    row_offsets,
                    col_indices,
                    weights,
                }
            }
            GraphSymmetry::SymmetricMean | GraphSymmetry::SymmetricUnnormalized => {
                let mut counts = vec![0usize; n];
                for i in 0..n {
                    let row = flat.row(i);
                    counts[i] += row.len();
                    for e in row {
                        counts[e.id as usize] += 1;
                    }
                }
                let mut raw_offsets = Vec::with_capacity(n + 1);
                raw_offsets.push(0usize);
                for &c in &counts {
                    raw_offsets.push(raw_offsets.last().unwrap() + c);
                }
                let total_raw = *raw_offsets.last().unwrap();
                let mut scratch = vec![(0u32, 0.0f32); total_raw];
                let mut write_cursor = raw_offsets[..n].to_vec();

                for i in 0..n {
                    let row = flat.row(i);
                    let w_dir = match symmetry {
                        GraphSymmetry::SymmetricMean => {
                            if row.is_empty() {
                                0.0
                            } else {
                                0.5 / (row.len() as f32)
                            }
                        }
                        GraphSymmetry::SymmetricUnnormalized => 1.0,
                        GraphSymmetry::DirectedNormalized => unreachable!(),
                    };
                    for e in row {
                        let j = e.id as usize;
                        let ci = write_cursor[i];
                        scratch[ci] = (j as u32, w_dir);
                        write_cursor[i] = ci + 1;

                        let cj = write_cursor[j];
                        scratch[cj] = (i as u32, w_dir);
                        write_cursor[j] = cj + 1;
                    }
                }

                let mut row_offsets = Vec::with_capacity(n + 1);
                row_offsets.push(0);
                let mut col_indices = Vec::with_capacity(total_raw);
                let mut weights = Vec::with_capacity(total_raw);

                for i in 0..n {
                    let s = raw_offsets[i];
                    let e = raw_offsets[i + 1];
                    let row = &mut scratch[s..e];
                    row.sort_unstable_by(|a, b| {
                        a.0.cmp(&b.0).then_with(|| canonical_f32_cmp(a.1, b.1))
                    });
                    let mut idx = 0;
                    while idx < row.len() {
                        let col = row[idx].0;
                        let mut w_acc = row[idx].1;
                        idx += 1;
                        while idx < row.len() && row[idx].0 == col {
                            if symmetry == GraphSymmetry::SymmetricMean {
                                w_acc += row[idx].1;
                            }
                            idx += 1;
                        }
                        col_indices.push(col);
                        weights.push(w_acc);
                    }
                    row_offsets.push(col_indices.len());
                }

                Self {
                    row_offsets,
                    col_indices,
                    weights,
                }
            }
        }
    }

    #[inline]
    pub fn row_edges(&self, i: usize) -> (&[u32], &[f32]) {
        let s = self.row_offsets[i];
        let e = self.row_offsets[i + 1];
        (&self.col_indices[s..e], &self.weights[s..e])
    }

    /// Second spatial moment `M_2(k_hat) = < sum_j w_ij ((x_j - x_i) . k_hat)^2 >_i` (§2.1A).
    pub fn second_spatial_moment(&self, positions: &[Vec3], k_hat: Vec3) -> f32 {
        let n = positions.len();
        if n == 0 {
            return 0.0;
        }
        let mut total = 0.0f32;
        for (i, &xi) in positions.iter().enumerate() {
            let (cols, ws) = self.row_edges(i);
            let mut row_sum = 0.0f32;
            for (&j, &w) in cols.iter().zip(ws.iter()) {
                let proj = (positions[j as usize] - xi).dot(k_hat);
                row_sum += w * (proj * proj);
            }
            total += row_sum;
        }
        total / (n as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::{keyed_rng, normal_vec3, uniform_open01, STREAM_INIT};

    #[test]
    fn lbvh_matches_brute_force_on_ties_coincident_zero_and_sparse_fov() {
        // 1. Exact symmetric ring around bird 0 (6-way distance tie) + coincident bird 1 == bird 0
        let mut positions = vec![
            Vec3::new(0.0, 0.0, 0.0),  // 0
            Vec3::new(0.0, 0.0, 0.0),  // 1: coincident with 0!
            Vec3::new(2.0, 0.0, 0.0),  // 2: dist^2 = 4
            Vec3::new(-2.0, 0.0, 0.0), // 3: dist^2 = 4 (in blind cone for +EX if blind_cos = -0.5)
            Vec3::new(0.0, 2.0, 0.0),  // 4: dist^2 = 4
            Vec3::new(0.0, -2.0, 0.0), // 5: dist^2 = 4
            Vec3::new(0.0, 0.0, 2.0),  // 6: dist^2 = 4
            Vec3::new(0.0, 0.0, -2.0), // 7: dist^2 = 4
            Vec3::new(500.0, 0.0, 0.0), // 8: isolated straggler facing +EX (0 eligible neighbours in narrow forward FoV!)
        ];
        // Add random cloud to force multi-level BVH splits
        for s in 9..160u32 {
            let mut rng = keyed_rng(42, STREAM_INIT, 0, s);
            positions.push(normal_vec3(&mut rng) * 15.0 + Vec3::new(-40.0, 0.0, 0.0));
        }
        let headings: Vec<Vec3> = (0..positions.len())
            .map(|idx| {
                if idx == 8 || idx == 0 {
                    Vec3::EX
                } else {
                    let mut rng = keyed_rng(99, STREAM_INIT, 1, idx as u32);
                    normal_vec3(&mut rng).normalized_or(Vec3::EX)
                }
            })
            .collect();

        for &blind_cos in &[-1.0f32, -0.5f32, 0.0f32, 0.70710677f32] {
            let rule = NeighbourRule::ExactTopoK { k: 7, blind_cos };
            let oracle = query_neighbours_brute_force(&positions, &headings, rule);
            let lbvh = LbvhIndex::build(&positions).query_all(&positions, &headings, rule);
            assert_eq!(oracle, lbvh, "Oracle mismatch for blind_cos={blind_cos}");
        }
        // Verify straggler bird 8 facing +EX with narrow forward FoV (`blind_cos = 0.7`) has 0 eligible neighbours
        let narrow = NeighbourRule::ExactTopoK {
            k: 7,
            blind_cos: 0.7,
        };
        let lbvh_narrow = LbvhIndex::build(&positions).query_all(&positions, &headings, narrow);
        assert!(lbvh_narrow[8].is_empty());

        // Verify sparse flock (`N = 5 < k = 7`) returns <= 4 neighbours identically
        let sparse_pos = positions[0..5].to_vec();
        let sparse_u = headings[0..5].to_vec();
        let rule7 = NeighbourRule::ExactTopoK {
            k: 7,
            blind_cos: -0.5,
        };
        assert_eq!(
            query_neighbours_brute_force(&sparse_pos, &sparse_u, rule7),
            LbvhIndex::build(&sparse_pos).query_all(&sparse_pos, &sparse_u, rule7)
        );
    }

    #[test]
    fn lbvh_matches_brute_force_across_four_density_regimes() {
        let n = 400usize;
        for regime in 0..4 {
            let mut pos = Vec::with_capacity(n);
            let mut u = Vec::with_capacity(n);
            for i in 0..n {
                let mut rng = keyed_rng(1000 + regime, STREAM_INIT, 0, i as u32);
                let p = match regime {
                    // 0: Calm ribbon
                    0 => {
                        let v = normal_vec3(&mut rng);
                        Vec3::new(v.x * 15.0, v.y * 7.5, v.z * 2.5)
                    }
                    // 1: 5x Compacted core
                    1 => normal_vec3(&mut rng) * 0.8,
                    // 2: 3x Diluted cloud
                    2 => normal_vec3(&mut rng) * 35.0,
                    // 3: Split sub-flocks (200 m gap) + isolated stragglers
                    _ => {
                        if i < n - 6 {
                            let offset = if i % 2 == 0 { -100.0 } else { 100.0 };
                            normal_vec3(&mut rng) * 6.0 + Vec3::new(offset, 0.0, 0.0)
                        } else {
                            let u01 = uniform_open01(&mut rng);
                            Vec3::new(400.0 * u01, -350.0, 120.0)
                        }
                    }
                };
                pos.push(p);
                u.push(normal_vec3(&mut rng).normalized_or(Vec3::EX));
            }
            let rule = NeighbourRule::ExactTopoK {
                k: 7,
                blind_cos: -0.5,
            };
            let oracle = query_neighbours_brute_force(&pos, &u, rule);
            let accel = LbvhIndex::build(&pos).query_all(&pos, &u, rule);
            assert_eq!(oracle, accel, "Mismatch in density regime {regime}");
        }
    }
}
