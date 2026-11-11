use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const EX: Self = Self { x: 1.0, y: 0.0, z: 0.0 };
    pub const EY: Self = Self { x: 0.0, y: 1.0, z: 0.0 };
    pub const EZ: Self = Self { x: 0.0, y: 0.0, z: 1.0 };

    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn dot(self, rhs: Self) -> f32 {
        (self.x * rhs.x) + (self.y * rhs.y) + (self.z * rhs.z)
    }

    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: (self.y * rhs.z) - (self.z * rhs.y),
            y: (self.z * rhs.x) - (self.x * rhs.z),
            z: (self.x * rhs.y) - (self.y * rhs.x),
        }
    }

    #[inline]
    pub fn norm_sq(self) -> f32 {
        self.dot(self)
    }

    #[inline]
    pub fn norm(self) -> f32 {
        libm::sqrtf(self.norm_sq())
    }

    #[inline]
    pub fn normalized_or(self, fallback: Self) -> Self {
        let n2 = self.norm_sq();
        if n2 > 1e-24 {
            let inv = 1.0 / libm::sqrtf(n2);
            self * inv
        } else {
            fallback
        }
    }

    /// Orthogonal projection perpendicular to unit vector `u`: `P_perp(u) * self`.
    #[inline]
    pub fn project_perp(self, u_unit: Self) -> Self {
        self - u_unit * self.dot(u_unit)
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl AddAssign for Vec3 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl SubAssign for Vec3 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}

impl Mul<Vec3> for f32 {
    type Output = Vec3;
    #[inline]
    fn mul(self, v: Vec3) -> Vec3 {
        v * self
    }
}

impl Div<f32> for Vec3 {
    type Output = Self;
    #[inline]
    fn div(self, s: f32) -> Self {
        let inv = 1.0 / s;
        self * inv
    }
}

impl Neg for Vec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

/// Canonical total comparison for non-NaN `f32` that normalizes `-0.0` to `+0.0` before comparing.
#[inline]
pub fn canonical_f32_cmp(a: f32, b: f32) -> std::cmp::Ordering {
    let ca = if a == 0.0 { 0.0f32 } else { a };
    let cb = if b == 0.0 { 0.0f32 } else { b };
    ca.total_cmp(&cb)
}

/// Exact Rodrigues rotation `exp([axis_angle]_x) * v` using pure-Rust `libm` transcendentals
/// and a smooth Taylor expansion for `|axis_angle| < 1e-4` rad.
#[inline]
pub fn rodrigues_rotate(v: Vec3, axis_angle: Vec3) -> Vec3 {
    let theta_sq = axis_angle.norm_sq();
    let (sinc, one_minus_cos_over_sq) = if theta_sq < 1e-8 {
        (1.0 - theta_sq * (1.0 / 6.0), 0.5 - theta_sq * (1.0 / 24.0))
    } else {
        let theta = libm::sqrtf(theta_sq);
        (
            libm::sinf(theta) / theta,
            (1.0 - libm::cosf(theta)) / theta_sq,
        )
    };
    let w_cross_v = axis_angle.cross(v);
    let w_cross_w_cross_v = axis_angle.cross(w_cross_v);
    v + w_cross_v * sinc + w_cross_w_cross_v * one_minus_cos_over_sq
}

/// Rotate a vector of fixed magnitude `target_speed` around `axis_angle` without norm drift.
#[inline]
pub fn rodrigues_rotate_fixed_speed(v: Vec3, axis_angle: Vec3, target_speed: f32) -> Vec3 {
    rodrigues_rotate(v, axis_angle).normalized_or(Vec3::EX) * target_speed
}

/// Gravity-consistent normal lift bound (`||a_perp + g * P_perp(u) e_z|| <= l_max_specific`, §2.3).
/// When exceeded, preserves vertical weight-supporting lift (`l_req` along `P_perp(u) e_z`) whenever
/// feasible (`|l_vert| <= l_max_specific`) and clamps lateral turning acceleration to
/// `sqrt(l_max_specific^2 - l_vert^2)` (so a horizontal level turn saturates at `sqrt((L_max/m)^2 - g^2)`
/// without losing altitude); falls back to radial scaling if `|l_vert| > l_max_specific`.
pub fn clamp_normal_lift_with_gravity(
    u_unit: Vec3,
    a_perp: Vec3,
    gravity_mps2: f32,
    l_max_specific: f32,
) -> (Vec3, bool, f32) {
    let g_perp_dir = Vec3::EZ.project_perp(u_unit);
    let g_perp = g_perp_dir * gravity_mps2;
    let l_req = a_perp + g_perp;
    let l_req_mag = l_req.norm();
    if l_req_mag <= l_max_specific {
        return (a_perp, false, 0.0);
    }

    let g_dir_norm = g_perp_dir.norm();
    let l_clamped = if g_dir_norm > 1e-4 {
        let dorsal_0 = g_perp_dir / g_dir_norm;
        let l_vert = l_req.dot(dorsal_0);
        if l_vert.abs() < l_max_specific {
            let l_lat_vec = l_req - dorsal_0 * l_vert;
            let l_lat_mag = l_lat_vec.norm();
            let max_lat = libm::sqrtf((l_max_specific * l_max_specific - l_vert * l_vert).max(0.0));
            let lat_clamped = if l_lat_mag > 1e-12 {
                l_lat_vec * (max_lat / l_lat_mag)
            } else {
                Vec3::ZERO
            };
            dorsal_0 * l_vert + lat_clamped
        } else if l_req_mag > 1e-12 {
            l_req * (l_max_specific / l_req_mag)
        } else {
            Vec3::ZERO
        }
    } else if l_req_mag > 1e-12 {
        l_req * (l_max_specific / l_req_mag)
    } else {
        Vec3::ZERO
    };

    let achieved_a_perp = l_clamped - g_perp;
    (achieved_a_perp, true, l_req_mag - l_max_specific)
}

/// Mode K kinematic bank frame: computes horizon reference `(right_0, dorsal_0)` and curvature+gravity
/// inferred bank angle `phi_bank` (`atan2(l_req . right_0, l_req . dorsal_0)`).
pub fn compute_mode_k_bank_angle(
    u_unit: Vec3,
    a_perp: Vec3,
    gravity_mps2: f32,
    prev_right: Vec3,
) -> (Vec3, Vec3, f32) {
    let horiz = u_unit.cross(Vec3::EZ);
    let right_0 = if horiz.norm_sq() > 1e-4 {
        horiz.normalized_or(Vec3::EY)
    } else {
        prev_right.project_perp(u_unit).normalized_or(Vec3::EY)
    };
    let dorsal_0 = right_0.cross(u_unit).normalized_or(Vec3::EZ);
    let l_req = a_perp + Vec3::EZ.project_perp(u_unit) * gravity_mps2;
    let lat = l_req.dot(right_0);
    let dor = l_req.dot(dorsal_0);
    let phi_bank = if lat * lat + dor * dor > 1e-12 {
        libm::atan2f(lat, dor)
    } else {
        0.0
    };
    (right_0, dorsal_0, phi_bank)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rodrigues_conserves_orthogonality_and_norm() {
        let v = Vec3::new(12.0, 0.0, 0.0);
        let s = Vec3::new(0.0, 0.0, 3.5);
        let v_rot = rodrigues_rotate_fixed_speed(v, s * 0.005, 12.0);
        assert!((v_rot.norm() - 12.0).abs() < 1e-5);
        // Rotation around `s` leaves `s . v` unchanged (0 -> 0)
        assert!(s.dot(v_rot).abs() < 1e-6);
    }

    #[test]
    fn lift_clamp_accounts_for_gravity_in_level_turn() {
        let u = Vec3::EX;
        let g = 9.81f32;
        let l_max = 2.0 * g; // 2g max specific lift => max horizontal turn = sqrt(3)*g
        let max_horiz_turn = libm::sqrtf(l_max * l_max - g * g);
        let a_req = Vec3::new(0.0, -3.0 * g, 0.0);
        let (a_clamped, hit, _) = clamp_normal_lift_with_gravity(u, a_req, g, l_max);
        assert!(hit);
        let l_tot = (a_clamped + Vec3::EZ * g).norm();
        assert!((l_tot - l_max).abs() < 1e-4);
        assert!(a_clamped.norm() <= max_horiz_turn + 1e-3);
    }

    #[test]
    fn mode_k_bank_angle_45_deg_for_1g_right_turn() {
        let u = Vec3::EX; // East; right wing is -EY (`EX x EZ = -EY`)
        let g = 9.81f32;
        let a_perp = Vec3::new(0.0, -g, 0.0); // 1g turn to the right (-EY)
        let (_, _, phi) = compute_mode_k_bank_angle(u, a_perp, g, Vec3::new(0.0, -1.0, 0.0));
        assert!((phi - core::f32::consts::FRAC_PI_4).abs() < 1e-5);
    }
}
