// Shared WGSL shader for instanced 3D starling rendering + screen-space velocity buffer
// + M_shutter linear-light directional motion blur + IEC 61966-2-1 linear -> sRGB (§4.2, §4.4).

struct SceneUniforms {
  view_proj: mat4x4<f32>,
  prev_view_proj: mat4x4<f32>,
  cam_pos_and_half_span: vec4<f32>, // xyz = cam world pos, w = halfSpanM (0.191 m)
  sun_dir_and_exposure: vec4<f32>,  // xyz = normalized sun dir, w = exposure
  sky_zenith: vec4<f32>,            // xyz = linear RGB zenith, w = shutter_strength (0 = off, 1 = on)
  sky_horizon: vec4<f32>,           // xyz = linear RGB dusk horizon, w = output_needs_srgb_encode (1 or 0)
  mesh_meta: vec4<u32>,             // x = vertex_count (316), y = flap_frame_count (37), z = glide_frame_idx (37), w = bound_frame_idx (38)
};

struct BirdInstance {
  pos_speed: vec4<f32>,   // xyz = world position (m), w = speed (m/s)
  quat_xyzw: vec4<f32>,   // unit quaternion rotating (+X=beak, +Y=left wing, +Z=dorsal) -> world
  gait_params: vec4<f32>, // x = flap_phase_01, y = glide_w, z = bound_w, w = bank_angle_rad
};

struct PoseVertex {
  pos_pad: vec4<f32>,
  norm_pad: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u_scene: SceneUniforms;
@group(0) @binding(1) var<storage, read> s_poses: array<PoseVertex>;
@group(0) @binding(2) var<storage, read> s_curr_instances: array<BirdInstance>;
@group(0) @binding(3) var<storage, read> s_prev_instances: array<BirdInstance>;
@group(0) @binding(4) var<storage, read> s_draw_bird_ids: array<u32>;

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}

fn eval_starling_local_pose(vert_id: u32, gait: vec4<f32>) -> PoseVertex {
  let v_count = u_scene.mesh_meta.x;
  let flap_frames = u_scene.mesh_meta.y; // 37 (0..36)
  let glide_frame = u_scene.mesh_meta.z;
  let bound_frame = u_scene.mesh_meta.w;

  let phase = clamp(gait.x, 0.0, 0.999999) * f32(flap_frames - 1u);
  let k0 = min(u32(floor(phase)), flap_frames - 1u);
  let k1 = min(k0 + 1u, flap_frames - 1u);
  let frac = phase - floor(phase);

  let f0 = s_poses[k0 * v_count + vert_id];
  let f1 = s_poses[k1 * v_count + vert_id];
  var p = mix(f0.pos_pad.xyz, f1.pos_pad.xyz, frac);
  var n = mix(f0.norm_pad.xyz, f1.norm_pad.xyz, frac);

  let w_glide = clamp(gait.y, 0.0, 1.0);
  let w_bound = clamp(gait.z, 0.0, 1.0);
  let w_flap = max(0.0, 1.0 - w_glide - w_bound);
  let w_sum = max(1e-5, w_flap + w_glide + w_bound);

  let g = s_poses[glide_frame * v_count + vert_id];
  let b = s_poses[bound_frame * v_count + vert_id];

  p = (p * w_flap + g.pos_pad.xyz * w_glide + b.pos_pad.xyz * w_bound) / w_sum;
  n = normalize((n * w_flap + g.norm_pad.xyz * w_glide + b.norm_pad.xyz * w_bound) / w_sum);

  return PoseVertex(vec4<f32>(p, 0.0), vec4<f32>(n, 0.0));
}

struct VsOut {
  @builtin(position) clip_pos: vec4<f32>,
  @location(0) world_pos: vec3<f32>,
  @location(1) world_norm: vec3<f32>,
  @location(2) curr_ndc: vec3<f32>,
  @location(3) prev_ndc: vec3<f32>,
  @location(4) local_y_and_bank: vec2<f32>,
};

@vertex
fn vs_starling(
  @location(0) vert_idx_u32: u32,
  @builtin(instance_index) inst_slot: u32,
) -> VsOut {
  let bird_id = s_draw_bird_ids[inst_slot];
  let curr = s_curr_instances[bird_id];
  let prev = s_prev_instances[bird_id];

  let half_span = u_scene.cam_pos_and_half_span.w;
  let curr_pose = eval_starling_local_pose(vert_idx_u32, curr.gait_params);
  let prev_pose = eval_starling_local_pose(vert_idx_u32, prev.gait_params);

  let world_p = curr.pos_speed.xyz + quat_rotate(curr.quat_xyzw, curr_pose.pos_pad.xyz * half_span);
  let prev_world_p = prev.pos_speed.xyz + quat_rotate(prev.quat_xyzw, prev_pose.pos_pad.xyz * half_span);
  let world_n = normalize(quat_rotate(curr.quat_xyzw, curr_pose.norm_pad.xyz));

  let clip_curr = u_scene.view_proj * vec4<f32>(world_p, 1.0);
  let clip_prev = u_scene.prev_view_proj * vec4<f32>(prev_world_p, 1.0);

  var out: VsOut;
  out.clip_pos = clip_curr;
  out.world_pos = world_p;
  out.world_norm = world_n;
  out.curr_ndc = vec3<f32>(clip_curr.xy, max(clip_curr.w, 1e-4));
  out.prev_ndc = vec3<f32>(clip_prev.xy, max(clip_prev.w, 1e-4));
  out.local_y_and_bank = vec2<f32>(abs(curr_pose.pos_pad.y), curr.gait_params.w);
  return out;
}

struct FsSceneOut {
  @location(0) hdr_rgba: vec4<f32>,
  @location(1) vel_uv: vec2<f32>,
};

@fragment
fn fs_starling(in: VsOut, @builtin(front_facing) is_front: bool) -> FsSceneOut {
  let n = select(-normalize(in.world_norm), normalize(in.world_norm), is_front);
  let sun_dir = normalize(u_scene.sun_dir_and_exposure.xyz);
  let view_dir = normalize(u_scene.cam_pos_and_half_span.xyz - in.world_pos);

  // Dark Sturnus vulgaris plumage in linear space: dark soot-charcoal body with subtle dorsal sheen,
  // producing realistic silhouette contrast against bright twilight sky when banked broadside (E8).
  let wing_tip_fade = smoothstep(0.15, 0.95, in.local_y_and_bank.x);
  let base_albedo = mix(vec3<f32>(0.022, 0.024, 0.028), vec3<f32>(0.036, 0.033, 0.030), wing_tip_fade);

  let ndotl = max(dot(n, sun_dir), 0.0);
  let sky_hemi = 0.5 + 0.5 * n.z;
  let ambient = mix(vec3<f32>(0.09, 0.07, 0.08), u_scene.sky_zenith.xyz * 0.45, sky_hemi);
  let sun_light = vec3<f32>(1.35, 0.88, 0.55) * ndotl;

  let half_vec = normalize(sun_dir + view_dir);
  let spec = pow(max(dot(n, half_vec), 0.0), 20.0) * 0.035;
  let rim = pow(1.0 - max(dot(n, view_dir), 0.0), 3.0) * 0.028;

  let color_lin = (base_albedo * (ambient + sun_light) + vec3<f32>(spec + rim)) * u_scene.sun_dir_and_exposure.w;

  // Screen-space UV velocity in [-0.5, 0.5]
  let uv_curr = (in.curr_ndc.xy / in.curr_ndc.z) * vec2<f32>(0.5, -0.5);
  let uv_prev = (in.prev_ndc.xy / in.prev_ndc.z) * vec2<f32>(0.5, -0.5);
  let v_uv = clamp(uv_curr - uv_prev, vec2<f32>(-0.035), vec2<f32>(0.035));

  var out: FsSceneOut;
  out.hdr_rgba = vec4<f32>(color_lin, 1.0);
  out.vel_uv = v_uv;
  return out;
}

// ============================================================================
// Pass 2: Twilight Sky + M_shutter Linear-Light Motion Blur + Linear -> sRGB
// ============================================================================

@group(0) @binding(0) var<uniform> u_comp: SceneUniforms;
@group(0) @binding(1) var t_hdr: texture_2d<f32>;
@group(0) @binding(2) var t_vel: texture_2d<f32>;
@group(0) @binding(3) var s_linear: sampler;

struct FullscreenOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> FullscreenOut {
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(3.0, 1.0)
  );
  let p = pos[vid];
  var out: FullscreenOut;
  out.pos = vec4<f32>(p, 0.0, 1.0);
  out.uv = vec2<f32>(0.5 * (p.x + 1.0), 0.5 * (1.0 - p.y));
  return out;
}

fn sky_gradient_linear(uv: vec2<f32>) -> vec3<f32> {
  // uv.y = 0 at top (zenith), uv.y = 1 at bottom (horizon)
  let h = clamp(uv.y, 0.0, 1.0);
  let band = pow(h, 1.65);
  let base = mix(u_comp.sky_zenith.xyz, u_comp.sky_horizon.xyz, band);
  // Warm dusk glow near lower-left solar azimuth
  let sun_uv = vec2<f32>(0.24, 0.86);
  let d = length((uv - sun_uv) * vec2<f32>(1.6, 1.0));
  let glow = exp(-d * 3.8) * vec3<f32>(0.26, 0.11, 0.04);
  return base + glow;
}

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linear_to_srgb_iec(c_in: vec3<f32>) -> vec3<f32> {
  let c = clamp(c_in, vec3<f32>(0.0), vec3<f32>(1.0));
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3<f32>(0.0031308));
}

@fragment
fn fs_composite(in: FullscreenOut) -> @location(0) vec4<f32> {
  let shutter = u_comp.sky_zenith.w;
  let v_center = textureSampleLevel(t_vel, s_linear, in.uv, 0.0).xy * shutter;
  let speed_uv = length(v_center);

  var accum_lin = vec3<f32>(0.0);
  if (speed_uv < 1e-4) {
    let fg = textureSampleLevel(t_hdr, s_linear, in.uv, 0.0);
    let sky = sky_gradient_linear(in.uv);
    accum_lin = mix(sky, fg.rgb, fg.a);
  } else {
    // 5-tap 180-degree rotary shutter integration in linear radiance space (§4.4.1)
    let m_taps = 5;
    for (var k = 0; k < m_taps; k = k + 1) {
      let t = (f32(k) / f32(m_taps - 1)) - 0.5;
      let sample_uv = clamp(in.uv + v_center * (0.5 * t), vec2<f32>(0.001), vec2<f32>(0.999));
      let fg = textureSampleLevel(t_hdr, s_linear, sample_uv, 0.0);
      let sky = sky_gradient_linear(sample_uv);
      accum_lin += mix(sky, fg.rgb, fg.a);
    }
    accum_lin /= f32(m_taps);
  }

  let mapped = aces_tonemap(accum_lin);
  let out_rgb = select(mapped, linear_to_srgb_iec(mapped), u_comp.sky_horizon.w > 0.5);
  return vec4<f32>(out_rgb, 1.0);
}
