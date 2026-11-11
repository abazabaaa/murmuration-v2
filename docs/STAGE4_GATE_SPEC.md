# Stage 4 Gate Specification — Grounded Horizon Camera, Predator Pursuit & Validated Optical Evaluation

This branch tracks **Stage 4** (stacked atop `stage-3-wasm-webgpu-viewer`).

## 1. Interactive Viewer Comfort & Ribbon Morphology Fixes (Pre-Search Prerequisite)

Interactive review of the Stage 3 viewer surfaced two critical human-factors and morphology gaps that must be resolved before running Stage 4 optical parameter search:

1. **Stationary Ground-Observer Camera + Dusk Horizon Reference (Eliminating Motion Sickness)**:
   - Replace default center-of-mass camera translation (`Follow Flock`) and continuous zoom breathing (`Auto-Frame`) with a **stationary ground-level / tripod observer** (`~140-220 m` from the roost, fixed FOV, optional gentle pan deadzone only near viewport margins).
   - Render a **static dusk horizon and ground/reedbed silhouette** so human observers have a stable vestibular visual anchor and perceive the murmuration's true `12 m/s` collective flow across the sky.
2. **Finite Wave-Transit Ribbon Regime (`L / c_s` ~ `1.5-4.0 s`)**:
   - Prevent stiff rigid-body lump turning (where `L ~ 25 m` and `c_s ~ 25 m/s` cross the entire group in `< 1 s`) by supporting elongated ribbon/stream aspect ratios, asynchronous cognitive neighbour refresh (`StarEscape` `20/50 ms` hold, which also cuts per-tick LBVH cost by `4x-10x`), and localized edge/predator turn initiations so the head folds against the trailing stream.

## 2. Core Stage 4 Physics & Observation Deliverables

1. **Predator Pursuit (`Mills et al. 2018`)**:
   - Pure Proportional Navigation (`PPN`) peregrine falcon agent with sensorimotor delay / filter timescale `tau_p = 50 ms`, realistic speed/load limits, and comparative active-pursuit vs. passive-flyby trajectories (`E10`).
2. **Collective Escape Cascades (`Papadopoulou et al. 2022`, *StarEscape*)**:
   - Asynchronous perception hold (`20 ms` / `50 ms`) + Markov escape state machine (alert -> evasive banked zig -> neighbour copying -> refractory recovery) producing flash expansion, vacuoles, split-and-rejoin, and post-attack compression/dilution (`E9` over full `35 s` windows).
3. **Closed-Loop Flight Actuation (`Mode K` vs. `Mode A`)**:
   - Side-by-side comparison of `Mode K` (kinematic curvature+gravity bank visualization) against `Mode A` (roll-rate-limited lift vector + gravity + drag actuation with logged saturation counts).
4. **Shared-Shader Headless `wgpu` Observer & 4-Condition Optical Isolation (`E8`)**:
   - Native headless `wgpu` renderer sharing `web/src/shaders/starling.wgsl` and starling vertex/pose buffers (`PSNR > 45 dB` vs. browser WebGPU).
   - Four controlled render passes on identical trajectories to prevent degenerate metric gaming:
     - `C_roll`: frozen positions + live roll/wing attitude (isolates planform broadside darkening).
     - `C_pos`: fixed attitude / isotropic proxy + live positions (isolates density & occlusion darkening).
     - `C_full`: full coupled render (comparing `Mode K/A` banked turns vs. single-sided zig vs. double-band zigzag control).
     - `C_cam`: multi-azimuth/elevation & exposure sweep from grounded observer viewpoints.
5. **Strict 3-Way Optimization Split (`Train` / `Validation` / `Final Test`)**:
   - Disjoint seed banks, flock sizes, and attack azimuths across `Train` (`CMA-ES` / `NSGA-II`), `Validation` (`35 s` video review + detector thresholds), and one-shot frozen `Final Test`.

## 3. Quantitative Gate Acceptance Criteria

- [ ] **Comfort & Grounded Flow**: Default viewer operates from a stationary horizon-anchored camera with zero continuous zoom-pumping (`Follow Flock` / `Auto-Frame` opt-in only).
- [ ] **Renderer Parity**: Native `wgpu` vs. browser WebGPU keyframe `PSNR > 45 dB`.
- [ ] **Optical Isolation (`E8`)**: `C_roll`, `C_pos`, `C_full`, and `C_cam` cleanly separate attitude vs. projected-density darkening and single-band vs. double-band zig controls.
- [ ] **Escape & Recovery Dynamics (`E9`, `E10`)**: Active PPN pursuit triggers statistically higher agitation/split response than matched passive flyby (`E10`), full `35 s` post-attack dilution/re-cohesion trajectory captured (`E9`), zero silent speed clamps, and frozen `Final Test` report committed.
