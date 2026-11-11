# murmuration-v2

Physics-grounded 3D starling murmuration simulation, multi-rate quantitative evaluation harness, and instanced WebGPU viewer.

## Workspace Architecture (Stages 1–3)

- `crates/sim-core`: Pure-Rust (`libm` deterministic floating-point, counter-keyed `ChaCha8Rng` streams) simulation, neighbour search, instance packing, and observable core:
  - **Reference Models (`Stage 1`)**:
    - `Ref-ISM`: Constant-speed Inertial Spin Model (Cavagna et al. 2015) + matched constant-speed Overdamped Vicsek comparator, with explicit graph reciprocity switches (`SymmetricMean`, `DirectedNormalized`, `SymmetricUnnormalized`).
    - `Ref-MarginalSPP`: Cartesian variable-speed self-propelled particle model (Cavagna et al. 2022) comparing octic marginal speed confinement against linear Gaussian speed control with zero artificial speed clamps.
  - **Coupled Speed–Heading SDE (`Stage 2`)**:
    - `1/v`-scaled polar/attitude `(v_i, u_i, omega_i)` SDE with mandatory Itô spherical drift `+2 D_v / v_i^*`, symplectic exponential-Euler bivariate Ornstein–Uhlenbeck angular integration across all `tau_omega >= 0`, and safeguarded drift-implicit 1D speed integration with zero hard clamps.
  - **Exact Biological Neighbour Rules vs. Spatial Acceleration**:
    - `ExactTopoK` (`k = 7` in field-of-view over `R^3`) strictly decoupled from named biological model variants (`BoundedTopoK`, `OctantTopoK`, `MetricRadius`).
    - Variance-split `LbvhIndex` verified for 100% exact ID, order, and squared-distance parity against the `O(N^2)` `query_neighbours_brute_force` oracle across exact ties, coincident positions, `0` and `< 7` eligible neighbours, and four density stress regimes.
  - **Shared 48-Byte GPU Instance Packing & Rayner (2001) Gait (`Stage 3`)**:
    - `BirdInstanceGpu` (`48 bytes`), `Mode K` curvature+gravity bank quaternion builder, Rayner et al. (2001) intermittent flap/glide/bound scheduler, and double-buffered sub-tick pose interpolation (`alpha = accumulator / dt_sim`).
- `crates/sim-cli`: Multi-threaded (`rayon`) native verification and scenario runner (`verify-stage1`, `verify-stage2`, `stage3-native-parity`, `run`).
- `crates/sim-wasm`: Single-threaded `wasm32-unknown-unknown +simd128` WebAssembly module with zero external imports and no `SharedArrayBuffer` or `COOP`/`COEP` header requirement.
- `web/`: TypeScript + Vite + WebGPU instanced 3D starling renderer (`316` vertices, `440` near / `48` far LOD triangles, `37` cyclic flap frames + glide + bound poses, `rgba16float` linear radiance + 5-tap velocity-buffer shutter motion blur + IEC linear-to-sRGB output).

## Quick Start

Enable the privacy pre-commit and pre-push hooks before committing:

```sh
git config core.hooksPath .githooks
node scripts/privacy-check.js
```

Run native tests and Stage 1 + Stage 2 verification suites:

```sh
cargo test --workspace --release
cargo run --release -p sim-cli -- verify-stage1
cargo run --release -p sim-cli -- verify-stage2
cargo run --release -p sim-cli -- run scenarios/coupled_calm_inertial_marginal.toml
```

Build the `wasm32 +simd128` module + WebGPU viewer and run Stage 3 verification suites:

```sh
cargo build --release --target wasm32-unknown-unknown -p sim-wasm
node scripts/verify-stage3-wasm.mjs
pnpm --dir web install
pnpm --dir web build
node scripts/verify-stage3-webgpu.mjs
```

Launch the interactive WebGPU viewer locally:

```sh
pnpm --dir web dev
```

## License

MIT
