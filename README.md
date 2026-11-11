# murmuration-v2

Physics-grounded 3D starling murmuration simulation, multi-rate quantitative evaluation harness, and instanced WebGPU viewer.

## Staged Gate Architecture

Development is organized into four sequential, quantitatively gated stages:

1. **Stage 1 — Native Rust Reference Harness**: Exact `Ref-ISM` (Cavagna et al. 2015) and `Ref-MarginalSPP` (Cavagna et al. 2022), variance-split `ExactTopoK` LBVH verified 100% against an `O(N^2)` brute-force oracle, and multi-rate kinematic evaluators (`200 Hz` / `40 Hz` / `10 Hz`).
2. **Stage 2 — Coupled Speed–Heading SDE & Factorial Ablations**: `1/v`-scaled coupled `(v_i, u_i, omega_i)` SDE with mandatory Itô spherical Jacobian drift (`+2 D_v / v_i^* dt`), deterministic/stochastic reduction proofs, and `2x2` factorial attribution (`Exp 2`, `Exp 3`) with zero artificial speed clamps.
3. **Stage 3 — `wasm32 +simd128` Parity & Instanced WebGPU Viewer**: Standalone WebAssembly module (zero external imports, no `SharedArrayBuffer` / `COOP`+`COEP`), 100% IEEE-754 bitwise Native <-> Wasm trajectory parity, and 12-cell multi-tick WebGPU frame-budget suite with 3D `Mode K` starling geometry and Rayner (2001) intermittent flap/glide/bound gait.
4. **Stage 4 — Grounded Horizon Camera, Predator Pursuit & Validated Optical Evaluation**: Stationary ground-observer camera with horizon reference, Mills (2018) PPN falcon stoop, `StarEscape` collective escape waves, `Mode K` vs. `Mode A` closed-loop flight actuation, shared-shader headless `wgpu` optical detectors (`C_roll`, `C_pos`, `C_full`, `C_cam`), and `Train / Validation / Final Test` parameter search.

## Privacy Hook Setup

```sh
git config core.hooksPath .githooks
node scripts/privacy-check.js
```

## License

MIT
