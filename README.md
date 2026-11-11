# murmuration-v2

Physics-grounded 3D starling murmuration simulation and quantitative evaluation harness.

## Stage 1 Architecture (Native Rust Reference Harness)

- `crates/sim-core`: Pure-Rust (`libm` deterministic floating-point, counter-keyed `ChaCha8Rng` streams) simulation and observable core:
  - **Exact Reference Models**:
    - `Ref-ISM`: Constant-speed Inertial Spin Model (Cavagna et al. 2015) + matched constant-speed Overdamped Vicsek comparator, with explicit graph reciprocity switches (`SymmetricMean`, `DirectedNormalized`, `SymmetricUnnormalized`).
    - `Ref-MarginalSPP`: Cartesian variable-speed self-propelled particle model (Cavagna et al. 2022) comparing octic marginal speed confinement against linear Gaussian speed control with zero artificial speed clamps.
  - **Exact Biological Neighbour Rules vs. Spatial Acceleration**:
    - `ExactTopoK` (`k = 7` in field-of-view over `R^3`) decoupled from explicit biological model variants (`BoundedTopoK`, `OctantTopoK`, `MetricRadius`).
    - Morton-ordered `LbvhIndex` verified for 100% exact ID, order, and squared-distance parity against the `O(N^2)` `query_neighbours_brute_force` oracle across exact ties, coincident positions, `0` and `< 7` eligible neighbours, and four density stress regimes.
  - **Multi-Rate Kinematic Evaluators**:
    - `200 Hz` (`dt = 5 ms`) full-tick directional turn-wave propagation speed (`E7`),
    - `40 Hz` centre-of-mass (`E6a`) and mutual-neighbour (`E6b`) internal diffusion,
    - `10 Hz` polarization (`E1`), topological anisotropy `gamma(n)` (`E2`), principal-axis bounding-box spatial proportions `L1 : L2 : L3` (`E3`), orientation/speed correlation lengths (`E4`), single-bird and group speed distributions across `N = 10..3000` (`E5`), and neighbour retention `Q_M(t)` (`E6c`).
- `crates/sim-cli`: Multi-threaded (`rayon`) verification and scenario runner.

## Quick Start

Enable the privacy pre-commit and pre-push hooks before committing:

```sh
git config core.hooksPath .githooks
node scripts/privacy-check.js
```

Run unit tests and the full Stage 1 verification suite:

```sh
cargo test --workspace
cargo run --release -p sim-cli -- verify-stage1
cargo run --release -p sim-cli -- run scenarios/calm_marginal_spp.toml
```

## License

MIT
