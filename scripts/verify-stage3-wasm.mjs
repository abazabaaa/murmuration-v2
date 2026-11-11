#!/usr/bin/env node
/**
 * Stage 3 Gate 1 Verification Suite:
 * 1. Native `aarch64` vs. `wasm32-unknown-unknown +simd128` 1,000-tick (5.0 s) bitwise & 3.0 s trajectory parity
 *    across `Ref-ISM`, `Ref-MarginalSPP`, and `CoupledFlock`.
 * 2. 64-seed paired bootstrap Two-One-Sided Equivalence Tests (TOST, 30.0 s rollouts) against §4.1.3 margins.
 * 3. Single-threaded `wasm32 +simd128` vs. Native simulation tick, 3/4-tick frame, and instance-packing
 *    timings across all 4 spatial stress scenarios at N in {1000, 5000, 10000}.
 */

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const WASM_PATH = path.join(
  ROOT,
  "target",
  "wasm32-unknown-unknown",
  "release",
  "sim_wasm.wasm"
);
const CLI_PATH = path.join(ROOT, "target", "release", "sim-cli");
const OUT_DIR = path.join(ROOT, "out");
const OUT_JSON = path.join(OUT_DIR, "stage3_wasm_parity_and_bench.json");

function round4(v) {
  return Math.round(v * 10000) / 10000;
}

function round6(v) {
  return Math.round(v * 1e6) / 1e6;
}

function splitU64(seedBig) {
  const b = BigInt(seedBig);
  const lo = Number(b & 0xffffffffn) >>> 0;
  const hi = Number((b >> 32n) & 0xffffffffn) >>> 0;
  return [lo, hi];
}

function u32ToF32(u) {
  const buf = new ArrayBuffer(4);
  const uView = new Uint32Array(buf);
  const fView = new Float32Array(buf);
  uView[0] = u >>> 0;
  return fView[0];
}

function ulpDistanceF32(uA, uB) {
  const a = uA >>> 0;
  const b = uB >>> 0;
  if (a === b) return 0;
  // Normalize -0.0 (0x80000000) and +0.0 (0x00000000)
  const normZero = (x) => (x === 0x80000000 ? 0 : x);
  const na = normZero(a);
  const nb = normZero(b);
  if (na === nb) return 0;
  const toLex = (x) => ((x & 0x80000000) !== 0 ? (0x80000000 - (x & 0x7fffffff)) : x + 0x80000000);
  return Math.abs(toLex(na) - toLex(nb));
}

function maxPosVelDiff(natBits, wasmBits) {
  const nBirds = Math.floor(natBits.length / 6);
  let maxPosErr = 0;
  let maxVelErr = 0;
  let differingFloats = 0;
  let differingIgnoreSignedZero = 0;
  let maxUlp = 0;

  for (let k = 0; k < natBits.length; k++) {
    const ua = natBits[k] >>> 0;
    const ub = wasmBits[k] >>> 0;
    if (ua !== ub) {
      differingFloats++;
      const ulp = ulpDistanceF32(ua, ub);
      if (ulp > 0) differingIgnoreSignedZero++;
      if (ulp > maxUlp) maxUlp = ulp;
    }
  }

  for (let i = 0; i < nBirds; i++) {
    const base = i * 6;
    const dx = u32ToF32(natBits[base]) - u32ToF32(wasmBits[base]);
    const dy = u32ToF32(natBits[base + 1]) - u32ToF32(wasmBits[base + 1]);
    const dz = u32ToF32(natBits[base + 2]) - u32ToF32(wasmBits[base + 2]);
    const dvx = u32ToF32(natBits[base + 3]) - u32ToF32(wasmBits[base + 3]);
    const dvy = u32ToF32(natBits[base + 4]) - u32ToF32(wasmBits[base + 4]);
    const dvz = u32ToF32(natBits[base + 5]) - u32ToF32(wasmBits[base + 5]);
    const pe = Math.hypot(dx, dy, dz);
    const ve = Math.hypot(dvx, dvy, dvz);
    if (pe > maxPosErr) maxPosErr = pe;
    if (ve > maxVelErr) maxVelErr = ve;
  }

  return {
    exact_u32_bitwise_equal: differingFloats === 0,
    ieee754_value_equal: differingIgnoreSignedZero === 0,
    differing_floats: differingFloats,
    max_ulp_diff: maxUlp,
    max_pos_err_m: round6(maxPosErr),
    max_vel_err_mps: round6(maxVelErr),
  };
}

// Deterministic xorshift32 for reproducible paired percentile bootstrap (2,000 resamples)
function makeBootstrapRng(seed = 0x20261110) {
  let x = seed >>> 0;
  return () => {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    return (x >>> 0) / 4294967296;
  };
}

function pairedBootstrapCi95(diffs, nBoot = 2000) {
  const n = diffs.length;
  const rng = makeBootstrapRng(0x51a6e301);
  const means = new Float64Array(nBoot);
  for (let b = 0; b < nBoot; b++) {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const idx = Math.min(n - 1, Math.floor(rng() * n));
      sum += diffs[idx];
    }
    means[b] = sum / n;
  }
  means.sort();
  let obsSum = 0;
  for (const d of diffs) obsSum += d;
  const meanDiff = obsSum / n;
  const lo = means[Math.floor(0.025 * (nBoot - 1))];
  const hi = means[Math.ceil(0.975 * (nBoot - 1))];
  return {
    mean_diff: round6(meanDiff),
    ci95_low: round6(lo),
    ci95_high: round6(hi),
    max_abs_ci95: round6(Math.max(Math.abs(lo), Math.abs(hi))),
  };
}

function wasserstein1dSorted(a, b) {
  const sa = Float64Array.from(a).sort();
  const sb = Float64Array.from(b).sort();
  const m = Math.max(sa.length, sb.length);
  let sum = 0;
  for (let k = 0; k < m; k++) {
    const u = (k + 0.5) / m;
    const ia = Math.min(sa.length - 1, Math.floor(u * sa.length));
    const ib = Math.min(sb.length - 1, Math.floor(u * sb.length));
    sum += Math.abs(sa[ia] - sb[ib]);
  }
  return sum / m;
}

async function main() {
  if (!fs.existsSync(WASM_PATH) || !fs.existsSync(CLI_PATH)) {
    throw new Error(
      `Missing compiled artifacts (${WASM_PATH} or ${CLI_PATH}). Run cargo build --release first.`
    );
  }

  console.error("[1/4] Running native sim-cli stage3-native-parity...");
  const tNat0 = performance.now();
  const nativeStdout = execFileSync(CLI_PATH, ["stage3-native-parity"], {
    cwd: ROOT,
    maxBuffer: 64 * 1024 * 1024,
    encoding: "utf8",
  });
  const nativeWallS = (performance.now() - tNat0) / 1000;
  const nativeBundle = JSON.parse(nativeStdout);

  console.error(`[2/4] Loading wasm32 +simd128 module (${path.basename(WASM_PATH)})...`);
  const wasmBytes = fs.readFileSync(WASM_PATH);
  const compiled = await WebAssembly.compile(wasmBytes);
  const importsReq = WebAssembly.Module.imports(compiled);
  if (importsReq.length > 0) {
    throw new Error(
      `Expected standalone zero-import Wasm module, found imports: ${JSON.stringify(importsReq)}`
    );
  }
  const instance = await WebAssembly.instantiate(compiled, {});
  const ex = instance.exports;
  const mem = ex.memory;

  console.error("[3/4] Running 1,000-tick (5.0 s) state parity + 64-seed 30.0 s TOST suite in Wasm...");
  const parityModelsReport = [];
  let allModelsBitwise1000 = true;
  let allModels3sBoundOk = true;

  for (const m of nativeBundle.models_1000_ticks) {
    const [sLo, sHi] = splitU64(m.seed);
    const dumpPtr = ex.wasm_parity_run_1000_ticks(m.model_kind, m.n_birds, sLo, sHi);
    const fLen = ex.wasm_parity_float_len(dumpPtr) >>> 0;
    const p600 = ex.wasm_parity_tick600_ptr(dumpPtr) >>> 0;
    const p1000 = ex.wasm_parity_tick1000_ptr(dumpPtr) >>> 0;

    const wasm600 = Array.from(new Uint32Array(mem.buffer, p600, fLen));
    const wasm1000 = Array.from(new Uint32Array(mem.buffer, p1000, fLen));
    ex.wasm_parity_free(dumpPtr);

    const cmp600 = maxPosVelDiff(m.tick_600_u32_bits, wasm600);
    const cmp1000 = maxPosVelDiff(m.tick_1000_u32_bits, wasm1000);
    if (!cmp1000.exact_u32_bitwise_equal) allModelsBitwise1000 = false;
    if (!(cmp600.max_pos_err_m < 1e-3)) allModels3sBoundOk = false;

    parityModelsReport.push({
      model_name: m.model_name,
      model_kind: m.model_kind,
      n_birds: m.n_birds,
      seed: m.seed,
      tick_600_3s: cmp600,
      tick_1000_5s: cmp1000,
    });
  }

  // 64-seed TOST suite
  const dPhi = [];
  const dV = [];
  const dXiOrientRel = [];
  const dXiSpeedRel = [];
  const dCsRel = [];
  const w1PerSeed = [];
  const pooledNatSpeeds = [];
  const pooledWasmSpeeds = [];
  let tostExactBitwiseSeeds = 0;

  const tWasmTost0 = performance.now();
  for (const row of nativeBundle.tost_seeds) {
    const [sLo, sHi] = splitU64(row.seed);
    const vecHandle = ex.wasm_tost_run_seed(sLo, sHi, nativeBundle.tost_n_birds);
    const len = ex.wasm_vec_len(vecHandle) >>> 0;
    const ptr = ex.wasm_vec_ptr(vecHandle) >>> 0;
    const wasmBits = Array.from(new Uint32Array(mem.buffer, ptr, len));
    const wasmVals = Array.from(new Float32Array(mem.buffer, ptr, len));
    ex.wasm_vec_free(vecHandle);

    let seedBitwise = wasmBits.length === row.u32_bits.length;
    if (seedBitwise) {
      for (let k = 0; k < wasmBits.length; k++) {
        if ((wasmBits[k] >>> 0) !== (row.u32_bits[k] >>> 0)) {
          seedBitwise = false;
          break;
        }
      }
    }
    if (seedBitwise) tostExactBitwiseSeeds++;

    const natVals = row.f32_values;
    dPhi.push(natVals[0] - wasmVals[0]);
    dV.push(natVals[1] - wasmVals[1]);
    dXiOrientRel.push((natVals[2] - wasmVals[2]) / Math.max(Math.abs(natVals[2]), 1e-6));
    dXiSpeedRel.push((natVals[3] - wasmVals[3]) / Math.max(Math.abs(natVals[3]), 1e-6));
    dCsRel.push((natVals[4] - wasmVals[4]) / Math.max(Math.abs(natVals[4]), 1e-6));

    const natQ = natVals.slice(5);
    const wasmQ = wasmVals.slice(5);
    w1PerSeed.push(wasserstein1dSorted(natQ, wasmQ));
    for (const v of natQ) pooledNatSpeeds.push(v);
    for (const v of wasmQ) pooledWasmSpeeds.push(v);
  }
  const wasmTostWallS = (performance.now() - tWasmTost0) / 1000;

  const ciPhi = pairedBootstrapCi95(dPhi);
  const ciV = pairedBootstrapCi95(dV);
  const ciXiOrient = pairedBootstrapCi95(dXiOrientRel);
  const ciXiSpeed = pairedBootstrapCi95(dXiSpeedRel);
  const ciCs = pairedBootstrapCi95(dCsRel);
  const ciW1 = pairedBootstrapCi95(w1PerSeed);
  const pooledW1 = round6(wasserstein1dSorted(pooledNatSpeeds, pooledWasmSpeeds));

  const tostPassed =
    allModels3sBoundOk &&
    ciPhi.max_abs_ci95 < 0.005 &&
    ciV.max_abs_ci95 < 0.05 &&
    ciXiOrient.max_abs_ci95 < 0.05 &&
    ciXiSpeed.max_abs_ci95 < 0.05 &&
    ciCs.max_abs_ci95 < 0.05 &&
    ciW1.max_abs_ci95 < 0.05 &&
    pooledW1 < 0.05;

  console.error("[4/4] Running single-threaded wasm32 +simd128 stress benchmarks (N = 1000, 5000, 10000)...");
  const stressBenchmarks = [];
  for (const natRow of nativeBundle.native_single_thread_benchmarks) {
    const n = natRow.n_birds;
    const rIdx = natRow.regime_idx;
    const nTicks = n <= 1000 ? 24 : n <= 5000 ? 16 : 10;
    const [sLo, sHi] = splitU64(9000);
    const h = ex.sim_create(rIdx, n, sLo, sHi);

    // Warm up 2 ticks
    ex.sim_step_ticks(h, 2);

    const tickMs = [];
    for (let t = 0; t < nTicks; t++) {
      const t0 = performance.now();
      ex.sim_step_ticks(h, 1);
      tickMs.push(performance.now() - t0);
    }
    tickMs.sort((a, b) => a - b);
    const med = tickMs[Math.floor(tickMs.length / 2)];
    const p95 = tickMs[Math.min(tickMs.length - 1, Math.round((tickMs.length - 1) * 0.95))];

    const packIters = 10;
    const tp0 = performance.now();
    let ptrCheck = 0;
    for (let k = 0; k < packIters; k++) {
      ptrCheck ^= ex.sim_pack_instances(h, (k + 0.5) / packIters) >>> 0;
    }
    const packMs = (performance.now() - tp0) / packIters;
    const byteLen = ex.sim_instances_byte_len(h) >>> 0;
    if (byteLen !== 48 * n || ptrCheck === 0xffffffff) {
      throw new Error(`Unexpected instance buffer size ${byteLen} for N=${n}`);
    }
    ex.sim_destroy(h);

    stressBenchmarks.push({
      regime: natRow.regime,
      n_birds: n,
      instance_upload_bytes: byteLen,
      native_aarch64: {
        t_sim_tick_median_ms: natRow.t_sim_tick_median_ms,
        t_sim_tick_p95_ms: natRow.t_sim_tick_p95_ms,
        t_sim_frame_3ticks_ms: natRow.t_sim_frame_3ticks_ms,
        t_sim_frame_4ticks_ms: natRow.t_sim_frame_4ticks_ms,
        t_pack_interp_ms: natRow.t_pack_interp_ms,
      },
      wasm32_simd128: {
        t_sim_tick_median_ms: round4(med),
        t_sim_tick_p95_ms: round4(p95),
        t_sim_frame_3ticks_ms: round4(med * 3),
        t_sim_frame_4ticks_ms: round4(med * 4),
        t_pack_interp_ms: round4(packMs),
      },
      wasm_over_native_tick_ratio: round4(
        med / Math.max(natRow.t_sim_tick_median_ms, 1e-6)
      ),
    });
  }

  const gate1Passed = allModelsBitwise1000 || tostPassed;
  const report = {
    wasm_binary_bytes: wasmBytes.byteLength,
    zero_external_wasm_imports: true,
    coop_coep_required: false,
    shared_array_buffer_used: false,
    native_suite_wall_s: round4(nativeWallS),
    wasm_tost_64_seeds_wall_s: round4(wasmTostWallS),
    gate1_parity: {
      primary_1000_tick_exact_bitwise_passed: allModelsBitwise1000,
      short_horizon_3s_pos_bound_1e_3m_passed: allModels3sBoundOk,
      fallback_64_seed_tost_equivalence_passed: tostPassed,
      overall_gate1_parity_passed: gate1Passed,
      models_1000_ticks: parityModelsReport,
      tost_64_seeds: {
        n_seeds: nativeBundle.tost_seeds.length,
        n_birds: nativeBundle.tost_n_birds,
        rollout_duration_s: 30.0,
        exact_bitwise_matching_seeds: tostExactBitwiseSeeds,
        margins_and_ci95: {
          delta_phi: { margin: 0.005, ...ciPhi, passed: ciPhi.max_abs_ci95 < 0.005 },
          delta_mean_speed_mps: { margin: 0.05, ...ciV, passed: ciV.max_abs_ci95 < 0.05 },
          rel_delta_xi_orient_over_l: {
            margin: 0.05,
            ...ciXiOrient,
            passed: ciXiOrient.max_abs_ci95 < 0.05,
          },
          rel_delta_xi_speed_over_l: {
            margin: 0.05,
            ...ciXiSpeed,
            passed: ciXiSpeed.max_abs_ci95 < 0.05,
          },
          rel_delta_cs: { margin: 0.05, ...ciCs, passed: ciCs.max_abs_ci95 < 0.05 },
          wasserstein_w1_speed_mps: {
            margin: 0.05,
            pooled_w1_mps: pooledW1,
            ...ciW1,
            passed: ciW1.max_abs_ci95 < 0.05 && pooledW1 < 0.05,
          },
        },
      },
    },
    stress_benchmarks_4_regimes: stressBenchmarks,
  };

  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(OUT_JSON, JSON.stringify(report, null, 2) + "\n");
  console.log(JSON.stringify(report, null, 2));

  if (!gate1Passed) {
    process.exit(1);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
