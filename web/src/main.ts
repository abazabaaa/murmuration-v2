import { CoupledParams, WasmFlockSimulation } from "./wasmSim";
import { CameraState, LodPolicy, StarlingWebGpuRenderer } from "./webgpuRenderer";

export interface Gate2CellResult {
  regime: string;
  regimeIdx: number;
  nBirds: number;
  framesMeasured: number;
  nearTrianglesPerBird: number;
  farTrianglesPerBird: number;
  avgNearBirdsRendered: number;
  avgFarBirdsRendered: number;
  t_sim_tick_median_ms: number;
  t_sim_frame_median_ms: number;
  t_sim_frame_4ticks_p95_ms: number;
  t_upload_mean_ms: number;
  t_render_gpu_median_ms: number;
  t_frame_total_median_ms: number;
  t_frame_total_p95_ms: number;
  debt_clamp_events: number;
}

export interface Gate2SuiteReport {
  adapter: string;
  viewportWidthPx: number;
  viewportHeightPx: number;
  shutterMotionBlurEnabled: boolean;
  lodPolicy: LodPolicy;
  cells: Gate2CellResult[];
}

declare global {
  interface Window {
    __STAGE3_BENCH_RESULT__?: Gate2SuiteReport;
    __STAGE3_BENCH_ERROR__?: string;
  }
}

const REGIME_NAMES = [
  "calm_ribbon",
  "compacted_core_5x",
  "diluted_cloud_3x",
  "split_subflocks_stragglers",
] as const;

function round3(x: number): number {
  return Math.round(x * 1000) / 1000;
}

function percentile(arr: number[], q: number): number {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.round((sorted.length - 1) * q)));
  return sorted[idx];
}

function meanOf(arr: number[]): number {
  if (arr.length === 0) return 0;
  let s = 0;
  for (const v of arr) s += v;
  return s / arr.length;
}

function defaultDistanceForRegime(regimeIdx: number, nBirds: number): number {
  const cbrt = Math.cbrt(nBirds);
  switch (regimeIdx) {
    case 1:
      return Math.max(22, cbrt * 2.4);
    case 2:
      return Math.max(55, cbrt * 9.5);
    case 3:
      return 260;
    default:
      return Math.max(38, cbrt * 5.2);
  }
}

async function main(): Promise<void> {
  const canvas = document.getElementById("gpu-canvas") as HTMLCanvasElement;
  const errBanner = document.getElementById("error-banner") as HTMLDivElement;

  const showError = (msg: string) => {
    errBanner.style.display = "block";
    errBanner.textContent = msg;
    window.__STAGE3_BENCH_ERROR__ = msg;
  };

  let sim: WasmFlockSimulation;
  let renderer: StarlingWebGpuRenderer;
  try {
    [sim, renderer] = await Promise.all([
      WasmFlockSimulation.load("./wasm/sim_wasm.wasm"),
      StarlingWebGpuRenderer.create(canvas, "./assets/starling-mesh.json"),
    ]);
  } catch (err) {
    showError(err instanceof Error ? err.message : String(err));
    return;
  }

  const selBirds = document.getElementById("sel-birds") as HTMLSelectElement;
  const selRegime = document.getElementById("sel-regime") as HTMLSelectElement;
  const selLod = document.getElementById("sel-lod") as HTMLSelectElement;
  const chkFollow = document.getElementById("chk-follow") as HTMLInputElement;
  const chkAutoZoom = document.getElementById("chk-auto-zoom") as HTMLInputElement;
  const chkBlur = document.getElementById("chk-blur") as HTMLInputElement;
  const btnFrame = document.getElementById("btn-frame") as HTMLButtonElement;
  const btnWave = document.getElementById("btn-wave") as HTMLButtonElement;
  const btnPause = document.getElementById("btn-pause") as HTMLButtonElement;
  const btnReset = document.getElementById("btn-reset") as HTMLButtonElement;
  const btnBench = document.getElementById("btn-bench") as HTMLButtonElement;
  const benchStatus = document.getElementById("bench-status") as HTMLDivElement;

  const readParamsFromUi = (): CoupledParams => ({
    jSpp: parseFloat((document.getElementById("rng-jspp") as HTMLInputElement).value),
    tauOmega: parseFloat((document.getElementById("rng-tau") as HTMLInputElement).value),
    lambda: parseFloat((document.getElementById("rng-lambda") as HTMLInputElement).value),
    v0: parseFloat((document.getElementById("rng-v0") as HTMLInputElement).value),
    dV: parseFloat((document.getElementById("rng-dv") as HTMLInputElement).value),
    kCoh: 0.55,
    kRep: 3.40,
    verticalCohGain: parseFloat((document.getElementById("rng-gz") as HTMLInputElement).value),
    roostRadiusM: parseFloat((document.getElementById("rng-roost") as HTMLInputElement).value),
    roostPullMps2: 7.5,
  });

  const sliderBindings: Array<[string, string, number]> = [
    ["rng-jspp", "val-jspp", 1],
    ["rng-tau", "val-tau", 2],
    ["rng-lambda", "val-lambda", 2],
    ["rng-v0", "val-v0", 1],
    ["rng-dv", "val-dv", 2],
    ["rng-gz", "val-gz", 2],
    ["rng-roost", "val-roost", 0],
  ];
  for (const [rngId, valId, digits] of sliderBindings) {
    const inp = document.getElementById(rngId) as HTMLInputElement;
    const lbl = document.getElementById(valId) as HTMLSpanElement;
    inp.addEventListener("input", () => {
      lbl.textContent = parseFloat(inp.value).toFixed(digits);
      sim.setParams(readParamsFromUi());
    });
  }

  const cam: CameraState = {
    target: [0, 0, 0],
    distance: 52,
    azimuthRad: -1.18,
    elevationRad: 0.22,
    fovYRad: (48 * Math.PI) / 180,
  };

  let paused = false;
  let benchmarkRunning = false;
  let zoomScale = 1.0;

  const computeFramedDistance = (coreRmsM: number): number => {
    const base = Math.max(24, Math.min(260, coreRmsM * 2.35));
    return Math.max(8, Math.min(1800, base * zoomScale));
  };

  const snapFrameFlock = () => {
    const tel = sim.readTelemetry();
    zoomScale = 1.0;
    chkFollow.checked = true;
    cam.target = [...tel.centroid];
    cam.distance = computeFramedDistance(tel.coreRmsRadiusM);
  };

  const initFlockFromUi = () => {
    const n = parseInt(selBirds.value, 10);
    const r = parseInt(selRegime.value, 10);
    zoomScale = 1.0;
    sim.reset(r, n, 20261110n, readParamsFromUi());
    renderer.resetHistory();
    const tel = sim.readTelemetry();
    cam.target = [...tel.centroid];
    cam.distance = r === 3 ? 240 : computeFramedDistance(tel.coreRmsRadiusM);
  };

  selBirds.addEventListener("change", initFlockFromUi);
  selRegime.addEventListener("change", initFlockFromUi);
  btnReset.addEventListener("click", initFlockFromUi);
  btnFrame.addEventListener("click", snapFrameFlock);
  window.addEventListener("keydown", (e) => {
    if (e.key === "f" || e.key === "F") {
      snapFrameFlock();
    }
  });
  btnWave.addEventListener("click", () => {
    const initCount = Math.max(8, Math.round(sim.birdCount * 0.02));
    sim.triggerTurnWave(initCount, 24.0, 0.24);
  });
  btnPause.addEventListener("click", () => {
    paused = !paused;
    btnPause.textContent = paused ? "Resume" : "Pause";
  });

  // Orbit & zoom pointer interaction
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  canvas.addEventListener("pointerdown", (e) => {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    cam.azimuthRad -= dx * 0.006;
    cam.elevationRad = Math.max(-1.35, Math.min(1.35, cam.elevationRad + dy * 0.006));
  });
  canvas.addEventListener("pointerup", () => {
    dragging = false;
  });
  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      const factor = Math.exp(e.deltaY * 0.0012);
      if (chkAutoZoom.checked) {
        zoomScale = Math.max(0.25, Math.min(6.0, zoomScale * factor));
      } else {
        cam.distance = Math.max(6, Math.min(2400, cam.distance * factor));
      }
    },
    { passive: false }
  );

  const runStressBenchmark12Cells = async (): Promise<Gate2SuiteReport> => {
    benchmarkRunning = true;
    btnBench.disabled = true;
    const cells: Gate2CellResult[] = [];
    const lodPolicy = (selLod.value as LodPolicy) || "auto";
    const blurOn = chkBlur.checked;
    const birdCounts = [1000, 5000, 10000];

    for (const n of birdCounts) {
      const measuredFrames = n <= 1000 ? 36 : n <= 5000 ? 24 : 15;
      for (let rIdx = 0; rIdx < 4; rIdx++) {
        const rName = REGIME_NAMES[rIdx];
        benchStatus.textContent = `Benchmarking ${rName} @ N=${n}...`;
        await new Promise((r) => setTimeout(r, 0));

        sim.reset(rIdx, n, 9000n, readParamsFromUi());
        renderer.resetHistory();
        const tel0 = sim.readTelemetry();
        const benchCam: CameraState = {
          target: [...tel0.centroid],
          distance: defaultDistanceForRegime(rIdx, n),
          azimuthRad: -1.18,
          elevationRad: 0.20,
          fovYRad: (48 * Math.PI) / 180,
        };

        // Warm up 3 frames
        for (let w = 0; w < 3; w++) {
          const st = sim.advanceFrame(1 / 60, false);
          renderer.renderFrame(
            sim.wasmMemoryBuffer,
            st.instanceByteOffset,
            st.instanceByteLength,
            n,
            benchCam,
            lodPolicy,
            blurOn
          );
        }
        await renderer.device.queue.onSubmittedWorkDone();

        const simTickArr: number[] = [];
        const simFrameArr: number[] = [];
        const simFrame4Arr: number[] = [];
        const upArr: number[] = [];
        const renGpuArr: number[] = [];
        const totArr: number[] = [];
        let sumNear = 0;
        let sumFar = 0;

        for (let f = 0; f < measuredFrames; f++) {
          const tFrame0 = performance.now();
          // Exact 60 Hz display cadence (16.6667 ms -> 3 ticks on 2/3 of frames, 4 ticks on 1/3)
          const st = sim.advanceFrame(1 / 60, false);
          const tRenStart = performance.now();
          const rt = renderer.renderFrame(
            sim.wasmMemoryBuffer,
            st.instanceByteOffset,
            st.instanceByteLength,
            n,
            benchCam,
            lodPolicy,
            blurOn
          );
          await renderer.device.queue.onSubmittedWorkDone();
          const tAfterGpu = performance.now();

          const gpuRenderMs = Math.max(0, tAfterGpu - tRenStart - rt.tUploadMs);
          const totalMs = tAfterGpu - tFrame0;

          if (st.tSimTickAvgMs > 0) simTickArr.push(st.tSimTickAvgMs);
          simFrameArr.push(st.tSimFrameMs);
          if (st.ticksExecuted >= 4) simFrame4Arr.push(st.tSimFrameMs);
          upArr.push(rt.tUploadMs);
          renGpuArr.push(gpuRenderMs);
          totArr.push(totalMs);
          sumNear += rt.nearBirdCount;
          sumFar += rt.farBirdCount;
        }

        cells.push({
          regime: rName,
          regimeIdx: rIdx,
          nBirds: n,
          framesMeasured: measuredFrames,
          nearTrianglesPerBird: 440,
          farTrianglesPerBird: 48,
          avgNearBirdsRendered: Math.round(sumNear / measuredFrames),
          avgFarBirdsRendered: Math.round(sumFar / measuredFrames),
          t_sim_tick_median_ms: round3(percentile(simTickArr, 0.5)),
          t_sim_frame_median_ms: round3(percentile(simFrameArr, 0.5)),
          t_sim_frame_4ticks_p95_ms: round3(
            percentile(simFrame4Arr.length > 0 ? simFrame4Arr : simFrameArr, 0.95)
          ),
          t_upload_mean_ms: round3(meanOf(upArr)),
          t_render_gpu_median_ms: round3(percentile(renGpuArr, 0.5)),
          t_frame_total_median_ms: round3(percentile(totArr, 0.5)),
          t_frame_total_p95_ms: round3(percentile(totArr, 0.95)),
          debt_clamp_events: sim.debtClampEvents,
        });
      }
    }

    // Leave a visually rich 2,500-bird calm ribbon frame with an active turn wave on screen
    selBirds.value = "2500";
    selRegime.value = "0";
    initFlockFromUi();
    sim.triggerTurnWave(40, 26.0, 0.30);
    for (let k = 0; k < 18; k++) {
      const st = sim.advanceFrame(1 / 60, false);
      const tel = sim.readTelemetry();
      cam.target = [...tel.centroid];
      renderer.renderFrame(
        sim.wasmMemoryBuffer,
        st.instanceByteOffset,
        st.instanceByteLength,
        sim.birdCount,
        cam,
        lodPolicy,
        blurOn
      );
    }
    await renderer.device.queue.onSubmittedWorkDone();

    const report: Gate2SuiteReport = {
      adapter: renderer.adapterInfoSummary,
      viewportWidthPx: canvas.width,
      viewportHeightPx: canvas.height,
      shutterMotionBlurEnabled: blurOn,
      lodPolicy,
      cells,
    };
    window.__STAGE3_BENCH_RESULT__ = report;
    benchStatus.textContent = `Benchmark complete (${cells.length} cells).`;
    btnBench.disabled = false;
    benchmarkRunning = false;
    return report;
  };

  btnBench.addEventListener("click", () => {
    void runStressBenchmark12Cells();
  });

  initFlockFromUi();

  const urlParams = new URLSearchParams(window.location.search);
  if (urlParams.get("bench") === "1") {
    await runStressBenchmark12Cells();
  }

  // Rolling HUD metrics window
  const elSimTick = document.getElementById("m-sim-tick") as HTMLSpanElement;
  const elTicks = document.getElementById("m-ticks") as HTMLSpanElement;
  const elSimFrame = document.getElementById("m-sim-frame") as HTMLSpanElement;
  const elUpload = document.getElementById("m-upload") as HTMLSpanElement;
  const elRender = document.getElementById("m-render") as HTMLSpanElement;
  const elTotal = document.getElementById("m-total") as HTMLSpanElement;
  const elPhi = document.getElementById("m-phi") as HTMLSpanElement;
  const elSpeed = document.getElementById("m-speed") as HTMLSpanElement;
  const elRadius = document.getElementById("m-radius") as HTMLSpanElement;
  const elTickIdx = document.getElementById("m-tick-idx") as HTMLSpanElement;

  const recentTotals: number[] = [];
  const recentUploads: number[] = [];
  const recentRenders: number[] = [];
  let frameCounter = 0;
  let lastNow = performance.now();

  const frameLoop = (now: number) => {
    const dtS = Math.min(0.1, Math.max(0.0, (now - lastNow) / 1000));
    lastNow = now;

    if (!benchmarkRunning) {
      const t0 = performance.now();
      const stepRes = sim.advanceFrame(dtS, paused);
      const tel = sim.readTelemetry();
      if (chkFollow.checked) {
        const followRate = 1.0 - Math.exp(-dtS * 6.0);
        cam.target[0] += (tel.centroid[0] - cam.target[0]) * followRate;
        cam.target[1] += (tel.centroid[1] - cam.target[1]) * followRate;
        cam.target[2] += (tel.centroid[2] - cam.target[2]) * followRate;
      }
      if (chkAutoZoom.checked) {
        const zoomRate = 1.0 - Math.exp(-dtS * 3.0);
        const desiredDist = computeFramedDistance(tel.coreRmsRadiusM);
        cam.distance += (desiredDist - cam.distance) * zoomRate;
      }
      const rt = renderer.renderFrame(
        sim.wasmMemoryBuffer,
        stepRes.instanceByteOffset,
        stepRes.instanceByteLength,
        sim.birdCount,
        cam,
        (selLod.value as LodPolicy) || "auto",
        chkBlur.checked
      );
      const totalMs = performance.now() - t0;
      recentTotals.push(totalMs);
      recentUploads.push(rt.tUploadMs);
      recentRenders.push(rt.tRenderEncodeMs);
      if (recentTotals.length > 90) {
        recentTotals.shift();
        recentUploads.shift();
        recentRenders.shift();
      }

      frameCounter++;
      if (frameCounter % 8 === 0) {
        elSimTick.textContent = `${stepRes.tSimTickAvgMs.toFixed(2)} ms`;
        elTicks.textContent = `${stepRes.ticksExecuted} (${stepRes.debtClampEvents})`;
        elSimFrame.textContent = `${stepRes.tSimFrameMs.toFixed(2)} ms`;
        elUpload.textContent = `${meanOf(recentUploads).toFixed(3)} ms`;
        elRender.textContent = `${meanOf(recentRenders).toFixed(3)} ms`;
        elTotal.textContent = `${percentile(recentTotals, 0.5).toFixed(2)} / ${percentile(recentTotals, 0.95).toFixed(2)} ms`;
        elPhi.textContent = tel.polarizationPhi.toFixed(3);
        elSpeed.textContent = `${tel.meanSpeedMps.toFixed(1)} [${tel.minSpeedMps.toFixed(1)}..${tel.maxSpeedMps.toFixed(1)}] m/s`;
        elRadius.textContent = tel.rmsRadiusM.toFixed(1);
        elTickIdx.textContent = String(tel.tickCount);
      }
    }

    requestAnimationFrame(frameLoop);
  };

  requestAnimationFrame(frameLoop);
}

void main();
