export interface SimExports extends WebAssembly.Exports {
  memory: WebAssembly.Memory;
  sim_create(regimeIdx: number, nBirds: number, seedLo: number, seedHi: number): number;
  sim_destroy(handle: number): void;
  sim_set_params(
    handle: number,
    jSpp: number,
    tauOmega: number,
    lambda: number,
    v0: number,
    dV: number,
    kCoh: number,
    kRep: number,
    verticalCohGain: number
  ): void;
  sim_set_roost_params(handle: number, roostRadiusM: number, roostPullMps2: number): void;
  sim_trigger_turn_pulse(
    handle: number,
    initiatorCount: number,
    ax: number,
    ay: number,
    az: number,
    durationS: number
  ): void;
  sim_step_ticks(handle: number, nTicks: number): number;
  sim_pack_instances(handle: number, alpha: number): number;
  sim_instances_byte_len(handle: number): number;
  sim_bird_count(handle: number): number;
  sim_tick_count(handle: number): number;
  sim_write_telemetry(handle: number): number;
}

export interface CoupledParams {
  jSpp: number;
  tauOmega: number;
  lambda: number;
  v0: number;
  dV: number;
  kCoh: number;
  kRep: number;
  verticalCohGain: number;
  roostRadiusM: number;
  roostPullMps2: number;
}

export interface LiveTelemetry {
  meanSpeedMps: number;
  minSpeedMps: number;
  maxSpeedMps: number;
  centroid: [number, number, number];
  rmsRadiusM: number;
  coreRmsRadiusM: number;
  polarizationPhi: number;
  tickCount: number;
}

export interface StepFrameResult {
  ticksExecuted: number;
  alpha: number;
  tSimFrameMs: number;
  tSimTickAvgMs: number;
  instanceByteOffset: number;
  instanceByteLength: number;
  debtClampEvents: number;
  debtClampedSeconds: number;
}

export const DT_SIM_S = 0.005; // 200 Hz fixed simulation timestep (§4.1.1)
export const MAX_SUBSTEPS_PER_FRAME = 6; // Prevents background-tab spiral of death (§4.2.1)

export class WasmFlockSimulation {
  private readonly ex: SimExports;
  private handle = 0;
  private accumulatorS = 0;
  public debtClampEvents = 0;
  public debtClampedSeconds = 0;

  private constructor(exports: SimExports) {
    this.ex = exports;
  }

  static async load(wasmUrl = "./wasm/sim_wasm.wasm"): Promise<WasmFlockSimulation> {
    const resp = await fetch(wasmUrl);
    if (!resp.ok) {
      throw new Error(`Failed to fetch Wasm module at ${wasmUrl}: HTTP ${resp.status}`);
    }
    const bytes = await resp.arrayBuffer();
    const mod = await WebAssembly.compile(bytes);
    const imports = WebAssembly.Module.imports(mod);
    if (imports.length > 0) {
      throw new Error(`Expected standalone Wasm module with 0 imports, got ${imports.length}`);
    }
    const inst = await WebAssembly.instantiate(mod, {});
    return new WasmFlockSimulation(inst.exports as SimExports);
  }

  reset(regimeIdx: number, nBirds: number, seed = 20261110n, params?: CoupledParams): void {
    if (this.handle !== 0) {
      this.ex.sim_destroy(this.handle);
      this.handle = 0;
    }
    const seedLo = Number(seed & 0xffffffffn) >>> 0;
    const seedHi = Number((seed >> 32n) & 0xffffffffn) >>> 0;
    this.handle = this.ex.sim_create(regimeIdx >>> 0, nBirds >>> 0, seedLo, seedHi);
    this.accumulatorS = 0;
    this.debtClampEvents = 0;
    this.debtClampedSeconds = 0;
    if (params) {
      this.setParams(params);
    }
  }

  destroy(): void {
    if (this.handle !== 0) {
      this.ex.sim_destroy(this.handle);
      this.handle = 0;
    }
  }

  get wasmMemoryBuffer(): ArrayBuffer {
    return this.ex.memory.buffer;
  }

  get birdCount(): number {
    return this.handle !== 0 ? this.ex.sim_bird_count(this.handle) >>> 0 : 0;
  }

  setParams(p: CoupledParams): void {
    if (this.handle === 0) return;
    this.ex.sim_set_params(
      this.handle,
      p.jSpp,
      p.tauOmega,
      p.lambda,
      p.v0,
      p.dV,
      p.kCoh,
      p.kRep,
      p.verticalCohGain
    );
    this.ex.sim_set_roost_params(this.handle, p.roostRadiusM, p.roostPullMps2);
  }

  triggerTurnWave(initiatorCount = 16, accelMag = 24.0, durationS = 0.22): void {
    if (this.handle === 0) return;
    this.ex.sim_trigger_turn_pulse(this.handle, initiatorCount >>> 0, 0.0, accelMag, 3.5, durationS);
  }

  /**
   * Advances the 200 Hz fixed-step accumulator by `frameDtS` (3 ticks on 2/3 of 60 fps frames,
   * 4 ticks on 1/3 of frames, capped at `MAX_SUBSTEPS_PER_FRAME = 6` with explicit debt logging)
   * and packs sub-step-interpolated 48-byte GPU instances directly in Wasm linear memory.
   */
  advanceFrame(frameDtS: number, paused = false): StepFrameResult {
    if (this.handle === 0) {
      throw new Error("Simulation handle not initialized");
    }

    let ticksExecuted = 0;
    const t0 = performance.now();

    if (!paused) {
      this.accumulatorS += Math.max(0, frameDtS);
      const maxAccum = MAX_SUBSTEPS_PER_FRAME * DT_SIM_S;
      if (this.accumulatorS > maxAccum + 1e-7) {
        this.debtClampEvents += 1;
        this.debtClampedSeconds += this.accumulatorS - maxAccum;
        this.accumulatorS = maxAccum;
      }
      ticksExecuted = Math.min(
        MAX_SUBSTEPS_PER_FRAME,
        Math.floor(this.accumulatorS / DT_SIM_S + 1e-7)
      );
      if (ticksExecuted > 0) {
        this.ex.sim_step_ticks(this.handle, ticksExecuted);
        this.accumulatorS = Math.max(0, this.accumulatorS - ticksExecuted * DT_SIM_S);
      }
    }

    const alpha = paused ? 1.0 : Math.min(1.0, Math.max(0.0, this.accumulatorS / DT_SIM_S));
    const ptr = this.ex.sim_pack_instances(this.handle, alpha) >>> 0;
    const byteLen = this.ex.sim_instances_byte_len(this.handle) >>> 0;
    const tSimFrameMs = performance.now() - t0;

    return {
      ticksExecuted,
      alpha,
      tSimFrameMs,
      tSimTickAvgMs: ticksExecuted > 0 ? tSimFrameMs / ticksExecuted : 0,
      instanceByteOffset: ptr,
      instanceByteLength: byteLen,
      debtClampEvents: this.debtClampEvents,
      debtClampedSeconds: this.debtClampedSeconds,
    };
  }

  readTelemetry(): LiveTelemetry {
    if (this.handle === 0) {
      return {
        meanSpeedMps: 0,
        minSpeedMps: 0,
        maxSpeedMps: 0,
        centroid: [0, 0, 0],
        rmsRadiusM: 10,
        coreRmsRadiusM: 10,
        polarizationPhi: 0,
        tickCount: 0,
      };
    }
    const ptr = this.ex.sim_write_telemetry(this.handle) >>> 0;
    const view = new Float32Array(this.ex.memory.buffer, ptr, 9);
    return {
      meanSpeedMps: view[0],
      minSpeedMps: view[1],
      maxSpeedMps: view[2],
      centroid: [view[3], view[4], view[5]],
      rmsRadiusM: view[6],
      polarizationPhi: view[7],
      coreRmsRadiusM: view[8],
      tickCount: this.ex.sim_tick_count(this.handle) >>> 0,
    };
  }
}
