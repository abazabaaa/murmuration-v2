import shaderSource from "./shaders/starling.wgsl?raw";

export interface StarlingMeshAsset {
  format: string;
  halfSpanM: number;
  lengthM: number;
  vertexCount: number;
  indicesNear: number[];
  indicesFar: number[];
  flapTimes: number[];
  flapFrames: Array<{ positions: number[]; normals: number[] }>;
  glideFrame: { positions: number[]; normals: number[] };
  boundFrame: { positions: number[]; normals: number[] };
}

export type LodPolicy = "auto" | "near" | "far";

export interface CameraState {
  target: [number, number, number];
  distance: number;
  azimuthRad: number;
  elevationRad: number;
  fovYRad: number;
}

export interface RenderTimings {
  tUploadMs: number;
  tRenderEncodeMs: number;
  nearBirdCount: number;
  farBirdCount: number;
}

function mat4Perspective(fovY: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1.0 / Math.tan(0.5 * fovY);
  const nf = 1.0 / (near - far);
  // Column-major WebGPU [0, 1] depth clip matrix
  const out = new Float32Array(16);
  out[0] = f / Math.max(aspect, 1e-4);
  out[5] = f;
  out[10] = far * nf;
  out[11] = -1.0;
  out[14] = near * far * nf;
  return out;
}

function mat4LookAtZUp(
  eye: [number, number, number],
  target: [number, number, number]
): Float32Array {
  let fx = target[0] - eye[0];
  let fy = target[1] - eye[1];
  let fz = target[2] - eye[2];
  const fl = Math.hypot(fx, fy, fz) || 1;
  fx /= fl;
  fy /= fl;
  fz /= fl;

  // World up = +Z (0, 0, 1); right = f x up = (fy, -fx, 0)
  let rx = fy;
  let ry = -fx;
  let rz = 0;
  let rl = Math.hypot(rx, ry, rz);
  if (rl < 1e-5) {
    rx = 1;
    ry = 0;
    rz = 0;
    rl = 1;
  }
  rx /= rl;
  ry /= rl;

  // True camera up u = r x f
  const ux = ry * fz - rz * fy;
  const uy = rz * fx - rx * fz;
  const uz = rx * fy - ry * fx;

  const out = new Float32Array(16);
  out[0] = rx;
  out[1] = ux;
  out[2] = -fx;
  out[3] = 0;

  out[4] = ry;
  out[5] = uy;
  out[6] = -fy;
  out[7] = 0;

  out[8] = rz;
  out[9] = uz;
  out[10] = -fz;
  out[11] = 0;

  out[12] = -(rx * eye[0] + ry * eye[1] + rz * eye[2]);
  out[13] = -(ux * eye[0] + uy * eye[1] + uz * eye[2]);
  out[14] = fx * eye[0] + fy * eye[1] + fz * eye[2];
  out[15] = 1;
  return out;
}

function mat4Multiply(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    const b0 = b[col * 4 + 0];
    const b1 = b[col * 4 + 1];
    const b2 = b[col * 4 + 2];
    const b3 = b[col * 4 + 3];
    out[col * 4 + 0] = a[0] * b0 + a[4] * b1 + a[8] * b2 + a[12] * b3;
    out[col * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9] * b2 + a[13] * b3;
    out[col * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
    out[col * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
  }
  return out;
}

export function computeCameraEye(cam: CameraState): [number, number, number] {
  const ce = Math.cos(cam.elevationRad);
  const se = Math.sin(cam.elevationRad);
  const ca = Math.cos(cam.azimuthRad);
  const sa = Math.sin(cam.azimuthRad);
  return [
    cam.target[0] + cam.distance * ce * ca,
    cam.target[1] + cam.distance * ce * sa,
    cam.target[2] + cam.distance * se,
  ];
}

export class StarlingWebGpuRenderer {
  public readonly device: GPUDevice;
  public readonly adapterInfoSummary: string;
  private readonly context: GPUCanvasContext;
  private readonly canvas: HTMLCanvasElement;
  private readonly presentationFormat: GPUTextureFormat;
  private readonly mesh: StarlingMeshAsset;

  private readonly uniformBuf: GPUBuffer;
  private readonly uniformData = new ArrayBuffer(224); // 14 * 16 bytes (SceneUniforms = 208 bytes rounded to 224)
  private readonly poseBuf: GPUBuffer;
  private readonly vertIdBuf: GPUBuffer;
  private readonly indexNearBuf: GPUBuffer;
  private readonly indexFarBuf: GPUBuffer;
  private readonly indexNearCount: number;
  private readonly indexFarCount: number;

  private readonly sceneBgl: GPUBindGroupLayout;
  private readonly compBgl: GPUBindGroupLayout;
  private readonly scenePipeline: GPURenderPipeline;
  private readonly compPipeline: GPURenderPipeline;
  private readonly linearSampler: GPUSampler;

  private instanceCapacity = 0;
  private instBufA!: GPUBuffer;
  private instBufB!: GPUBuffer;
  private drawBirdIdsBuf!: GPUBuffer;
  private drawIdsScratch = new Uint32Array(0);
  private nearIdsScratch = new Uint32Array(0);
  private farIdsScratch = new Uint32Array(0);
  private sceneBindGroupAB!: GPUBindGroup; // curr = A, prev = B
  private sceneBindGroupBA!: GPUBindGroup; // curr = B, prev = A
  private pingPong = 0;
  private hasPrevInstances = false;

  private texWidth = 0;
  private texHeight = 0;
  private hdrTex!: GPUTexture;
  private velTex!: GPUTexture;
  private depthTex!: GPUTexture;
  private hdrView!: GPUTextureView;
  private velView!: GPUTextureView;
  private depthView!: GPUTextureView;
  private compBindGroup!: GPUBindGroup;

  private prevViewProj: Float32Array | null = null;

  private constructor(
    canvas: HTMLCanvasElement,
    device: GPUDevice,
    context: GPUCanvasContext,
    format: GPUTextureFormat,
    mesh: StarlingMeshAsset,
    adapterSummary: string
  ) {
    this.canvas = canvas;
    this.device = device;
    this.context = context;
    this.presentationFormat = format;
    this.mesh = mesh;
    this.adapterInfoSummary = adapterSummary;

    this.uniformBuf = device.createBuffer({
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Pack 37 flap frames + 1 glide frame + 1 bound frame = 39 frames * vertexCount * 8 floats (pos_pad + norm_pad)
    const vCount = mesh.vertexCount;
    const totalFrames = mesh.flapFrames.length + 2;
    const poseFloats = new Float32Array(totalFrames * vCount * 8);
    const writeFrame = (frameIdx: number, pos: number[], norm: number[]) => {
      const base = frameIdx * vCount * 8;
      for (let v = 0; v < vCount; v++) {
        const dst = base + v * 8;
        poseFloats[dst + 0] = pos[3 * v + 0];
        poseFloats[dst + 1] = pos[3 * v + 1];
        poseFloats[dst + 2] = pos[3 * v + 2];
        poseFloats[dst + 3] = 0;
        poseFloats[dst + 4] = norm[3 * v + 0];
        poseFloats[dst + 5] = norm[3 * v + 1];
        poseFloats[dst + 6] = norm[3 * v + 2];
        poseFloats[dst + 7] = 0;
      }
    };
    mesh.flapFrames.forEach((f, idx) => writeFrame(idx, f.positions, f.normals));
    writeFrame(mesh.flapFrames.length, mesh.glideFrame.positions, mesh.glideFrame.normals);
    writeFrame(mesh.flapFrames.length + 1, mesh.boundFrame.positions, mesh.boundFrame.normals);

    this.poseBuf = device.createBuffer({
      size: poseFloats.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.poseBuf, 0, poseFloats);

    const vertIds = new Uint32Array(vCount);
    for (let i = 0; i < vCount; i++) vertIds[i] = i;
    this.vertIdBuf = device.createBuffer({
      size: vertIds.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.vertIdBuf, 0, vertIds);

    const makeIndexU16 = (arr: number[]) => {
      const paddedLen = arr.length + (arr.length % 2);
      const u16 = new Uint16Array(paddedLen);
      u16.set(arr);
      const buf = device.createBuffer({
        size: u16.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buf, 0, u16);
      return buf;
    };
    this.indexNearBuf = makeIndexU16(mesh.indicesNear);
    this.indexFarBuf = makeIndexU16(mesh.indicesFar);
    this.indexNearCount = mesh.indicesNear.length;
    this.indexFarCount = mesh.indicesFar.length;

    const shaderMod = device.createShaderModule({ code: shaderSource });

    this.sceneBgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      ],
    });

    this.compBgl = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      ],
    });

    this.scenePipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.sceneBgl] }),
      vertex: {
        module: shaderMod,
        entryPoint: "vs_starling",
        buffers: [
          {
            arrayStride: 4,
            stepMode: "vertex",
            attributes: [{ shaderLocation: 0, offset: 0, format: "uint32" }],
          },
        ],
      },
      fragment: {
        module: shaderMod,
        entryPoint: "fs_starling",
        targets: [{ format: "rgba16float" }, { format: "rg16float" }],
      },
      primitive: {
        topology: "triangle-list",
        cullMode: "none", // Thin two-sided avian wing geometry
      },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    });

    this.compPipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.compBgl] }),
      vertex: {
        module: shaderMod,
        entryPoint: "vs_fullscreen",
      },
      fragment: {
        module: shaderMod,
        entryPoint: "fs_composite",
        targets: [{ format: this.presentationFormat }],
      },
      primitive: { topology: "triangle-list" },
    });

    this.linearSampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });

    this.ensureInstanceCapacity(1024);
    this.ensureAttachments();
  }

  static async create(
    canvas: HTMLCanvasElement,
    meshUrl = "./assets/starling-mesh.json"
  ): Promise<StarlingWebGpuRenderer> {
    if (!navigator.gpu) {
      throw new Error("WebGPU (`navigator.gpu`) is not available in this browser.");
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) {
      throw new Error("WebGPU adapter request returned null.");
    }
    const device = await adapter.requestDevice();
    const info = adapter.info;
    const summary = [info?.vendor, info?.architecture, info?.description]
      .filter(Boolean)
      .join(" / ") || "WebGPU Adapter";

    const context = canvas.getContext("webgpu");
    if (!context) {
      throw new Error("Failed to acquire `webgpu` context on canvas.");
    }
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
      device,
      format,
      alphaMode: "opaque",
    });

    const resp = await fetch(meshUrl);
    if (!resp.ok) {
      throw new Error(`Failed to fetch starling mesh asset at ${meshUrl}: HTTP ${resp.status}`);
    }
    const mesh = (await resp.json()) as StarlingMeshAsset;
    return new StarlingWebGpuRenderer(canvas, device, context, format, mesh, summary);
  }

  resetHistory(): void {
    this.hasPrevInstances = false;
    this.prevViewProj = null;
  }

  private ensureInstanceCapacity(nBirds: number): void {
    if (nBirds <= this.instanceCapacity) return;
    if (this.instBufA) this.instBufA.destroy();
    if (this.instBufB) this.instBufB.destroy();
    if (this.drawBirdIdsBuf) this.drawBirdIdsBuf.destroy();

    const cap = Math.max(nBirds, 1024);
    const byteSize = cap * 48;
    this.instBufA = this.device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.instBufB = this.device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.drawBirdIdsBuf = this.device.createBuffer({
      size: cap * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.drawIdsScratch = new Uint32Array(cap);
    this.nearIdsScratch = new Uint32Array(cap);
    this.farIdsScratch = new Uint32Array(cap);

    this.sceneBindGroupAB = this.device.createBindGroup({
      layout: this.sceneBgl,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: { buffer: this.poseBuf } },
        { binding: 2, resource: { buffer: this.instBufA } },
        { binding: 3, resource: { buffer: this.instBufB } },
        { binding: 4, resource: { buffer: this.drawBirdIdsBuf } },
      ],
    });

    this.sceneBindGroupBA = this.device.createBindGroup({
      layout: this.sceneBgl,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: { buffer: this.poseBuf } },
        { binding: 2, resource: { buffer: this.instBufB } },
        { binding: 3, resource: { buffer: this.instBufA } },
        { binding: 4, resource: { buffer: this.drawBirdIdsBuf } },
      ],
    });

    this.instanceCapacity = cap;
    this.hasPrevInstances = false;
  }

  private ensureAttachments(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (w === this.texWidth && h === this.texHeight) return;

    this.canvas.width = w;
    this.canvas.height = h;
    if (this.hdrTex) this.hdrTex.destroy();
    if (this.velTex) this.velTex.destroy();
    if (this.depthTex) this.depthTex.destroy();

    this.hdrTex = this.device.createTexture({
      size: { width: w, height: h },
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.velTex = this.device.createTexture({
      size: { width: w, height: h },
      format: "rg16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.depthTex = this.device.createTexture({
      size: { width: w, height: h },
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.hdrView = this.hdrTex.createView();
    this.velView = this.velTex.createView();
    this.depthView = this.depthTex.createView();

    this.compBindGroup = this.device.createBindGroup({
      layout: this.compBgl,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: this.hdrView },
        { binding: 2, resource: this.velView },
        { binding: 3, resource: this.linearSampler },
      ],
    });

    this.texWidth = w;
    this.texHeight = h;
  }

  /**
   * Uploads `48 * N` bytes directly from `wasmMemoryBuffer` via `device.queue.writeBuffer`
   * (zero intermediate JS copy of instance structs) and encodes + submits both render passes.
   */
  renderFrame(
    wasmMemoryBuffer: ArrayBuffer,
    instanceByteOffset: number,
    instanceByteLength: number,
    nBirds: number,
    cam: CameraState,
    lodPolicy: LodPolicy,
    shutterBlurEnabled: boolean
  ): RenderTimings {
    this.ensureAttachments();
    this.ensureInstanceCapacity(nBirds);

    const eye = computeCameraEye(cam);
    const aspect = this.texWidth / Math.max(1, this.texHeight);
    const proj = mat4Perspective(cam.fovYRad, aspect, 0.4, 2400.0);
    const view = mat4LookAtZUp(eye, cam.target);
    const viewProj = mat4Multiply(proj, view);
    const prevVp = this.prevViewProj ?? viewProj;
    this.prevViewProj = viewProj;

    // Partition birds into Near LOD (< 75 m) and Far LOD (>= 75 m) directly from Wasm Float32Array view
    let nearCount = 0;
    let farCount = 0;
    if (lodPolicy === "near") {
      for (let i = 0; i < nBirds; i++) this.drawIdsScratch[i] = i;
      nearCount = nBirds;
    } else if (lodPolicy === "far") {
      for (let i = 0; i < nBirds; i++) this.drawIdsScratch[i] = i;
      farCount = nBirds;
    } else {
      const fView = new Float32Array(wasmMemoryBuffer, instanceByteOffset, nBirds * 12);
      const nearThreshSq = 75.0 * 75.0;
      for (let i = 0; i < nBirds; i++) {
        const base = i * 12;
        const dx = fView[base] - eye[0];
        const dy = fView[base + 1] - eye[1];
        const dz = fView[base + 2] - eye[2];
        if (dx * dx + dy * dy + dz * dz < nearThreshSq) {
          this.nearIdsScratch[nearCount++] = i;
        } else {
          this.farIdsScratch[farCount++] = i;
        }
      }
      this.drawIdsScratch.set(this.nearIdsScratch.subarray(0, nearCount), 0);
      this.drawIdsScratch.set(this.farIdsScratch.subarray(0, farCount), nearCount);
    }

    // Pack SceneUniforms (208 bytes)
    const fUni = new Float32Array(this.uniformData);
    const uUni = new Uint32Array(this.uniformData);
    fUni.set(viewProj, 0);
    fUni.set(prevVp, 16);
    fUni[32] = eye[0];
    fUni[33] = eye[1];
    fUni[34] = eye[2];
    fUni[35] = this.mesh.halfSpanM;

    // Low golden-hour twilight sun vector
    fUni[36] = -0.62;
    fUni[37] = -0.58;
    fUni[38] = 0.528;
    fUni[39] = 1.15; // exposure

    // Linear-space dusk sky zenith & horizon colors
    const needsManualSrgb = this.presentationFormat.endsWith("-srgb") ? 0.0 : 1.0;
    fUni[40] = 0.075;
    fUni[41] = 0.135;
    fUni[42] = 0.285;
    fUni[43] = shutterBlurEnabled && this.hasPrevInstances ? 1.0 : 0.0;

    fUni[44] = 0.56;
    fUni[45] = 0.31;
    fUni[46] = 0.20;
    fUni[47] = needsManualSrgb;

    const flapCount = this.mesh.flapFrames.length;
    uUni[48] = this.mesh.vertexCount;
    uUni[49] = flapCount;
    uUni[50] = flapCount; // glide frame index
    uUni[51] = flapCount + 1; // bound frame index

    // Direct CPU -> GPU staging transfer (`t_upload`) from Wasm linear memory
    const tUp0 = performance.now();
    const currBuf = this.pingPong === 0 ? this.instBufA : this.instBufB;
    const prevBuf = this.pingPong === 0 ? this.instBufB : this.instBufA;
    this.device.queue.writeBuffer(
      currBuf,
      0,
      wasmMemoryBuffer,
      instanceByteOffset,
      instanceByteLength
    );
    if (!this.hasPrevInstances) {
      this.device.queue.writeBuffer(
        prevBuf,
        0,
        wasmMemoryBuffer,
        instanceByteOffset,
        instanceByteLength
      );
      this.hasPrevInstances = true;
    }
    this.device.queue.writeBuffer(this.drawBirdIdsBuf, 0, this.drawIdsScratch, 0, nBirds);
    this.device.queue.writeBuffer(this.uniformBuf, 0, this.uniformData, 0, 208);
    const tUploadMs = performance.now() - tUp0;

    // Encode & submit Pass 1 (instanced starling geometry + motion vectors) + Pass 2 (composite + shutter blur)
    const tRen0 = performance.now();
    const sceneBg = this.pingPong === 0 ? this.sceneBindGroupAB : this.sceneBindGroupBA;
    this.pingPong ^= 1;

    const encoder = this.device.createCommandEncoder();
    const pass1 = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.hdrView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        },
        {
          view: this.velView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: this.depthView,
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "discard",
      },
    });
    pass1.setPipeline(this.scenePipeline);
    pass1.setBindGroup(0, sceneBg);
    pass1.setVertexBuffer(0, this.vertIdBuf);
    if (nearCount > 0) {
      pass1.setIndexBuffer(this.indexNearBuf, "uint16");
      pass1.drawIndexed(this.indexNearCount, nearCount, 0, 0, 0);
    }
    if (farCount > 0) {
      pass1.setIndexBuffer(this.indexFarBuf, "uint16");
      pass1.drawIndexed(this.indexFarCount, farCount, 0, 0, nearCount);
    }
    pass1.end();

    const swapView = this.context.getCurrentTexture().createView();
    const pass2 = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: swapView,
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass2.setPipeline(this.compPipeline);
    pass2.setBindGroup(0, this.compBindGroup);
    pass2.draw(3, 1, 0, 0);
    pass2.end();

    this.device.queue.submit([encoder.finish()]);
    const tRenderEncodeMs = performance.now() - tRen0;

    return {
      tUploadMs,
      tRenderEncodeMs,
      nearBirdCount: nearCount,
      farBirdCount: farCount,
    };
  }
}
