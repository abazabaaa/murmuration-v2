#!/usr/bin/env node
/**
 * Extracts a clean, self-contained 3D starling mesh + cyclic flap/glide/bound pose table
 * (`web/public/assets/starling-mesh.json`) with precomputed smooth vertex normals and flat
 * near/far LOD uint16 index arrays, stripping any workspace-local input paths.
 *
 * Usage: node scripts/extract-starling-asset.mjs <source-starling-motion.json>
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const srcPath = process.argv[2];
if (!srcPath) {
  console.error("Usage: node scripts/extract-starling-asset.mjs <source-starling-motion.json>");
  process.exit(2);
}

const raw = JSON.parse(fs.readFileSync(srcPath, "utf8"));
const nVerts = raw.clips.glide.frames[0].length / 3;

function computeNormals(posFlat, facesTriples) {
  const normals = new Float64Array(nVerts * 3);
  for (const [a, b, c] of facesTriples) {
    const ax = posFlat[3 * a], ay = posFlat[3 * a + 1], az = posFlat[3 * a + 2];
    const bx = posFlat[3 * b], by = posFlat[3 * b + 1], bz = posFlat[3 * b + 2];
    const cx = posFlat[3 * c], cy = posFlat[3 * c + 1], cz = posFlat[3 * c + 2];
    const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
    const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
    let nx = e1y * e2z - e1z * e2y;
    let ny = e1z * e2x - e1x * e2z;
    let nz = e1x * e2y - e1y * e2x;
    // Orient thin wing/body face normal upward (+Z dorsal) or outward from spine (y=0, z=0)
    const mx = (ax + bx + cx) / 3;
    const my = (ay + by + cy) / 3;
    const mz = (az + bz + cz) / 3;
    const refZ = Math.abs(my) > 0.22 ? (nz >= 0 ? 1 : -1) : (nx * (mx + 0.2) + ny * my + nz * mz >= 0 ? 1 : -1);
    if (refZ < 0) {
      nx = -nx;
      ny = -ny;
      nz = -nz;
    }
    for (const idx of [a, b, c]) {
      normals[3 * idx] += nx;
      normals[3 * idx + 1] += ny;
      normals[3 * idx + 2] += nz;
    }
  }
  const out = new Array(nVerts * 3);
  for (let i = 0; i < nVerts; i++) {
    const x = normals[3 * i], y = normals[3 * i + 1], z = normals[3 * i + 2];
    const len = Math.hypot(x, y, z);
    if (len > 1e-12) {
      out[3 * i] = Math.round((x / len) * 1e5) / 1e5;
      out[3 * i + 1] = Math.round((y / len) * 1e5) / 1e5;
      out[3 * i + 2] = Math.round((z / len) * 1e5) / 1e5;
    } else {
      out[3 * i] = 0;
      out[3 * i + 1] = 0;
      out[3 * i + 2] = 1;
    }
  }
  return out;
}

function roundPositions(posFlat) {
  return posFlat.map((v) => Math.round(v * 1e5) / 1e5);
}

const nearFaces = raw.faces.near;
const farFaces = raw.faces.far;

const flapFrames = raw.clips.flap.frames.map((f) => ({
  positions: roundPositions(f),
  normals: computeNormals(f, nearFaces),
}));
const glideFrame = {
  positions: roundPositions(raw.clips.glide.frames[0]),
  normals: computeNormals(raw.clips.glide.frames[0], nearFaces),
};
const boundFrame = {
  positions: roundPositions(raw.clips.bound.frames[0]),
  normals: computeNormals(raw.clips.bound.frames[0], nearFaces),
};

const cleanAsset = {
  format: "murmuration-v2-starling-mesh-v1",
  halfSpanM: raw.halfSpanM,
  lengthM: raw.lengthM,
  vertexCount: nVerts,
  indicesNear: nearFaces.flat(),
  indicesFar: farFaces.flat(),
  flapTimes: raw.clips.flap.times.map((t) => Math.round(t * 1e6) / 1e6),
  flapFrames,
  glideFrame,
  boundFrame,
  doiReferences: (raw.reference?.sources ?? []).filter((s) => s.startsWith("https://doi.org/")),
};

const outPath = path.join(ROOT, "web", "public", "assets", "starling-mesh.json");
fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, JSON.stringify(cleanAsset));
console.log(
  `Wrote ${path.relative(ROOT, outPath)} (${(fs.statSync(outPath).size / 1024).toFixed(1)} KB, ${nVerts} vertices, ${nearFaces.length} near / ${farFaces.length} far tris, ${flapFrames.length} flap frames)`
);
