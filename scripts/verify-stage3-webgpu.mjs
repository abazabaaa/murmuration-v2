#!/usr/bin/env node
/**
 * Stage 3 Gate 2 Live WebGPU Frame-Budget & Keyframe Verification Suite:
 * - Serves `web/dist` on an ephemeral `127.0.0.1:0` HTTP port WITHOUT COOP/COEP headers
 *   (proving static GitHub Pages compatibility with zero SharedArrayBuffer requirement).
 * - Spawns an isolated headless Chromium/Chrome instance on an ephemeral DevTools port (`0`),
 *   runs `?bench=1` across all 12 stress cells (4 spatial regimes x N in {1000, 5000, 10000})
 *   with actual 440/48-tri 3D starling meshes + 5-tap linear-light shutter motion blur,
 *   saves `out/stage3_webgpu_frame_budget.json` and `out/stage3_webgpu_keyframe.png`,
 *   and tears down ONLY its own child process and temporary profile directory.
 */

import { spawn } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const DIST_DIR = path.join(ROOT, "web", "dist");
const OUT_DIR = path.join(ROOT, "out");
const OUT_JSON = path.join(OUT_DIR, "stage3_webgpu_frame_budget.json");
const OUT_PNG = path.join(OUT_DIR, "stage3_webgpu_keyframe.png");

const MIME_MAP = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm",
  ".css": "text/css; charset=utf-8",
  ".png": "image/png",
};

function findChromeBinary() {
  if (process.env.CHROME_BIN && fs.existsSync(process.env.CHROME_BIN)) {
    return process.env.CHROME_BIN;
  }
  const candidates = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium-browser",
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  throw new Error("Could not locate Chrome/Chromium binary. Set CHROME_BIN.");
}

function startEphemeralStaticServer(rootDir) {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      const urlObj = new URL(req.url || "/", "http://127.0.0.1");
      let rel = decodeURIComponent(urlObj.pathname);
      if (rel === "/") rel = "/index.html";
      const safePath = path.normalize(path.join(rootDir, rel));
      if (!safePath.startsWith(rootDir) || !fs.existsSync(safePath) || fs.statSync(safePath).isDirectory()) {
        res.writeHead(404);
        res.end("Not found");
        return;
      }
      const ext = path.extname(safePath);
      // Intentionally NO Cross-Origin-Opener-Policy / Cross-Origin-Embedder-Policy headers (§4.2.2)
      res.writeHead(200, {
        "Content-Type": MIME_MAP[ext] || "application/octet-stream",
        "Cache-Control": "no-store",
      });
      fs.createReadStream(safePath).pipe(res);
    });
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const addr = server.address();
      resolve({ server, port: addr.port });
    });
  });
}

function launchHeadlessChrome(chromeBin, userDataDir) {
  return new Promise((resolve, reject) => {
    const args = [
      "--headless=new",
      "--enable-unsafe-webgpu",
      "--ignore-gpu-blocklist",
      "--window-size=1280,800",
      "--no-first-run",
      "--no-default-browser-check",
      "--disable-background-timer-throttling",
      "--disable-renderer-backgrounding",
      `--user-data-dir=${userDataDir}`,
      "--remote-debugging-port=0",
      "about:blank",
    ];
    const child = spawn(chromeBin, args, { stdio: ["ignore", "pipe", "pipe"] });
    let stderrBuf = "";
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error(`Timed out waiting for Chrome DevTools WS URL. Stderr:\n${stderrBuf}`));
    }, 15000);

    child.stderr.on("data", (chunk) => {
      stderrBuf += chunk.toString("utf8");
      const m = stderrBuf.match(/DevTools listening on (ws:\/\/127\.0\.0\.1:\d+\/devtools\/browser\/[^\s]+)/);
      if (m) {
        clearTimeout(timer);
        resolve({ child, browserWsUrl: m[1] });
      }
    });
    child.on("error", (err) => {
      clearTimeout(timer);
      reject(err);
    });
  });
}

class CdpSession {
  constructor(ws) {
    this.ws = ws;
    this.nextId = 1;
    this.pending = new Map();
    this.events = [];
    ws.addEventListener("message", (ev) => {
      const msg = JSON.parse(typeof ev.data === "string" ? ev.data : ev.data.toString());
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(JSON.stringify(msg.error)));
        else resolve(msg.result);
      } else if (msg.method) {
        this.events.push(msg);
      }
    });
  }

  static connect(wsUrl) {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(wsUrl);
      ws.addEventListener("open", () => resolve(new CdpSession(ws)));
      ws.addEventListener("error", (e) => reject(new Error(`WebSocket error: ${e.message || e}`)));
    });
  }

  send(method, params = {}, sessionId = undefined) {
    const id = this.nextId++;
    const payload = sessionId ? { id, method, params, sessionId } : { id, method, params };
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify(payload));
    });
  }

  close() {
    try {
      this.ws.close();
    } catch {}
  }
}

async function main() {
  if (!fs.existsSync(path.join(DIST_DIR, "index.html"))) {
    throw new Error("Missing web/dist/index.html. Run `pnpm --dir web build` first.");
  }

  const chromeBin = findChromeBinary();
  const tmpProfile = fs.mkdtempSync(path.join(os.tmpdir(), "murm-v2-webgpu-"));
  const { server, port } = await startEphemeralStaticServer(DIST_DIR);
  let chromeChild = null;
  let cdp = null;

  try {
    const launched = await launchHeadlessChrome(chromeBin, tmpProfile);
    chromeChild = launched.child;
    cdp = await CdpSession.connect(launched.browserWsUrl);

    const { targetInfos } = await cdp.send("Target.getTargets");
    const pageTarget = targetInfos.find((t) => t.type === "page");
    if (!pageTarget) throw new Error("No page target found in headless Chrome");

    const { sessionId } = await cdp.send("Target.attachToTarget", {
      targetId: pageTarget.targetId,
      flatten: true,
    });

    await cdp.send("Page.enable", {}, sessionId);
    await cdp.send("Runtime.enable", {}, sessionId);
    await cdp.send("Log.enable", {}, sessionId);
    await cdp.send(
      "Emulation.setDeviceMetricsOverride",
      { width: 1280, height: 800, deviceScaleFactor: 1, mobile: false },
      sessionId
    );

    const targetUrl = `http://127.0.0.1:${port}/?bench=1`;
    console.error(`Navigating headless Chrome WebGPU to ${targetUrl} ...`);
    await cdp.send("Page.navigate", { url: targetUrl }, sessionId);

    const deadline = Date.now() + 120000;
    let benchResult = null;
    while (Date.now() < deadline) {
      const evalRes = await cdp.send(
        "Runtime.evaluate",
        {
          expression:
            "JSON.stringify({ err: window.__STAGE3_BENCH_ERROR__ || null, res: window.__STAGE3_BENCH_RESULT__ || null })",
          returnByValue: true,
        },
        sessionId
      );
      const parsed = JSON.parse(evalRes.result.value || "{}");
      if (parsed.err) {
        throw new Error(`Browser WebGPU error: ${parsed.err}`);
      }
      if (parsed.res) {
        benchResult = parsed.res;
        break;
      }
      await new Promise((r) => setTimeout(r, 500));
    }

    if (!benchResult) {
      throw new Error("Timed out waiting for window.__STAGE3_BENCH_RESULT__");
    }

    // Collect any console errors or GPU validation warnings
    const consoleErrors = cdp.events
      .filter(
        (e) =>
          (e.method === "Runtime.exceptionThrown") ||
          (e.method === "Log.entryAdded" && e.params?.entry?.level === "error")
      )
      .map((e) => e.params);

    const shot = await cdp.send("Page.captureScreenshot", { format: "png" }, sessionId);
    fs.mkdirSync(OUT_DIR, { recursive: true });
    fs.writeFileSync(OUT_PNG, Buffer.from(shot.data, "base64"));

    const finalReport = {
      static_hosting_without_coop_coep_verified: true,
      console_or_webgpu_errors: consoleErrors.length,
      screenshot_saved: "out/stage3_webgpu_keyframe.png",
      ...benchResult,
    };
    fs.writeFileSync(OUT_JSON, JSON.stringify(finalReport, null, 2) + "\n");
    console.log(JSON.stringify(finalReport, null, 2));

    if (consoleErrors.length > 0) {
      console.error("Console/WebGPU errors:", JSON.stringify(consoleErrors, null, 2));
      process.exitCode = 1;
    }
  } finally {
    if (cdp) cdp.close();
    if (chromeChild) {
      chromeChild.kill("SIGTERM");
    }
    server.close();
    try {
      fs.rmSync(tmpProfile, { recursive: true, force: true });
    } catch {}
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
