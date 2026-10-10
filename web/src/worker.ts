import type { PyodideInterface } from "pyodide";
import type { EngineStatus } from "./types";
import { PYODIDE_VERSION } from "./version";

const MIRROR_INDEX = `https://cdn.npmmirror.com/packages/pyodide/${PYODIDE_VERSION}/files/`;
const CDN_INDEX = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const FASTLY_INDEX = `https://fastly.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

// 本地 pyodide-lock.json 不可达时的兜底文件名(与 lock 内 numpy.file_name 一致)
const NUMPY_FILE_FALLBACK = "numpy-2.4.6-cp314-cp314-pyemscripten_2026_0_wasm32.whl";

const IMPORT_TIMEOUT_MS = 20_000;
const RUNTIME_TIMEOUT_MS = 150_000;
const LOCK_TIMEOUT_MS = 15_000;
const NUMPY_TIMEOUT_MS = 60_000;
const WHEEL_TIMEOUT_MS = 60_000;

const ctx = self as unknown as {
  postMessage(message: unknown): void;
  onmessage: ((event: MessageEvent) => void) | null;
  location: Location;
};

type WebApi = {
  handle: (request: string) => string;
  set_progress_cb: (cb: unknown) => void;
};

type Runtime = {
  loadPyodide: (options: { indexURL: string }) => Promise<PyodideInterface>;
  setInterruptBuffer?: (buffer: Int32Array) => void;
};

type Source = {
  name: string;
  loader: string;
  indexURL: string;
};

let py: PyodideInterface | null = null;
let mod: WebApi | null = null;
let runtimeModule: Runtime | null = null;
let interruptBuffer: Int32Array | null = null;

function errText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function status(stage: string, detail?: string) {
  const payload: EngineStatus = { stage, detail };
  ctx.postMessage({ type: "status", status: payload });
}

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(
      () => reject(new Error(`${label} 超时(${Math.round(ms / 1000)}s)`)),
      ms,
    );
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}

async function fetchWithTimeout(url: string, ms: number): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  try {
    return await fetch(url, { signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function fetchBytes(url: string, ms: number, stage: string): Promise<Uint8Array> {
  const response = await fetchWithTimeout(url, ms);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const total = Number(response.headers.get("Content-Length") ?? 0);
  if (!response.body || !(total > 0)) {
    return new Uint8Array(await response.arrayBuffer());
  }
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  let reported = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value) continue;
    chunks.push(value);
    loaded += value.byteLength;
    if (loaded - reported >= 256 * 1024) {
      reported = loaded;
      ctx.postMessage({ type: "download", stage, loaded, total });
    }
  }
  const out = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  ctx.postMessage({ type: "download", stage, loaded, total });
  return out;
}

function preferredSource(): string {
  const override = import.meta.env.VITE_PYODIDE_SOURCE as string | undefined;
  if (override) return override;
  return import.meta.env.DEV ? "local" : "mirror";
}

function runtimeSources(base: string): Source[] {
  const local = {
    name: "local",
    loader: new URL("pyodide/pyodide.mjs", base).href,
    indexURL: new URL("pyodide/", base).href,
  };
  const mirror = {
    name: "mirror",
    loader: `${MIRROR_INDEX}pyodide.mjs`,
    indexURL: MIRROR_INDEX,
  };
  const cdn = {
    name: "cdn",
    loader: `${CDN_INDEX}pyodide.mjs`,
    indexURL: CDN_INDEX,
  };
  const fastly = {
    name: "fastly",
    loader: `${FASTLY_INDEX}pyodide.mjs`,
    indexURL: FASTLY_INDEX,
  };
  const all = [local, mirror, cdn, fastly];
  const order = [preferredSource(), "mirror", "local", "cdn", "fastly"];
  const seen = new Set<string>();
  const out: Source[] = [];
  for (const name of order) {
    const src = all.find((s) => s.name === name);
    if (src && !seen.has(name)) {
      seen.add(name);
      out.push(src);
    }
  }
  for (const src of all) {
    if (!seen.has(src.name)) out.push(src);
  }
  return out;
}

async function loadRuntime(base: string): Promise<PyodideInterface> {
  let lastError: unknown = null;
  for (const source of runtimeSources(base)) {
    try {
      status("core", `${source.name} · pyodide ${PYODIDE_VERSION}`);
      const module = await withTimeout(
        import(/* @vite-ignore */ source.loader) as Promise<Runtime>,
        IMPORT_TIMEOUT_MS,
        `${source.name} 模块`,
      );
      const instance = await withTimeout(
        module.loadPyodide({ indexURL: source.indexURL }),
        RUNTIME_TIMEOUT_MS,
        `${source.name} 运行时`,
      );
      runtimeModule = module;
      return instance;
    } catch (error) {
      lastError = error;
      status("fallback", `${source.name}: ${errText(error)}`);
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

async function loadNumpy(base: string) {
  if (!py) throw new Error("运行时未初始化");
  let file = NUMPY_FILE_FALLBACK;
  const notes: string[] = [];
  try {
    const response = await fetchWithTimeout(
      new URL("pyodide/pyodide-lock.json", base).href,
      LOCK_TIMEOUT_MS,
    );
    if (response.ok) {
      const lock = (await response.json()) as {
        packages?: { numpy?: { file_name?: string } };
      };
      const name = lock.packages?.numpy?.file_name;
      if (name) file = name;
    } else {
      notes.push(`lock HTTP ${response.status}`);
    }
  } catch (error) {
    notes.push(`lock ${errText(error)}`);
  }
  const attempts: Source[] = [
    { name: "local", loader: new URL(`pyodide/${file}`, base).href, indexURL: "" },
    { name: "fastly", loader: `${FASTLY_INDEX}${file}`, indexURL: "" },
    { name: "cdn", loader: `${CDN_INDEX}${file}`, indexURL: "" },
  ];
  for (const attempt of attempts) {
    try {
      status("numpy", `${attempt.name} · ${file}`);
      await withTimeout(py.loadPackage(attempt.loader), NUMPY_TIMEOUT_MS, `numpy(${attempt.name})`);
      return;
    } catch (error) {
      notes.push(`${attempt.name}: ${errText(error)}`);
      status("fallback", `numpy ${attempt.name}: ${errText(error)}`);
    }
  }
  throw new Error(`numpy 加载失败(${notes.join("; ")})`);
}

function enableInterrupts() {
  if (!interruptBuffer || !py) return;
  const api = py as unknown as { setInterruptBuffer?: (buffer: Int32Array) => void };
  const fn = typeof api.setInterruptBuffer === "function"
    ? api.setInterruptBuffer.bind(api)
    : runtimeModule?.setInterruptBuffer?.bind(runtimeModule);
  if (!fn) {
    interruptBuffer = null;
    return;
  }
  try {
    fn(interruptBuffer);
  } catch {
    interruptBuffer = null;   // 运行时不支持:退化为不可中断
  }
}

async function boot(base: string) {
  status("core");
  py = await loadRuntime(base);
  enableInterrupts();
  status("numpy");
  await loadNumpy(base);
  status("engine");
  const wheel = await fetchBytes(new URL("antifive.whl", base).href, WHEEL_TIMEOUT_MS, "engine");
  py.FS.writeFile("/tmp/antifive.whl", wheel);
  py.runPython(
    'import sys\nif "/tmp/antifive.whl" not in sys.path:\n    sys.path.insert(0, "/tmp/antifive.whl")',
  );
  mod = py.pyimport("antifive.web_api") as unknown as WebApi;
  mod.set_progress_cb((nodes: number) =>
    ctx.postMessage({ type: "progress", nodes: Number(nodes) }),
  );
  status("ready");
  ctx.postMessage({ type: "ready" });
}

ctx.onmessage = async (event: MessageEvent) => {
  const msg = event.data;
  try {
    if (msg.type === "init") {
      if (msg.interruptBuffer) interruptBuffer = new Int32Array(msg.interruptBuffer);
      await boot(new URL(msg.base, ctx.location.origin).href);
    } else if (msg.type === "call") {
      if (!mod) throw new Error("引擎尚未就绪");
      if (interruptBuffer) Atomics.store(interruptBuffer, 0, 0);   // 清除过期中断信号
      const value = JSON.parse(mod.handle(JSON.stringify({ cmd: msg.cmd, ...msg.payload })));
      if (!value.ok) throw new Error(value.error);
      ctx.postMessage({ type: "result", id: msg.id, ok: true, value });
    }
  } catch (error) {
    const text = errText(error);
    if (msg.type === "init") {
      ctx.postMessage({ type: "fatal", error: text });
    } else {
      ctx.postMessage({ type: "result", id: msg.id, ok: false, error: text });
    }
  }
};
