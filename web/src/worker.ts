import type { PyodideInterface } from "pyodide";
import type { EngineStatus } from "./types";

const PYODIDE_VERSION = "314.0.7";
const MIRROR_INDEX = `https://cdn.npmmirror.com/packages/pyodide/${PYODIDE_VERSION}/files/`;
const CDN_INDEX = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

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
};

let py: PyodideInterface | null = null;
let mod: WebApi | null = null;

function status(stage: string, detail?: string) {
  const payload: EngineStatus = { stage, detail };
  ctx.postMessage({ type: "status", status: payload });
}

function preferredSource(): string {
  const override = import.meta.env.VITE_PYODIDE_SOURCE as string | undefined;
  if (override) return override;
  return import.meta.env.DEV ? "local" : "mirror";
}

function runtimeSources(base: string): { name: string; loader: string; indexURL: string }[] {
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
  const order = [preferredSource(), "mirror", "local", "cdn"];
  const all = [local, mirror, cdn];
  const seen = new Set<string>();
  const out: typeof all = [];
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
      const module = (await import(/* @vite-ignore */ source.loader)) as Runtime;
      return await module.loadPyodide({ indexURL: source.indexURL });
    } catch (error) {
      lastError = error;
      status("fallback", `${source.name}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

async function loadNumpy(base: string) {
  if (!py) throw new Error("运行时未初始化");
  try {
    const response = await fetch(new URL("pyodide/pyodide-lock.json", base));
    if (response.ok) {
      const lock = (await response.json()) as { packages?: { numpy?: { file_name?: string } } };
      const file = lock.packages?.numpy?.file_name;
      if (file) {
        await py.loadPackage(new URL(`pyodide/${file}`, base).href);
        return;
      }
    }
  } catch {
    // 本地 numpy wheel 不可用时退回 indexURL 解析
  }
  await py.loadPackage("numpy");
}

async function boot(base: string) {
  status("core");
  py = await loadRuntime(base);
  status("numpy");
  await loadNumpy(base);
  status("engine");
  const response = await fetch(new URL("antifive.whl", base));
  if (!response.ok) throw new Error(`antifive.whl 加载失败: HTTP ${response.status}`);
  py.FS.writeFile("/tmp/antifive.whl", new Uint8Array(await response.arrayBuffer()));
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
      await boot(new URL(msg.base, ctx.location.origin).href);
    } else if (msg.type === "call") {
      if (!mod) throw new Error("引擎尚未就绪");
      const value = JSON.parse(mod.handle(JSON.stringify({ cmd: msg.cmd, ...msg.payload })));
      if (!value.ok) throw new Error(value.error);
      ctx.postMessage({ type: "result", id: msg.id, ok: true, value });
    }
  } catch (error) {
    const text = error instanceof Error ? error.message : String(error);
    if (msg.type === "init") {
      ctx.postMessage({ type: "fatal", error: text });
    } else {
      ctx.postMessage({ type: "result", id: msg.id, ok: false, error: text });
    }
  }
};
