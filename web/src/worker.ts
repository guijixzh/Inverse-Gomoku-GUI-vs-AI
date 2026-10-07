import type { PyodideInterface } from "pyodide";
import type { EngineStatus } from "./types";

const PYODIDE_VERSION = "314.0.7";
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

let py: PyodideInterface | null = null;
let mod: WebApi | null = null;

function status(stage: string, detail?: string) {
  const payload: EngineStatus = { stage, detail };
  ctx.postMessage({ type: "status", status: payload });
}

async function loadRuntime(base: string): Promise<PyodideInterface> {
  const indexURL = new URL("pyodide/", base).href;
  try {
    const local = (await import(/* @vite-ignore */ new URL("pyodide/pyodide.mjs", base).href)) as {
      loadPyodide: (options: { indexURL: string }) => Promise<PyodideInterface>;
    };
    return await local.loadPyodide({ indexURL });
  } catch (error) {
    status("fallback", error instanceof Error ? error.message : String(error));
    const remote = (await import(/* @vite-ignore */ `${CDN_INDEX}pyodide.mjs`)) as {
      loadPyodide: (options: { indexURL: string }) => Promise<PyodideInterface>;
    };
    return await remote.loadPyodide({ indexURL: CDN_INDEX });
  }
}

async function boot(base: string) {
  status("core");
  py = await loadRuntime(base);
  status("numpy");
  await py.loadPackage("numpy");
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
