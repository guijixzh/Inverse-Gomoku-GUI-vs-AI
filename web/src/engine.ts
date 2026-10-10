import type { EngineStatus } from "./types";

type PendingCall = {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
};

const DEFAULT_CALL_TIMEOUT_MS = 300_000;

export class Engine {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<number, PendingCall>();
  private readyResolve!: () => void;
  private readyReject!: (error: Error) => void;
  private settled = false;
  private dead = false;
  private interruptBuffer: Int32Array | null = null;

  readonly ready: Promise<void>;
  onProgress: ((nodes: number) => void) | null = null;
  onStatus: ((status: EngineStatus) => void) | null = null;
  onDownload: ((stage: string, loaded: number, total: number) => void) | null = null;
  onFatal: ((error: Error) => void) | null = null;

  constructor(base: string) {
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    this.ready = new Promise<void>((resolve, reject) => {
      this.readyResolve = resolve;
      this.readyReject = reject;
    });
    const isolated = (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true;
    if (isolated && typeof SharedArrayBuffer !== "undefined") {
      try {
        this.interruptBuffer = new Int32Array(new SharedArrayBuffer(4));
      } catch {
        this.interruptBuffer = null;
      }
    }
    this.worker.onmessage = (event: MessageEvent) => {
      const msg = event.data;
      if (msg.type === "ready") {
        this.settled = true;
        this.readyResolve();
      } else if (msg.type === "status") {
        this.onStatus?.(msg.status);
      } else if (msg.type === "progress") {
        this.onProgress?.(msg.nodes);
      } else if (msg.type === "download") {
        this.onDownload?.(msg.stage, msg.loaded, msg.total);
      } else if (msg.type === "result") {
        const entry = this.pending.get(msg.id);
        if (!entry) return;
        this.pending.delete(msg.id);
        if (msg.ok) entry.resolve(msg.value);
        else entry.reject(new Error(msg.error));
      } else if (msg.type === "fatal") {
        this.fail(new Error(msg.error));
      }
    };
    this.worker.onerror = (event: ErrorEvent) => {
      event.preventDefault?.();
      this.fail(new Error(event.message || "引擎 Worker 加载失败(资源可能已更新,请刷新页面)"));
    };
    this.worker.onmessageerror = () => {
      this.fail(new Error("引擎 Worker 消息解码失败"));
    };
    this.worker.postMessage({
      type: "init",
      base,
      interruptBuffer: this.interruptBuffer?.buffer ?? null,
    });
  }

  get canInterrupt(): boolean {
    return this.interruptBuffer !== null;
  }

  /** 中断 Worker 中正在执行的 Python 搜索(需跨源隔离 + SharedArrayBuffer)。 */
  interrupt(): void {
    if (!this.interruptBuffer) return;
    Atomics.store(this.interruptBuffer, 0, 2);   // SIGINT → KeyboardInterrupt
  }

  private fail(error: Error) {
    if (this.dead) return;
    this.dead = true;
    if (!this.settled) {
      this.settled = true;
      this.readyReject(error);
    } else {
      this.onFatal?.(error);
    }
    for (const entry of this.pending.values()) entry.reject(error);
    this.pending.clear();
  }

  call<T>(
    cmd: string,
    payload: Record<string, unknown> = {},
    timeoutMs = DEFAULT_CALL_TIMEOUT_MS,
  ): Promise<T> {
    if (this.dead) return Promise.reject(new Error("引擎已停止"));
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      const timer = timeoutMs > 0
        ? setTimeout(() => {
            this.pending.delete(id);
            reject(new Error(`引擎调用超时: ${cmd}`));
          }, timeoutMs)
        : null;
      this.pending.set(id, {
        resolve: (value) => {
          if (timer !== null) clearTimeout(timer);
          resolve(value as T);
        },
        reject: (error) => {
          if (timer !== null) clearTimeout(timer);
          reject(error);
        },
      });
      this.worker.postMessage({ type: "call", id, cmd, payload });
    });
  }
}
