import type { EngineStatus } from "./types";

type PendingCall = {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
};

export class Engine {
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<number, PendingCall>();
  private readyResolve!: () => void;
  private readyReject!: (error: Error) => void;

  readonly ready: Promise<void>;
  onProgress: ((nodes: number) => void) | null = null;
  onStatus: ((status: EngineStatus) => void) | null = null;

  constructor(base: string) {
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    this.ready = new Promise<void>((resolve, reject) => {
      this.readyResolve = resolve;
      this.readyReject = reject;
    });
    this.worker.onmessage = (event: MessageEvent) => {
      const msg = event.data;
      if (msg.type === "ready") {
        this.readyResolve();
      } else if (msg.type === "status") {
        this.onStatus?.(msg.status);
      } else if (msg.type === "progress") {
        this.onProgress?.(msg.nodes);
      } else if (msg.type === "result") {
        const entry = this.pending.get(msg.id);
        if (!entry) return;
        this.pending.delete(msg.id);
        if (msg.ok) entry.resolve(msg.value);
        else entry.reject(new Error(msg.error));
      } else if (msg.type === "fatal") {
        this.readyReject(new Error(msg.error));
      }
    };
    this.worker.postMessage({ type: "init", base });
  }

  call<T>(cmd: string, payload: Record<string, unknown> = {}): Promise<T> {
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (value: unknown) => void, reject });
      this.worker.postMessage({ type: "call", id, cmd, payload });
    });
  }
}
