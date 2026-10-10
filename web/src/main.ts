import { App } from "./app";
import { PYODIDE_VERSION } from "./version";
import "./style.css";

declare global {
  interface Window {
    __antifive?: App;
  }
}

const RELOAD_KEY = "antifive-coi-reload";

// GitHub Pages 不能自定义响应头,sw.js 给文档补 COOP/COEP 以启用跨源隔离。
// 首次注册 SW 后需要刷新一次让隔离生效;不支持时静默退化为无 SharedArrayBuffer
// (AI 搜索无法中断,悔棋走乐观回退)。
async function ensureCrossOriginIsolation(): Promise<boolean> {
  if (window.crossOriginIsolated) return true;
  if (!import.meta.env.PROD || !("serviceWorker" in navigator)) return true;
  try {
    await navigator.serviceWorker.register(`${import.meta.env.BASE_URL}sw.js?v=${PYODIDE_VERSION}`);
    if (!navigator.serviceWorker.controller) {
      await new Promise<void>((resolve) => {
        const timer = window.setTimeout(resolve, 4000);
        navigator.serviceWorker.addEventListener(
          "controllerchange",
          () => {
            window.clearTimeout(timer);
            resolve();
          },
          { once: true },
        );
      });
      if (!window.crossOriginIsolated && !sessionStorage.getItem(RELOAD_KEY)) {
        sessionStorage.setItem(RELOAD_KEY, "1");
        window.location.reload();
        return false;
      }
    }
  } catch {
    // Service Worker 不可用:继续,无隔离
  }
  return true;
}

void ensureCrossOriginIsolation().then((proceed) => {
  if (!proceed) return; // 正在刷新以启用跨源隔离
  window.__antifive = new App(import.meta.env.BASE_URL);
});
