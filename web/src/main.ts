import { App } from "./app";
import "./style.css";

declare global {
  interface Window {
    __antifive?: App;
  }
}

window.__antifive = new App(import.meta.env.BASE_URL);

if (import.meta.env.PROD && "serviceWorker" in navigator) {
  navigator.serviceWorker
    .register(`${import.meta.env.BASE_URL}sw.js`)
    .catch(() => undefined);
}
