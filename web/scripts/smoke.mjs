import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const here = path.dirname(fileURLToPath(import.meta.url));
const webRoot = path.resolve(here, "..");
const repoRoot = path.resolve(webRoot, "..");
const dist = path.join(webRoot, "dist");
const base = process.env.SMOKE_BASE ?? "/";
const samplePath = path.join(repoRoot, "data", "samples", "sample_black_win.afg");

if (!process.env.SMOKE_URL && !fs.existsSync(path.join(dist, "index.html"))) {
  console.error("dist 不存在,请先运行 npm run build");
  process.exit(1);
}

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".wasm": "application/wasm",
  ".whl": "application/octet-stream",
  ".zip": "application/zip",
  ".svg": "image/svg+xml",
};

const externalUrl = process.env.SMOKE_URL;
let server = null;
let url = externalUrl;

if (!externalUrl) {
  server = http.createServer((req, res) => {
    const reqUrl = new URL(req.url ?? "/", "http://localhost");
    let pathname = decodeURIComponent(reqUrl.pathname);
    if (base !== "/" && pathname.startsWith(base)) pathname = pathname.slice(base.length - 1);
    const rel = pathname.replace(/^\/+/, "");
    const file = path.join(dist, rel === "" ? "index.html" : rel);
    if (!file.startsWith(dist) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404);
      res.end("not found");
      return;
    }
    res.writeHead(200, { "content-type": MIME[path.extname(file)] ?? "application/octet-stream" });
    fs.createReadStream(file).pipe(res);
  });
  const port = await new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve(server.address().port));
  });
  url = `http://127.0.0.1:${port}${base}`;
}

function findChrome() {
  const candidates = [
    process.env.CHROME_PATH,
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  ];
  return candidates.find((p) => p && fs.existsSync(p));
}

const chrome = findChrome();
if (!chrome) {
  console.error("未找到 Chrome/Edge,请设置环境变量 CHROME_PATH");
  server?.close();
  process.exit(1);
}

const browser = await puppeteer.launch({
  executablePath: chrome,
  headless: true,
  args: ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
});
const page = await browser.newPage();
page.on("pageerror", (e) => console.log("[pageerror]", e.message));

const failures = [];
const check = (ok, label) => {
  console.log(`${ok ? "ok  " : "FAIL"} ${label}`);
  if (!ok) failures.push(label);
};

const wait = async (fn, timeout = 240000, step = 300) => {
  const t0 = Date.now();
  for (;;) {
    const v = await page.evaluate(fn);
    if (v) return v;
    if (Date.now() - t0 > timeout) throw new Error(`timeout: ${fn}`);
    await new Promise((r) => setTimeout(r, step));
  }
};

await page.goto(url, { waitUntil: "domcontentloaded" });
await wait(() => document.getElementById("loading").classList.contains("hidden"));
check(true, "engine ready");

const initial = await page.evaluate(() => ({
  player: window.__antifive.state.player,
  count: window.__antifive.state.move_count,
  legal: window.__antifive.state.legal.length,
}));
check(initial.player === 1 && initial.count === 0 && initial.legal === 225, "initial state");

const box = await (await page.$("#board")).boundingBox();
await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
await wait(() => window.__antifive.state.move_count >= 2, 120000);
const afterAi = await page.evaluate(() => ({
  count: window.__antifive.state.move_count,
  player: window.__antifive.state.player,
}));
check(afterAi.count === 2 && afterAi.player === 1, "human move + AI reply");

await page.keyboard.press("u");
await wait(() => window.__antifive.state.move_count === 0);
check(true, "undo");

await page.keyboard.press("d");
check(await page.evaluate(() => window.__antifive.showDanger), "danger toggle");

await page.keyboard.press("4");
await wait(() => window.__antifive.mode === 3 && window.__antifive.state.move_count >= 4, 180000);
check(true, "ai vs ai");

await page.evaluate((text) => window.__antifive.importRecord(new File([text], "sample.afg")), fs.readFileSync(samplePath, "utf8"));
await wait(() => window.__antifive.review && window.__antifive.variation.length > 0);
const imported = await page.evaluate(() => ({
  len: window.__antifive.variation.length,
  over: window.__antifive.state.game_over,
}));
check(imported.len > 0 && imported.over, "import .afg");

await page.evaluate(() => window.__antifive.reviewJump(0));
await wait(() => window.__antifive.pos === 0);
await page.evaluate(() => window.__antifive.reviewJump(10));
await wait(() => window.__antifive.pos === 10);
check(true, "review jump");

const branch = await page.evaluate(async () => {
  const app = window.__antifive;
  await app.reviewTry(app.state.legal[0]);
  return { divergence: app.divergence, pos: app.pos };
});
check(branch.divergence === 10 && branch.pos === 11, "branch trial");

await page.evaluate(() => window.__antifive.reviewReset());
await wait(() => window.__antifive.pos === 10);
check(true, "review reset");

const afg = await page.evaluate(async () => {
  const app = window.__antifive;
  const res = await app.engine.call("export", { moves: app.recordMoves, result: app.reviewResult });
  return res.text;
});
check(afg.startsWith("[AntiFive 1.0]"), "export .afg");

await browser.close();
server?.close();

if (failures.length) {
  console.error(`\n${failures.length} 项失败`);
  process.exit(1);
}
console.log("\nSMOKE OK");
