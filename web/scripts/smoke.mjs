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
await wait(() => window.__antifive?.state, 60000, 50);
check(true, "engine ready");

const visibilityRedraw = await page.evaluate(() => {
  const setHidden = (value) => {
    Object.defineProperty(document, "hidden", { configurable: true, get: () => value });
    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      get: () => (value ? "hidden" : "visible"),
    });
    document.dispatchEvent(new Event("visibilitychange"));
  };
  window.__antifive.needsDraw = false;
  setHidden(true);
  setHidden(false);
  const redrawn = window.__antifive.needsDraw === true;
  delete document.hidden;
  delete document.visibilityState;
  return redrawn;
});
check(visibilityRedraw, "visibilitychange redraws board");

await wait(async () => (await navigator.serviceWorker.getRegistrations()).length > 0, 15000, 200);
check(true, "service worker registered");
check(await page.evaluate(() => window.crossOriginIsolated === true),
  "cross-origin isolated (SharedArrayBuffer)");

const initial = await page.evaluate(() => ({
  player: window.__antifive.state.player,
  count: window.__antifive.state.move_count,
  legal: window.__antifive.state.legal.length,
}));
check(initial.player === 1 && initial.count === 0 && initial.legal === 225, "initial state");

const defaults = await page.evaluate(() => ({
  tier: window.__antifive.tier,
  limit: window.__antifive.limitSeconds,
  mode: window.__antifive.limitMode,
  infoLimit: document.getElementById("info-limit").textContent,
  param: document.getElementById("info-param").textContent,
}));
check(defaults.tier === 2 && defaults.limit === 12 && defaults.mode === "none" &&
  defaults.infoLimit === "不限时" &&
  defaults.param === "高级·VCF开 · 深度 8 层",
  "default advanced+VCF / depth 8 / 不限时");

await page.keyboard.press("e");
await page.click('input[name="limit"][value="ai"]');
await page.$eval("#limit-seconds", (el) => {
  el.value = "20";
});
// 版本/VCF 开关:初级时 VCF 自动禁用;切到中级再切回高级并关 VCF 后应生效
await page.click('input[name="hver"][value="beginner"]');
check(await page.$eval('input[name="hvcf"]', (el) => el.disabled),
  "vcf disabled for beginner version");
await page.click('input[name="hver"][value="mid"]');
check(!(await page.$eval('input[name="hvcf"]', (el) => el.disabled)),
  "vcf enabled for mid version");
await page.click('input[name="hver"][value="advanced"]');
await page.click('input[name="hvcf"][value="off"]');
await page.click("#settings-apply");
const applied = await page.evaluate(() => ({
  ver: window.__antifive.hver,
  vcf: window.__antifive.hvcf,
  param: document.getElementById("info-param").textContent,
}));
check(applied.ver === "advanced" && applied.vcf === false &&
  applied.param.includes("VCF关"), "version/vcf applied");

const box = await (await page.$("#board")).boundingBox();
const cx = box.x + box.width / 2;
const cy = box.y + box.height / 2;
await page.mouse.click(cx, cy);
const ghost = await page.evaluate(() => ({
  on: window.__antifive.confirmMove,
  idx: window.__antifive.confirmIdx,
  count: window.__antifive.state.move_count,
}));
check(ghost.on === true && ghost.idx !== null && ghost.count === 0,
  "confirm move: first click only fixes ghost");
await page.mouse.click(cx, cy);
await wait(() => window.__antifive.thinking === true, 30000, 50);
const remain = await page.evaluate(() => (window.__antifive.deadline - performance.now()) / 1000);
check(remain > 16 && remain <= 20.5, `limit seconds applied (${remain.toFixed(1)}s)`);
await wait(() => window.__antifive.state.move_count >= 2 && !window.__antifive.thinking, 120000);
const afterAi = await page.evaluate(() => ({
  count: window.__antifive.state.move_count,
  player: window.__antifive.state.player,
}));
check(afterAi.count === 2 && afterAi.player === 1, "human move + AI reply");

await page.evaluate(() => {
  const app = window.__antifive;
  app.limitMode = "none";
  app.tier = 0;
  app.syncPanel();
});

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
  const idx = app.state.legal[0];
  await app.reviewTry(idx);
  const ghost = { idx: app.confirmIdx, divergence: app.divergence, pos: app.pos };
  await app.reviewTry(idx);
  return { ghost, divergence: app.divergence, pos: app.pos };
});
check(branch.ghost.idx !== null && branch.ghost.divergence === null &&
  branch.ghost.pos === 10, "review trial confirm gate");
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

// 引擎中断:跨源隔离下 SharedArrayBuffer 应能中止正在搜索的 AI(悔棋即时生效)
check(await page.evaluate(() => window.__antifive.engine.canInterrupt), "engine interrupt available");
const interrupt = await page.evaluate(async () => {
  const app = window.__antifive;
  await app.engine.call("new", {});
  for (const idx of [112, 52, 172, 34, 190, 76]) {
    await app.engine.call("move", { idx });
  }
  const t0 = performance.now();
  const p = app.engine.call("ai", { depth: 8, budget: 60, engine: "advanced", vcf: true }, 120000);
  await new Promise((r) => setTimeout(r, 1500));
  const nodes = app.nodes;
  app.engine.interrupt();
  try {
    await p;
    return { ok: false, ms: performance.now() - t0, nodes };
  } catch {
    return { ok: true, ms: performance.now() - t0, nodes };
  }
});
check(interrupt.ok && interrupt.ms < 8000,
  `engine interrupt aborts AI search (${Math.round(interrupt.ms)}ms, nodes=${interrupt.nodes})`);

// 认输:二次确认(第一次点击只变红,点其他按钮取消,再点确认)
await page.evaluate(() => {
  const app = window.__antifive;
  app.mode = 0;
  app.newGame();
});
await wait(() => window.__antifive.state.move_count === 0 && !window.__antifive.review);
await page.click('button[data-action="resign"]');
const armed = await page.evaluate(() => ({
  shown: window.__antifive.resignArmedShown,
  over: window.__antifive.state.game_over,
  text: document.querySelector('button[data-action="resign"]').textContent,
}));
check(armed.shown === true && !armed.over && armed.text === "确定？",
  "resign first click arms only");
await page.click('button[data-action="undo"]');
await wait(() => window.__antifive.resignArmedShown === false);
check(await page.$eval('button[data-action="resign"]', (b) => b.textContent === "认输"),
  "clicking elsewhere disarms resign");
await page.click('button[data-action="resign"]');
await page.click('button[data-action="resign"]');
await wait(() => window.__antifive.state.game_over === true);
const resigned = await page.evaluate(() => ({
  loser: window.__antifive.state.loser,
  resigned: window.__antifive.state.resigned,
  result: window.__antifive.state.result,
  review: window.__antifive.review,
}));
check(resigned.loser === 1 && resigned.resigned === 1 &&
  resigned.result === "黑认输" && resigned.review, "resign committed + review");

await browser.close();
server?.close();

if (failures.length) {
  console.error(`\n${failures.length} 项失败`);
  process.exit(1);
}
console.log("\nSMOKE OK");
