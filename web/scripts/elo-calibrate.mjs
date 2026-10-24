// 网页端语义的五档 Elo 标定:编排器。
//
// 用法示例(12s/步,五档全循环):
//   node web/scripts/elo-calibrate.mjs --budget 12 --depth 8 --workers 20 \
//        --out records/web_elo_12s --deadline 03:30
//
// 规则:
//   * 每个无序对 (a,b) 生成 N 盘,N 为偶数;黑白各半(每局开局独立随机,
//     用 seed 区分,避免同色对局完全重复)。
//   * 一局一个 worker 线程,worker 内部用 web/src/engine/api.ts 的 EngineSession
//     (与浏览器完全同一份代码),每步 ai({depth,budget,engine,vcf})。
//   * 断点续跑:OUT/games/<pair>/*.json 已存在的对局直接跳过。
//   * --deadline 到点后不再开新局,已开的跑完;结果仍可出报告。
import { Worker } from "node:worker_threads";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";

const LEVELS = ["beginner", "mid", "advanced", "hell", "legend"];

function parseArgs(argv) {
  const out = {
    budget: 12, depth: 8, vcf: 1, workers: Math.max(2, Math.min(20, os.cpus().length - 12)),
    openPlies: 4, out: "records/web_elo_12s", deadline: "", pairs: "",
    uniformDepth: true, variety: "off", nodes: 0,
    pairGames: {
      "beginner-mid": 40, "beginner-advanced": 32, "beginner-hell": 20,
      "beginner-legend": 20, "mid-advanced": 48, "mid-hell": 28,
      "mid-legend": 28, "advanced-hell": 48, "advanced-legend": 48,
      "hell-legend": 60,
    },
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === "--budget") out.budget = Number(next());
    else if (a === "--depth") out.depth = Number(next());
    else if (a === "--vcf") out.vcf = Number(next());
    else if (a === "--workers") out.workers = Number(next());
    else if (a === "--open-plies") out.openPlies = Number(next());
    else if (a === "--out") out.out = next();
    else if (a === "--deadline") out.deadline = next();
    else if (a === "--uniform-depth") out.uniformDepth = true;
    else if (a === "--legacy-depth") out.uniformDepth = false;
    else if (a === "--nodes") out.nodes = Number(argv[++i]) || 0;
    else if (a === "--variety") out.variety = next();
    else if (a === "--pairs") out.pairs = next();
    else if (a === "--pair-games") {
      // 形如 beginner-mid=40,hell-legend=60
      out.pairGames = {};
      for (const part of String(next()).split(",")) {
        const [k, v] = part.split("=");
        if (k && v) out.pairGames[k.trim()] = Number(v);
      }
    } else if (a === "-h" || a === "--help") {
      console.log("见文件头注释"); process.exit(0);
    }
  }
  return out;
}

const args = parseArgs(process.argv.slice(2));
const OUT = path.resolve(args.out);
const GAMES = path.join(OUT, "games");
fs.mkdirSync(GAMES, { recursive: true });

function pairId(a, b) {
  const i = LEVELS.indexOf(a), j = LEVELS.indexOf(b);
  return i < j ? `${a}-${b}` : `${b}-${a}`;
}

/** 目标对局列表;每对 2N 盘(黑白各 N)。 */
function buildJobs() {
  const jobs = [];
  const wanted = args.pairs
    ? args.pairs.split(",").map((s) => s.trim()).filter(Boolean)
    : Object.keys(args.pairGames);
  for (const pid of wanted) {
    const [a, b] = pid.split("-");
    if (!LEVELS.includes(a) || !LEVELS.includes(b)) throw new Error(`未知对:${pid}`);
    const n = args.pairGames[pid] ?? 20;
    if (n % 2) throw new Error(`${pid} 局数必须为偶数(黑白互换),收到 ${n}`);
    const half = n / 2;
    for (let k = 0; k < half; k++) {
      const seed = hashSeed(pid, k);
      for (const aIsBlack of [true, false]) {
        const tag = `${k.toString().padStart(3, "0")}${aIsBlack ? "aB" : "aW"}`;
        const file = path.join(GAMES, pid, `g${tag}.json`);
        fs.mkdirSync(path.dirname(file), { recursive: true });
        jobs.push({ pid, a, b, aIsBlack, seed, file });
      }
    }
  }
  return jobs;
}

function hashSeed(pid, k) {
  let h = 2166136261;
  for (const ch of `${pid}#${k}`) {
    h ^= ch.charCodeAt(0);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return (h >>> 0) || 1;
}

function existing(file) {
  try {
    const rec = JSON.parse(fs.readFileSync(file, "utf8"));
    return ["a", "b", "draw"].includes(rec.winner) ? rec : null;
  } catch { return null; }
}

function deadlineTs(text) {
  if (!text) return Infinity;
  const [hh, mm] = text.split(":").map(Number);
  const d = new Date();
  d.setHours(hh, mm, 0, 0);
  if (d.getTime() <= Date.now()) d.setDate(d.getDate() + 1);
  return d.getTime();
}

const allJobs = buildJobs();
const done = [];
for (const j of allJobs) {
  const rec = existing(j.file);
  if (rec) done.push(rec);
}
const todo = allJobs.filter((j) => !existing(j.file));
const started = Date.now();

// 已存在的比分,用于实时进度显示
const score = new Map();
function noteScore(rec) {
  const k = pairId(rec.a, rec.b);
  const s = score.get(k) ?? { a: 0, b: 0, d: 0, n: 0 };
  s.n++;
  if (rec.winner === "draw") s.d++;
  else if (rec.winner === "a") s.a++;
  else if (rec.winner === "b") s.b++;
  score.set(k, s);
}
for (const r of done) noteScore(r);

function fmt(ms) {
  const m = Math.round(ms / 60000);
  return m >= 60 ? `${Math.floor(m / 60)}h${String(m % 60).padStart(2, "0")}m` : `${m}m`;
}

fs.writeFileSync(path.join(OUT, "run_meta.json"), JSON.stringify({
  engine: "web/src/engine/api.ts EngineSession",
  budget: args.budget, depth: args.depth, vcf: args.vcf, nodes: args.nodes,
  openPlies: args.openPlies, workers: args.workers,
  variety: args.variety, uniformDepth: args.uniformDepth,
  pairGames: args.pairGames, totalGames: allJobs.length,
  started: new Date().toISOString(),
}, null, 1));

console.log(`对局总数 ${allJobs.length},已完成 ${done.length},待跑 ${todo.length}`);
console.log(`budget=${args.budget}s depth=${args.depth} vcf=${args.vcf} workers=${args.workers} openPlies=${args.openPlies}`);
console.log(`输出目${OUT}` + (args.deadline ? `,截止 ${args.deadline}` : ""));

const dl = deadlineTs(args.deadline);
let finished = 0;
let failed = 0;
let cursor = 0;

function launch() {
  if (cursor >= todo.length) return Promise.resolve();
  if (Date.now() > dl) { console.log("已到截止时间,不再开新局"); return Promise.resolve(); }
  const job = todo[cursor++];
  return new Promise((resolve) => {
    const w = new Worker(new URL("./elo-calibrate-worker.mjs", import.meta.url), {
      workerData: {
        a: job.a, b: job.b, aIsBlack: job.aIsBlack, seed: job.seed,
        openPlies: args.openPlies, depth: args.depth, budget: args.budget, vcf: args.vcf,
        nodes: args.nodes,
        variety: args.variety, uniformDepth: args.uniformDepth,
      },
    });
    const kill = setTimeout(() => {
      console.log(`[超时] ${job.pid} seed=${job.seed} — 终止该线程`);
      w.terminate();
    }, (args.budget * 260 + 600) * 1000);
    w.on("message", (rec) => {
      clearTimeout(kill);
      const out = { ...rec, pair: job.pid, budget: args.budget, depth: args.depth,
                    vcf: args.vcf, openPlies: args.openPlies, ts: new Date().toISOString() };
      fs.writeFileSync(job.file, JSON.stringify(out, null, 1));
      if (rec.winner === "error") { failed++; console.log(`[错误] ${job.pid} seed=${job.seed}: ${String(rec.error).split("\n")[0]}`); }
      else { noteScore(out); finished++; }
      const el = Date.now() - started;
      const rate = finished / (el / 60000);
      const s = score.get(job.pid) ?? { a: 0, b: 0, d: 0 };
      console.log(`[${done.length + finished}/${allJobs.length}] ${job.pid} ${rec.winner === "error" ? "ERR" : rec.winner === "a" ? `${job.a}胜` : rec.winner === "b" ? `${job.b}胜` : "和"} ` +
        `(${s.a}:${s.b}${s.d ? `:+${s.d}` : ""}) ${rec.turns}手 ${fmt(rec.msA + rec.msB)} | ` +
        `速度 ${rate.toFixed(2)} 局/分,剩 ${todo.length - finished} 局 ≈ ${fmt((todo.length - finished) / Math.max(rate, 0.01) * 60000)}`);
      resolve();
    });
    w.on("error", (err) => {
      clearTimeout(kill);
      failed++;
      console.log(`[线程崩溃] ${job.pid} seed=${job.seed}: ${err}`);
      fs.writeFileSync(job.file, JSON.stringify({ ...job, winner: "error", error: String(err) }, null, 1));
      resolve();
    });
    w.on("exit", (code) => { if (code !== 0 && code !== undefined) { clearTimeout(kill); resolve(); } });
  });
}

async function main() {
  const running = new Set();
  while (cursor < todo.length || running.size) {
    while (running.size < args.workers && cursor < todo.length && Date.now() < dl) {
      const p = launch().then(() => running.delete(p));
      running.add(p);
    }
    if (!running.size) break;
    await Promise.race(running);
  }
  // 汇总
  const games = [];
  for (const j of allJobs) {
    const rec = existing(j.file);
    if (rec) games.push(rec);
  }
  const summary = { budget: args.budget, depth: args.depth, vcf: args.vcf, openPlies: args.openPlies,
    nodes: args.nodes,
                    games: games.length, failed, elapsed_min: (Date.now() - started) / 60000,
                    score: Object.fromEntries([...score].map(([k, v]) => [k, v])) };
  fs.writeFileSync(path.join(OUT, "results.jsonl"), games.map((g) => JSON.stringify(g)).join("\n") + "\n");
  fs.writeFileSync(path.join(OUT, "summary.json"), JSON.stringify(summary, null, 1));
  console.log(`完成 ${games.length} 局(失败 ${failed}),用时 ${fmt(Date.now() - started)}`);
  process.exit(0);
}

main();
