// 探针:同一局面下网页端 15s 实际完成深度(SearchResult.depth)
import fs from "node:fs";
import path from "node:path";
import { EngineSession } from "../src/engine/api.ts";
import { searchBest } from "../src/engine/search.ts";

const dir = process.argv[2] ?? "records/web_elo_15s/games";
const budgetMs = Number(process.argv[3] ?? 12000);
const plies = [12, 20, 28];

function firstGame(root) {
  for (const pair of fs.readdirSync(root)) {
    const d = path.join(root, pair);
    if (!fs.statSync(d).isDirectory()) continue;
    for (const f of fs.readdirSync(d)) {
      if (f.endsWith(".json")) return path.join(d, f);
    }
  }
  throw new Error("no games");
}

const file = firstGame(dir);
const rec = JSON.parse(fs.readFileSync(file, "utf8"));
const sess = new EngineSession();
sess.handle({ cmd: "new" });

const out = [];
for (let i = 0; i < rec.moves.length; i++) {
  const ply = i + 1;
  sess.handle({ cmd: "move", idx: rec.moves[i] });
  if (!plies.includes(ply)) continue;
  for (const engine of ["advanced", "hell"]) {
    const r = searchBest(sess.pos, sess.config, {
      maxDepth: 16, timeMs: budgetMs, engine,
    });
    out.push({
      ply, engine, depth: r.depth, nodes: r.nodes,
      sec: +(r.elapsed / 1000).toFixed(2),
    });
  }
}
console.log(path.basename(file));
for (const o of out) {
  console.log(`ply ${o.ply} ${o.engine}: depth ${o.depth}, nodes ${o.nodes}, ${o.sec}s`);
}
