// 探针:导出局面并把两端 evaluate 摆在一起比较
import fs from "node:fs";
import path from "node:path";
import { EngineSession } from "../src/engine/api.ts";
import { evaluate } from "../src/engine/eval.ts";

const root = process.argv[2] ?? "records/web_elo_15s/games";
const plies = [12, 20, 28, 36];
const outPath = process.argv[3] ?? "records/web_vs_py_20261023/positions.json";

function firstGame(r) {
  for (const pair of fs.readdirSync(r)) {
    const d = path.join(r, pair);
    if (!fs.statSync(d).isDirectory()) continue;
    for (const f of fs.readdirSync(d)) if (f.endsWith(".json")) return path.join(d, f);
  }
  throw new Error("no games");
}

const file = firstGame(root);
const rec = JSON.parse(fs.readFileSync(file, "utf8"));
const sess = new EngineSession();
sess.handle({ cmd: "new" });
const rows = [];
for (let i = 0; i < rec.moves.length; i++) {
  sess.handle({ cmd: "move", idx: rec.moves[i] });
  if (!plies.includes(i + 1)) continue;
  const pos = sess.pos;
  const ev = evaluate(pos, sess.config);
  rows.push({
    ply: i + 1,
    board: Array.from(pos.board),
    player: pos.player,
    pending: pos.pending,
    turnCount: pos.turnCount,
    whiteTurns: pos.whiteTurns,
    webEval: ev,
  });
}
fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, JSON.stringify(rows, null, 1));
console.log(`wrote ${outPath} (${rows.length} positions)`);
for (const r of rows) {
  console.log(`ply ${r.ply} player ${r.player} pending ${r.pending} webEval ${r.webEval.toFixed(3)}`);
}
