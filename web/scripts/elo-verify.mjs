// 校验 elo-calibrate 产出的对局:把 moves 在本引擎里重放一遍,确认终局判定与记录一致。
// 用法: node web/scripts/elo-verify.mjs --out records/web_elo_12s [--sample 20]
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const args = { out: "records/web_elo_12s", sample: 0 };
for (let i = 2; i < process.argv.length; i++) {
  const a = process.argv[i];
  if (a === "--out") args.out = process.argv[++i];
  else if (a === "--sample") args.sample = Number(process.argv[++i]);
}

const apiUrl = pathToFileURL(path.resolve(import.meta.dirname, "..", "src", "engine", "api.ts")).href;
const { EngineSession } = await import(apiUrl);

const root = path.resolve(args.out, "games");
const files = [];
for (const pid of fs.readdirSync(root)) {
  for (const f of fs.readdirSync(path.join(root, pid))) {
    if (f.endsWith(".json")) files.push(path.join(root, pid, f));
  }
}
let list = files;
if (args.sample > 0 && args.sample < files.length) {
  list = [];
  const step = files.length / args.sample;
  for (let k = 0; k < args.sample; k++) list.push(files[Math.floor(k * step)]);
}

let ok = 0, bad = 0, skipped = 0;
for (const f of list) {
  const rec = JSON.parse(fs.readFileSync(f, "utf8"));
  if (!["a", "b", "draw"].includes(rec.winner)) { skipped++; continue; }
  const s = new EngineSession();
  s.handle({ cmd: "new" });
  let err = null;
  try {
    for (const mv of rec.moves) s.handle({ cmd: "move", idx: mv });
  } catch (e) { err = String(e); }
  if (err) { bad++; console.log(`非法棋谱 ${path.basename(f)}: ${err}`); continue; }
  const st = s.state();
  const aColor = rec.aIsBlack ? 1 : 2;
  const replay = !st.game_over ? "unfinished" : st.is_draw ? "draw" : (st.winner === aColor ? "a" : "b");
  if (replay === rec.winner) ok++;
  else {
    bad++;
    console.log(`不一致 ${path.basename(f)}: 记录 ${rec.winner} vs 重放 ${replay}(moves=${rec.moves.length})`);
  }
}
console.log(`重放 ${list.length} 局:一致 ${ok},不一致 ${bad},跳过 ${skipped}`);
