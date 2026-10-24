// 网页端引擎的进程内桥:给「Python 引擎 vs 网页引擎」的对拍当对端用。
//
// 协议:stdin 一行一个 JSON 请求,stdout 一行一个 JSON 应答。
//   {"cmd":"new"}                                  -> {"ok":true}
//   {"cmd":"move","idx":123}                       -> {"ok":true}        (对手着法喂进来)
//   {"cmd":"gen","engine":"advanced","nodes":7000} -> {"move":42,"nodes":7003,"elapsed":812.3}
//   {"cmd":"state"}                                -> {"player":1,"game_over":false,"winner":0,"legal":[...]}
//   {"cmd":"quit"}                                 -> 退出
//
// gen 里的 nodes 是「每步节点上限」,用来和 Python 引擎的节点预算对齐;
// budget 只是时间上限(VCF/迭代加深的兜底),真正生效的是 nodeLimit。
import path from "node:path";
import { pathToFileURL } from "node:url";
import readline from "node:readline";

const apiUrl = pathToFileURL(
  path.resolve(import.meta.dirname, "..", "src", "engine", "api.ts"),
).href;
const { EngineSession } = await import(apiUrl);

const session = new EngineSession();
session.handle({ cmd: "new" });

function reply(obj) {
  process.stdout.write(JSON.stringify(obj) + "\n");
}

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
rl.on("line", (line) => {
  const text = line.trim();
  if (!text) return;
  let req;
  try {
    req = JSON.parse(text);
  } catch (e) {
    reply({ error: `bad json: ${e.message}` });
    return;
  }
  try {
    if (req.cmd === "new") {
      session.handle({ cmd: "new" });
      reply({ ok: true });
    } else if (req.cmd === "move") {
      session.handle({ cmd: "move", idx: req.idx });
      reply({ ok: true });
    } else if (req.cmd === "state") {
      const st = session.state();
      reply({ player: st.player, game_over: st.game_over, is_draw: st.is_draw,
        winner: st.winner, legal: st.legal, turn_count: st.turn_count });
    } else if (req.cmd === "gen") {
      const t0 = performance.now();
      const r = session.ai({
        depth: req.depth ?? 8,
        budget: req.budget ?? 15,
        engine: req.engine,
        vcf: req.vcf === undefined ? true : Boolean(req.vcf),
        variety: req.variety ?? "off",
        uniformDepth: true,
        ...(req.nodes ? { nodeLimit: req.nodes } : {}),
      });
      reply({ move: r.move, nodes: r.nodes, elapsed: performance.now() - t0 });
    } else if (req.cmd === "hint") {
      const t0 = performance.now();
      const r = session.handle({
        cmd: "hint",
        depth: req.depth ?? 8,
        budget: req.budget ?? 3,
        engine: req.engine,
        vcf: req.vcf === undefined ? true : Boolean(req.vcf),
        k: req.k ?? 8,
        uniformDepth: true,
      });
      reply({ pos_v: r.pos_v, moves: r.moves, elapsed: performance.now() - t0 });
    } else if (req.cmd === "quit") {
      reply({ ok: true });
      process.exit(0);
    } else {
      reply({ error: `unknown cmd ${req.cmd}` });
    }
  } catch (e) {
    reply({ error: String(e && e.message ? e.message : e) });
  }
});
