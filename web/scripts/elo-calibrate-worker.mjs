// elo-calibrate.mjs 的 worker:一个 worker 用「网页端语义」下完一整盘两个档位之间的对局。
//
// 关键点:直接调用 web/src/engine/api.ts 的 EngineSession(与浏览器里跑的是同一份代码),
// 每步用 ai({ depth, budget, engine, vcf, variety, uniformDepth }),因此 vcfSeconds /
// dynamic depth / 开局库 / 变化抽样 / 蒸馏 这些「网页端行为」全部原样生效;
// uniformDepth=true 时改为「统一有效深度」(与 Python heuristic_versions.choose 对齐),
// variety 传页面默认档位("mid")时对局带随机扰动。
import { parentPort, workerData } from "node:worker_threads";
import path from "node:path";
import { pathToFileURL } from "node:url";

const { a, b, aIsBlack, seed, openPlies, depth, budget, vcf,
  variety = "off", uniformDepth = false, nodes = 0 } = workerData;
const apiUrl = pathToFileURL(
  path.resolve(import.meta.dirname, "..", "src", "engine", "api.ts"),
).href;
const { EngineSession } = await import(apiUrl);

/** 与 mulberry32 同族的简单 LCG:同一 seed 必出同一开局。 */
function rngFrom(seed) {
  let s = (seed >>> 0) || 0x2545f491;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function play() {
  const rnd = rngFrom(seed);
  const SA = new EngineSession();
  const SB = new EngineSession();
  SA.reseed(seed || 1);
  SB.reseed((seed ^ 0x9e3779b9) >>> 0);
  for (const s of [SA, SB]) s.handle({ cmd: "new" });

  const aColor = aIsBlack ? 1 : 2;
  const moves = [];
  let msA = 0, msB = 0, nodesA = 0, nodesB = 0, turns = 0;
  let winner = "unfinished";

  try {
    // 同一条随机开局分别喂给两边,保证同 seed 的「黑白互换」两局开局一致。
    for (let k = 0; k < openPlies; k++) {
      const st = SA.state();
      if (st.game_over || !st.legal || !st.legal.length) break;
      const idx = st.legal[Math.floor(rnd() * st.legal.length)];
      SA.handle({ cmd: "move", idx });
      SB.handle({ cmd: "move", idx });
      moves.push(idx);
      turns++;
    }

    while (turns < 225) {
      const st = SA.state();
      if (st.game_over) break;
      const player = st.player;
      const aTurn = player === aColor;
      const actor = aTurn ? SA : SB;
      const other = aTurn ? SB : SA;
      const t0 = performance.now();
      const r = actor.ai({
        depth,
        budget,
        engine: aTurn ? a : b,
        vcf,
        variety,
        uniformDepth,
        ...(nodes ? { nodeLimit: nodes } : {}),
      });
      const ms = performance.now() - t0;
      if (aTurn) { msA += ms; nodesA += r.nodes | 0; } else { msB += ms; nodesB += r.nodes | 0; }
      if (!(r.move >= 0)) throw new Error(`engine ${aTurn ? a : b} returned no move`);
      other.handle({ cmd: "move", idx: r.move });
      moves.push(r.move);
      turns++;
      const st2 = actor.state();
      if (st2.game_over) {
        winner = st2.is_draw ? "draw" : (st2.winner === aColor ? "a" : "b");
        break;
      }
    }
    if (winner === "unfinished") {
      const st = SA.state();
      winner = st.game_over
        ? (st.is_draw ? "draw" : (st.winner === aColor ? "a" : "b"))
        : "draw";
    }
    return { a, b, aIsBlack, seed, winner, turns, moves, msA, msB, nodesA, nodesB };
  } catch (err) {
    return {
      a, b, aIsBlack, seed, winner: "error", turns, moves,
      msA, msB, nodesA, nodesB, error: String(err && err.stack ? err.stack : err),
    };
  }
}

parentPort.postMessage(play());
