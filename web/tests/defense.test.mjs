// 大师/传奇档防守机制（移植自 Python heuristic.py）的回归测试。
import test from "node:test";
import assert from "node:assert/strict";
import {
  DEFAULT_CONFIG, createPosition, applyStep, legalMoves,
} from "../src/engine/rules.ts";
import { profileFor } from "../src/engine/profiles.ts";
import { pregiftCover } from "../src/engine/candidates.ts";
import { searchBest, defenseStats } from "../src/engine/search.ts";

const replay = (moves) => {
  const p = createPosition();
  for (const m of moves) {
    if (p.gameOver) break;
    applyStep(p, m, DEFAULT_CONFIG);
  }
  return p;
};

// 实战第 33 步(records/web_elo_15s/games/advanced-hell/g000aB.json):
// 此局面对手可以借己方棋子投送成四,22 / 50 是应当封堵的射线空格。
const TACTICAL = [5, 6, 6, 126, 19, 19, 99, 6, 81, 6, 51, 5, 80, 5, 35, 3, 99,
  96, 5, 85, 5, 69, 5, 37, 4, 37, 127, 5, 21, 4, 20, 3, 63];

test("档位开关：只有大师/传奇启用防守机制", () => {
  assert.equal(profileFor("hell").defense, true);
  assert.equal(profileFor("legend").defense, true);
  assert.equal(profileFor("advanced").defense, false);
  assert.equal(profileFor("mid").defense, false);
  assert.equal(profileFor("beginner").defense, false);
});

test("先手礼防御：投送危险点后方的射线空格进入封堵覆盖", () => {
  const p = replay(TACTICAL);
  assert.equal(p.pending, -1, "测试局面应为完整回合的起点");
  const cover = pregiftCover(p, DEFAULT_CONFIG);
  assert.ok(cover.size > 0, "应当找到封堵格");
  assert.ok(cover.has(22) && cover.has(50), `封堵格应含 22/50，实际 ${[...cover.keys()]}`);
});

test("防守时间片：大师/传奇仍按时限返回合法着法，高级不跑过滤", () => {
  for (const tier of ["hell", "legend", "advanced"]) {
    const p = replay(TACTICAL);
    const before = defenseStats.calls;
    const r = searchBest(p, DEFAULT_CONFIG, { maxDepth: 8, timeMs: 600, engine: tier });
    assert.ok(legalMoves(replay(TACTICAL), DEFAULT_CONFIG).includes(r.move), `${tier} 着法合法`);
    assert.ok(r.elapsed < 2500, `${tier} 没超时过多: ${r.elapsed}ms`);
    if (tier === "advanced") assert.equal(defenseStats.calls, before, "高级不跑防守过滤");
  }
});
