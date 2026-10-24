/** Python `_candidates` 的逐步移植：线图、威胁路线、静态分与分类选取。 */
import { EMPTY, checkWinAt, legalMoves, other } from "./rules.ts";
import type { Config, Position } from "./rules.ts";
import { features } from "./eval.ts";
import type { Four } from "./eval.ts";
import { jumpThrough, killCaptures, killPlacementsConfig } from "./tactics.ts";

export interface Candidate {
  score: number;
  move: number;
}

const O_KILL = 1e5;
const O_DEFEND = 9e4;
const O_ROUTE = 8e4;
const O_BUILD4 = 150;
const O_BUILD3 = 40;
const O_BREAK4 = -250;
const O_BREAK3 = -80;
const O_OWN4 = -400;
const O_OWN3 = -50;
const O_FIGHT = 2;
const O_PUSH_OPP = 2;
const O_PUSH_CENTER = 5;
const O_PUSH_SELF = 1;
const O_BLOCK_SELF4 = 30;
const O_BLOCK_SELF3 = 10;
const O_GIFT_D2 = 3;
const O_GIFT_D3 = 8;
// Python `heuristic.py` 的 P1 预防性封堵常量。
const O_PREVENT = 2e3;
const PREGIFT_K = 3;
const PREGIFT_HUB_CAP = 48;
const PREGIFT_SIM_CAP = 96;

const DIRS8: ReadonlyArray<readonly [number, number]> = [
  [-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1],
];
// Python `_line_maps_combo_full` 的历史轴对：对角两条是交叉统计。
const PAIRS: ReadonlyArray<readonly [number, number]> = [[0, 1], [2, 3], [4, 5], [6, 7]];

interface Stats { lt: number; lt2: number; nb: number }

function statsAt(board: Int8Array, idx: number, color: number): Stats {
  const r = (idx / 15) | 0;
  const c = idx % 15;
  let first = 1, second = 1;
  for (const [d1, d2] of PAIRS) {
    const dr = DIRS8[d1][0], dc = DIRS8[d1][1];
    let f = 0, b = 0;
    let nr = r + dr, nc = c + dc;
    while (nr >= 0 && nr < 15 && nc >= 0 && nc < 15 && board[nr * 15 + nc] === color) {
      f++; nr += dr; nc += dc;
    }
    const dr2 = DIRS8[d2][0], dc2 = DIRS8[d2][1];
    nr = r - dr; nc = c - dc;
    while (nr >= 0 && nr < 15 && nc >= 0 && nc < 15 && board[nr * 15 + nc] === color) {
      b++; nr += dr2; nc += dc2;
    }
    const run = 1 + f + b;
    if (run >= first) { second = first; first = run; }
    else if (run > second) second = run;
  }
  let nb = 0;
  for (let dr = -1; dr <= 1; dr++) {
    for (let dc = -1; dc <= 1; dc++) {
      if (!dr && !dc) continue;
      const nr = r + dr, nc = c + dc;
      if (nr >= 0 && nr < 15 && nc >= 0 && nc < 15 && board[nr * 15 + nc] === color) nb++;
    }
  }
  return { lt: first, lt2: second, nb };
}

function center(idx: number): number {
  const r = (idx / 15) | 0;
  const c = idx % 15;
  return 1 - Math.max(Math.abs(r - 7), Math.abs(c - 7)) / 7;
}

interface Threats { ends: Set<number>; route: Set<number>; killable: boolean; oppFours: number | null }

export function threats(pos: Position, config: Config): Threats {
  const ends = new Set<number>();
  const route = new Set<number>();
  if (pos.turnCount < config.loss_start_turns) return { ends, route, killable: false, oppFours: null };
  const opp = other(pos.player);
  if (opp === 2 && pos.whiteTurns < config.white_restrict_turns) {
    return { ends, route, killable: false, oppFours: null };
  }
  const f = features(pos, config);
  const mine = pos.player === 1 ? f.black : f.white;
  const fours: Four[] = pos.player === 1 ? f.foursBlack : f.foursWhite;
  for (const four of fours) {
    for (const target of four.ends) {
      const tr = (target / 15) | 0;
      const tc = target % 15;
      for (const [dr, dc] of DIRS8) {
        const mid: number[] = [];
        let bridge = false;
        for (let k = 1; k < 15; k++) {
          const nr = tr + dr * k, nc = tc + dc * k;
          if (nr < 0 || nr >= 15 || nc < 0 || nc >= 15) break;
          const cell = nr * 15 + nc;
          const v = pos.board[cell];
          if (v === EMPTY) { mid.push(cell); continue; }
          if (v === pos.player && !four.run.has(cell)) bridge = true;
          break;
        }
        if (bridge) {
          ends.add(target);
          for (const cell of mid) route.add(cell);
        }
      }
    }
  }
  return { ends, route, killable: mine.killable4 > 0, oppFours: (opp === 1 ? f.foursBlack : f.foursWhite).length };
}

export function placeTargets(board: Int8Array, capture: number): number[] {
  const out: number[] = [];
  const r = (capture / 15) | 0;
  const c = capture % 15;
  for (const [dr, dc] of DIRS8) {
    for (let k = 1; k < 15; k++) {
      const nr = r + dr * k, nc = c + dc * k;
      if (nr < 0 || nr >= 15 || nc < 0 || nc >= 15) break;
      const cell = nr * 15 + nc;
      if (board[cell] !== EMPTY) break;
      out.push(cell);
    }
  }
  return out;
}

function placementScore(idx: number, self: number, stat: (i: number, c: number) => Stats): number {
  const opp = other(self);
  const s = stat(idx, opp);
  const m = stat(idx, self);
  let v = s.lt >= 5 ? O_KILL : s.lt === 4 ? O_BUILD4 : s.lt === 3 ? O_BUILD3 : 0;
  v += s.nb * O_PUSH_OPP + center(idx) * O_PUSH_CENTER - m.nb * O_PUSH_SELF;
  if (m.lt >= 4) v += O_BLOCK_SELF4;
  else if (m.lt === 3) v += O_BLOCK_SELF3;
  if (s.lt >= 2 && s.lt2 >= 2) v += O_GIFT_D2;
  if (s.lt >= 3 && s.lt2 >= 3) v += O_GIFT_D3;
  return v;
}

function nbAt(board: Int8Array, idx: number, color: number): number {
  const r = (idx / 15) | 0, c = idx % 15;
  let n = 0;
  for (let dr = -1; dr <= 1; dr++) {
    for (let dc = -1; dc <= 1; dc++) {
      if (!dr && !dc) continue;
      const nr = r + dr, nc = c + dc;
      if (nr >= 0 && nr < 15 && nc >= 0 && nc < 15 && board[nr * 15 + nc] === color) n++;
    }
  }
  return n;
}

function sortCandidates(out: Candidate[]): Candidate[] {
  // Python `sorted(out, reverse=True)`：分值降序，同分 move 降序。
  return out.sort((a, b) => b.score - a.score || b.move - a.move);
}

function selectTop<T>(items: T[], key: (x: T) => number, k: number): T[] {
  const n = Math.min(Math.max(0, k), items.length);
  if (n === 0) return [];
  const top: T[] = [];
  for (const x of items) {
    const kx = key(x);
    if (top.length === n && kx <= key(top[n - 1])) continue;
    let at = top.length;
    while (at > 0 && key(top[at - 1]) < kx) at--;
    top.splice(at, 0, x);
    if (top.length > n) top.pop();
  }
  return top;
}

function appendDiversity(out: Candidate[], legal: number[], self: number, stat: (i: number, c: number) => Stats): void {
  const present = new Set(out.map((x) => x.move));
  const pool = legal.filter((m) => !present.has(m));
  if (!pool.length) return;
  const scored = pool.map((m) => [stat(m, self).nb, m] as const);
  scored.sort((a, b) => b[0] - a[0]);
  for (const [nb, m] of scored.slice(0, 2)) out.push({ score: nb * 1e-3, move: m });
}

/** 先手礼防御用的临时局面：只保留 threats()/features() 需要的字段。 */
function probePosition(board: Int8Array, player: number, turnCount: number,
                       whiteTurns: number): Position {
  return {
    board, player, pending: -1, turnCount, whiteTurns,
    moveCount: 0, gameOver: false, loser: 0, isDraw: false, resigned: 0, hash: 0,
  };
}

/** 模拟“先手礼”：己方 c 离盘、t 落己方子后，是否立刻成五或造出必杀四。 */
function giftLethal(board: Int8Array, player: number, c: number, t: number,
                    turnCount: number, whiteTurns: number, config: Config): boolean {
  if (board[c] !== player || board[t] !== EMPTY) return false;
  const b2 = Int8Array.from(board);
  b2[c] = EMPTY;
  b2[t] = player;
  if (checkWinAt(b2, t, player)) return true;
  return threats(probePosition(b2, player, turnCount, whiteTurns), config).ends.size > 0;
}

/**
 * 预防性封堵（Python `_pregift_defense`）：对方如果在空点 t 落子、把己方某个
 * 棋子 c “投送”成四连，t 就是危险点；把 t 与该己方子之间射线上的空格记为覆盖格。
 */
export function pregiftCover(pos: Position, config: Config): Map<number, number> {
  const cover = new Map<number, number>();
  if (pos.pending >= 0) return cover;
  if (pos.turnCount < config.loss_start_turns) return cover;
  const board = pos.board;
  const self = pos.player;
  const hubs: number[] = [];
  for (let i = 0; i < 225; i++) {
    if (board[i] !== EMPTY) continue;
    const r = (i / 15) | 0;
    const c = i % 15;
    if (jumpThrough(board, r, c, self) >= 4) hubs.push(i);
  }
  if (!hubs.length) return cover;
  let sims = 0;
  for (const t of hubs.slice(0, PREGIFT_HUB_CAP)) {
    const tr = (t / 15) | 0;
    const tc = t % 15;
    for (const [dr, dc] of DIRS8) {
      const path: number[] = [];
      let stone = -1;
      for (let k = 1; k < 15; k++) {
        const nr = tr + dr * k, nc = tc + dc * k;
        if (nr < 0 || nr >= 15 || nc < 0 || nc >= 15) break;
        const cell = nr * 15 + nc;
        const v = board[cell];
        if (v === EMPTY) { path.push(cell); continue; }
        if (v === self && path.length) stone = cell;
        break;
      }
      if (stone < 0) continue;
      if (sims >= PREGIFT_SIM_CAP) return cover;
      sims++;
      if (!giftLethal(board, self, stone, t, pos.turnCount, pos.whiteTurns, config)) continue;
      for (const cell of path) cover.set(cell, (cover.get(cell) ?? 0) + 1);
    }
  }
  return cover;
}

/** Python `_append_pregift`：覆盖格按 (-w, -cell) 排序取前 3，补进根候选。 */
function appendPregift(out: Candidate[], cover: Map<number, number>, legal: number[]): void {
  if (!cover.size) return;
  const present = new Set(out.map((x) => x.move));
  const legalSet = new Set(legal);
  const items: Array<[number, number]> = [];
  for (const [cell, w] of cover) {
    if (!legalSet.has(cell) || present.has(cell)) continue;
    items.push([w, cell]);
  }
  if (!items.length) return;
  items.sort((a, b) => b[0] - a[0] || b[1] - a[1]);
  for (const [w, cell] of items.slice(0, PREGIFT_K)) {
    out.push({ score: O_PREVENT + Math.min(w, 99), move: cell });
  }
}

export interface CandidateOptions {
  /** 大师/传奇档：把预防性封堵的覆盖格补进根候选。 */
  pregift?: boolean;
}

export function candidates(
  pos: Position,
  config: Config,
  k: number,
  diversity = false,
  coreK: number | null = null,
  history: Float64Array | null = null,
  oppLast = -1,
  legalInput?: number[],
  opts: CandidateOptions = {},
): Candidate[] {
  const kk = coreK === null ? k : Math.min(coreK, k);
  const self = pos.player;
  const opp = other(self);
  const board = pos.board;
  const legal = legalInput ?? legalMoves(pos, config);
  if (!legal.length) return [];
  const statsB: Array<Stats | undefined> = new Array(225);
  const statsW: Array<Stats | undefined> = new Array(225);
  const stat = (idx: number, color: number): Stats => {
    const arr = color === 1 ? statsB : statsW;
    let v = arr[idx];
    if (v === undefined) { v = statsAt(board, idx, color); arr[idx] = v; }
    return v;
  };
  const out: Candidate[] = [];
  const th = threats(pos, config);
  const hist = (m: number): number => history ? Math.min(history[m], 10000) * 0.05 : 0;
  const finish = (): Candidate[] => {
    if (oppLast >= 0 && board[oppLast] === opp && legal.includes(oppLast)
        && !out.some((x) => x.move === oppLast)) {
      out.push({ score: 1e-6, move: oppLast });
    }
    return sortCandidates(out);
  };

  if (pos.pending >= 0) {
    const kills = killPlacementsConfig(pos, config.loss_start_turns, legal);
    if (kills.length) {
      return kills.map((m) => ({ score: O_KILL, move: m })).sort((a, b) => b.move - a.move);
    }
    for (const m of legal) {
      if (th.ends.has(m)) out.push({ score: O_DEFEND, move: m });
      else if (th.route.has(m)) out.push({ score: O_ROUTE, move: m });
    }
    const rest = legal.filter((m) => !th.ends.has(m) && !th.route.has(m));
    const scored = rest.map((m) => {
      const sc = placementScore(m, self, stat);
      const snb = stat(m, self).nb + stat(m, opp).nb;
      return { score: sc, key: sc + snb * 1e-4, move: m };
    });
    for (const x of selectTop(scored, (x) => x.key + hist(x.move), kk)) out.push({ score: x.score, move: x.move });
    if (diversity) appendDiversity(out, legal, self, stat);
    if (diversity && opts.pregift) appendPregift(out, pregiftCover(pos, config), legal);
    return finish();
  }

  const kc = killCaptures(pos, config, legal);
  const killCells = new Set(kc.map((x) => x[0]));
  if (killCells.size) {
    for (const m of killCells) out.push({ score: O_KILL, move: m });
    return finish();
  }
  if (th.ends.size || th.route.size) {
    for (const c of legal) {
      if (board[c] !== opp) continue;
      for (const t of placeTargets(board, c)) {
        if (th.ends.has(t)) { out.push({ score: O_DEFEND, move: c }); break; }
        if (th.route.has(t)) { out.push({ score: O_ROUTE, move: c }); break; }
      }
    }
    for (const t of legal) {
      if (board[t] === EMPTY && th.route.has(t)) out.push({ score: O_ROUTE - 1, move: t });
    }
  }
  if (th.ends.size && th.killable) return finish();

  const rest = legal.filter((m) => !killCells.has(m));
  const empties = rest.filter((m) => board[m] === EMPTY);
  const caps = rest.filter((m) => board[m] !== EMPTY);
  const emptyScored = empties.map((m) => {
    const s = stat(m, self);
    const oNb = nbAt(board, m, opp);
    const base = (s.lt >= 4 ? O_OWN4 : s.lt === 3 ? O_OWN3 : 0)
      + oNb * O_FIGHT - s.nb;
    return { score: base, key: base - m * 1e-9, move: m };
  });
  for (const x of selectTop(emptyScored, (x) => x.key + hist(x.move), kk)) out.push({ score: x.score, move: x.move });

  const capScored = caps.map((m) => {
    const s = stat(m, self);
    const o = stat(m, opp);
    let sc = s.lt >= 4 ? O_OWN4 : s.lt === 3 ? O_OWN3 : 0;
    sc += o.lt >= 4 ? O_BREAK4 : o.lt === 3 ? O_BREAK3 : 0;
    let best = -Infinity;
    for (const t of placeTargets(board, m)) {
      const v = placementScore(t, self, stat);
      if (v > best) best = v;
    }
    if (best > -Infinity) sc += best;
    return { score: sc, key: sc - m * 1e-9, move: m };
  });
  for (const x of selectTop(capScored, (x) => x.key + hist(x.move), kk)) out.push({ score: x.score, move: x.move });
  if (diversity) appendDiversity(out, legal, self, stat);
  if (diversity && opts.pregift) appendPregift(out, pregiftCover(pos, config), legal);
  return finish();
}
