import {
  BLACK,
  BOARD_PX,
  BOARD_SIZE,
  CELL,
  EMPTY,
  MARGIN,
} from "./types";
import type { GameState } from "./types";

const STONE_R = CELL / 2 - 3;
const DIRS: [number, number][] = [
  [-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1],
];
const COLOR_WOOD = "#deb280";
const COLOR_WOOD_DARK = "#a6723a";
const COLOR_LINE = "#4e3414";
const COLOR_DANGER_SELF = "rgba(220, 60, 60, 0.35)";
const COLOR_DANGER_OPP = "rgba(228, 170, 40, 0.35)";
const COLOR_TARGET = "#3cc85a";
const COLOR_LAST = "#eb4632";
const COLOR_ARROW = "#2d6ee6";
const COLOR_RAY = "rgba(140, 110, 230, 0.8)";
const ARROW_FLY_MS = 500;

export interface ArrowView {
  from: number;
  to: number;
  t0: number | null;
}

export interface BoardView {
  state: GameState;
  hover: { r: number; c: number } | null;
  held: { x: number; y: number } | null;
  showDanger: boolean;
  showNumbers: boolean;
  showRays: boolean;
  arrow: ArrowView | null;
  hoverTarget: number | null;
  confirmIdx: number | null;
}

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function reachableRays(board: number[], from: number): number[] {
  const r0 = Math.floor(from / BOARD_SIZE);
  const c0 = from % BOARD_SIZE;
  const out: number[] = [];
  for (const [dr, dc] of DIRS) {
    let r = r0 + dr;
    let c = c0 + dc;
    while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE) {
      const idx = r * BOARD_SIZE + c;
      if (board[idx] !== EMPTY) break;
      out.push(idx);
      r += dr;
      c += dc;
    }
  }
  return out;
}

function cellX(c: number) {
  return MARGIN + c * CELL;
}

function cellY(r: number) {
  return MARGIN + r * CELL;
}

export class BoardCanvas {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private bg: HTMLCanvasElement | null = null;
  private scale = 1;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("canvas 2d context 不可用");
    this.ctx = ctx;
    this.resize();
  }

  resize(): boolean {
    const parent = this.canvas.parentElement;
    if (!parent) return false;
    const size = Math.max(240, Math.floor(Math.min(parent.clientWidth, window.innerHeight * 0.82)));
    const dpr = window.devicePixelRatio || 1;
    const backing = Math.round(size * dpr);
    const changed = this.canvas.width !== backing || this.canvas.height !== backing;
    this.canvas.style.width = `${size}px`;
    this.canvas.style.height = `${size}px`;
    if (changed) {
      this.canvas.width = backing;
      this.canvas.height = backing;
      this.scale = size / BOARD_PX;
      this.bg = this.buildBackground(backing, this.scale * dpr);
    }
    return changed;
  }

  get pixelSize(): number {
    return this.canvas.width;
  }

  toVirtual(clientX: number, clientY: number): { x: number; y: number } {
    const rect = this.canvas.getBoundingClientRect();
    const k = BOARD_PX / rect.width;
    return { x: (clientX - rect.left) * k, y: (clientY - rect.top) * k };
  }

  pick(clientX: number, clientY: number): number | null {
    const { x, y } = this.toVirtual(clientX, clientY);
    const c = Math.round((x - MARGIN) / CELL);
    const r = Math.round((y - MARGIN) / CELL);
    if (r < 0 || r >= BOARD_SIZE || c < 0 || c >= BOARD_SIZE) return null;
    if (Math.abs(x - cellX(c)) > CELL * 0.45) return null;
    if (Math.abs(y - cellY(r)) > CELL * 0.45) return null;
    return r * BOARD_SIZE + c;
  }

  private buildBackground(backing: number, transform: number): HTMLCanvasElement {
    const bg = document.createElement("canvas");
    bg.width = backing;
    bg.height = backing;
    const g = bg.getContext("2d");
    if (!g) return bg;
    g.setTransform(transform, 0, 0, transform, 0, 0);
    g.fillStyle = COLOR_WOOD;
    g.fillRect(0, 0, BOARD_PX, BOARD_PX);
    const rng = mulberry32(7);
    const [wr, wg, wb] = [222, 178, 128];
    for (let i = 0; i < 220; i++) {
      const x = rng() * BOARD_PX;
      const y = rng() * BOARD_PX;
      const w = 30 + rng() * 130;
      const h = 1 + rng() * 2;
      const a = rng() * Math.PI;
      const c = Math.floor(rng() * 24) - 14;
      const r = Math.max(0, Math.min(255, wr + c));
      const gg = Math.max(0, Math.min(255, wg + c));
      const b = Math.max(0, Math.min(255, wb + c));
      g.save();
      g.translate(x, y);
      g.rotate(a);
      g.fillStyle = `rgba(${r}, ${gg}, ${b}, 0.18)`;
      g.fillRect(-w / 2, -h / 2, w, h);
      g.restore();
    }
    g.strokeStyle = COLOR_WOOD_DARK;
    g.lineWidth = 6;
    g.strokeRect(3, 3, BOARD_PX - 6, BOARD_PX - 6);
    g.strokeStyle = COLOR_LINE;
    g.lineWidth = 1;
    for (let i = 0; i < BOARD_SIZE; i++) {
      const p = MARGIN + i * CELL;
      g.beginPath();
      g.moveTo(MARGIN, p);
      g.lineTo(BOARD_PX - MARGIN, p);
      g.moveTo(p, MARGIN);
      g.lineTo(p, BOARD_PX - MARGIN);
      g.stroke();
    }
    g.fillStyle = COLOR_LINE;
    for (const [r, c] of [[3, 3], [3, 11], [11, 3], [11, 11], [7, 7]]) {
      g.beginPath();
      g.arc(cellX(c), cellY(r), 5, 0, Math.PI * 2);
      g.fill();
    }
    g.fillStyle = "#785028";
    g.font = "14px 'Microsoft YaHei', 'PingFang SC', sans-serif";
    g.textAlign = "center";
    g.textBaseline = "middle";
    for (let i = 0; i < BOARD_SIZE; i++) {
      const label = String.fromCharCode(65 + i);
      g.fillText(label, cellX(i), 18);
      g.fillText(label, cellX(i), BOARD_PX - 18);
      g.fillText(String(i), 18, cellY(i));
      g.fillText(String(i), BOARD_PX - 18, cellY(i));
    }
    return bg;
  }

  private stone(x: number, y: number, color: number, alpha = 1) {
    const g = this.ctx;
    const dark = color === BLACK;
    g.save();
    g.globalAlpha = alpha;
    const grad = g.createRadialGradient(
      x - STONE_R * 0.35, y - STONE_R * 0.35, STONE_R * 0.15,
      x, y, STONE_R,
    );
    if (dark) {
      grad.addColorStop(0, "#5a5a64");
      grad.addColorStop(0.6, "#22222a");
      grad.addColorStop(1, "#3c3c46");
    } else {
      grad.addColorStop(0, "#ffffff");
      grad.addColorStop(0.65, "#f2f0eb");
      grad.addColorStop(1, "#c9c4ba");
    }
    g.fillStyle = grad;
    g.beginPath();
    g.arc(x, y, STONE_R, 0, Math.PI * 2);
    g.fill();
    g.restore();
  }

  private arrow(from: number, to: number, progress: number | null, alpha: number) {
    const g = this.ctx;
    const fr = Math.floor(from / BOARD_SIZE);
    const fc = from % BOARD_SIZE;
    const tr = Math.floor(to / BOARD_SIZE);
    const tc = to % BOARD_SIZE;
    const x1 = cellX(fc);
    const y1 = cellY(fr);
    const x2 = cellX(tc);
    const y2 = cellY(tr);
    if (progress !== null && progress < 1) {
      const sx = x1 + (x2 - x1) * progress;
      const sy = y1 + (y2 - y1) * progress;
      const dist = Math.hypot(x2 - x1, y2 - y1);
      let ux = 0;
      let uy = 0;
      if (dist > 1) {
        ux = (x2 - x1) / dist;
        uy = (y2 - y1) / dist;
      }
      const tipX = sx + ux * (STONE_R + 3);
      const tipY = sy + uy * (STONE_R + 3);
      const baseX = sx - ux * (STONE_R - 2);
      const baseY = sy - uy * (STONE_R - 2);
      const px = -uy;
      const py = ux;
      g.fillStyle = COLOR_ARROW;
      g.beginPath();
      g.moveTo(tipX, tipY);
      g.lineTo(baseX + 9 * px, baseY + 9 * py);
      g.lineTo(baseX - 9 * px, baseY - 9 * py);
      g.closePath();
      g.fill();
      return { x: sx, y: sy };
    }
    const ang = Math.atan2(y2 - y1, x2 - x1);
    const hl = 15;
    const hw = 9;
    const hx = x2;
    const hy = y2;
    const bx = hx - hl * Math.cos(ang);
    const by = hy - hl * Math.sin(ang);
    const px = -Math.sin(ang);
    const py = Math.cos(ang);
    g.save();
    g.globalAlpha = alpha;
    g.fillStyle = COLOR_ARROW;
    g.beginPath();
    g.moveTo(hx, hy);
    g.lineTo(bx + hw * px, by + hw * py);
    g.lineTo(bx - hw * px, by - hw * py);
    g.closePath();
    g.fill();
    g.strokeStyle = COLOR_ARROW;
    g.lineWidth = 4;
    g.beginPath();
    g.moveTo(x1, y1);
    g.lineTo(bx, by);
    g.stroke();
    g.restore();
    return null;
  }

  draw(view: BoardView, now: number) {
    const g = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    g.setTransform(1, 0, 0, 1, 0, 0);
    if (this.bg) g.drawImage(this.bg, 0, 0);
    const s = this.scale * dpr;
    g.setTransform(s, 0, 0, s, 0, 0);
    const st = view.state;
    if (view.showDanger) {
      for (let i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
        if (st.danger_me[i]) {
          g.fillStyle = COLOR_DANGER_SELF;
          g.beginPath();
          g.arc(cellX(i % BOARD_SIZE), cellY(Math.floor(i / BOARD_SIZE)), STONE_R - 6, 0, Math.PI * 2);
          g.fill();
        } else if (st.danger_opp[i]) {
          g.fillStyle = COLOR_DANGER_OPP;
          g.beginPath();
          g.arc(cellX(i % BOARD_SIZE), cellY(Math.floor(i / BOARD_SIZE)), STONE_R - 6, 0, Math.PI * 2);
          g.fill();
        }
      }
    }
    if (st.pending >= 0) {
      for (const i of st.legal) {
        const x = cellX(i % BOARD_SIZE);
        const y = cellY(Math.floor(i / BOARD_SIZE));
        g.strokeStyle = COLOR_TARGET;
        g.lineWidth = 2;
        g.beginPath();
        g.arc(x, y, STONE_R - 4, 0, Math.PI * 2);
        g.stroke();
        g.fillStyle = "rgba(60, 200, 90, 0.47)";
        g.beginPath();
        g.arc(x, y, 7, 0, Math.PI * 2);
        g.fill();
      }
      if (view.held) {
        this.stone(view.held.x, view.held.y, st.pending_color, 0.6);
      }
    }
    let hoverRays: { src: number; targets: number[] } | null = null;
    if (view.showRays && view.hover && st.pending < 0) {
      const idx = view.hover.r * BOARD_SIZE + view.hover.c;
      if (st.board[idx] !== EMPTY) {
        hoverRays = { src: idx, targets: reachableRays(st.board, idx) };
      }
    }
    if (hoverRays) {
      const x1 = cellX(hoverRays.src % BOARD_SIZE);
      const y1 = cellY(Math.floor(hoverRays.src / BOARD_SIZE));
      g.strokeStyle = COLOR_RAY;
      g.lineWidth = 2;
      for (const i of hoverRays.targets) {
        const x2 = cellX(i % BOARD_SIZE);
        const y2 = cellY(Math.floor(i / BOARD_SIZE));
        const dx = x2 - x1;
        const dy = y2 - y1;
        const dist = Math.hypot(dx, dy);
        if (dist > 1) {
          const ux = dx / dist;
          const uy = dy / dist;
          g.beginPath();
          g.moveTo(x1 + ux * (STONE_R + 2), y1 + uy * (STONE_R + 2));
          g.lineTo(x2 - ux * 11, y2 - uy * 11);
          g.stroke();
        }
        g.fillStyle = "rgba(140, 110, 230, 0.8)";
        g.beginPath();
        g.arc(x2, y2, 7, 0, Math.PI * 2);
        g.fill();
        g.strokeStyle = "rgba(140, 110, 230, 0.31)";
        g.lineWidth = 1;
        g.beginPath();
        g.arc(x2, y2, 11, 0, Math.PI * 2);
        g.stroke();
      }
      g.strokeStyle = "rgba(140, 110, 230, 0.67)";
      g.lineWidth = 2;
      g.beginPath();
      g.arc(x1, y1, STONE_R + 2, 0, Math.PI * 2);
      g.stroke();
    }
    const flying = view.arrow && view.arrow.t0 !== null && now - view.arrow.t0 < ARROW_FLY_MS
      ? view.arrow : null;
    const flyProgress = flying && flying.t0 !== null ? (now - flying.t0) / ARROW_FLY_MS : null;
    let flyStone: { x: number; y: number } | null = null;
    if (flying && flyProgress !== null) {
      flyStone = this.arrow(flying.from, flying.to, flyProgress, 1);
    }
    const last = st.moves.length ? st.moves[st.moves.length - 1] : -1;
    for (let i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
      const v = st.board[i];
      if (v === EMPTY) continue;
      if (flying && i === flying.to) continue;
      const x = cellX(i % BOARD_SIZE);
      const y = cellY(Math.floor(i / BOARD_SIZE));
      const isConfirm = i === view.confirmIdx;
      this.stone(x, y, v, isConfirm ? 0.55 : 1);
      if (isConfirm) {                          // 占领目标:半透明+绿环示意将被拿起
        g.strokeStyle = COLOR_TARGET;
        g.lineWidth = 2;
        g.beginPath();
        g.arc(x, y, STONE_R + 2, 0, Math.PI * 2);
        g.stroke();
      }
      if (i === last) {
        g.strokeStyle = COLOR_LAST;
        g.lineWidth = 3;
        g.beginPath();
        g.arc(x, y, STONE_R - 1, 0, Math.PI * 2);
        g.stroke();
      }
      if (view.showNumbers && st.numbers[i] > 0) {
        g.fillStyle = v === BLACK ? "#ffffff" : "#281c10";
        g.font = "14px 'Microsoft YaHei', 'PingFang SC', sans-serif";
        g.textAlign = "center";
        g.textBaseline = "middle";
        g.fillText(String(st.numbers[i]), x, y);
      }
    }
    if (view.confirmIdx !== null && st.board[view.confirmIdx] === EMPTY) {
      // 落子确认虚影:空位=半透明棋子(安置待定时为手中棋子颜色)+绿环
      const x = cellX(view.confirmIdx % BOARD_SIZE);
      const y = cellY(Math.floor(view.confirmIdx / BOARD_SIZE));
      this.stone(x, y, st.pending >= 0 ? st.pending_color : st.player, 0.55);
      g.strokeStyle = COLOR_TARGET;
      g.lineWidth = 2;
      g.beginPath();
      g.arc(x, y, STONE_R + 2, 0, Math.PI * 2);
      g.stroke();
    }
    if (flying && flyStone) {
      this.stone(flyStone.x, flyStone.y, st.board[flying.to]);
      g.strokeStyle = COLOR_ARROW;
      g.lineWidth = 2;
      g.beginPath();
      g.arc(flyStone.x, flyStone.y, STONE_R + 3, 0, Math.PI * 2);
      g.stroke();
    }
    if (view.arrow && !flying) {
      this.arrow(view.arrow.from, view.arrow.to, null, 1);
    }
    if (view.hoverTarget !== null && st.pending >= 0) {
      const from = st.pending;
      const fr = Math.floor(from / BOARD_SIZE);
      const fc = from % BOARD_SIZE;
      const tr = Math.floor(view.hoverTarget / BOARD_SIZE);
      const tc = view.hoverTarget % BOARD_SIZE;
      const x1 = cellX(fc);
      const y1 = cellY(fr);
      const x2 = cellX(tc);
      const y2 = cellY(tr);
      const ang = Math.atan2(y2 - y1, x2 - x1);
      const bx = x2 - 15 * Math.cos(ang);
      const by = y2 - 15 * Math.sin(ang);
      const px = -Math.sin(ang);
      const py = Math.cos(ang);
      g.save();
      g.globalAlpha = 0.59;
      g.fillStyle = COLOR_ARROW;
      g.beginPath();
      g.moveTo(x2, y2);
      g.lineTo(bx + 9 * px, by + 9 * py);
      g.lineTo(bx - 9 * px, by - 9 * py);
      g.closePath();
      g.fill();
      g.strokeStyle = COLOR_ARROW;
      g.lineWidth = 4;
      g.beginPath();
      g.moveTo(x1, y1);
      g.lineTo(bx, by);
      g.stroke();
      g.restore();
    }
    if (view.hover && st.pending < 0 && !hoverRays) {
      g.strokeStyle = "rgba(60, 200, 90, 0.43)";
      g.lineWidth = 2;
      g.beginPath();
      g.arc(cellX(view.hover.c), cellY(view.hover.r), STONE_R - 4, 0, Math.PI * 2);
      g.stroke();
    }
  }
}
