import { BoardCanvas } from "./board";
import type { ArrowView, BoardView } from "./board";
import { Engine } from "./engine";
import { Sounds } from "./audio";
import { BLACK, BOARD_SIZE, EMPTY, WHITE } from "./types";
import type { AiResult, GameState } from "./types";

const TIERS = [
  { name: "简单", depth: 2, budget: 1.5 },
  { name: "中等", depth: 4, budget: 5 },
  { name: "困难", depth: 8, budget: 12 },
] as const;

const VERSION_LABELS: Record<string, string> = {
  beginner: "初级", mid: "中级", advanced: "高级",
};

const MODE_NAMES = ["人执黑", "人执白", "人人对战", "机机观战"];

function $(id: string): HTMLElement {
  const el = document.getElementById(id);
  if (!el) throw new Error(`缺少元素 #${id}`);
  return el;
}

function fmtClock(seconds: number): string {
  const s = Math.max(0, seconds);
  const m = Math.floor(s / 60);
  const rest = s - m * 60;
  return `${m}:${rest.toFixed(1).padStart(4, "0")}`;
}

function errText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export class App {
  private engine: Engine;
  private sounds = new Sounds();
  private board: BoardCanvas;
  private timeline: HTMLCanvasElement;
  private timelineCtx: CanvasRenderingContext2D;

  private state: GameState | null = null;
  private mode = 0;
  private epoch = 0;
  private ready = false;

  private thinking = false;
  private deadline: number | null = null;
  private nodes = 0;

  private tier = 2;
  private limitMode: "none" | "ai" = "none";
  private limitSeconds = 12;
  private hver: "beginner" | "mid" | "advanced" = "advanced";
  private hvcf = true;

  private showDanger = false;
  private showNumbers = false;
  private showArrows = true;
  private showRays = true;
  private confirmMove = true;
  private confirmIdx: number | null = null;

  private arrow: ArrowView | null = null;
  private hover: { r: number; c: number } | null = null;
  private pointer: { x: number; y: number } | null = null;

  private review = false;
  private recordMoves: number[] = [];
  private variation: number[] = [];
  private pos = 0;
  private divergence: number | null = null;
  private reviewResult = "未完";

  private statusText = "";
  private statusUntil = 0;
  private needsDraw = true;

  constructor(base: string) {
    this.board = new BoardCanvas($("board") as HTMLCanvasElement);
    this.timeline = $("timeline") as HTMLCanvasElement;
    const tctx = this.timeline.getContext("2d");
    if (!tctx) throw new Error("timeline 2d context 不可用");
    this.timelineCtx = tctx;
    this.buildButtons();
    this.bindEvents();
    this.engine = new Engine(base);
    this.engine.onStatus = (s) => {
      const texts: Record<string, string> = {
        core: "正在加载 Pyodide 运行时…",
        numpy: "正在加载 numpy…",
        engine: "正在加载规则引擎…",
        fallback: "资源源不可用，切换备用源…",
      };
      $("loading-text").textContent = texts[s.stage] ?? "正在加载…";
      $("loading-detail").textContent = s.detail ?? "";
    };
    this.engine.onProgress = (nodes) => {
      this.nodes = nodes;
    };
    this.engine.ready
      .then(() => {
        this.ready = true;
        $("loading").classList.add("hidden");
        this.newGame();
      })
      .catch((error) => {
        $("loading-text").textContent = "引擎加载失败";
        $("loading-detail").textContent = errText(error);
      });
    window.requestAnimationFrame(this.tick);
    window.setInterval(() => this.updateClock(), 100);
  }

  private buildButtons() {
    const buttons = $("buttons");
    const actions: [string, string][] = [
      ["settings", "设定"],
      ["mode-0", MODE_NAMES[0]],
      ["mode-1", MODE_NAMES[1]],
      ["mode-2", MODE_NAMES[2]],
      ["mode-3", MODE_NAMES[3]],
      ["new", "新局"],
      ["undo", "悔棋"],
      ["save", "保存棋谱"],
      ["load", "读取棋谱"],
    ];
    for (const [action, label] of actions) {
      const btn = document.createElement("button");
      btn.textContent = label;
      btn.dataset.action = action;
      buttons.appendChild(btn);
    }
    buttons.addEventListener("click", (event) => {
      const target = (event.target as HTMLElement).closest("button");
      if (!target?.dataset.action) return;
      const action = target.dataset.action;
      if (action.startsWith("mode-")) {
        this.mode = Number(action.slice(5));
        this.newGame();
      } else if (action === "settings") {
        this.openSettings();
      } else if (action === "new") {
        this.newGame();
      } else if (action === "undo") {
        void this.undo();
      } else if (action === "save") {
        void this.saveRecord();
      } else if (action === "load") {
        ($("file-input") as HTMLInputElement).click();
      }
    });

    const toggles = $("toggle-btns");
    const toggleDefs: [string, string][] = [
      ["危险提示", "danger"],
      ["手数显示", "numbers"],
      ["箭头指示", "arrows"],
      ["辅助射线", "rays"],
      ["落子确认", "confirm"],
      ["音效", "sound"],
    ];
    for (const [label, key] of toggleDefs) {
      const btn = document.createElement("button");
      btn.textContent = label;
      btn.dataset.toggle = key;
      toggles.appendChild(btn);
    }
    toggles.addEventListener("click", (event) => {
      const target = (event.target as HTMLElement).closest("button");
      const key = target?.dataset.toggle;
      if (!key) return;
      if (key === "danger") this.showDanger = !this.showDanger;
      if (key === "numbers") this.showNumbers = !this.showNumbers;
      if (key === "arrows") {
        this.showArrows = !this.showArrows;
        if (this.showArrows) this.rebuildArrowStatic();
        else this.arrow = null;
      }
      if (key === "rays") this.showRays = !this.showRays;
      if (key === "confirm") {
        this.confirmMove = !this.confirmMove;
        if (!this.confirmMove) this.confirmIdx = null;
      }
      if (key === "sound") this.sounds.enabled = !this.sounds.enabled;
      this.syncPanel();
      this.needsDraw = true;
    });

    const reviewButtons = $("review-buttons");
    const reviewDefs: [string, () => void][] = [
      ["开局", () => void this.reviewJump(0)],
      ["上一手", () => void this.reviewJump(this.pos - 1)],
      ["下一手", () => void this.reviewJump(this.pos + 1)],
      ["终局", () => void this.reviewJump(this.variation.length)],
      ["原谱", () => this.reviewReset()],
    ];
    for (const [label, fn] of reviewDefs) {
      const btn = document.createElement("button");
      btn.textContent = label;
      btn.addEventListener("click", fn);
      reviewButtons.appendChild(btn);
    }

    const tierRadios = $("tier-radios");
    TIERS.forEach((tier, i) => {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "radio";
      input.name = "tier";
      input.value = String(i);
      input.checked = i === this.tier;
      label.append(input, document.createTextNode(`${tier.name}(深度 ${tier.depth})`));
      tierRadios.appendChild(label);
    });
    const hverRadios = $("hver-radios");
    for (const [value, text] of [["beginner", "初级(经典)"], ["mid", "中级(强化)"], ["advanced", "高级(+蒸馏)"]] as const) {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "radio";
      input.name = "hver";
      input.value = value;
      input.checked = value === this.hver;
      input.addEventListener("change", () => this.syncVcfEnabled());
      label.append(input, document.createTextNode(text));
      hverRadios.appendChild(label);
    }
    const hvcfRadios = $("hvcf-radios");
    for (const [value, text] of [["on", "开(强制杀链)"], ["off", "关"]] as const) {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "radio";
      input.name = "hvcf";
      input.value = value;
      input.checked = (value === "on") === this.hvcf;
      label.append(input, document.createTextNode(text));
      hvcfRadios.appendChild(label);
    }
    this.syncVcfEnabled();
    const limitRadios = $("limit-radios");
    for (const [value, text] of [["none", "不限时"], ["ai", "AI 读秒"]] as const) {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "radio";
      input.name = "limit";
      input.value = value;
      input.checked = value === this.limitMode;
      label.append(input, document.createTextNode(text));
      limitRadios.appendChild(label);
    }
    ($("limit-seconds") as HTMLInputElement).value = String(this.limitSeconds);
    $("settings-apply").addEventListener("click", () => this.applySettings());
    $("settings-cancel").addEventListener("click", () => this.closeSettings());
  }

  private bindEvents() {
    const canvas = $("board") as HTMLCanvasElement;
    canvas.addEventListener("pointermove", (event) => {
      const idx = this.board.pick(event.clientX, event.clientY);
      this.hover = idx === null ? null : { r: Math.floor(idx / BOARD_SIZE), c: idx % BOARD_SIZE };
      this.pointer = this.board.toVirtual(event.clientX, event.clientY);
      this.needsDraw = true;
    });
    canvas.addEventListener("pointerleave", () => {
      this.hover = null;
      this.pointer = null;
      this.needsDraw = true;
    });
    canvas.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      const idx = this.board.pick(event.clientX, event.clientY);
      if (idx !== null) void this.onBoardClick(idx);
    });
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    this.timeline.addEventListener("pointerdown", (event) => {
      const seek = (clientX: number) => {
        const rect = this.timeline.getBoundingClientRect();
        const t = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
        void this.reviewJump(Math.round(t * this.variation.length));
      };
      seek(event.clientX);
      const move = (e: PointerEvent) => seek(e.clientX);
      const up = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    });
    window.addEventListener("resize", () => {
      if (this.board.resize()) this.needsDraw = true;
    });
    window.addEventListener("keydown", (event) => this.onKey(event));
    const file = $("file-input") as HTMLInputElement;
    file.addEventListener("change", () => {
      const selected = file.files?.[0];
      if (selected) void this.importRecord(selected);
      file.value = "";
    });
    const settings = $("settings");
    settings.addEventListener("click", (event) => {
      if (event.target === settings) this.closeSettings();
    });
  }

  private onKey(event: KeyboardEvent) {
    if (event.target instanceof HTMLInputElement) return;
    const key = event.key.toLowerCase();
    if (key === "r") this.newGame();
    else if (key === "u") void this.undo();
    else if (key === "d") this.showDanger = !this.showDanger;
    else if (key === "n") this.showNumbers = !this.showNumbers;
    else if (key === "a") {
      this.showArrows = !this.showArrows;
      if (this.showArrows) this.rebuildArrowStatic();
      else this.arrow = null;
    } else if (key === "h") this.showRays = !this.showRays;
    else if (key === "c") {
      this.confirmMove = !this.confirmMove;
      if (!this.confirmMove) this.confirmIdx = null;
    } else if (key === "m") this.sounds.enabled = !this.sounds.enabled;
    else if (key === "s") void this.saveRecord();
    else if (key === "l") ($("file-input") as HTMLInputElement).click();
    else if (key === "e") this.openSettings();
    else if (key === "escape") {
      if (!$("settings").classList.contains("hidden")) this.closeSettings();
    } else if (["1", "2", "3", "4"].includes(key)) {
      this.mode = Number(key) - 1;
      this.newGame();
    } else if (this.review && key === "arrowleft") void this.reviewJump(this.pos - 1);
    else if (this.review && key === "arrowright") void this.reviewJump(this.pos + 1);
    else return;
    this.syncPanel();
    this.needsDraw = true;
  }

  private isAiTurn(): boolean {
    if (!this.state) return false;
    if (this.mode === 0) return this.state.player === WHITE;
    if (this.mode === 1) return this.state.player === BLACK;
    if (this.mode === 3) return true;
    return false;
  }

  private newGame() {
    if (!this.ready) return;
    this.epoch++;
    this.thinking = false;
    this.deadline = null;
    this.nodes = 0;
    this.review = false;
    this.recordMoves = [];
    this.variation = [];
    this.pos = 0;
    this.divergence = null;
    this.arrow = null;
    this.confirmIdx = null;
    this.hideOverlay();
    this.engine
      .call<{ state: GameState }>("new", {})
      .then((res) => {
        this.state = res.state;
        this.afterStateChange();
      })
      .catch((error) => this.setStatus(`新局失败: ${errText(error)}`));
  }

  private afterStateChange() {
    this.syncPanel();
    this.needsDraw = true;
    if (this.state?.stuck && !this.state.game_over && !this.review) {
      this.engine
        .call<{ state: GameState }>("stuck", {})
        .then((res) => {
          this.state = res.state;
          this.syncPanel();
          this.needsDraw = true;
        })
        .catch(() => undefined);
    }
    this.maybeRunAi();
  }

  private maybeRunAi() {
    if (!this.ready || this.review || !this.state || this.state.game_over) return;
    if (!this.isAiTurn() || this.thinking) return;
    void this.runAi();
  }

  private async runAi() {
    const epoch = this.epoch;
    const tier = TIERS[this.tier];
    const budget = this.limitMode === "ai" ? this.limitSeconds : tier.budget;
    const prev = this.state;
    if (!prev) return;
    this.thinking = true;
    this.nodes = 0;
    this.deadline = this.limitMode === "ai" ? performance.now() + budget * 1000 : null;
    this.syncPanel();
    try {
      const res = await this.engine.call<AiResult>("ai", {
        depth: tier.depth,
        budget,
        engine: this.hver,
        vcf: this.hvcf,
      });
      if (epoch !== this.epoch) return;
      this.applyMove(prev, res.state, res.move);
    } catch (error) {
      if (epoch === this.epoch) this.setStatus(`AI 出错: ${errText(error)}`);
    } finally {
      if (epoch === this.epoch) {
        this.thinking = false;
        this.deadline = null;
        this.syncPanel();
      }
    }
    if (epoch !== this.epoch) return;
    if (this.mode === 3 && this.state && !this.state.game_over && !this.review) {
      window.setTimeout(() => this.maybeRunAi(), 350);
    } else {
      this.maybeRunAi();
    }
  }

  private applyMove(prev: GameState, next: GameState, moveIdx: number | null) {
    this.confirmIdx = null;
    if (prev.pending >= 0 && moveIdx !== null) {
      this.arrow = { from: prev.pending, to: moveIdx, t0: performance.now() };
    } else if (next.pending < 0) {
      this.arrow = null;
    }
    if (moveIdx !== null) {
      if (prev.pending >= 0 || prev.board[moveIdx] !== EMPTY) this.sounds.move();
      else this.sounds.place();
    }
    this.state = next;
    if (next.game_over) {
      this.sounds.win();
      this.enterReview(true);
    } else {
      this.afterStateChange();
    }
  }

  private confirmGate(idx: number): boolean {
    if (!this.confirmMove) {
      this.confirmIdx = null;
      return true;
    }
    if (this.confirmIdx === idx) {
      this.confirmIdx = null;
      return true;
    }
    this.confirmIdx = idx;
    this.needsDraw = true;
    this.syncPanel();
    return false;
  }

  private clearGhost() {
    if (this.confirmIdx === null) return;
    this.confirmIdx = null;
    this.needsDraw = true;
    this.syncPanel();
  }

  private async onBoardClick(idx: number) {
    if (!this.ready || !this.state) return;
    if (this.review) {
      await this.reviewTry(idx);
      return;
    }
    if (this.state.game_over || this.thinking || this.isAiTurn()) return;
    if (!this.state.legal.includes(idx)) {
      this.clearGhost();
      return;
    }
    if (!this.confirmGate(idx)) return;
    const prev = this.state;
    const epoch = ++this.epoch;
    try {
      const res = await this.engine.call<{ state: GameState }>("move", { idx });
      if (epoch !== this.epoch) return;
      this.applyMove(prev, res.state, idx);
    } catch (error) {
      this.setStatus(`落子失败: ${errText(error)}`);
    }
  }

  private async undo() {
    if (!this.ready || !this.state) return;
    if (this.review) {
      await this.reviewJump(this.pos - 1);
      return;
    }
    if (!this.state.moves.length) return;
    const epoch = ++this.epoch;
    this.thinking = false;
    this.deadline = null;
    this.confirmIdx = null;
    try {
      let res = await this.engine.call<{ state: GameState }>("undo", {});
      if (epoch !== this.epoch) return;
      this.state = res.state;
      while (this.state.moves.length > 0 && this.isAiTurn() && !this.state.game_over) {
        res = await this.engine.call<{ state: GameState }>("undo", {});
        if (epoch !== this.epoch) return;
        this.state = res.state;
      }
      this.arrow = null;
      this.rebuildArrowStatic();
      this.hideOverlay();
      this.afterStateChange();
    } catch (error) {
      this.setStatus(`悔棋失败: ${errText(error)}`);
    }
  }

  private enterReview(overlay: boolean) {
    if (!this.state) return;
    this.review = true;
    this.recordMoves = [...this.state.moves];
    this.variation = [...this.state.moves];
    this.pos = this.variation.length;
    this.divergence = null;
    this.reviewResult = this.state.result;
    this.rebuildArrowStatic();
    if (overlay) {
      const text = this.state.is_draw
        ? "和棋"
        : this.state.loser === BLACK
          ? "白棋获胜"
          : "黑棋获胜";
      const el = $("result-overlay");
      el.textContent = text;
      el.classList.remove("hidden");
      el.classList.add("show");
      window.setTimeout(() => el.classList.remove("show"), 2800);
      window.setTimeout(() => el.classList.add("hidden"), 3700);
    }
    this.afterStateChange();
  }

  private hideOverlay() {
    const el = $("result-overlay");
    el.classList.remove("show");
    el.classList.add("hidden");
  }

  private rebuildArrowStatic() {
    if (!this.state) return;
    const { kinds, moves, pending } = this.state;
    const n = kinds.length;
    if (this.showArrows && n >= 2 && kinds[n - 1] === "place2") {
      this.arrow = { from: moves[n - 2], to: moves[n - 1], t0: null };
    } else if (pending < 0) {
      this.arrow = null;
    }
  }

  private async reviewJump(k: number) {
    if (!this.ready || !this.state) return;
    const target = Math.max(0, Math.min(k, this.variation.length));
    if (target === this.pos && this.review) {
      this.needsDraw = true;
      return;
    }
    this.pos = target;
    this.confirmIdx = null;
    const epoch = ++this.epoch;
    this.thinking = false;
    this.deadline = null;
    try {
      const res = await this.engine.call<{ state: GameState; pos: number }>("goto", {
        moves: this.variation,
        pos: target,
      });
      if (epoch !== this.epoch) return;
      this.state = res.state;
      this.hideOverlay();
      this.rebuildArrowStatic();
      this.afterStateChange();
    } catch (error) {
      this.setStatus(`复盘跳转失败: ${errText(error)}`);
    }
  }

  private async reviewTry(idx: number) {
    if (!this.state || this.state.game_over) return;
    if (!this.state.legal.includes(idx)) {
      this.clearGhost();
      return;
    }
    if (!this.confirmGate(idx)) return;
    if (this.divergence === null) this.divergence = this.pos;
    this.variation = this.variation.slice(0, this.pos).concat(idx);
    await this.reviewJump(this.pos + 1);
  }

  private reviewReset() {
    this.variation = [...this.recordMoves];
    const target = this.divergence ?? this.variation.length;
    void this.reviewJump(target);
  }

  private async saveRecord() {
    if (!this.ready || !this.state) return;
    try {
      const payload: Record<string, unknown> = {};
      if (this.review && this.recordMoves.length) {
        payload.moves = this.recordMoves;
        payload.result = this.reviewResult;
      }
      const res = await this.engine.call<{ text: string }>("export", payload);
      const blob = new Blob([res.text], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      const now = new Date();
      const stamp = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, "0")}${String(now.getDate()).padStart(2, "0")}-${String(now.getHours()).padStart(2, "0")}${String(now.getMinutes()).padStart(2, "0")}`;
      a.href = url;
      a.download = `对局-${stamp}.afg`;
      a.click();
      URL.revokeObjectURL(url);
      this.setStatus(`棋谱已保存 (${this.state.result})`);
    } catch (error) {
      this.setStatus(`保存失败: ${errText(error)}`);
    }
  }

  private async importRecord(file: File) {
    try {
      const text = await file.text();
      const res = await this.engine.call<{ state: GameState; result: string }>("import", { text });
      this.epoch++;
      this.thinking = false;
      this.state = res.state;
      this.review = true;
      this.recordMoves = [...res.state.moves];
      this.variation = [...res.state.moves];
      this.pos = this.variation.length;
      this.divergence = null;
      this.reviewResult = res.result;
      this.arrow = null;
      this.confirmIdx = null;
      this.rebuildArrowStatic();
      this.hideOverlay();
      this.setStatus(`已载入 ${res.state.moves.length} 手 [${res.result}] (${file.name})`);
      this.afterStateChange();
    } catch (error) {
      this.setStatus(`读取失败: ${errText(error)}`);
    }
  }

  private setStatus(text: string, duration = 6000) {
    this.statusText = text;
    this.statusUntil = performance.now() + duration;
    this.syncPanel();
  }

  private syncPanel() {
    for (const btn of document.querySelectorAll<HTMLButtonElement>("#buttons button")) {
      const action = btn.dataset.action;
      if (action?.startsWith("mode-")) {
        btn.classList.toggle("active", Number(action.slice(5)) === this.mode);
      }
    }
    for (const btn of document.querySelectorAll<HTMLButtonElement>("#toggle-btns button")) {
      const key = btn.dataset.toggle;
      const on =
        key === "danger" ? this.showDanger :
        key === "numbers" ? this.showNumbers :
        key === "arrows" ? this.showArrows :
        key === "rays" ? this.showRays :
        key === "confirm" ? this.confirmMove :
        key === "sound" ? this.sounds.enabled : false;
      btn.classList.toggle("active", on);
    }
    $("review-bar").classList.toggle("hidden", !this.review);
    const st = this.state;
    const tier = TIERS[this.tier];
    const vtxt = this.hver === "beginner"
      ? "初级"
      : `${VERSION_LABELS[this.hver]}·${this.hvcf ? "VCF开" : "VCF关"}`;
    $("info-param").textContent = `${vtxt} · 深度 ${tier.depth} 层`;
    const infoLimit = $("info-limit");
    infoLimit.textContent = this.limitMode === "ai" ? `读 ${this.limitSeconds} 秒` : "不限时";
    if (!st) return;
    if (this.review) {
      $("info-step-label").textContent = "复盘";
      $("info-step").textContent = `${this.pos} / ${this.variation.length}`;
      if (st.game_over) {
        $("info-mover").textContent = "已终局";
        this.setStrip(`结果: ${st.result}`, "result");
      } else if (this.confirmIdx !== null) {
        $("info-mover").textContent = `${st.player === BLACK ? "黑" : "白"}·${st.pending >= 0 ? "移子" : "落子"}`;
        this.setStrip("试下:已选点(虚影) · 再点同位置确认", "divergence");
      } else if (this.isDiverged()) {
        $("info-mover").textContent = `${st.player === BLACK ? "黑" : "白"}·${st.pending >= 0 ? "移子" : "落子"}`;
        this.setStrip(`试下中 · 原谱 ${this.reviewResult}`, "divergence");
      } else {
        $("info-mover").textContent = `${st.player === BLACK ? "黑" : "白"}·${st.pending >= 0 ? "移子" : "落子"}`;
        this.setStrip("点击棋盘可试下 · 点轴回看", "review");
      }
    } else {
      $("info-step-label").textContent = "手数";
      $("info-step").textContent = `${st.turn_count + 1} / ${st.move_count}`;
      const who = this.isAiTurn() ? "AI" : "你";
      const action = st.pending >= 0 ? "移子" : "落子";
      $("info-mover").textContent = `${st.player === BLACK ? "黑" : "白"}(${who})·${action}`;
      if (st.game_over) this.setStrip(`结果: ${st.result}`, "result");
      else if (this.confirmIdx !== null) {
        const r = Math.floor(this.confirmIdx / BOARD_SIZE);
        const c = this.confirmIdx % BOARD_SIZE;
        this.setStrip(`已选点(${String.fromCharCode(65 + c)}${r}) · 再点同位置确认`, "pending");
      }
      else if (st.pending >= 0) this.setStrip("移子待定 · 点击目标位置安置", "pending");
      else this.setStrip("落子待定 · 可占敌移子", "normal");
    }
    const notify = $("notify");
    notify.textContent = this.statusText && performance.now() < this.statusUntil ? this.statusText : "暂无提示";
  }

  private isDiverged(): boolean {
    if (this.variation.length !== this.recordMoves.length) return true;
    for (let i = 0; i < this.variation.length; i++) {
      if (this.variation[i] !== this.recordMoves[i]) return true;
    }
    return false;
  }

  private setStrip(text: string, kind: string) {
    const strip = $("status-strip");
    strip.textContent = text;
    strip.dataset.kind = kind;
  }

  private updateClock() {
    const st = this.state;
    if (!st) return;
    const clock = $("clock");
    const indicator = $("ai-indicator");
    const aiState = $("ai-state");
    const compute = $("ai-compute");
    if (this.limitMode !== "ai") {
      clock.textContent = "不限时";
      clock.classList.remove("warn");
    } else if (this.review || st.game_over) {
      clock.textContent = "--:--";
      clock.classList.remove("warn");
    } else if (this.isAiTurn()) {
      const remain = this.deadline !== null
        ? (this.deadline - performance.now()) / 1000
        : this.limitSeconds;
      clock.textContent = fmtClock(remain);
      clock.classList.toggle("warn", remain < 5);
    } else {
      clock.textContent = fmtClock(this.limitSeconds);
      clock.classList.remove("warn");
    }
    indicator.classList.toggle("thinking", this.thinking);
    aiState.textContent = this.thinking ? "AI思考中" : "AI空闲";
    compute.textContent = this.thinking ? `节点 ${this.nodes}` : "";
    const notify = $("notify");
    if (this.statusText && performance.now() >= this.statusUntil && notify.textContent !== "暂无提示") {
      notify.textContent = "暂无提示";
    }
  }

  private tick = () => {
    const now = performance.now();
    if (this.arrow && this.arrow.t0 !== null) {
      if (now - this.arrow.t0 < 500) this.needsDraw = true;
      else {
        this.arrow = { ...this.arrow, t0: null };
        this.needsDraw = true;
      }
    }
    if (this.needsDraw && this.state) {
      this.drawBoard(now);
      this.needsDraw = false;
    }
    window.requestAnimationFrame(this.tick);
  };

  private drawBoard(now: number) {
    const st = this.state;
    if (!st) return;
    let held: { x: number; y: number } | null = null;
    if (st.pending >= 0 && this.pointer) held = this.pointer;
    let hoverTarget: number | null = null;
    if (st.pending >= 0 && this.hover) {
      const idx = this.hover.r * BOARD_SIZE + this.hover.c;
      if (st.legal.includes(idx)) hoverTarget = idx;
    }
    const view: BoardView = {
      state: st,
      hover: this.hover,
      held,
      showDanger: this.showDanger,
      showNumbers: this.showNumbers,
      showRays: this.showRays,
      arrow: this.showArrows ? this.arrow : null,
      hoverTarget,
      confirmIdx: this.confirmIdx,
    };
    this.board.draw(view, now);
    if (this.review) this.drawTimeline();
  }

  private drawTimeline() {
    const canvas = this.timeline;
    const parent = canvas.parentElement;
    if (!parent) return;
    const width = Math.max(200, Math.floor(parent.clientWidth));
    const height = 44;
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    }
    const g = this.timelineCtx;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, width, height);
    const x0 = 10;
    const x1 = width - 10;
    const axisY = 16;
    const n = Math.max(1, this.variation.length);
    g.strokeStyle = "#a6723a";
    g.lineWidth = 2;
    g.beginPath();
    g.moveTo(x0, axisY);
    g.lineTo(x1, axisY);
    g.stroke();
    const thin = this.variation.length <= 160 ? 1 : Math.ceil(this.variation.length / 160);
    g.fillStyle = "#281c10";
    g.font = "11px 'Microsoft YaHei', 'PingFang SC', sans-serif";
    g.textAlign = "center";
    g.textBaseline = "top";
    for (let k = 0; k <= this.variation.length; k += thin) {
      const tx = x0 + ((x1 - x0) * k) / n;
      const length = k % 10 === 0 ? 10 : k % 5 === 0 ? 7 : 4;
      g.strokeStyle = "#a6723a";
      g.lineWidth = 1;
      g.beginPath();
      g.moveTo(tx, axisY);
      g.lineTo(tx, axisY + length);
      g.stroke();
      if (k % 10 === 0) g.fillText(String(k), tx, axisY + 13);
    }
    const t = x0 + ((x1 - x0) * this.pos) / n;
    g.strokeStyle = "#eb4632";
    g.lineWidth = 3;
    g.beginPath();
    g.moveTo(t, axisY - 9);
    g.lineTo(t, axisY + 9);
    g.stroke();
    $("review-pos").textContent = `${this.pos} / ${this.variation.length}`;
  }

  private selectedVersion(): "beginner" | "mid" | "advanced" {
    const el = document.querySelector<HTMLInputElement>('input[name="hver"]:checked');
    const v = el?.value;
    return v === "beginner" || v === "mid" ? v : "advanced";
  }

  private syncVcfEnabled(): void {
    const beginner = this.selectedVersion() === "beginner";
    for (const input of document.querySelectorAll<HTMLInputElement>('input[name="hvcf"]')) {
      input.disabled = beginner;
      input.closest("label")?.classList.toggle("disabled", beginner);
    }
  }

  private openSettings() {
    const settings = $("settings");
    for (const input of document.querySelectorAll<HTMLInputElement>('input[name="tier"]')) {
      input.checked = Number(input.value) === this.tier;
    }
    for (const input of document.querySelectorAll<HTMLInputElement>('input[name="hver"]')) {
      input.checked = input.value === this.hver;
    }
    for (const input of document.querySelectorAll<HTMLInputElement>('input[name="hvcf"]')) {
      input.checked = (input.value === "on") === this.hvcf;
    }
    this.syncVcfEnabled();
    for (const input of document.querySelectorAll<HTMLInputElement>('input[name="limit"]')) {
      input.checked = input.value === this.limitMode;
    }
    ($("limit-seconds") as HTMLInputElement).value = String(this.limitSeconds);
    settings.classList.remove("hidden");
  }

  private closeSettings() {
    $("settings").classList.add("hidden");
  }

  private applySettings() {
    const tierInput = document.querySelector<HTMLInputElement>('input[name="tier"]:checked');
    if (tierInput) this.tier = Number(tierInput.value);
    this.hver = this.selectedVersion();
    const vcfInput = document.querySelector<HTMLInputElement>('input[name="hvcf"]:checked');
    if (vcfInput) this.hvcf = vcfInput.value === "on";
    if (this.hver === "beginner") this.hvcf = false;
    const limitInput = document.querySelector<HTMLInputElement>('input[name="limit"]:checked');
    if (limitInput) this.limitMode = limitInput.value === "ai" ? "ai" : "none";
    const seconds = Number(($("limit-seconds") as HTMLInputElement).value);
    if (Number.isFinite(seconds)) this.limitSeconds = Math.max(1, Math.min(600, Math.round(seconds)));
    this.closeSettings();
    const vtxt = this.hver === "beginner"
      ? "初级"
      : `${VERSION_LABELS[this.hver]}·${this.hvcf ? "VCF开" : "VCF关"}`;
    this.setStatus(`设定已应用: ${vtxt} / ${TIERS[this.tier].name} / ${this.limitMode === "ai" ? `读秒 ${this.limitSeconds}s` : "不限时"}`);
    this.syncPanel();
    this.maybeRunAi();
  }
}
