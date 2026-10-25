import { ARROW_FLY_MS, BoardCanvas } from "./board";
import type { ArrowView, BoardView } from "./board";
import { Engine } from "./engine";
import { Sounds } from "./audio";
import { BLACK, BOARD_SIZE, EMPTY, WHITE } from "./types";
import type { AiResult, GameState } from "./types";
import { PARAM_FIELDS, sanitizeParams, varietyFor } from "./engine/variety";
import type { VarietyLevel, VarietyParams } from "./engine/variety";

const VARIETY_LEVELS: readonly VarietyLevel[] = ["off", "low", "mid", "high", "custom"];

function isVarietyLevel(v: string): v is VarietyLevel {
  return (VARIETY_LEVELS as readonly string[]).includes(v);
}

interface Tier {
  name: string;
  depth: number;
}

// AI 难度 = 搜索深度(三档);「大师」「传奇」是引擎版本(见 hver 单选),不放在
// 难度里,避免同一入口重复且互相覆盖(选大师/传奇难度会静默忽略版本单选)。
const TIERS: Tier[] = [
  { name: "简单", depth: 2 },
  { name: "中等", depth: 4 },
  { name: "困难", depth: 8 },
];

const VERSION_LABELS: Record<string, string> = {
  beginner: "初级", mid: "中级", advanced: "高级", hell: "大师", legend: "传奇",
};

const MODE_NAMES = ["人执黑", "人执白", "人人对战", "机机观战"];

const RESIGN_CONFIRM_MS = 4000;   // 认输二次确认窗口:首次点击后需在此时长内再点
const INTRO_KEY = "antifive-intro-skip";   // 勾选「下次不再显示」后写入 localStorage

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

/** Worker 侧被中断时抛「搜索已中断」(api.ts 在抛出前不会改动局面)。 */
function isInterruptError(error: unknown): boolean {
  return errText(error).includes("中断");
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
  private limitMode: "none" | "ai" = "ai";
  private limitSeconds = 12;
  private hver: "beginner" | "mid" | "advanced" | "hell" | "legend" = "advanced";
  private hvcf = true;
  private whiteRestrict = 2;        // 白棋前 N 手禁推(0=无禁推,新局生效)
  // AI 变化:同局面不再只给同一应手(见 engine/variety.ts)。off=每步最强。
  private hvariety: VarietyLevel = "mid";
  /** custom 档的可调参数(预设档位忽略)。 */
  private hvarietyParams: VarietyParams = sanitizeParams(varietyFor("custom"));
  private static readonly VARIETY_LABELS: Record<VarietyLevel, string> = {
    off: "关(每步最强)", low: "低", mid: "中(推荐)", high: "高", custom: "自定义",
  };
  private readonly VARIETY_SHORT: Record<VarietyLevel, string> = {
    off: "关", low: "低", mid: "中", high: "高", custom: "自定",
  };

  private showDanger = false;
  private showNumbers = false;
  private showArrows = true;
  private showRays = true;
  private confirmMove = true;
  private confirmIdx: number | null = null;
  private resignArmedUntil = 0;
  private resignArmedShown = false;

  private arrow: ArrowView | null = null;
  private hover: { r: number; c: number } | null = null;
  private pointer: { x: number; y: number } | null = null;

  private hiddenSince: number | null = null;
  private hiddenDuringThink = 0;
  private hiddenProgress = 0;
  private suspendedDuringThink = false;
  private aiRedoArmed = false;
  private aiRedoEpoch = 0;
  private dprCheckedAt = 0;
  private undoBusy = false;
  private humanPrevState: GameState | null = null;
  private paused = false;        // 玩家点了「暂停AI」:停手停钟,不再自动落子
  private pauseAbort = false;    // 暂停引发的中断还没收尾(用于吞掉「搜索已中断」)
  // 暂停时提示语取自哪一档(思考中被中断 / 浏览器不支持中断 / 本来就没在思考),
  // 记下来是为了暂停期间改设定时能用新时长重新生成同一句提示。
  private pausedMode: "thinking" | "soft" | "idle" = "idle";

  private review = false;
  private recordMoves: number[] = [];
  private variation: number[] = [];
  private pos = 0;
  private divergence: number | null = null;
  private reviewResult = "未完";
  private playSeconds = 1;          // 复盘自动播放每手间隔秒数(设定弹窗可调)
  private playing = false;
  private playTimer: number | null = null;

  private statusText = "";
  private statusUntil = 0;
  private needsDraw = true;

  constructor(base: string) {
    this.board = new BoardCanvas($("board") as HTMLCanvasElement);
    this.board.onRestored = () => {
      this.needsDraw = true;
    };
    this.timeline = $("timeline") as HTMLCanvasElement;
    const tctx = this.timeline.getContext("2d");
    if (!tctx) throw new Error("timeline 2d context 不可用");
    this.timelineCtx = tctx;
    this.buildButtons();
    this.bindEvents();
    this.engine = new Engine(base);
    this.engine.onStatus = (s) => {
      const texts: Record<string, string> = {
        engine: "正在加载 TypeScript 引擎…",
      };
      $("loading-text").textContent = texts[s.stage] ?? "正在加载…";
      $("loading-detail").textContent = s.detail ?? "";
    };
    this.engine.onProgress = (nodes) => {
      this.nodes = nodes;
      this.hiddenProgress++;   // 进度心跳:切屏期间是否仍有消息 = Worker 是否被系统挂起
    };
    this.engine.onDownload = (stage, loaded, total) => {
      const mb = (n: number) => (n / 1048576).toFixed(1);
      $("loading-detail").textContent = `${stage} ${mb(loaded)} / ${mb(total)} MB`;
    };
    this.engine.onFatal = (error) => {
      $("loading-text").textContent = "引擎已停止";
      $("loading-detail").textContent = `${errText(error)}(请刷新页面重试)`;
      $("loading").classList.remove("hidden");
      this.showReloadButton();
    };
    this.engine.ready
      .then(() => {
        this.ready = true;
        $("loading").classList.add("hidden");
        const nav = navigator as Navigator & { deviceMemory?: number };
        this.logDiag("ready", {
          ua: navigator.userAgent,
          cores: navigator.hardwareConcurrency,
          memoryGB: nav.deviceMemory,
          dpr: window.devicePixelRatio,
        });
        this.maybeShowIntro();
        this.newGame();
      })
      .catch((error) => {
        $("loading-text").textContent = "引擎加载失败";
        $("loading-detail").textContent = errText(error);
        this.showReloadButton();
      });
    window.requestAnimationFrame(this.tick);
    window.setInterval(() => this.updateClock(), 100);
  }

  private buildButtons() {
    const buttons = $("buttons");
    const actions: [string, string][] = [
      ["settings", "设定"],
      ["new", "新局"],
      ["mode-0", MODE_NAMES[0]],
      ["mode-1", MODE_NAMES[1]],
      ["mode-2", MODE_NAMES[2]],
      ["mode-3", MODE_NAMES[3]],
      ["undo", "悔棋"],
      ["resign", "认输"],
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
      } else if (action === "resign") {
        void this.resign();
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
      ["自动播放", () => this.toggleAutoPlay()],
      ["下一手", () => void this.reviewJump(this.pos + 1)],
      ["终局", () => void this.reviewJump(this.variation.length)],
      ["原谱", () => this.reviewReset()],
    ];
    for (const [label, fn] of reviewDefs) {
      const btn = document.createElement("button");
      btn.textContent = label;
      if (label === "自动播放") btn.dataset.playToggle = "1";
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
    for (const [value, text] of [["beginner", "初级(经典)"], ["mid", "中级(强化)"], ["advanced", "高级(+蒸馏)"], ["hell", "大师(防守强化)"], ["legend", "传奇(+蒸馏v2)"]] as const) {
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
    const hvarietyRadios = $("hvariety-radios");
    for (const value of VARIETY_LEVELS) {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "radio";
      input.name = "hvariety";
      input.value = value;
      input.checked = value === this.hvariety;
      label.append(input, document.createTextNode(App.VARIETY_LABELS[value]));
      hvarietyRadios.appendChild(label);
    }
    // 自定义参数(仅「自定义」档可见):改任一数值即自动切到自定义。
    const paramBox = $("hvariety-params");
    for (const field of PARAM_FIELDS) {
      const label = document.createElement("label");
      label.className = "seconds";
      const input = document.createElement("input");
      input.type = "number";
      input.id = `variety-${field.key}`;
      input.min = String(field.min);
      input.max = String(field.max);
      input.step = String(field.step);
      input.value = String(this.hvarietyParams[field.key] ?? "");
      input.addEventListener("input", () => this.onVarietyParamInput());
      label.append(document.createTextNode(field.label), input);
      paramBox.appendChild(label);
    }
    const hint = document.createElement("div");
    hint.className = "variety-hint";
    hint.textContent = "T 越大越随机；window 是允许比最优差多少（0.9 ≈ 一个可杀四）；K 只在前 K 个着法里抽。必胜/必杀不参与随机。";
    paramBox.appendChild(hint);
    hvarietyRadios.addEventListener("change", () => this.onVarietyPresetChange());
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
    ($("play-seconds") as HTMLInputElement).value = String(this.playSeconds);
    ($("restrict-turns") as HTMLInputElement).value = String(this.whiteRestrict);
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
    // 元素缺失时不要让整个事件绑定崩掉(旧 HTML + 新 JS 的缓存错配场景)
    document.getElementById("ai-pause")?.addEventListener("click", () => this.togglePauseAi());
    document.addEventListener(
      "pointerdown",
      (event) => {
        const target = event.target as HTMLElement | null;
        if (!target?.closest?.('button[data-action="resign"]')) this.disarmResign();
      },
      true,
    );
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
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) {
        this.hiddenSince = performance.now();
        this.hiddenProgress = 0;
        return;
      }
      if (this.hiddenSince !== null) {
        const hiddenMs = performance.now() - this.hiddenSince;
        if (this.thinking) {
          this.hiddenDuringThink += hiddenMs;
          // 挂起判定:隐藏超过 2s 且期间一条搜索进度都没有(Worker 被冻结),
          // 时间预算会被挂起时长空烧,结果只是浅层搜索
          if (hiddenMs > 2000 && this.hiddenProgress === 0) this.suspendedDuringThink = true;
        }
        this.hiddenSince = null;
      }
      // 切屏/锁屏后 canvas 后备存储可能被系统回收,回到前台强制重建重绘
      this.board.resize();
      this.needsDraw = true;
      if (this.aiRedoArmed) {
        this.aiRedoArmed = false;
        if (this.epoch === this.aiRedoEpoch && !this.thinking && !this.review
            && this.state && !this.state.game_over) {
          this.maybeRunAi();
        }
      } else {
        this.maybeRunAi();
      }
    });
    window.addEventListener("pageshow", () => {
      // bfcache 恢复时画面可能整体失效,同样强制重绘
      this.board.resize();
      this.needsDraw = true;
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
    else if (this.review && key === " " && !(event.target instanceof HTMLButtonElement)) {
      event.preventDefault();          // 避免页面滚动;按钮聚焦时留给按钮自身
      this.toggleAutoPlay();
    } else return;
    this.syncPanel();
    this.needsDraw = true;
  }

  private isAiTurnState(st: GameState): boolean {
    if (this.mode === 0) return st.player === WHITE;
    if (this.mode === 1) return st.player === BLACK;
    if (this.mode === 3) return true;
    return false;
  }

  private isAiTurn(): boolean {
    if (!this.state) return false;
    return this.isAiTurnState(this.state);
  }

  private newGame() {
    if (!this.ready) return;
    this.engine.interrupt();     // 正在搜索的 AI 立即中断,避免新局排队等待
    this.epoch++;
    this.thinking = false;
    this.deadline = null;
    this.paused = false;
    this.pauseAbort = false;
    this.nodes = 0;
    this.humanPrevState = null;
    this.review = false;
    this.stopAutoPlay();
    this.recordMoves = [];
    this.variation = [];
    this.pos = 0;
    this.divergence = null;
    this.arrow = null;
    this.confirmIdx = null;
    this.disarmResign();
    this.hideOverlay();
    this.engine
      .call<{ state: GameState }>("new", {
        config: { white_restrict_turns: this.whiteRestrict },
      })
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
      const epoch = this.epoch;
      this.engine
        .call<{ state: GameState }>("stuck", {})
        .then((res) => {
          if (epoch !== this.epoch) return;
          this.state = res.state;
          this.syncPanel();
          this.needsDraw = true;
        })
        .catch(() => undefined);
    }
    this.maybeRunAi();
  }

  private maybeRunAi() {
    if (!this.ready || this.review || this.paused || !this.state || this.state.game_over) return;
    if (!this.isAiTurn() || this.thinking || this.undoBusy) return;
    if (document.hidden) return;   // 切屏时先不启动,回到前台再思考(避免预算被挂起空烧)
    void this.runAi();
  }

  private async runAi(retry = 0) {
    const epoch = this.epoch;
    const tier = TIERS[this.tier];
    // 读秒模式传墙钟预算;「不限时」传 0,引擎按难度深度完整搜索,不做动态加深。
    const budget = this.limitMode === "ai" ? this.limitSeconds : 0;
    const prev = this.state;
    if (!prev) return;
    // 先占位:下面要等推子飞完,这段时间不能再进第二个 runAi(否则同一手会被算两次)。
    this.thinking = true;
    // 上一手的推子还在飞:先让它飞完再开始本轮思考。否则下面的 arrow=null 会把飞行
    // 截在半空(机机观战 350ms 一手,而飞行 500ms,必现),而且清箭头不重绘会让画布
    // 停在中途那一帧,直到下一手落下才跳到正确位置。
    const flyWait = this.arrowRemainingMs();
    if (flyWait > 0) {
      await new Promise<void>((resolve) => setTimeout(resolve, flyWait + 16));
      if (epoch !== this.epoch || this.paused) {
        this.thinking = false;
        this.deadline = null;
        this.syncPanel();
        return;
      }
    }
    // 对方一开始思考就撤掉上一手留下的移子箭头，避免整段思考期间残留。
    this.arrow = null;
    this.needsDraw = true;   // 清箭头必须重绘,否则画布停在飞行中的那一帧
    this.nodes = 0;
    this.hiddenDuringThink = 0;
    this.suspendedDuringThink = false;
    this.deadline = this.limitMode === "ai" ? performance.now() + budget * 1000 : null;
    this.syncPanel();
    let degraded = false;
    try {
      const res = await this.engine.call<AiResult>(
        "ai",
        {
          depth: tier.depth, budget, engine: this.hver, vcf: this.hvcf,
          variety: this.hvariety,
          ...(this.hvariety === "custom" ? { varietyParams: this.hvarietyParams } : {}),
        },
        Math.round((budget + 300) * 1000),
      );
      if (epoch !== this.epoch) return;
      // 暂停之后才到达的结果(中断窗口极小 / 浏览器不支持中断):直接丢弃,不算落子
      if (this.paused) {
        this.logDiag("ai result discarded while paused", { move: res.move, nodes: res.nodes });
        return;
      }
      // 切屏/锁屏期间 Worker 被系统挂起(搜索心跳中断):时间预算被空烧,
      // 结果只是浅层搜索,悔掉后在前台用完整预算重搜一次
      if (retry === 0 && this.suspendedDuringThink) {
        degraded = true;
        this.logDiag("ai degraded by suspend, redo", {
          budget,
          elapsed: Number(res.elapsed.toFixed(2)),
          hiddenMs: Math.round(this.hiddenDuringThink),
          nodes: res.nodes,
        });
        const undoRes = await this.engine.call<{ state: GameState }>("undo", {});
        if (epoch !== this.epoch) return;
        this.state = undoRes.state;
        this.needsDraw = true;
      } else {
        this.applyMove(prev, res.state, res.move);
        this.logDiag("ai move", {
          move: res.move,
          nodes: res.nodes,
          elapsed: Number(res.elapsed.toFixed(2)),
          hiddenMs: Math.round(this.hiddenDuringThink),
        });
      }
    } catch (error) {
      if (this.pauseAbort && isInterruptError(error)) {
        // 暂停主动中断:不算错误;api.ts 在抛出前没改局面,继续后整轮重算
      } else if (epoch === this.epoch) {
        this.setStatus(`AI 出错: ${errText(error)}`);
      }
    } finally {
      this.pauseAbort = false;
      if (epoch === this.epoch) {
        this.thinking = false;
        this.deadline = null;
        this.syncPanel();
      }
    }
    if (epoch !== this.epoch) return;
    if (degraded) {
      if (document.hidden) {
        this.aiRedoArmed = true;
        this.aiRedoEpoch = epoch;
        return;
      }
      await this.runAi(retry + 1);
      return;
    }
    if (this.mode === 3 && this.state && !this.state.game_over && !this.review) {
      window.setTimeout(() => this.maybeRunAi(), 350);
    } else {
      this.maybeRunAi();
    }
  }

  // ---- 暂停 AI(A 方案:作废本轮思考,继续后重新计时) ----
  // AI 跑在同步 Worker 里,主线程没法「冻结」它,只能让它中断退出。
  // 所以暂停 = 停手 + 停钟(读秒回到设定值);继续 = 按设定时长重新计时并整轮重算。

  private togglePauseAi() {
    if (!this.ready || this.review || !this.state || this.state.game_over) return;
    if (this.paused) this.resumeAi();
    else this.pauseAi();
  }

  private pauseAi() {
    if (this.paused) return;
    this.paused = true;
    const wasThinking = this.thinking;
    const interrupting = wasThinking && this.engine.canInterrupt;
    this.pausedMode = !wasThinking ? "idle" : interrupting ? "thinking" : "soft";
    if (interrupting) {
      this.pauseAbort = true;
      this.engine.interrupt();   // Worker 在下一个检查点(每 1024 节点)退出
    }
    this.deadline = null;
    this.syncPanel();
    this.setStatus(this.pausedNoticeText(), this.pausedMode === "soft" ? 8000 : 12000);
  }

  /** 暂停提示语:时长直接取当前设定,不在暂停那一刻写死。 */
  private pausedNoticeText(): string {
    const limit = this.clockLimitText();
    switch (this.pausedMode) {
      case "thinking":
        return `已暂停：本轮思考作废，继续后 AI 重新计时并重算（${limit}）`;
      case "soft":
        // 没有跨源隔离时中断不可用:这一轮会跑完,但结果照样丢弃不落子
        return `已暂停：当前浏览器不支持中断，这一轮会跑完但不落子，继续后重新计时（${limit}）`;
      default:
        return `已暂停 AI，继续后重新计时（${limit}）`;
    }
  }

  private resumeAi() {
    if (!this.paused) return;
    this.paused = false;
    this.pausedMode = "idle";
    this.syncPanel();
    this.setStatus(`继续思考：重新计时（${this.clockLimitText()}）`, 4000);
    // 上一轮中断若还没收尾,thinking 仍为 true,由 runAi 收尾后接着启动
    this.maybeRunAi();
  }

  private clockLimitText(): string {
    return this.limitMode === "ai" ? `${this.limitSeconds} 秒` : "不限时";
  }

  private syncPauseButton() {
    const btn = document.getElementById("ai-pause") as HTMLButtonElement | null;
    if (!btn) return;
    const live = this.ready && !!this.state && !this.state.game_over && !this.review;
    btn.hidden = !live;
    btn.disabled = !live;
    btn.textContent = this.paused ? "继续" : "暂停";
    btn.classList.toggle("paused", this.paused);
    btn.setAttribute("aria-pressed", this.paused ? "true" : "false");
    btn.title = this.paused
      ? "继续 AI 思考（读秒重新计时）"
      : "暂停 AI 思考与读秒（本轮思考作废，继续后重新计时）";
  }

  private applyMove(prev: GameState, next: GameState, moveIdx: number | null) {
    this.confirmIdx = null;
    if (prev.pending >= 0 && moveIdx !== null) {
      this.arrow = { from: prev.pending, to: moveIdx, t0: performance.now() };
    } else {
      // 对手开始下一步（普通落子或占领）时，上一步的移子箭头应立即消失；
      // 只有“自己完成安置”这一分支才保留/重建箭头。
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

  /** 当前推子动画还剩多少毫秒(没开箭头/没在飞时为 0)。 */
  private arrowRemainingMs(now = performance.now()): number {
    if (!this.showArrows || !this.arrow || this.arrow.t0 === null) return 0;
    const remain = ARROW_FLY_MS - (now - this.arrow.t0);
    return remain > 0 ? remain : 0;
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

  private syncResignButton() {
    const armed = performance.now() < this.resignArmedUntil;
    if (armed === this.resignArmedShown) return;
    this.resignArmedShown = armed;
    const btn = document.querySelector<HTMLButtonElement>('button[data-action="resign"]');
    if (!btn) return;
    btn.textContent = armed ? "确定？" : "认输";
    btn.classList.toggle("confirm", armed);
  }

  private disarmResign() {
    if (!this.resignArmedUntil && !this.resignArmedShown) return;
    this.resignArmedUntil = 0;
    this.syncResignButton();
  }

  private async resign() {
    if (!this.ready || !this.state || this.review || this.state.game_over) {
      this.disarmResign();
      return;
    }
    if (this.mode === 3) {
      this.disarmResign();
      this.setStatus("机机观战模式无需认输");
      return;
    }
    if (performance.now() >= this.resignArmedUntil) {
      this.resignArmedUntil = performance.now() + RESIGN_CONFIRM_MS;
      this.setStatus("再点一次「认输」确认结束对局", RESIGN_CONFIRM_MS);
      this.syncResignButton();
      return;
    }
    this.disarmResign();
    const loser = this.mode === 0 ? BLACK : this.mode === 1 ? WHITE : this.state.player;
    const epoch = ++this.epoch;
    this.engine.interrupt();     // 认输也要等 Worker 空闲,先中断在跑的搜索
    this.thinking = false;
    this.deadline = null;
    this.confirmIdx = null;
    try {
      const res = await this.engine.call<{ state: GameState }>("resign", { player: loser });
      if (epoch !== this.epoch) return;
      this.applyMove(this.state, res.state, null);
    } catch (error) {
      this.setStatus(`认输失败: ${errText(error)}`);
    }
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
    this.arrow = null;               // 人类开始行动，上一手移子箭头立即撤掉
    this.humanPrevState = prev;      // 悔棋乐观回退用:人类上一手之前的局面
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
    // 悔棋链必须独占:AI 思考中重复点击会让多条悔棋链交叉,被 epoch 丢弃的
    // 响应会让引擎(Python)继续回滚而界面状态停在半路,造成棋盘与引擎错位
    if (this.undoBusy) return;
    this.undoBusy = true;
    const wasThinking = this.thinking;
    const epoch = ++this.epoch;
    // 不立即清 thinking:AI 搜索仍在 Worker 里跑,清掉会让 visibilitychange /
    // afterStateChange 触发 maybeRunAi 排入新的 AI 调用,悔棋链与 AI 互相排队,
    // 棋盘长时间不回落(切屏回来还会"重新思考")。等链执行完再统一收尾。
    this.confirmIdx = null;
    this.disarmResign();
    if (wasThinking) {
      this.engine.interrupt();   // 立即中断 Worker 中的 AI 搜索(跨源隔离时可用)
      this.setStatus("悔棋中…(等待 AI 结束)", 8000);
      // 乐观回退:AI 搜索在 Worker 里无法打断(单线程 WASM),先把界面显示
      // 恢复到人类上一手之前,避免棋盘长时间不动(切屏回来也不受影响);
      // 引擎搜索结束后再由悔棋链的结果对齐。
      const snap = this.humanPrevState;
      if (snap && !snap.game_over && snap.moves.length + 1 === this.state.moves.length) {
        this.state = snap;
        this.arrow = null;
        this.rebuildArrowStatic();
        this.needsDraw = true;
        this.syncPanel();
      }
    }
    let ok = false;
    try {
      let res = await this.engine.call<{ state: GameState }>("undo", {});
      if (epoch !== this.epoch) return;
      let st: GameState = res.state;
      while (st.moves.length > 0 && this.isAiTurnState(st) && !st.game_over) {
        res = await this.engine.call<{ state: GameState }>("undo", {});
        if (epoch !== this.epoch) return;
        st = res.state;
      }
      this.state = st;
      ok = true;
    } catch (error) {
      if (epoch === this.epoch) {
        this.setStatus(`悔棋失败: ${errText(error)}`);
        try {
          const s = await this.engine.call<{ state: GameState }>("state", {});
          if (epoch === this.epoch) {
            this.state = s.state;
            this.needsDraw = true;
            this.syncPanel();
          }
        } catch {
          // 引擎不可用时保持现状
        }
      }
    } finally {
      if (epoch === this.epoch) {
        this.thinking = false;
        this.deadline = null;
      }
      this.undoBusy = false;
    }
    if (!ok || epoch !== this.epoch) return;
    this.arrow = null;
    this.rebuildArrowStatic();
    this.hideOverlay();
    this.afterStateChange();
  }

  private enterReview(overlay: boolean) {
    if (!this.state) return;
    this.review = true;
    this.paused = false;
    this.pauseAbort = false;
    this.stopAutoPlay();
    this.recordMoves = [...this.state.moves];
    this.variation = [...this.state.moves];
    this.pos = this.variation.length;
    this.divergence = null;
    this.reviewResult = this.state.result;
    this.disarmResign();
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
    const { kinds, moves } = this.state;
    // 回放到“占领步”或普通落子时，上一步的箭头可能还留在 this.arrow 里；
    // 必须先清空，不能只在 pending < 0 时清，否则占领待安置状态会显示错位箭头。
    this.arrow = null;
    const n = kinds.length;
    if (this.showArrows && n >= 2 && kinds[n - 1] === "place2") {
      this.arrow = { from: moves[n - 2], to: moves[n - 1], t0: null };
    }
  }

  private async reviewJump(k: number, keepPlaying = false) {
    if (!keepPlaying) this.stopAutoPlay();
    if (!this.ready || !this.state) return;
    const target = Math.max(0, Math.min(k, this.variation.length));
    if (target === this.pos && this.review) {
      this.needsDraw = true;
      return;
    }
    this.pos = target;
    this.confirmIdx = null;
    this.disarmResign();
    const gotoResult = this.isDiverged() ? null : this.reviewResult;
    const epoch = ++this.epoch;
    this.thinking = false;
    this.deadline = null;
    try {
      const res = await this.engine.call<{ state: GameState; pos: number }>("goto", {
        moves: this.variation,
        pos: target,
        result: gotoResult,
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

  private toggleAutoPlay() {
    if (this.playing) {
      this.stopAutoPlay();
      this.setStatus("自动播放已暂停");
      return;
    }
    if (!this.ready || !this.review || !this.variation.length) return;
    this.playing = true;
    this.setStatus(`自动播放中 · 每手 ${this.playSeconds} 秒`);
    this.syncPanel();
    this.needsDraw = true;
    void this.startAutoPlay();
  }

  private async startAutoPlay() {
    if (this.pos >= this.variation.length) {
      await this.reviewJump(0, true);   // 已到末尾:从头播放
      if (!this.playing) return;
    }
    this.scheduleAutoPlay();
  }

  private scheduleAutoPlay(delayMs = this.playSeconds * 1000) {
    this.clearPlayTimer();
    this.playTimer = window.setTimeout(() => {
      this.playTimer = null;
      void this.autoPlayStep();
    }, delayMs);
  }

  private async autoPlayStep() {
    if (!this.playing || !this.review) return;
    if (this.pos >= this.variation.length) {
      this.stopAutoPlay();              // 播到终局自动停
      return;
    }
    await this.reviewJump(this.pos + 1, true);
    if (this.playing && this.review) this.scheduleAutoPlay();
  }

  private stopAutoPlay() {
    if (!this.playing && this.playTimer === null) return;
    this.playing = false;
    this.clearPlayTimer();
    this.syncPanel();
    this.needsDraw = true;
  }

  private clearPlayTimer() {
    if (this.playTimer !== null) {
      window.clearTimeout(this.playTimer);
      this.playTimer = null;
    }
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
      this.engine.interrupt();   // 读取棋谱会替换整局,先中断在跑的搜索
      const res = await this.engine.call<{ state: GameState; result: string }>("import", { text });
      this.epoch++;
      this.thinking = false;
      this.deadline = null;
      this.paused = false;
      this.pauseAbort = false;
      this.humanPrevState = null;
      this.state = res.state;
      this.review = true;
      this.stopAutoPlay();
      this.recordMoves = [...res.state.moves];
      this.variation = [...res.state.moves];
      this.pos = this.variation.length;
      this.divergence = null;
      this.reviewResult = res.result;
      this.arrow = null;
      this.confirmIdx = null;
      this.disarmResign();
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

  private logDiag(message: string, data?: Record<string, unknown>) {
    console.info(`[antifive] ${message}`, data ?? "");
  }

  private showReloadButton() {
    if (document.getElementById("loading-retry")) return;
    const btn = document.createElement("button");
    btn.id = "loading-retry";
    btn.textContent = "重试 / 刷新";
    btn.addEventListener("click", () => window.location.reload());
    $("loading").appendChild(btn);
  }

  private maybeShowIntro() {
    try {
      if (localStorage.getItem(INTRO_KEY) === "1") return;
    } catch {
      // 隐私模式等禁用 localStorage:照常展示
    }
    const el = $("intro");
    el.classList.remove("hidden");
    $("intro-start").addEventListener("click", () => this.closeIntro(), { once: true });
    el.addEventListener("click", (event) => {
      if (event.target === el) this.closeIntro();
    });
  }

  private closeIntro() {
    const skip = document.getElementById("intro-skip") as HTMLInputElement | null;
    if (skip?.checked) {
      try {
        localStorage.setItem(INTRO_KEY, "1");
      } catch {
        // 忽略存储失败
      }
    }
    $("intro").classList.add("hidden");
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
    const playBtn = document.querySelector<HTMLButtonElement>(
      "#review-buttons button[data-play-toggle]");
    if (playBtn) {
      playBtn.textContent = this.playing ? "暂停" : "自动播放";
      playBtn.classList.toggle("active", this.playing);
    }
    this.syncPauseButton();
    const st = this.state;
    const tier = TIERS[this.tier];
    const vtxt = this.versionText();
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
    this.syncResignButton();
    const st = this.state;
    if (!st) return;
    const clock = $("clock");
    const indicator = $("ai-indicator");
    const aiState = $("ai-state");
    const compute = $("ai-compute");
    const paused = this.paused && !this.review && !st.game_over;
    if (paused) {
      // 暂停:读秒重置回设定时长(本轮搜索已作废,继续后按完整预算重算)
      clock.textContent = this.limitMode === "ai" ? fmtClock(this.limitSeconds) : "不限时";
      clock.classList.remove("warn");
    } else if (this.limitMode !== "ai") {
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
    indicator.classList.toggle("thinking", this.thinking && !paused);
    indicator.classList.toggle("paused", paused);
    aiState.textContent = paused ? "AI已暂停" : this.thinking ? "AI思考中" : "AI空闲";
    compute.textContent = this.thinking && !paused ? `节点 ${this.nodes}` : "";
    const notify = $("notify");
    if (this.statusText && performance.now() >= this.statusUntil && notify.textContent !== "暂无提示") {
      notify.textContent = "暂无提示";
    }
  }

  private tick = () => {
    const now = performance.now();
    if (now - this.dprCheckedAt > 1000) {
      this.dprCheckedAt = now;
      // 跨不同缩放的显示器拖动窗口时 devicePixelRatio 会变化,但可能不触发 resize
      if (this.board.checkDpr()) this.needsDraw = true;
    }
    if (this.arrow && this.arrow.t0 !== null) {
      if (now - this.arrow.t0 < ARROW_FLY_MS) this.needsDraw = true;
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

  private selectedVersion(): "beginner" | "mid" | "advanced" | "hell" | "legend" {
    const el = document.querySelector<HTMLInputElement>('input[name="hver"]:checked');
    const v = el?.value;
    return v === "beginner" || v === "mid" || v === "hell" || v === "legend"
      ? v : "advanced";
  }

  private versionText(): string {
    const variety = this.hvariety === "off" ? "" : `·变化${this.VARIETY_SHORT[this.hvariety]}`;
    if (this.hver === "beginner") return `初级${variety}`;
    return `${VERSION_LABELS[this.hver]}·${this.hvcf ? "VCF开" : "VCF关"}${variety}`;
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
    for (const input of document.querySelectorAll<HTMLInputElement>('input[name="hvariety"]')) {
      input.checked = input.value === this.hvariety;
    }
    this.syncVarietyParams();
    if (this.hvariety === "custom") this.writeVarietyParams();
    for (const input of document.querySelectorAll<HTMLInputElement>('input[name="limit"]')) {
      input.checked = input.value === this.limitMode;
    }
    ($("limit-seconds") as HTMLInputElement).value = String(this.limitSeconds);
    ($("play-seconds") as HTMLInputElement).value = String(this.playSeconds);
    ($("restrict-turns") as HTMLInputElement).value = String(this.whiteRestrict);
    settings.classList.remove("hidden");
  }

  private closeSettings() {
    $("settings").classList.add("hidden");
  }

  /** 收集自定义参数:夹紧后写回状态(输入框在「确定」时才回写,避免打断输入)。 */
  private readVarietyParams(): void {
    const raw: Record<string, number> = {};
    for (const field of PARAM_FIELDS) {
      const el = document.getElementById(`variety-${field.key}`) as HTMLInputElement | null;
      if (el) raw[field.key] = Number(el.value);
    }
    this.hvarietyParams = sanitizeParams(raw);
  }

  private writeVarietyParams(): void {
    const base = varietyFor("custom", this.hvarietyParams);
    for (const field of PARAM_FIELDS) {
      const el = document.getElementById(`variety-${field.key}`) as HTMLInputElement | null;
      if (el) el.value = String(base[field.key]);
    }
    this.hvarietyParams = sanitizeParams(base);
  }

  /** 预设档位 → 参数框(切到「自定义」时从上一个预设继续调)。 */
  private fillVarietyParams(level: VarietyLevel): void {
    const spec = varietyFor(level);
    this.hvarietyParams = sanitizeParams(spec);
    for (const field of PARAM_FIELDS) {
      const el = document.getElementById(`variety-${field.key}`) as HTMLInputElement | null;
      if (el) el.value = String(spec[field.key]);
    }
  }

  private syncVarietyParams(): void {
    // 以「面板上当前选中的档位」为准:选自定义还没点确定时也要能立刻调参数。
    $("hvariety-params").classList.toggle("hidden", this.selectedVarietyLevel() !== "custom");
  }

  private selectedVarietyLevel(): VarietyLevel {
    const sel = document.querySelector<HTMLInputElement>('input[name="hvariety"]:checked');
    return sel && isVarietyLevel(sel.value) ? sel.value : this.hvariety;
  }

  private onVarietyPresetChange(): void {
    const sel = document.querySelector<HTMLInputElement>('input[name="hvariety"]:checked');
    if (!sel || !isVarietyLevel(sel.value)) return;
    if (sel.value !== "custom") this.fillVarietyParams(sel.value);
    this.syncVarietyParams();
  }

  private onVarietyParamInput(): void {
    this.readVarietyParams();
    this.hvariety = "custom";
    const custom = document.querySelector<HTMLInputElement>('input[name="hvariety"][value="custom"]');
    if (custom) custom.checked = true;
    this.syncVarietyParams();
  }

  private applySettings() {
    const tierInput = document.querySelector<HTMLInputElement>('input[name="tier"]:checked');
    if (tierInput) this.tier = Number(tierInput.value);
    this.hver = this.selectedVersion();
    const vcfInput = document.querySelector<HTMLInputElement>('input[name="hvcf"]:checked');
    if (vcfInput) this.hvcf = vcfInput.value === "on";
    if (this.hver === "beginner") this.hvcf = false;
    const varietyInput = document.querySelector<HTMLInputElement>('input[name="hvariety"]:checked');
    if (varietyInput && isVarietyLevel(varietyInput.value)) this.hvariety = varietyInput.value;
    if (this.hvariety === "custom") {
      this.readVarietyParams();
      this.writeVarietyParams();   // 夹紧后的值回写输入框
    }
    const limitInput = document.querySelector<HTMLInputElement>('input[name="limit"]:checked');
    if (limitInput) this.limitMode = limitInput.value === "ai" ? "ai" : "none";
    const seconds = Number(($("limit-seconds") as HTMLInputElement).value);
    if (Number.isFinite(seconds)) this.limitSeconds = Math.max(1, Math.min(600, Math.round(seconds)));
    const playSeconds = Number(($("play-seconds") as HTMLInputElement).value);
    if (Number.isFinite(playSeconds)) {
      this.playSeconds = Math.round(Math.max(0.1, Math.min(10, playSeconds)) * 10) / 10;
    }
    const restrict = Number(($("restrict-turns") as HTMLInputElement).value);
    const restrictChanged = Number.isFinite(restrict) &&
      Math.max(0, Math.min(4, Math.round(restrict))) !== this.whiteRestrict;
    if (Number.isFinite(restrict)) {
      this.whiteRestrict = Math.max(0, Math.min(4, Math.round(restrict)));
    }
    this.closeSettings();
    if (this.paused) {
      // 暂停期间改设定:读秒已按新值冻结,提示语里的「（X 秒）」跟着重算;
      // 应用了哪些设定由面板的「参数/限时」行和读秒本身体现,这里优先提示「仍处于暂停」。
      this.setStatus(this.pausedNoticeText(), 12000);
    } else {
      this.setStatus(`设定已应用: ${this.versionText()} / ${TIERS[this.tier].name} / ${this.limitMode === "ai" ? `读秒 ${this.limitSeconds}s` : "不限时"} / 变化 ${App.VARIETY_LABELS[this.hvariety]} / 自动播放 ${this.playSeconds}s${restrictChanged ? ` / 白棋禁推 ${this.whiteRestrict} 手(新局生效)` : ""}`);
    }
    this.syncPanel();
    this.maybeRunAi();
  }
}
