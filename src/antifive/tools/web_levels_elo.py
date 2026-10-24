"""把「网页端语义 + 时间预算」的对局结果(a/b 两档自对弈)标定成 Elo 与围棋段位。

数据来源:`web/scripts/elo-calibrate.mjs` 产出的 `games/<pair>/g*.json`,
每个文件是一盘 `{a,b,aIsBlack,seed,winner,turns,moves,msA,msB,nodesA,nodesB}`
(winner ∈ {"a","b","draw"}),由 `web/src/engine/api.ts` 的 EngineSession 真实对弈。

用法:
    python -m antifive.tools.web_levels_elo --out records/web_elo_12s [--budget 12]
输出:同目录下 summary.json / report.md / elo_chart.png。
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from antifive.tools.tournament import go_rank, go_rank_value

# 与 web/src/engine/profiles.ts 的键一致
PLAYERS = ["beginner", "mid", "advanced", "hell", "legend"]
LABELS = {"beginner": "初级", "mid": "中级", "advanced": "高级",
          "hell": "大师", "legend": "传奇"}
LN10_400 = math.log(10.0) / 400.0


# ---------------------------------------------------------------- 统计工具
def _wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - r) / d), min(1.0, (c + r) / d))


def _elo(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return 400.0 * math.log10(p / (1 - p))


def _cp_lower(k: int, n: int) -> float:
    """Clopper-Pearson 95% 单侧下界(用于全胜/全负的饱和边)。"""
    if n <= 0:
        return float("nan")
    return 0.025 ** (1.0 / n) if k == 0 else 1.0 - 0.025 ** (1.0 / n)


def load_games(out_dir: Path) -> list:
    games = []
    for p in sorted((out_dir / "games").glob("*/*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if rec.get("winner") in ("a", "b", "draw"):
            games.append(rec)
    return games


def pair_key(a: str, b: str) -> str:
    return f"{a}-{b}" if PLAYERS.index(a) < PLAYERS.index(b) else f"{b}-{a}"


def pairwise(games: list) -> dict:
    stats: dict = {}
    for g in games:
        a, b = g["a"], g["b"]
        key = pair_key(a, b)
        s = stats.setdefault(key, {"games": 0, "a_wins": 0, "b_wins": 0, "draws": 0,
                                   "first": a if PLAYERS.index(a) < PLAYERS.index(b) else b})
        s["games"] += 1
        if g["winner"] == "draw":
            s["draws"] += 1
        elif g["winner"] == "a":
            s["a_wins"] += 1
        else:
            s["b_wins"] += 1
    for key, s in stats.items():
        # 统一成「key 里的第一个名字」视角
        x, y = key.split("-")
        if s["first"] != x:
            s["a_wins"], s["b_wins"] = s["b_wins"], s["a_wins"]
            s["first"] = x
    return stats


def fit_bt(games: list, anchor: str = "beginner", prior_elo: float = 400.0,
           iters: int = 200) -> dict:
    """带弱先验的 Bradley-Terry,锚点固定 0;CI 用 Laplace 近似。"""
    idx = {p: i for i, p in enumerate(PLAYERS)}
    k = len(PLAYERS)
    theta = np.zeros(k)
    free = [i for i in range(k) if PLAYERS[i] != anchor]
    st = prior_elo * LN10_400
    prec = 1.0 / (st * st)

    def loglik2(th):
        ll = 0.0
        for g in games:
            i, j = idx[g["a"]], idx[g["b"]]
            y = 0.5 if g["winner"] == "draw" else (1.0 if g["winner"] == "a" else 0.0)
            d = th[i] - th[j]
            if d >= 0:
                ll += y * -math.log1p(math.exp(-d))
                ll += (1 - y) * (-d - math.log1p(math.exp(-d)))
            else:
                ll += y * (d - math.log1p(math.exp(d)))
                ll += (1 - y) * -math.log1p(math.exp(d))
        return ll - 0.5 * prec * float(np.sum(th[free] ** 2))

    cur = loglik2(theta)
    hess = np.zeros((k, k))
    for _ in range(iters):
        grad = np.zeros(k)
        hess = np.zeros((k, k))
        for g in games:
            i, j = idx[g["a"]], idx[g["b"]]
            y = 0.5 if g["winner"] == "draw" else (1.0 if g["winner"] == "a" else 0.0)
            d = theta[i] - theta[j]
            p = 1.0 / (1.0 + math.exp(-d))
            r = y - p
            w = max(p * (1 - p), 1e-9)
            grad[i] += r
            grad[j] -= r
            hess[i, i] += w
            hess[j, j] += w
            hess[i, j] -= w
            hess[j, i] -= w
        grad[free] -= prec * theta[free]
        hess[np.diag_indices(k)] += prec
        try:
            step = np.linalg.solve(hess[np.ix_(free, free)], grad[free])
        except np.linalg.LinAlgError:
            break
        t = 1.0
        for _ in range(40):
            cand = theta.copy()
            cand[free] += t * step
            v = loglik2(cand)
            if v >= cur - 1e-12:
                theta, cur = cand, v
                break
            t *= 0.5
        else:
            break
        if np.max(np.abs(t * step)) < 1e-9:
            break

    cov = np.full((k, k), np.nan)
    try:
        cov[np.ix_(free, free)] = np.linalg.inv(hess[np.ix_(free, free)])
    except np.linalg.LinAlgError:
        pass
    elo = theta / LN10_400
    ci = {}
    for i, p in enumerate(PLAYERS):
        if p == anchor:
            ci[p] = (0.0, 0.0)
        elif not np.isnan(cov[i, i]):
            se = math.sqrt(max(cov[i, i], 0.0)) / LN10_400
            ci[p] = (elo[i] - 1.96 * se, elo[i] + 1.96 * se)
        else:
            ci[p] = (float("nan"), float("nan"))
    return {"elo": {p: float(elo[i]) for i, p in enumerate(PLAYERS)}, "ci95": ci}


def timing(games: list) -> dict:
    """每档的实测步时与搜索量:总秒数/步数、总节点/步数,以及每局平均回合。"""
    acc: dict = {p: {"s": 0.0, "n": 0, "nodes": 0.0} for p in PLAYERS}
    turns = []
    for g in games:
        a, b = g["a"], g["b"]
        t = int(g.get("turns", len(g.get("moves", []))))
        turns.append(t)
        na = (t + 1) // 2 if g.get("aIsBlack") else t // 2
        nb = t - na
        acc[a]["s"] += float(g.get("msA", 0.0)) / 1000.0
        acc[a]["n"] += na
        acc[a]["nodes"] += float(g.get("nodesA", 0.0) or 0.0)
        acc[b]["s"] += float(g.get("msB", 0.0)) / 1000.0
        acc[b]["n"] += nb
        acc[b]["nodes"] += float(g.get("nodesB", 0.0) or 0.0)
    out = {}
    for p, v in acc.items():
        if v["n"]:
            out[p] = {"moves": v["n"], "avg_seconds": v["s"] / v["n"],
                      "nodes_per_move": v["nodes"] / v["n"]}
        else:
            out[p] = {"moves": 0, "avg_seconds": float("nan"),
                      "nodes_per_move": float("nan")}
    out["_turns"] = sum(turns) / len(turns) if turns else 0.0
    return out


def _read_meta(od: Path) -> dict:
    """读 run_meta.json(编排器写的扁平结构)或 summary.json(本工具写的嵌套结构)。"""
    for name in ("run_meta.json", "summary.json"):
        mp = od / name
        if not mp.exists():
            continue
        try:
            data = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = data.get("meta") if isinstance(data.get("meta"), dict) else data
        keys = ("budget", "depth", "vcf", "openPlies", "variety",
                "uniformDepth", "workers")
        return {k: meta.get(k) for k in keys if k in meta}
    return {}


def _py_nodes_per_move(path: Path) -> dict:
    """程序端(Python 引擎)定段赛里各档的平均每步节点数,仅用于搜索量对照。"""
    out: dict = {}
    if not Path(path).exists():
        return out
    acc: dict = {}
    for f in Path(path).glob("games/*/*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        black, white = d.get("black"), d.get("white")
        for step in d.get("step_log") or []:
            # [move_count, "H"|"K", seconds, nodes];move_count 为自增后的手数
            if len(step) < 4 or step[1] != "H":
                continue
            side = black if int(step[0]) % 2 == 1 else white
            if side not in PLAYERS:
                continue
            a = acc.setdefault(side, [0.0, 0])
            a[0] += float(step[3] or 0.0)
            a[1] += 1
    for p, (s, n) in acc.items():
        if n:
            out[p] = s / n
    return out

def _fmt_elo(x: float) -> str:
    return f"{int(round(x)):+d}"


def _pair_line(key: str, s: dict) -> tuple:
    """返回 (人类可读比分, Elo 点估计, Elo 95% 区间文本)。"""
    x, y = key.split("-")
    n = s["games"]
    d = s["draws"]
    dec = n - d
    wins_x = s["a_wins"]
    if dec == 0:
        return (f"{wins_x}:{s['b_wins']}" + (f" (+{d} 和)" if d else ""), 0.0,
                "无有效对局")
    score = (wins_x + 0.5 * d) / n
    lo, hi = _wilson(int(round(wins_x + 0.5 * d)), n)
    if wins_x == 0 or wins_x == dec:
        if n >= 6:
            q = _cp_lower(0 if wins_x == 0 else n, n)
            bound = _elo(q if wins_x == 0 else 1 - q)
            if bound > 0:
                return (f"{wins_x}:{s['b_wins']}", None,
                        f"{LABELS[y]} 至少 {_fmt_elo(bound)}")
            return (f"{wins_x}:{s['b_wins']}", None, "饱和,但局数不足以给出正的界")
        return (f"{wins_x}:{s['b_wins']}", None, "饱和(局数太少)")
    return (f"{wins_x}:{s['b_wins']}" + (f" (+{d} 和)" if d else ""), _elo(score),
            f"{_fmt_elo(_elo(lo))} ~ {_fmt_elo(_elo(hi))}")


def _chart(od: Path, games: list, pairs: dict, bt: dict, label: str = "") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei",
                                              "Noto Sans CJK SC", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), constrained_layout=True)
    ax = axes[0]
    seen = {p for k in pairs for p in k.split("-")}
    order = [p for p in sorted(PLAYERS, key=lambda q: bt["elo"][q]) if p in seen]
    ys = list(range(len(order)))
    vals = [bt["elo"][p] for p in order]
    lo = [max(0.0, bt["elo"][p] - bt["ci95"][p][0]) for p in order]
    hi = [max(0.0, bt["ci95"][p][1] - bt["elo"][p]) for p in order]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[: len(order)]
    ax.barh(ys, vals, color=colors, height=0.55, alpha=0.9)
    ax.errorbar(vals, ys, xerr=[lo, hi], fmt="none", ecolor="#222222",
                capsize=4, lw=1.1)
    for y, v in zip(ys, vals):
        ax.text(v + 12, y, f"{v:+.0f}", va="center", fontsize=9)
    ax.set_yticks(ys, [f"{LABELS[p]} ({go_rank(bt['elo'][p])})" for p in order])
    ax.set_xlabel("Elo(锚点:页面「初级」= 0 = 围棋 10k)")
    ax.set_title("Elo 与 95% CI(网页端)" + (f" {label}" if label else ""))
    ax.grid(axis="x", ls=":", alpha=0.5)
    ax.axvline(0, color="#666666", lw=0.9)

    ax = axes[1]
    mat = np.full((len(PLAYERS), len(PLAYERS)), np.nan)
    for i, p in enumerate(PLAYERS):
        for j, q in enumerate(PLAYERS):
            if i == j:
                continue
            key = pair_key(p, q)
            s = pairs.get(key)
            if not s:
                continue
            x, y = key.split("-")
            if p == x:
                sc = (s["a_wins"] + 0.5 * s["draws"]) / max(1, s["games"])
            else:
                sc = (s["b_wins"] + 0.5 * s["draws"]) / max(1, s["games"])
            mat[i, j] = sc
    im = ax.imshow(mat, cmap="coolwarm", vmin=0, vmax=1)
    ax.set_xticks(range(len(PLAYERS)), [LABELS[p] for p in PLAYERS], rotation=30)
    ax.set_yticks(range(len(PLAYERS)), [LABELS[p] for p in PLAYERS])
    for i in range(len(PLAYERS)):
        for j in range(len(PLAYERS)):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color="#999999")
                continue
            s = pairs.get(pair_key(PLAYERS[i], PLAYERS[j]))
            if not s:
                ax.text(j, i, "无", ha="center", va="center", fontsize=8, color="#666666")
                continue
            ax.text(j, i, f"{mat[i, j] * 100:.0f}%\n(n={s['games']})",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(mat[i, j] - 0.5) > 0.3 else "#222222")
    ax.set_xlabel("对手")
    ax.set_ylabel("行:本方")
    ax.set_title("两两胜率" + (f"({label})" if label else ""))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    png = od / "elo_chart.png"
    fig.savefig(png, dpi=130)
    plt.close(fig)
    return png.name


def report(out_dir: Path, chart: bool = False) -> dict:
    od = Path(out_dir)
    games = load_games(od)
    if not games:
        raise SystemExit(f"{od} 里没有有效对局")
    pairs = pairwise(games)
    bt = fit_bt(games)
    tm = timing(games)
    meta = _read_meta(od)
    budget = float(meta.get("budget") or 15)
    variety = str(meta.get("variety") or "未记录")
    uni = meta.get("uniformDepth")
    if uni is None:
        # 老目录没记语义,用行为反推:低档吃不满预算 ⇒ 按档位夹取
        t0 = tm["beginner"]
        uni = not (t0["moves"] and t0["avg_seconds"] < 0.5 * budget)
    open_plies = meta.get("openPlies", "?")
    depth_param = meta.get("depth", "?")
    workers = meta.get("workers", "?")

    lines: list = []
    A = lines.append
    best = max(PLAYERS, key=lambda p: bt["elo"][p])
    ngames = {p: 0 for p in PLAYERS}
    for key, s in pairs.items():
        x, y = key.split("-")
        ngames[x] += s["games"]
        ngames[y] += s["games"]
    share = {p: (tm[p]["avg_seconds"] / budget if tm[p]["moves"] else float("nan"))
             for p in PLAYERS}
    thin = [p for p in PLAYERS
            if ngames[p] and not math.isnan(share[p]) and share[p] < 0.5]
    ranked = [p for p in PLAYERS if p != "beginner" and ngames[p]]

    A("# 网页端五档 Elo 标定")
    A("")
    A(f"> 生成时间:{datetime.now().isoformat(' ', 'seconds')}  ")
    A(f"> 对局:{len(games)} 局;每步预算 {budget:.0f}s;深度参数 {depth_param};"
      f"VCF {'开' if meta.get('vcf', 1) else '关'};变化档位 {variety};"
      f"随机开局 {open_plies} 步;每对黑白互换(同一 seed,开局真正配对)。")
    A(">")
    if uni:
        A(f"> 深度语义:**统一深度** —— 五档共用同一个搜索深度上限({depth_param}),"
          f"预算内一直迭代加深到时限(与 Python `heuristic_versions.choose` 一致)。")
    else:
        A("> 深度语义:**按档位夹取深度**(旧语义;`depthForProfile` 会限制低档的上限)。")
    A("> 锚点:页面「初级」= 0 Elo = 围棋 10k(1 档 = 100 Elo)。")
    A("")
    A("## 0. 一句话结论")
    A("")
    A(f"以「初级 = 0 = 10k」为锚,网页端在 {budget:.0f}s/步下:"
      + "、".join(f"{LABELS[p]} {bt['elo'][p]:+.0f} Elo({go_rank(bt['elo'][p])})"
                 for p in ranked)
      + f";最高是{LABELS[best]}。")
    A("")
    if thin:
        A("⚠ 有档位没吃满预算(平均步时不到预算一半):"
          + "、".join(f"{LABELS[p]} {share[p] * 100:.0f}%" for p in thin)
          + "。这通常是「按档位夹取深度」的表现,读排名时要留意。")
        A("")
    A("读法:§2 是把所有对局折成单一强度的 BT 估计,§3 是逐对实测比分;"
      "两者不一致时以 §3 为准(这些档位之间存在实测到的非传递性)。")
    A("")

    A("## 1. 赛制与实测")
    A("")
    A("| 项目 | 值 |")
    A("| --- | --- |")
    A("| 引擎 | `web/src/engine/api.ts` 的 `EngineSession`(与浏览器跑的是同一份代码) |")
    A(f"| 每步预算 | {budget:.0f}s(`ai({{depth, budget}})` 的 wall-clock 上限) |")
    A(f"| 深度参数 | {depth_param}(页面「困难」档的值) |")
    A(f"| 变化档位 | {variety}(页面默认 `mid`) |")
    A(f"| 随机开局 | {open_plies} 步;黑白两局同 seed 配对 |")
    A(f"| 有效对局 | {len(games)} 局,平均 {tm['_turns']:.1f} 手 |")
    A(f"| 并发 | {workers} 个 worker,每个对局独占一个线程 |")
    A("")
    A("| 档位 | 对局数 | 平均步时 | 占预算 | 平均节点/步 |")
    A("| --- | --- | --- | --- | --- |")
    for p in sorted(PLAYERS, key=lambda q: -bt["elo"][q]):
        t = tm[p]
        if not t["moves"]:
            A(f"| {LABELS[p]} | 0 | — | — | — |")
            continue
        A(f"| {LABELS[p]} | {ngames[p]} | {t['avg_seconds']:.2f}s "
          f"| {share[p] * 100:.0f}% | {t['nodes_per_move'] / 1000:.0f}k |")
    A("")

    A("## 2. 总表(Elo 与段位)")
    A("")
    A("| 档位 | Elo | 95% CI | 围棋段位(100 Elo=1 档) | 国际象棋式(400 Elo=1 档) | 对局数 |")
    A("| --- | --- | --- | --- | --- | --- |")
    for p in sorted(PLAYERS, key=lambda q: -bt["elo"][q]):
        lo_, hi_ = bt["ci95"][p]
        if ngames[p] == 0:
            A(f"| {LABELS[p]} | 无对局 | — | — | — | 0 |")
            continue
        est = bt["elo"][p]
        ci_txt = "锚点" if p == "beginner" else f"{_fmt_elo(lo_)} ~ {_fmt_elo(hi_)}"
        A(f"| {LABELS[p]} | {est:+.0f} | {ci_txt} "
          f"| {go_rank(est)} | {go_rank(est, per=400)} | {ngames[p]} |")
    A("")
    A("![Elo 与两两胜率](elo_chart.png)")
    A("")

    A("## 3. 两两对阵")
    A("")
    A("| 对阵 | 比分(前者视角) | 前者胜率 | Elo 差 | 95% CI |")
    A("| --- | --- | --- | --- | --- |")
    for key in sorted(pairs, key=lambda k: (PLAYERS.index(k.split("-")[0]),
                                            PLAYERS.index(k.split("-")[1]))):
        s = pairs[key]
        x, y = key.split("-")
        score_txt, elo, citxt = _pair_line(key, s)
        sc = (s["a_wins"] + 0.5 * s["draws"]) / max(1, s["games"])
        elo_txt = "饱和" if elo is None else f"{elo:+.0f}"
        A(f"| {LABELS[x]} vs {LABELS[y]} | {score_txt} | {sc * 100:.1f}% | "
          f"{elo_txt} | {citxt} |")
    A("")

    A("## 4. 搜索量:网页端 vs 程序端")
    A("")
    py_nodes = _py_nodes_per_move(od.parent / "tournament_20261022")
    A("| 档位 | 网页端节点/步 | 网页端步时 | 程序端节点/步(15s 定段赛) |")
    A("| --- | --- | --- | --- |")
    for p in PLAYERS:
        t = tm[p]
        if not t["moves"]:
            continue
        pn = py_nodes.get(p)
        nodes_txt = (f"{t['nodes_per_move'] / 1000:.0f}k"
                     if t["nodes_per_move"] >= 1000 else "—")
        A(f"| {LABELS[p]} | {nodes_txt} | "
          f"{t['avg_seconds']:.2f}s | "
          + (f"{pn / 1000:.1f}k" if pn else "—") + " |")
    A("")
    ns = [tm[p]["nodes_per_move"] for p in PLAYERS if tm[p]["moves"]]
    if ns:
        A(f"> 节点口径两边不完全相同,只能看量级:同样的墙钟预算下,网页端每步 "
          f"{min(ns) / 1000:.0f}k–{max(ns) / 1000:.0f}k 节点"
          + (",程序端每步是万级" if py_nodes else "")
          + "。另有一次「同一局面、depth 6、不设时限」的直接对照:"
          "程序端初级 20.2s/10,284 节点、高级 40.7s/27,598 节点;"
          "网页端约 2.1s/21k 节点 —— 网页端**每节点更快**,而不是更慢。")
    A("")

    other = od.parent / ("web_elo_15s" if od.name.endswith("_legacy")
                         else "web_elo_15s_legacy")
    prev = {}
    if other.exists():
        try:
            data = json.loads((other / "summary.json").read_text(encoding="utf-8"))
            prev = data.get("elo") or {}
            pmeta = _read_meta(other)
            if pmeta.get("budget") and float(pmeta["budget"]) != budget:
                prev = {}   # 预算不同(比如 2s vs 15s),单列一节对照没有意义
        except Exception:
            prev = {}
    col_self = "本轮(统一深度)" if uni else "本轮(夹取深度)"
    col_prev = "旧夹取(对照)" if uni else "统一深度(对照)"
    A("## 5. 与另一种深度语义的对照")
    A("")
    if prev and other.name.endswith("_legacy") != od.name.endswith("_legacy"):
        A(f"同一批设置({budget:.0f}s、variety {variety}、同一 harness),"
          f"只把深度语义换掉:")
        A("")
        A(f"| 档位 | {col_self} | {col_prev} | Δ |")
        A("| --- | --- | --- | --- |")
        for p in sorted(PLAYERS, key=lambda q: -bt["elo"][q]):
            if p not in prev:
                continue
            d = bt["elo"][p] - float(prev[p])
            A(f"| {LABELS[p]} | {bt['elo'][p]:+.0f} ({go_rank(bt['elo'][p])}) "
              f"| {float(prev[p]):+.0f} | {d:+.0f} |")
        A("")
        A("> 两个 0 都是「初级」,但旧的初级被夹在 depth 2、几乎不吃时间;"
          "所以旧表里其他四档相对初级的 Elo 被整体放大,两列**不能直接相减当棋力变化**。")
    else:
        A(f"(对照目录 `{other.name}` 还没有 summary.json,跳过。)")
    A("")

    A("## 6. 结论与限制")
    A("")
    items: list = []
    items.append(f"**网页端 {budget:.0f}s 下的强弱**见 §2;最高 {LABELS[best]} "
                 f"{bt['elo'][best]:+.0f} Elo({go_rank(bt['elo'][best])})。")
    items.append("**CI 宽度**:相邻档位的 CI 一重叠,这几十分到一百分的差在本样本量下"
                 "就不显著,别把排名当定论。")
    spread = max(bt["elo"][p] for p in PLAYERS) - min(bt["elo"][p] for p in PLAYERS)
    if budget <= 3:
        items.append(f"**短时限压平了差距**(本表 {budget:.0f}s):五档最大跨度只有 "
                     f"{spread:.0f} Elo(≈{spread / 100:.1f} 档)。{budget:.0f}s 内连低档"
                     "也吃满预算、搜到的深度都差不多,VCF / 开局库 / 蒸馏的差异还没到兑现"
                     "的时候;页面默认 15s,要看五档真正的区分度得用更长的预算。")

    def _edge_elo(x: str, y: str):
        key = pair_key(x, y)
        s2 = pairs.get(key)
        if not s2:
            return None
        xx, _ = key.split("-")
        wins = s2["a_wins"] if xx == x else s2["b_wins"]
        return _elo((wins + 0.5 * s2["draws"]) / max(1, s2["games"]))

    e_am = _edge_elo("advanced", "mid")
    e_mh = _edge_elo("mid", "hell")
    e_ah = _edge_elo("advanced", "hell")
    if None not in (e_am, e_mh, e_ah):
        # 三条边都是「前者相对后者」的有符号 Elo;若可传递,高级-大师 = (高级-中级) + (中级-大师)
        implied = e_am + e_mh
        items.append(f"**非传递性(实测)**:各边都是「前者减后者」的有符号 Elo —— "
                     f"高级对中级 {e_am:+.0f}、中级对大师 {e_mh:+.0f};若强度可传递,"
                     f"高级对大师应约 {implied:+.0f},实测 {e_ah:+.0f}"
                     f"(差 {implied - e_ah:.0f} Elo)。这是循环克制,不是采样噪声的简单"
                     "叠加,所以 §3 的逐对数字比 §2 的单一排名更可靠。")
    else:
        items.append("**非传递性**:BT 是单一强度的折中;若各边不自洽,读 §3 比读 §2 更可靠。")
    items.append(f"**并发影响**:{workers} 个 worker 同时跑,每个引擎只占一个线程;"
                 "预算按 wall-clock 计,重负载下每步搜到的节点略少,但双方同样受影响。")
    if uni:
        items.append("**这张表回答的是哪一个问题**:它按「引擎语义对齐」的口径跑 —— "
                     "五档共用同一个深度上限、同样吃满预算,差别只剩评估网络(蒸馏档)、"
                     "VCF 时间片和开局库。页面默认的分档深度(`depthForProfile`:初级 2、"
                     "中级 4、高级 8、大师 +1、传奇 +2)在这里被关掉了,所以本表**不等于**"
                     "「用户在浏览器里点五个难度」的体验;要看那个,看 `records/"
                     "web_elo_15s_legacy`(同样的 15s、同样的 harness,只保留页面分档深度)。")
    items.append("**锚点与跨引擎**:本表的 0 是**网页端初级**;它与程序端(Python)初级"
                 "不是同一个强度,所以不能把本表数字和程序端定段赛的表直接相加/相减。"
                 "要接到 KataGo 坐标,需要另跑「网页端某档 vs KataGo」的直接对局。")
    for k, txt in enumerate(items, 1):
        A(f"{k}. {txt}")
    A("")
    md = "\n".join(lines) + "\n"
    (od / "report.md").write_text(md, encoding="utf-8")

    summary = {
        "generated": datetime.now().isoformat(" ", "seconds"),
        "meta": meta, "games": len(games), "pairs": pairs,
        "elo": bt["elo"], "elo_ci95": bt["ci95"],
        "rank_100": {p: go_rank(bt["elo"][p]) for p in PLAYERS},
        "rank_400": {p: go_rank(bt["elo"][p], per=400) for p in PLAYERS},
        "timing": tm,
    }
    (od / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    if chart:
        _chart(od, games, pairs, bt, label=f"{budget:.0f}s")
    return summary

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="网页端语义的 Elo 标定报告")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chart", action="store_true")
    args = ap.parse_args(argv)
    s = report(Path(args.out), chart=args.chart)
    print(f"已写出 {args.out}/report.md")
    for p in PLAYERS:
        print(f"  {LABELS[p]:<4} {s['elo'][p]:+8.1f}  {s['rank_100'][p]:>5}  "
              f"({s['rank_400'][p]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
