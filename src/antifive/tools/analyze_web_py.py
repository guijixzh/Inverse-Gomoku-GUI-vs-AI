# -*- coding: utf-8 -*-
"""跨引擎对拍统计(通用版):网页端 vs Python 端。

只统计「双方都真的下过棋」的对局(steps 同时含 PY 与 WEB),并按棋谱重放重判胜负,
不信任棋谱里的 winner 字段。输出:终端表格 + report.md + 图。

用法:
    python records/web_vs_py_15s_v2/analyze.py --out records/web_vs_py_15s_v2
"""
import argparse
import glob
import json
import math
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from antifive.reversegomoku import WHITE, GameConfig, ReverseGomoku

TIERS = ["beginner", "mid", "advanced", "hell", "legend"]
LABELS = {"beginner": "初级", "mid": "中级", "advanced": "高级",
          "hell": "大师", "legend": "传奇"}
CFG = GameConfig(loss_start_turns=8, white_restrict_turns=2, mask_suicide=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - r) / d), min(1.0, (c + r) / d)


def elo(p):
    p = min(1 - 1e-6, max(1e-6, p))
    return -400.0 * math.log10(1.0 / p - 1.0)


def n_needed(p):
    if p <= 0.5:
        return None
    for n in range(1, 3001):
        lo, _ = wilson(round(p * n), n)
        if lo > 0.5:
            return n
    return None


def true_winner(g):
    gg = ReverseGomoku(CFG)
    for mv in g["moves"]:
        gg.make_move(mv, allow_suicide=True)
    if gg.loser == 0:
        return "draw"
    loser = "white" if gg.loser == WHITE else "black"
    return "web" if loser == g["py_color"] else "py"


def load(run):
    games, invalid = [], 0
    for f in sorted(glob.glob(os.path.join(run, "games", "*", "*.json"))):
        g = json.loads(open(f, encoding="utf-8").read())
        if {s[1] for s in g.get("steps", [])} != {"PY", "WEB"}:
            invalid += 1
            continue
        g["winner"] = true_winner(g)
        games.append(g)
    return games, invalid


def aggregate(games):
    out = {t: {"n": 0, "py": 0, "web": 0, "draw": 0} for t in TIERS}
    by_color = {"black": [0, 0], "white": [0, 0]}  # [py 胜, 总数]
    py_nodes, web_nodes, py_sec, web_sec = [], [], [], []
    for g in games:
        t = g["py"]
        if t not in out:
            continue
        d = out[t]
        d["n"] += 1
        d[g["winner"]] += 1
        if g["winner"] != "draw":
            c = g["py_color"]
            by_color[c][1] += 1
            if g["winner"] == "py":
                by_color[c][0] += 1
        for step in g["steps"]:
            if step[1] == "PY":
                py_nodes.append(step[3])
                py_sec.append(step[2])
            elif step[1] == "WEB":
                web_nodes.append(step[3])
                web_sec.append(step[2])
    return out, py_nodes, web_nodes, py_sec, web_sec, by_color


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="records/web_vs_py_15s_v2")
    ap.add_argument("--budget", type=float, default=15.0)
    args = ap.parse_args()
    run = args.out

    games, invalid = load(run)
    for g in games:
        pn = [x[3] for x in g["steps"] if x[1] == "PY"]
        g["_py_nodes_med"] = st.median(pn) if pn else 0.0
    s, py_nodes, web_nodes, py_sec, web_sec, by_color = aggregate(games)
    rows = []
    for t in TIERS:
        d = s[t]
        n = d["n"] - d["draw"]
        p = d["py"] / n if n else 0.0
        lo, hi = wilson(d["py"], n)
        rows.append((t, d, p, lo, hi))
    tot_py = sum(d["py"] for d in s.values())
    tot_web = sum(d["web"] for d in s.values())
    tot_n = tot_py + tot_web
    tp = tot_py / tot_n if tot_n else 0.0
    tlo, thi = wilson(tot_py, tot_n)

    for t, d, p, lo, hi in rows:
        n = d["n"] - d["draw"]
        need = n_needed(p)
        print(f"{t:9s} n={n:3d} py={d['py']:3d} web={d['web']:3d} {p:6.1%} "
              f"CI[{lo:.1%},{hi:.1%}] elo={elo(p):+6.0f} "
              f"CI[{elo(lo):+.0f},{elo(hi):+.0f}] "
              f"{'显著' if lo > 0.5 else ('显著(网页)' if hi < 0.5 else '打平')}"
              + (f"  需n≈{need}" if need else ""))
    print(f"合计 n={tot_n} py={tot_py} web={tot_web} {tp:.1%} "
          f"CI[{tlo:.1%},{thi:.1%}] elo={elo(tp):+.0f} CI[{elo(tlo):+.0f},{elo(thi):+.0f}]")
    print(f"nodes/step median: py={st.median(py_nodes):,.0f} "
          f"web={st.median(web_nodes):,.0f} (x{st.median(web_nodes)/max(1, st.median(py_nodes)):.1f})")
    print(f"sec/step median: py={st.median(py_sec):.2f} web={st.median(web_sec):.2f}")
    print(f"无效对局:{invalid}")

    # --- chart ---
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    xs = range(len(rows))
    vals = [r[2] * 100 for r in rows]
    errs = [[max(0.0, (r[2] - r[3]) * 100) for r in rows],
            [max(0.0, (r[4] - r[2]) * 100) for r in rows]]
    bars = ax.bar(xs, vals, 0.55, color="#4472c4", yerr=errs, capsize=4,
                  error_kw=dict(lw=1.0, ecolor="#333333"))
    for b, r in zip(bars, rows):
        n = r[1]["n"] - r[1]["draw"]
        ax.text(b.get_x() + b.get_width() / 2, 6.0,
                f"{r[2]:.0%}\n({r[1]['py']}:{n - r[1]['py']})", ha="center",
                va="bottom", fontsize=9, color="white", fontweight="bold")
    ax.axhline(50, ls="--", lw=0.9, color="#888888")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([LABELS[r[0]] for r in rows])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Python 端胜率 %")
    ns = [r[1]["n"] - r[1]["draw"] for r in rows]
    span = f"每档 {min(ns)} 局" if min(ns) == max(ns) else f"每档 {min(ns)}–{max(ns)} 局"
    ax.set_title(f"{args.budget:g}s 同墙钟:五档 网页端 vs Python 端"
                 f"({span},合计 {tot_n} 局,Python {tp:.0%})", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    chart = os.path.join(run, "web_vs_py_chart.png")
    fig.savefig(chart, dpi=150)

    # --- report ---
    L = []
    L.append(f"# {args.budget:g}s 同墙钟:五档 网页端 vs Python 端\n")
    if tlo > 0.5:
        verdict = f"**Python 端更强约 {elo(tp):.0f} Elo**"
    elif thi < 0.5:
        verdict = f"**网页端更强约 {abs(elo(tp)):.0f} Elo**"
    else:
        verdict = "**两端整体打平**(合计 95% CI 跨过 50%)"
    fast = [g for g in games if g.get("_py_nodes_med", 0) >= 25000 and g["winner"] != "draw"]
    slow = [g for g in games if g.get("_py_nodes_med", 0) < 20000 and g["winner"] != "draw"]
    fk = sum(1 for g in fast if g["winner"] == "py")
    sk = sum(1 for g in slow if g["winner"] == "py")
    load_note = ""
    if fast and slow:
        flo, fhi = wilson(fk, len(fast))
        load_note = (f"  但成绩对机器负载极敏感:安静档(≥25k 节点/步)Python "
                     f"{fk}/{len(fast)}={fk / len(fast):.1%}(CI {flo:.1%}–{fhi:.1%}),"
                     f"忙碌档(<20k)只有 {sk}/{len(slow)}={sk / len(slow):.1%}。")
    L.append(f"> 一句话结论:同档位、同 {args.budget:g}s/步、有效对局 {tot_n} 局,"
             f"**Python {tot_py} : 网页 {tot_web}**(Python 胜率 {tp:.1%},"
             f"折合 {elo(tp):+.0f} Elo,95% CI {elo(tlo):+.0f}~{elo(thi):+.0f}),{verdict}。"
             + load_note)
    L.append("")
    L.append("## 1. 方法\n")
    L.append(f"- 工具:`src/antifive/tools/match_web_py.py` + `web/scripts/engine-bridge.mjs`。")
    L.append(f"- 同档位对同档位,`depth=8`,4 步随机开局同步喂双方,黑白互换,10 路并发"
             "(每局独占一核——并发数是实验条件,不再随意提高)。")
    L.append(f"- 双方每步 **{args.budget:g} 秒墙钟**:Python `time_budget`,网页端 `budget`(不限节点)。")
    L.append(f"- 每局结束自检「必须同时出现 PY 与 WEB 步」;统计时只保留有效对局,"
             f"胜负按棋谱重放重判(本次剔除 {invalid} 局)。\n")
    L.append("## 2. 结果\n")
    L.append("| 档位 | Python : 网页 | n | Python 胜率 | 95% CI | Elo(Python − 网页) | 95% 结论 |")
    L.append("| --- | ---: | ---: | ---: | ---: | --- | --- |")
    for t, d, p, lo, hi in rows:
        n = d["n"] - d["draw"]
        tag = "Python 显著" if lo > 0.5 else ("网页显著" if hi < 0.5 else "打平/证据不足")
        L.append(f"| {LABELS[t]} | {d['py']} : {n - d['py']} | {n} | {p:.1%} | "
                 f"{lo:.1%}–{hi:.1%} | {elo(p):+.0f}({elo(lo):+.0f}~{elo(hi):+.0f}) | {tag} |")
    tag_all = "Python 显著" if tlo > 0.5 else ("网页显著" if thi < 0.5 else "打平/证据不足")
    L.append(f"| **合计** | **{tot_py} : {tot_web}** | **{tot_n}** | **{tp:.1%}** | "
             f"**{tlo:.1%}–{thi:.1%}** | **{elo(tp):+.0f}({elo(tlo):+.0f}~{elo(thi):+.0f})** | **{tag_all}** |")
    L.append("")
    L.append("![五档同墙钟](web_vs_py_chart.png)\n")
    L.append("## 3. 算力对照(每步)\n")
    L.append(f"- 节点/步中位:Python **{st.median(py_nodes):,.0f}**,网页端 "
             f"**{st.median(web_nodes):,.0f}**({st.median(web_nodes)/max(1, st.median(py_nodes)):.1f}×)。")
    L.append(f"- 秒/步中位:Python **{st.median(py_sec):.2f}s**,网页端 **{st.median(web_sec):.2f}s**"
             f"(都在 {args.budget:g}s 附近用满)。\n")
    L.append("## 3.1 颜色平衡")
    L.append("")
    L.append("| Python 执 | Python 胜 / 局数 | 胜率 |")
    L.append("| --- | ---: | ---: |")
    for c, name in (("black", "黑"), ("white", "白")):
        w, n = by_color[c]
        L.append(f"| {name} | {w} / {n} | {w / n:.1%} |" if n else f"| {name} | 0 / 0 | – |")
    L.append("")
    L.append("黑白必须各半,否则整张表会被先后手差异带偏。"
             "上一批 162 局的清洗方式恰好只留下了 Python 执黑的偶数局,"
             "所以那版得出的是「Python 43.2%」——那是偏样本,不能代表整体。")
    L.append("")
    L.append("## 3.2 负载分层:Python 端对机器负载非常敏感")
    L.append("")
    strata = [("快档(py ≥ 25k 节点/步)", lambda g: g["_py_nodes_med"] >= 25000),
              ("中档(20k–25k)", lambda g: 20000 <= g["_py_nodes_med"] < 25000),
              ("慢档(<20k)", lambda g: g["_py_nodes_med"] < 20000)]
    L.append("| 分段 | 局数 | Python 胜 | Python 胜率 | 95% CI | Python 节点/步(中位) |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, pick in strata:
        sel = [g for g in games if pick(g) and g["winner"] != "draw"]
        n = len(sel)
        k = sum(1 for g in sel if g["winner"] == "py")
        if not n:
            continue
        lo, hi = wilson(k, n)
        L.append(f"| {name} | {n} | {k} | {k / n:.1%} | {lo:.1%}–{hi:.1%} | "
                 f"{st.median([g['_py_nodes_med'] for g in sel]):,.0f} |")
    L.append("")
    L.append("机器越空,Python 端拿到的节点越多、成绩越好;网页端是单线程 JS,每节点快一个数量级,"
             "负载高时只是少走点深度,不会塌。所以「同 15s 谁强」本身就依赖机器负载:")
    L.append("安静机器上两端接近(快档 95% CI 跨 50%),忙碌机器上网页端明显占优。")
    L.append("")
    L.append("## 4. 结论\n")
    items = []
    if tlo > 0.5:
        items.append(f"同墙钟下 Python 端更强 **{elo(tp):.0f} Elo**(95% CI "
                     f"{elo(tlo):+.0f}~{elo(thi):+.0f})。")
    elif thi < 0.5:
        items.append(f"同墙钟下网页端更强 **{abs(elo(tp)):.0f} Elo**(95% CI "
                     f"{elo(tlo):+.0f}~{elo(thi):+.0f})。")
    else:
        items.append(f"同墙钟下两端没有显著差异(合计 {tp:.1%},95% CI {tlo:.1%}–{thi:.1%} 跨过 50%)。")
    sig_py = [t for t, d, p, lo, hi in rows if lo > 0.5]
    sig_web = [t for t, d, p, lo, hi in rows if hi < 0.5]
    if sig_py:
        items.append(f"单档 Python 显著领先:{'、'.join(LABELS[t] for t in sig_py)}。")
    if sig_web:
        items.append(f"单档网页端显著领先:{'、'.join(LABELS[t] for t in sig_web)}。")
    rest = [t for t, d, p, lo, hi in rows if lo <= 0.5 <= hi]
    if rest:
        span_n = f"{min(ns)}" if min(ns) == max(ns) else f"{min(ns)}–{max(ns)}"
        items.append(f"其余档位未达 95% 显著:{'、'.join(LABELS[t] for t in rest)}"
                     f"(单档 n={span_n},区间较宽)。")
    items.append("两端「同 X 秒」≠「同等搜索量」:网页端节点数多一个数量级,Python 端每节点更强"
                 "(等节点对拍:网页 7k 节点时 Python 14:4,13k 节点时 11:7)。")
    for i, it in enumerate(items, 1):
        L.append(f"{i}. {it}")
    L.append("")
    open(os.path.join(run, "report.md"), "w", encoding="utf-8").write("\n".join(L))
    print(f"写回 {run}/report.md 与 web_vs_py_chart.png")


if __name__ == "__main__":
    main()
