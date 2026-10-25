"""桌面端五档自标定:初级 / 中级 / 高级 / 大师 / 传奇(全部 15 秒等时)

用途:给主 README 提供「桌面端自身」的 Elo 表——以最低档(初级)= 0 为锚点,
五档两两对弈后用 Bradley-Terry 拟合。与 `match_web_py`(桌面端 vs 网页端)互补:
前者量桌面端的档位阶梯,后者量同名档位在两端的强弱差。

赛程:10 个unordered 配对,每对 n 局(黑白各半,同一开局用于黑白两局),
每步 15 秒、depth=8、white_restrict=2、loss_start=8(与 GUI 一致)。

用法:
    python -m antifive.tools.py_levels_elo run --out records/py_elo_15s \
        --games-per-pair 6 --time 15 --depth 8 --workers 10 --deadline 02:00
    python -m antifive.tools.py_levels_elo report --out records/py_elo_15s --chart

中断后对同一 --out 重跑即续跑(已完成对局自动跳过)。
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np

from .. import heuristic_versions as hv
from ..heuristic import _TT
from ..reversegomoku import BLACK, WHITE, GameConfig, ReverseGomoku
from .match_katago import generate_openings
from .web_levels_elo import LABELS, PLAYERS, fit_bt, load_games, pairwise

PAIRS = [f"{a}-{b}" for a, b in combinations(PLAYERS, 2)]


# --------------------------------------------------------------------------
# 对局
# --------------------------------------------------------------------------
def _play(job: dict) -> dict:
    try:
        import torch as _torch
        _torch.set_num_threads(1)
    except Exception:
        pass

    cfg = GameConfig(loss_start_turns=job["loss_start"],
                     white_restrict_turns=job["white_restrict"],
                     mask_suicide=True)
    g = ReverseGomoku(cfg)
    for m in job["opening"]:
        g.make_move(m)

    a_is_black = job["a_black"]
    black_v = job["a"] if a_is_black else job["b"]
    white_v = job["b"] if a_is_black else job["a"]
    tts = {BLACK: _TT(), WHITE: _TT()}
    rng = np.random.default_rng(job["seed"])
    secs = {BLACK: [], WHITE: []}
    nodes = {BLACK: [], WHITE: []}

    while not g.game_over and g.move_count < job["max_steps"]:
        player = g.current_player
        version = black_v if player == BLACK else white_v
        cnt = [0]
        t0 = time.perf_counter()
        mv = hv.choose(version, g.board, player, g.pending, g.turn_count,
                       g.white_turns, g.config, rng, depth=job["depth"],
                       time_budget=job["time"],
                       progress_cb=lambda n: cnt.__setitem__(0, n),
                       tt=tts[player],
                       use_vcf=(version != "beginner"))
        secs[player].append(round(time.perf_counter() - t0, 3))
        nodes[player].append(int(cnt[0]))
        if mv is None:
            g.game_over = True
            g.loser = player
            break
        g.make_move(mv, allow_suicide=True)

    if not g.game_over:
        g.game_over = True
        g.is_draw = True

    if g.is_draw:
        winner = "draw"
    else:
        if g.loser == BLACK:
            winner = "b" if a_is_black else "a"
        else:
            winner = "a" if a_is_black else "b"

    a_player = BLACK if a_is_black else WHITE
    rec = {
        "pair": job["pair"],
        "game_id": job["game_id"],
        "a": job["a"],
        "b": job["b"],
        "a_black": a_is_black,
        "seed": job["seed"],
        "winner": winner,
        "turns": g.turn_count,
        "moves": g.record(),
        "a_sec": secs[a_player],
        "b_sec": secs[BLACK if a_player == WHITE else WHITE],
        "a_nodes": nodes[a_player],
        "b_nodes": nodes[BLACK if a_player == WHITE else WHITE],
        "wall": round(sum(secs[BLACK]) + sum(secs[WHITE]), 1),
    }
    out = Path(job["out_dir"]) / "games" / job["pair"]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"g{job['game_id']:04d}.json").write_text(
        json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return rec


def _existing(out_dir: Path) -> set:
    return {(p.parent.name, p.stem) for p in (out_dir / "games").glob("*/*.json")}


def build_jobs(out_dir: str, openings: list, games_per_pair: int,
               per_pair: dict | None, *, time_s: float, depth: int,
               loss_start: int, white_restrict: int, max_steps: int,
               seed: int) -> list:
    jobs = []
    for pair in PAIRS:
        a, b = pair.split("-")
        n = (per_pair or {}).get(pair, games_per_pair)
        for i in range(n):
            op = openings[(i // 2) % len(openings)]
            jobs.append({
                "pair": pair, "a": a, "b": b,
                "game_id": i, "opening": op, "a_black": i % 2 == 0,
                "seed": seed + i * 7919,
                "time": time_s, "depth": depth, "loss_start": loss_start,
                "white_restrict": white_restrict, "max_steps": max_steps,
                "out_dir": out_dir,
            })
    random.Random(seed).shuffle(jobs)
    return jobs


def _parse_pair_games(text: str) -> dict:
    out = {}
    for chunk in (text or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, val = chunk.partition("=")
        out[name.strip()] = int(val)
    return out


def run(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = GameConfig(loss_start_turns=args.loss_start,
                     white_restrict_turns=args.white_restrict,
                     mask_suicide=True)
    openings = generate_openings(max(1, args.openings), args.open_plies,
                                 args.seed, cfg)
    jobs = build_jobs(str(out_dir), openings, args.games_per_pair,
                      _parse_pair_games(getattr(args, "pair_games", "")),
                      time_s=args.time, depth=args.depth,
                      loss_start=args.loss_start,
                      white_restrict=args.white_restrict,
                      max_steps=args.max_steps, seed=args.seed)
    done = _existing(out_dir)
    jobs = [j for j in jobs
            if (j["pair"], f"g{j['game_id']:04d}") not in done]
    print(f"待跑 {len(jobs)} 局,workers={args.workers}", flush=True)

    deadline = None
    if getattr(args, "deadline", ""):
        hh, mm = args.deadline.split(":")
        now = time.localtime()
        deadline = time.mktime((now.tm_year, now.tm_mon, now.tm_mday,
                                int(hh), int(mm), 0, 0, 0, -1))
        if deadline < time.time():
            deadline += 86400

    t_start = time.time()
    finished = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        it = iter(jobs)
        for _ in range(args.workers):
            job = next(it, None)
            if job is None:
                break
            futs[pool.submit(_play, job)] = job
        while futs:
            for fut in as_completed(list(futs)):
                job = futs.pop(fut)
                finished += 1
                try:
                    rec = fut.result()
                except Exception as exc:      # 单局失败不影响整体
                    print(f"[{finished}/{len(jobs)}] {job['pair']} 失败: {exc}",
                          flush=True)
                else:
                    print(f"[{finished}/{len(jobs)}] {job['pair']} "
                          f"{rec['winner']} {rec['turns']}手", flush=True)
                if deadline and time.time() > deadline:
                    for f in futs:
                        f.cancel()
                    print("到达截止时间,不再开新局", flush=True)
                    for f, j in list(futs.items()):
                        if f.running():
                            continue
                        futs.pop(f, None)
                    continue
                job = next(it, None)
                if job is not None:
                    futs[pool.submit(_play, job)] = job

    meta = {k: v for k, v in vars(args).items()
            if isinstance(v, (str, int, float, bool, type(None)))}
    meta["elapsed_min"] = round((time.time() - t_start) / 60, 1)
    meta["pairs"] = PAIRS
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"用时 {(time.time() - t_start) / 60:.1f} min -> {out_dir}",
          flush=True)


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def report(out_dir: str, chart: bool = False) -> dict:
    od = Path(out_dir)
    games = load_games(od)
    if not games:
        raise SystemExit(f"{od} 里没有对局")
    pairs = pairwise(games)
    bt = fit_bt(games, anchor="beginner")

    lines = [f"# 桌面端五档自标定({len(games)} 局)", ""]
    meta_path = od / "run_meta.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    per_step = float(meta.get("time", 0.0) or 0.0)
    step_txt = f"{per_step:g}s/步" if per_step else "限时"
    depth_txt = f",depth {int(meta.get('depth', 0))}" if meta.get("depth") else ""

    lines.append("| 档位 | Elo(初级=0) | 95% CI | 局数 |")
    lines.append("| --- | ---: | --- | ---: |")
    elo, ci = bt["elo"], bt["ci95"]
    for p in sorted(PLAYERS, key=lambda x: elo[x]):
        e = elo[p]
        lo, hi = ci[p]
        n = sum(s["games"] for k, s in pairs.items() if p in k.split("-"))
        lines.append(f"| {LABELS[p]} | {e:+.0f} | {lo:+.0f} ~ {hi:+.0f} | {n} |")
    lines.append("")
    lines.append("| 配对 | 局数 | 比分 | 胜率(前) | Elo 差 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for key in sorted(pairs, key=lambda k: -pairs[k]["games"]):
        s = pairs[key]
        x, y = key.split("-")
        n = s["games"]
        wx = s["a_wins"]
        p = (wx + 0.5 * s["draws"]) / n
        if p >= 1.0:
            diff = "≥ +800"
        elif p <= 0.0:
            diff = "≤ −800"
        else:
            diff = f"{400 * np.log10(p / (1 - p)):+.0f}"
        lines.append(f"| {LABELS[x]} vs {LABELS[y]} | {n} | {wx} : {s['b_wins']} "
                     f"| {100 * p:.1f}% | {diff} |")
    md = "\n".join(lines) + "\n"
    (od / "report.md").write_text(md, encoding="utf-8")

    if chart:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return bt
        matplotlib.rcParams["font.sans-serif"] = [
            "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        order = sorted(PLAYERS, key=lambda x: elo[x])
        vals = [elo[p] for p in order]
        err = [[max(0.0, elo[p] - ci[p][0]) for p in order],
               [max(0.0, ci[p][1] - elo[p]) for p in order]]
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
        ax.barh([LABELS[p] for p in order], vals, xerr=err,
                color="#4f81bd", error_kw={"capsize": 3, "lw": 1})
        ax.axvline(0, color="#888", lw=0.8, ls="--")
        ax.set_xlabel("Elo(初级 = 0)")
        ax.set_title(f"桌面端五档 Elo({len(games)} 局,{step_txt}{depth_txt})")
        fig.tight_layout()
        fig.savefig(od / "py_elo_chart.png")
        plt.close(fig)
    return bt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="antifive.tools.py_levels_elo")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--games-per-pair", type=int, default=6)
    r.add_argument("--pair-games", default="")
    r.add_argument("--time", type=float, default=15.0)
    r.add_argument("--depth", type=int, default=8)
    r.add_argument("--openings", type=int, default=8)
    r.add_argument("--open-plies", type=int, default=4)
    r.add_argument("--loss-start", type=int, default=8)
    r.add_argument("--white-restrict", type=int, default=2)
    r.add_argument("--max-steps", type=int, default=225)
    r.add_argument("--workers", type=int, default=10)
    r.add_argument("--seed", type=int, default=20261025)
    r.add_argument("--deadline", default="")
    r.set_defaults(func=run)

    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    p.add_argument("--chart", action="store_true")
    p.set_defaults(func=lambda a: report(a.out, a.chart))

    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
