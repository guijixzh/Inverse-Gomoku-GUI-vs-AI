"""网页端引擎 vs Python 引擎,在「同等搜索量」下的对拍。

两边的着法都来自正式的引擎实现:

* Python 侧:``antifive.heuristic_versions.choose(version, ...)``,``time_budget`` 秒;
* 网页端侧:``web/scripts/engine-bridge.mjs``(直接 import ``web/src/engine/api.ts``,
  与浏览器里跑的是同一份代码),用 ``nodeLimit`` 卡每步节点数。

「同等搜索量」指网页端的每步节点上限 ≈ Python 侧实测的每步节点数(默认 7000,可用
``--web-nodes`` 调),而不是同等墙钟——网页端每节点快 15–20 倍,同墙钟下它会多搜一个量级。

用法::

    python -m antifive.tools.match_web_py run --out records/web_vs_py_nodes \\
        --pairs advanced:advanced,legend:legend --py-time 6 --web-nodes 7000 \\
        --games-per-pair 20 --workers 10

产物:``games/<pair>/g####.json``(含双方每步的耗时/节点)、``results.jsonl``、
``summary.json``。支持断点续跑(已存在的对局跳过)。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .. import record as record_mod
from ..reversegomoku import BLACK, WHITE, GameConfig, ReverseGomoku

WEB_ROOT = Path(__file__).resolve().parents[3] / "web"
BRIDGE = WEB_ROOT / "scripts" / "engine-bridge.mjs"


# ---------------------------------------------------------------- 网页端桥


class WebBridge:
    """一个网页端引擎实例(GTP 风格的行协议,每局一个进程)。"""

    def __init__(self, engine: str, nodes: int, depth: int = 8, budget: float = 15.0):
        self.engine = engine
        self.nodes = nodes
        self.depth = depth
        self.budget = budget
        self.proc = subprocess.Popen(
            ["node", str(BRIDGE)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1, cwd=str(WEB_ROOT),
        )
        self._cmd({"cmd": "new"})

    def _cmd(self, req: dict) -> dict:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("网页端引擎进程退出")
        resp = json.loads(line)
        if "error" in resp:
            raise RuntimeError(f"网页端引擎报错:{resp['error']}")
        return resp

    def move(self, idx: int) -> None:
        self._cmd({"cmd": "move", "idx": int(idx)})

    def gen(self) -> tuple[int, int, float]:
        t0 = time.perf_counter()
        resp = self._cmd({"cmd": "gen", "engine": self.engine, "nodes": self.nodes,
                          "depth": self.depth, "budget": self.budget})
        return int(resp["move"]), int(resp.get("nodes") or 0), time.perf_counter() - t0

    def close(self) -> None:
        try:
            self._cmd({"cmd": "quit"})
        except Exception:
            pass
        try:
            self.proc.kill()
        except Exception:
            pass


# ---------------------------------------------------------------- 一局


def _play(job: dict) -> dict:
    from .. import heuristic_versions as hv
    from ..heuristic import _TT

    py_version = job["py"]
    web_version = job["web"]
    cfg = GameConfig(loss_start_turns=job["loss_start"],
                     white_restrict_turns=job["white_restrict"],
                     mask_suicide=True)
    g = ReverseGomoku(cfg)
    rng = np.random.default_rng(job["seed"])
    opening = []
    for _ in range(job["open_plies"]):
        legal = np.flatnonzero(g.legal_mask())
        if len(legal) == 0:
            break
        m = int(legal[int(rng.integers(0, len(legal)))])
        g.make_move(m)
        opening.append(m)

    web = WebBridge(web_version, job["web_nodes"], depth=job["depth"],
                    budget=job["web_budget"])
    # 把已经下过的开局手同步给网页端
    for m in opening:
        web.move(m)

    tt = _TT()
    ms_py = ms_web = 0.0
    nodes_py = nodes_web = 0
    steps = []
    resigned = False
    try:
        while not g.game_over and g.move_count < job["max_steps"]:
            is_web = g.current_player != job["py_color"]
            if is_web:
                idx, nodes, dt = web.gen()
                if int(idx) < 0:
                    g.game_over = True
                    g.loser = g.current_player
                    break
                g.make_move(int(idx), allow_suicide=True)
                ms_web += dt * 1000.0
                nodes_web += nodes
                steps.append([g.move_count, "WEB", round(dt, 3), nodes])
            else:
                nodes = [0]
                t0 = time.perf_counter()
                a = hv.choose(py_version, g.board, g.current_player, g.pending,
                              g.turn_count, g.white_turns, g.config, rng,
                              depth=job["depth"], time_budget=job["py_time"],
                              progress_cb=lambda n: nodes.__setitem__(0, n),
                              tt=tt, use_vcf=True)
                dt = time.perf_counter() - t0
                if a is None:
                    g.game_over = True
                    g.loser = g.current_player
                    break
                web.move(a)
                g.make_move(a)
                ms_py += dt * 1000.0
                nodes_py += int(nodes[0])
                steps.append([g.move_count, "PY", round(dt, 3), int(nodes[0])])
        if not g.game_over:
            g.game_over = True
            g.is_draw = True
        if g.is_draw:
            winner = "draw"
        else:
            winner = "web" if g.loser == job["py_color"] else "py"
        moves = g.record()
        if g.pending >= 0 and moves:
            moves = moves[:-1]
        # 自检:两个引擎都必须真的下过棋(py_color 配错时会退化成网页端自战)
        sides = {st[1] for st in steps}
        if sides != {"PY", "WEB"}:
            raise RuntimeError(f"对局无效:只出现 {sorted(sides)} 步 game={job['game_id']}")
        # 自检:重放必须复现终局
        if not g.is_draw:
            chk = ReverseGomoku(cfg)
            for m in moves:
                chk.make_move(m, allow_suicide=True)
            if not (chk.game_over and chk.loser == g.loser):
                raise RuntimeError(f"记录自检失败 game={job['game_id']}")
    finally:
        web.close()
        # 根并行池是本进程的常驻资源;对局结束时主动收尾,
        # 否则 Pool worker 被强杀时会留下孤儿子进程。
        try:
            from ..root_parallel import shutdown_pools as _shutdown_pools
            _shutdown_pools()
        except Exception:
            pass

    return {
        "game_id": job["game_id"], "py": py_version, "web": web_version,
        "py_color": "black" if job["py_color"] == BLACK else "white",
        "winner": winner, "turns": g.move_count,
        "result": record_mod.game_result(g), "moves": moves,
        "ms_py": round(ms_py, 1), "ms_web": round(ms_web, 1),
        "nodes_py": nodes_py, "nodes_web": nodes_web,
        "steps": steps, "ts": datetime.now().isoformat(" ", "seconds"),
    }


# ---------------------------------------------------------------- 编排


def _deadline_ts(text: str) -> float:
    if not text:
        return float("inf")
    now = datetime.now()
    hh, mm = (int(x) for x in text.split(":"))
    target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target.timestamp()


def _wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    z = 1.96
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def _elo(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return 400.0 * math.log10(p / (1 - p))


def build_jobs(args, out_dir: Path) -> list[dict]:
    """按「每对轮流」交错排列,避免某一对先跑光、其余对饿死。"""
    per_pair: dict[str, list[dict]] = {}
    seed0 = args.seed
    for pair in args.pairs:
        py_v, web_v = pair.split(":")
        jobs = per_pair.setdefault(pair, [])
        for i in range(args.games_per_pair):
            gid = f"{py_v}-vs-web{web_v}_g{i:03d}"
            path = out_dir / "games" / f"{py_v}-web{web_v}" / f"g{i:03d}.json"
            if path.exists():
                continue
            jobs.append({
                "game_id": gid, "py": py_v, "web": web_v,
                "py_color": BLACK if i % 2 == 0 else WHITE,
                "seed": seed0 + i * 7919,
                "open_plies": args.open_plies, "depth": args.depth,
                "py_time": args.py_time, "web_nodes": args.web_nodes,
                "web_budget": args.web_budget,
                "loss_start": args.loss_start, "white_restrict": args.white_restrict,
                "max_steps": args.max_steps,
                "path": str(path),
            })
    out: list[dict] = []
    i = 0
    while True:
        added = False
        for pair in args.pairs:
            lst = per_pair[pair]
            if i < len(lst):
                out.append(lst[i])
                added = True
        if not added:
            return out
        i += 1


def _task(job: dict) -> dict:
    rec = _play(job)
    path = Path(job["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return {"game_id": rec["game_id"], "pair": f"{rec['py']}-web{rec['web']}",
            "winner": rec["winner"], "turns": rec["turns"], "path": str(path)}


def load(out_dir: Path) -> list[dict]:
    games = []
    for p in sorted((out_dir / "games").glob("*/*.json")):
        games.append(json.loads(p.read_text(encoding="utf-8")))
    return games


def summarize(games: list[dict]) -> dict:
    per_pair: dict[str, dict] = {}
    for g in games:
        pair = f"{g['py']}-web{g['web']}"
        d = per_pair.setdefault(pair, {"games": 0, "py_wins": 0, "web_wins": 0, "draws": 0})
        d["games"] += 1
        if g["winner"] == "py":
            d["py_wins"] += 1
        elif g["winner"] == "web":
            d["web_wins"] += 1
        else:
            d["draws"] += 1
    for pair, d in per_pair.items():
        n = d["games"] - d["draws"]
        p = d["py_wins"] / n if n else 0.0
        lo, hi = _wilson(d["py_wins"], n)
        d["py_score"] = round(p, 4)
        d["py_score_ci95"] = [round(lo, 4), round(hi, 4)]
        d["elo_py_vs_web"] = round(_elo(p), 1)
        d["elo_ci95"] = [round(_elo(lo), 1), round(_elo(hi), 1)]
    py_nodes = [s[3] for g in games for s in g["steps"] if s[1] == "PY"]
    web_nodes = [s[3] for g in games for s in g["steps"] if s[1] == "WEB"]
    py_ms = [s[2] for g in games for s in g["steps"] if s[1] == "PY"]
    web_ms = [s[2] for g in games for s in g["steps"] if s[1] == "WEB"]

    def med(xs):
        xs = sorted(xs)
        return round(xs[len(xs) // 2], 3) if xs else None

    return {"games": len(games), "pairs": per_pair,
            "timing": {"py_nodes_median": med(py_nodes), "web_nodes_median": med(web_nodes),
                       "py_sec_median": med(py_ms), "web_sec_median": med(web_ms),
                       "py_moves": len(py_nodes), "web_moves": len(web_nodes)}}


def run(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    args.pairs = pairs
    jobs = build_jobs(args, out_dir)
    total = len(jobs)
    print(f"待跑 {total} 局 | 每局 Python {args.py_time}s/步、网页端 {args.web_nodes} 节点/步")
    deadline = _deadline_ts(args.deadline)
    meta = {"start": datetime.now().isoformat(" ", "seconds"),
            "pairs": pairs, "py_time": args.py_time, "web_nodes": args.web_nodes,
            "depth": args.depth, "open_plies": args.open_plies,
            "workers": args.workers, "games_per_pair": args.games_per_pair}
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    from multiprocessing import Pool

    t0 = time.perf_counter()
    done = 0
    results = []
    with Pool(processes=args.workers) as pool:
        for rec in pool.imap_unordered(_task, jobs):
            done += 1
            results.append(rec)
            el = time.perf_counter() - t0
            rate = done / el * 60 if el > 0 else 0
            print(f"[{done}/{total}] {rec['pair']} {rec['winner']} {rec['turns']}手 "
                  f"| {rate:.1f} 局/分", flush=True)
            if time.time() > deadline:
                print("到达截止时间,停止派发")
                pool.terminate()
                break

    games = load(out_dir)
    summary = summarize(games)
    summary["meta"] = meta
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["pairs"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["timing"], ensure_ascii=False))
    print(f"用时 {(time.perf_counter() - t0) / 60:.1f} min -> {out_dir / 'summary.json'}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="网页端 vs Python 引擎(同等搜索量)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--pairs", default="advanced:advanced,legend:legend",
                   help="python档:网页档,逗号分隔")
    r.add_argument("--games-per-pair", type=int, default=20)
    r.add_argument("--py-time", type=float, default=6.0, help="Python 每步时限(秒)")
    r.add_argument("--web-nodes", type=int, default=7000, help="网页端每步节点上限")
    r.add_argument("--web-budget", type=float, default=15.0)
    r.add_argument("--depth", type=int, default=8)
    r.add_argument("--open-plies", type=int, default=0)
    r.add_argument("--loss-start", type=int, default=8)
    r.add_argument("--white-restrict", type=int, default=2)
    r.add_argument("--max-steps", type=int, default=225)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--seed", type=int, default=20261023)
    r.add_argument("--deadline", default="")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        run(args)


if __name__ == "__main__":
    main()
