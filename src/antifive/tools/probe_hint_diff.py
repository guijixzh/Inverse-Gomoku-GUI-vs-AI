# -*- coding: utf-8 -*-
"""网页端 vs Python 端:同局面的「分析层」对照(排序一致性 / 候选覆盖)。

对每盘真实对局取一个中盘局面,两端各跑一次局面分析:
  Python: heuristic_versions.analysis(version, ..., depth=6, time_budget=None, k=K)
  网页端: engine-bridge 的 hint(depth=8, budget=3, k=K)
比较 top-1 一致率、top-5 重合度、彼此在对方眼里的排名、pos_v 符号一致率。
"""
import argparse
import glob
import json
import os
import random
import re
import statistics as stats
import subprocess
import sys
import time
from pathlib import Path

from ..reversegomoku import BLACK, GameConfig, ReverseGomoku, WHITE
from .. import heuristic_versions as hv
from .. import record as record_mod

ROOT = Path(__file__).resolve().parents[3]
BRIDGE = ROOT / "web" / "scripts" / "engine-bridge.mjs"


def pick_positions(run_dir: Path, n: int, seed: int, min_ply: int, max_ply: int):
    rng = random.Random(seed)
    files = sorted(glob.glob(str(run_dir / "games" / "*" / "*.json")))
    rng.shuffle(files)
    out = []
    for f in files:
        d = json.loads(open(f, encoding="utf-8").read())
        moves = d.get("moves") or []
        if len(moves) < min_ply + 6:
            continue
        ply = rng.randint(min_ply, min(max_ply, len(moves) - 4))
        out.append({"file": f, "game_id": d["game_id"], "moves": moves, "ply": ply})
        if len(out) >= n:
            break
    return out


def python_analysis(job):
    pos, cfg_kw, depth, k = job
    moves = pos["moves"][: pos["ply"]]
    cfg = GameConfig(**cfg_kw)
    g = ReverseGomoku(cfg)
    for mv in moves:
        g.make_move(mv, allow_suicide=True)
    board = g.board.copy()
    t0 = time.time()
    v, mv = hv.analysis("advanced", board, g.current_player, g.pending,
                        g.turn_count, g.white_turns, cfg, random.Random(12345),
                        depth=depth, time_budget=None, k=k, use_vcf=True)
    return {"pos_v": float(v), "moves": [(int(i), float(x)) for i, x in mv],
            "sec": time.time() - t0, "nodes": 0}


def bridge_client():
    proc = subprocess.Popen(
        ["node", str(BRIDGE)], cwd=str(ROOT / "web"),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, encoding="utf-8", bufsize=1)
    return proc


def call(proc, obj):
    proc.stdin.write(json.dumps(obj) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("bridge died")
    r = json.loads(line)
    if "error" in r:
        raise RuntimeError(r["error"])
    return r


def web_analysis(proc, pos, depth, budget, k):
    call(proc, {"cmd": "new"})
    for mv in pos["moves"][: pos["ply"]]:
        call(proc, {"cmd": "move", "idx": int(mv)})
    r = call(proc, {"cmd": "hint", "engine": "advanced", "depth": depth,
                    "budget": budget, "k": k})
    return {"pos_v": float(r["pos_v"]),
            "moves": [(int(m["idx"]), float(m["v"])) for m in r["moves"]]}


def spearman(pairs):
    if len(pairs) < 3:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    rx = {v: i for i, v in enumerate(sorted(xs))}
    ry = {v: i for i, v in enumerate(sorted(ys))}
    a = [rx[v] for v in xs]
    b = [ry[v] for v in ys]
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da and db else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="records/web_vs_py_15s")
    ap.add_argument("--out", default="records/web_vs_py_diff")
    ap.add_argument("--positions", type=int, default=30)
    ap.add_argument("--py-depth", type=int, default=6)
    ap.add_argument("--web-depth", type=int, default=8)
    ap.add_argument("--web-budget", type=float, default=3.0)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--min-ply", type=int, default=10)
    ap.add_argument("--max-ply", type=int, default=30)
    args = ap.parse_args()

    run_dir = Path(args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    positions = pick_positions(run_dir, args.positions, args.seed,
                               args.min_ply, args.max_ply)
    cfg_kw = {"loss_start_turns": 8, "white_restrict_turns": 2, "mask_suicide": True}
    jobs = [(p, cfg_kw, args.py_depth, args.k) for p in positions]

    print(f"局面 {len(positions)} 个 (py depth={args.py_depth} 无时限, "
          f"web depth={args.web_depth} budget={args.web_budget}s, k={args.k})", flush=True)

    py_res = []
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.workers) as pool:
            py_res = pool.map(python_analysis, jobs, chunksize=1)
    else:
        py_res = [python_analysis(j) for j in jobs]

    proc = bridge_client()
    web_res = []
    try:
        for i, p in enumerate(positions):
            web_res.append(web_analysis(proc, p, args.web_depth,
                                        args.web_budget, args.k))
            print(f"  web {i+1}/{len(positions)}", flush=True)
    finally:
        try:
            call(proc, {"cmd": "quit"})
        except Exception:
            proc.kill()

    rows = []
    for p, pr, wr in zip(positions, py_res, web_res):
        py_moves = [m for m, _ in pr["moves"]]
        web_moves = [m for m, _ in wr["moves"]]
        top1_same = bool(py_moves) and bool(web_moves) and py_moves[0] == web_moves[0]
        overlap = len(set(py_moves[:5]) & set(web_moves[:5]))
        web_top1_rank = (py_moves.index(web_moves[0]) if web_moves and web_moves[0] in py_moves
                         else None)
        py_top1_rank = (web_moves.index(py_moves[0]) if py_moves and py_moves[0] in web_moves
                        else None)
        in_top5 = bool(web_moves) and web_moves[0] in py_moves[:5]
        common = [(m, v) for m, v in pr["moves"] if m in dict(wr["moves"])]
        wdict = dict(wr["moves"])
        rho = spearman([(v, wdict[m]) for m, v in common]) if len(common) >= 3 else None
        sign_same = (pr["pos_v"] > 0) == (wr["pos_v"] > 0)
        rows.append({
            "py_sec": pr.get("sec"),
            "game_id": p["game_id"], "ply": p["ply"], "player": "black",
            "py_top1": py_moves[0] if py_moves else None,
            "web_top1": web_moves[0] if web_moves else None,
            "top1_same": top1_same, "overlap5": overlap,
            "web_top1_rank_in_py": web_top1_rank,
            "py_top1_rank_in_web": py_top1_rank,
            "web_top1_in_py_top5": in_top5,
            "pos_v_py": pr["pos_v"], "pos_v_web": wr["pos_v"],
            "sign_same": sign_same, "rho": rho,
            "py_moves": pr["moves"], "web_moves": wr["moves"],
        })

    n = len(rows)
    agree = sum(1 for r in rows if r["top1_same"]) / n
    ov = stats.mean(r["overlap5"] for r in rows)
    cov = sum(1 for r in rows if r["web_top1_in_py_top5"]) / n
    ranks = [r["web_top1_rank_in_py"] for r in rows if r["web_top1_rank_in_py"] is not None]
    rhos = [r["rho"] for r in rows if r["rho"] is not None]
    signs = sum(1 for r in rows if r["sign_same"]) / n

    summary = {
        "positions": n, "k": args.k,
        "py_depth": args.py_depth, "web_depth": args.web_depth,
        "top1_agreement": round(agree, 4),
        "mean_top5_overlap": round(ov, 3),
        "web_top1_in_py_top5_rate": round(cov, 4),
        "mean_rank_of_web_top1_in_py": round(stats.mean(ranks), 2) if ranks else None,
        "rank_corr_mean": round(stats.mean(rhos), 3) if rhos else None,
        "pos_v_sign_agreement": round(signs, 4),
        "py_moves_median_sec": None,
    }
    (out / "positions.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
