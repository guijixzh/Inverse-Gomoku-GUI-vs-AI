"""启发式 AI 搜索基准(固定局面池 + 确定性指标)

在同一批局面上比较不同实现/参数/深度的搜索代价与结果稳定性:

    python -m antifive.tools.heur_bench --depths 4,5,6
    python -m antifive.tools.heur_bench --module antifive.tools.heuristic_v2 --depths 4,5,6
    python -m antifive.tools.heur_bench --limit 12 --budget 2.0 --depths 8

指标:每个深度的总节点数、总耗时、节点/秒、平均单步耗时、不同选点数;
--budget 时以迭代加深实际达到的深度为准(节点数取自进度回调)。
"""

from __future__ import annotations

import argparse
import importlib
import time

import numpy as np

from ..paths import data_dir
from ..reversegomoku import GameConfig


def _load_states(pool_file: str, limit: int):
    z = np.load(pool_file)
    states = []
    for prefix in ("t", "v"):
        b = z[prefix + "b"]
        p = z[prefix + "p"]
        e = z[prefix + "e"]
        t = z[prefix + "t"]
        w = z[prefix + "w"]
        for i in range(len(b)):
            states.append((b[i].astype(np.int8), int(p[i]), int(e[i]),
                           int(t[i]), int(w[i])))
    if limit and len(states) > limit:
        step = max(1, len(states) // limit)
        states = states[::step][:limit]
    return states


def main():
    p = argparse.ArgumentParser(description="启发式 AI 搜索基准")
    p.add_argument("--pool", default=str(data_dir() / "eval_pool.npz"),
                   help="局面池 npz(param_tune 格式,默认 data/eval_pool.npz)")
    p.add_argument("--limit", type=int, default=24, help="最多使用局面数")
    p.add_argument("--depths", default="4,5,6", help="逗号分隔的深度列表")
    p.add_argument("--budget", type=float, default=0.0,
                   help="每步限时(秒,0=按深度完整搜索)")
    p.add_argument("--module", default="antifive.heuristic",
                   help="被测模块(默认 antifive.heuristic,可指定快照)")
    p.add_argument("--seed", type=int, default=20260811)
    args = p.parse_args()

    h = importlib.import_module(args.module)
    states = _load_states(args.pool, args.limit)
    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    budget = args.budget if args.budget > 0 else None
    cfg = GameConfig()
    print(f"模块: {args.module} | 局面 {len(states)} | 限时 {budget} | 深度 {depths}")

    for depth in depths:
        rng = np.random.default_rng(args.seed)
        t0 = time.perf_counter()
        nodes_total = 0
        moves = []
        for b, pl, pe, tc, wt in states:
            seen: list = []
            m = h.choose_heuristic_move(
                b, pl, pe, tc, wt, cfg, rng, depth=depth,
                time_budget=budget, progress_cb=seen.append)
            moves.append(-1 if m is None else int(m))
            if seen:
                nodes_total += seen[-1]
        dt = time.perf_counter() - t0
        avg_ms = dt / max(len(states), 1) * 1000
        nps = nodes_total / dt if dt > 0 else 0.0
        uniq = len(set(moves))
        print(f"depth {depth}: 总耗时 {dt:7.2f}s | 单步均值 {avg_ms:8.1f}ms | "
              f"累计节点 {nodes_total:8d} | 节点/秒 {nps:8.0f} | 不同选点 {uniq}")


if __name__ == "__main__":
    main()
