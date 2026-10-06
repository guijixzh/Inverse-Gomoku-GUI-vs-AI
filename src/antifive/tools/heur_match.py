"""启发式 AI 同深度不同限时对战(验证限时对棋力的影响)

用法:
    python -m antifive.tools.heur_match --depth 8 --budget-a 60 --budget-b 2 --games 10 --workers 10

两侧均用 heuristic.py 的 choose_heuristic_move(深度 --depth),
a 侧限时 --budget-a(秒/步), b 侧限时 --budget-b;交替先后手。
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np

from ..reversegomoku import GameConfig, ReverseGomoku


def play(choose_a, choose_b, config, rng, budget_a, budget_b, max_steps=450):
    """a 执黑、b 执白一局,返回结果文本(黑胜/白胜/和棋)。"""
    g = ReverseGomoku(config)
    while not g.game_over and g.move_count < max_steps:
        chooser = choose_a if g.current_player == 1 else choose_b
        budget = budget_a if g.current_player == 1 else budget_b
        a = chooser(g.board, g.current_player, g.pending,
                    g.turn_count, g.white_turns, config, rng,
                    time_budget=budget if budget > 0 else None)
        if a is None:
            g.game_over = True
            g.loser = g.current_player
            break
        g.make_move(a)
    if not g.game_over:
        g.game_over = True
        g.is_draw = True
    if g.is_draw:
        return "和棋"
    return "白胜" if g.loser == 1 else "黑胜"


def _task(t):
    seed, depth, budget_a, budget_b, a_first = t
    import heuristic as h

    def mk(budget):
        def choose(board, player, pending, tc, wt, config, rng, time_budget=None):
            return h.choose_heuristic_move(board, player, pending, tc, wt,
                                           config, rng, depth=depth,
                                           time_budget=budget if budget > 0 else None)
        return choose

    cfg = GameConfig()
    rng = np.random.default_rng(seed)
    ch_a, ch_b = mk(budget_a), mk(budget_b)
    if a_first:
        return play(ch_a, ch_b, cfg, rng, budget_a, budget_b)
    return play(ch_b, ch_a, cfg, rng, budget_b, budget_a)


def main():
    p = argparse.ArgumentParser(description="启发式 AI 不同限时对战")
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--budget-a", type=float, default=60.0, help="a 侧每步限时(秒)")
    p.add_argument("--budget-b", type=float, default=2.0, help="b 侧每步限时(秒)")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()

    tasks = [(args.seed + i * 131, args.depth, args.budget_a, args.budget_b, i % 2 == 0)
             for i in range(args.games)]
    stats = {"黑胜": 0, "白胜": 0, "和棋": 0}
    a_wins = b_wins = draws = 0
    t0 = time.perf_counter()

    def handle(i, res):
        nonlocal a_wins, b_wins, draws
        stats[res] += 1
        if res == "和棋":
            draws += 1
        elif (res == "黑胜") == (i % 2 == 0):   # a 执黑(i%2==0)且黑胜 → a 胜
            a_wins += 1
        else:
            b_wins += 1
        print(f"[{i + 1}/{args.games}] a({'黑' if i % 2 == 0 else '白'},"
              f"{args.budget_a}s) vs b({'白' if i % 2 == 0 else '黑'},"
              f"{args.budget_b}s) → {res}", flush=True)

    if args.workers > 1 and args.games > 1:
        with mp.Pool(args.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_task, tasks)):
                handle(i, res)
    else:
        for i, task in enumerate(tasks):
            handle(i, _task(task))

    print(f"\n{args.budget_a}s vs {args.budget_b}s 深度{args.depth} {args.games} 局: "
          f"{args.budget_a}s 胜 {a_wins} | {args.budget_b}s 胜 {b_wins} | "
          f"和棋 {draws} | 用时 {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
