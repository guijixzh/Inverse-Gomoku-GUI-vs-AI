"""新旧启发式 AI 对战(或任意两配置对弈)

用法:
    python -m antifive.tools.match_ai --games 20 --depth 4 --old-depth 4
    python -m antifive.tools.match_ai --games 10 --depth 4 --old-depth 2   # 新强旧弱对照
    python -m antifive.tools.match_ai --games 10 --depth 6 --budget 3      # 限时对战
    python -m antifive.tools.match_ai --games 10 --workers 10              # 10 局并发

说明:
- 默认新版 = antifive.heuristic,旧版 = antifive.tools.heuristic_v2(冻结快照),
  可用 --new-module/--old-module 指定任意实现模块;
- 双方交换先后手,统计 黑胜/白胜/和棋;
- 每步可加 time_budget,保证深搜索对局在可控时间内完成;
- --workers N:多进程并发对局(Windows 下 spawn,worker 各自建表)。
"""
from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import time

import numpy as np

from ..reversegomoku import GameConfig, ReverseGomoku


def play(choose_a, choose_b, config: GameConfig, rng: np.random.Generator,
         budget: float | None, max_steps: int = 450):
    """a 执黑、b 执白一局,返回结果文本(黑胜/白胜/和棋)。"""
    g = ReverseGomoku(config)
    while not g.game_over and g.move_count < max_steps:
        chooser = choose_a if g.current_player == 1 else choose_b
        a = chooser(g.board, g.current_player, g.pending, g.turn_count,
                    g.white_turns, config, rng, time_budget=budget)
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


def _match_task(task):
    """worker 任务:一局棋 → 结果文本(黑胜/白胜/和棋)。"""
    seed, new_depth, old_depth, budget, new_first, new_mod, old_mod = task
    new_h = importlib.import_module(new_mod)
    old_h = importlib.import_module(old_mod)
    cfg = GameConfig()
    rng = np.random.default_rng(seed)

    def mk(mod, depth):
        def choose(board, player, pending, tc, wt, config, rng, time_budget=None):
            return mod.choose_heuristic_move(board, player, pending, tc, wt,
                                             config, rng, depth=depth,
                                             time_budget=time_budget)
        return choose

    new_ch = mk(new_h, new_depth)
    old_ch = mk(old_h, old_depth)
    if new_first:
        res = play(new_ch, old_ch, cfg, rng, budget)       # 新版执黑
    else:
        res = play(old_ch, new_ch, cfg, rng, budget)       # 新版执白
    return new_first, res


def main():
    p = argparse.ArgumentParser(description="新旧启发式 AI 对战")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--depth", type=int, default=4, help="新版搜索深度")
    p.add_argument("--old-depth", type=int, default=None, help="旧版深度(默认=--depth)")
    p.add_argument("--budget", type=float, default=0.0, help="每步限时(秒,0=不限)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1, help="并发进程数(>1 时多局并行)")
    p.add_argument("--new-module", default="antifive.heuristic",
                   help="新版模块(默认 antifive.heuristic)")
    p.add_argument("--old-module", default="antifive.tools.heuristic_v2",
                   help="旧版模块(默认 antifive.tools.heuristic_v2 冻结快照)")
    args = p.parse_args()

    old_depth = args.old_depth if args.old_depth is not None else args.depth
    budget = args.budget if args.budget > 0 else None
    tasks = [(args.seed + i * 131, args.depth, old_depth, budget, i % 2 == 0,
              args.new_module, args.old_module) for i in range(args.games)]
    stats = {"黑胜": 0, "白胜": 0, "和棋": 0}
    new_wins = old_wins = draws = 0
    t0 = time.perf_counter()

    def handle(new_is_black, res):
        nonlocal new_wins, old_wins, draws
        stats[res] += 1
        if res == "和棋":
            draws += 1
        elif (res == "黑胜") == new_is_black:   # 新版执黑且黑胜 → 新版胜
            new_wins += 1
        else:
            old_wins += 1
        print(f"[{new_wins + old_wins + draws}/{args.games}] "
              f"{args.new_module}{'黑' if new_is_black else '白'} vs "
              f"{args.old_module}{'白' if new_is_black else '黑'} → {res}",
              flush=True)

    if args.workers > 1 and args.games > 1:
        with mp.Pool(args.workers) as pool:
            for new_is_black, res in pool.imap_unordered(_match_task, tasks):
                handle(new_is_black, res)
    else:
        for task in tasks:
            handle(*_match_task(task))

    dt = time.perf_counter() - t0
    total = max(new_wins + old_wins + draws, 1)
    print(f"\n新版(深度{args.depth}) vs 旧版(深度{old_depth}) "
          f"{args.games} 局: 新版胜 {new_wins} | 旧版胜 {old_wins} "
          f"| 和棋 {draws} | 新版胜率 {new_wins / total:.0%} | 用时 {dt:.0f}s")


if __name__ == "__main__":
    main()
