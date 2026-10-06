"""神经网络 vs 神经网络 / 启发式 AI 对战(诊断用)

用法:
    python -m antifive.tools.net_match --model-a models/best_model.pth \
        --model-b models/warm_model.pth --games 10
    python -m antifive.tools.net_match --model-a models/best_model.pth \
        --heur-depth 4 --games 10

说明:
- --sims-a/--sims-b:两侧 MCTS 模拟数(启发式侧忽略);
- 每局交替先后手(规则不对称);--show-moves 打印每局着法;
- 结果从 model-a(或 net)视角统计。
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

import numpy as np
import torch

from .. import heuristic as h
from ..mcts import MCTS
from ..network import PolicyValueNet
from ..record import idx_to_coord
from ..reversegomoku import BLACK, WHITE, GameConfig, ReverseGomoku


def _annotate_moves(game) -> list:
    """复盘:每手标注行棋方,返回 [f"{玩家}:{坐标}", ...]。"""
    g2 = ReverseGomoku(game.config)
    out = []
    for m in game.record():
        sym = "B" if g2.current_player == BLACK else "W"
        out.append(f"{sym}:{idx_to_coord(m)}")
        g2.make_move(m)
    return out


def _load(path, device) -> PolicyValueNet:
    net = PolicyValueNet.load(path, device)
    net.eval()
    return net


def _pick(mcts, game, temperature):
    """按温度选步:>0 时按访问分布采样(前 temp_moves 步用),否则 argmax。"""
    if temperature > 0:
        legal, p = mcts.get_action_probs(game, temperature=temperature)
        if legal is None:
            return None
        return int(np.random.choice(legal, p=p))
    return mcts.search(game)


def play_net_net(net_a, net_b, config, device, sims_a, sims_b, seed, temp_moves, show_moves):
    """net_a 与 net_b 对弈,交替执黑。返回 (结果文本, 步序, a_执黑)。"""
    rng = np.random.default_rng(seed)
    g = ReverseGomoku(config)
    mcts_a = MCTS(net_a, device, sims=sims_a, config=config)
    mcts_b = MCTS(net_b, device, sims=sims_b, config=config)
    a_is_black = (seed % 2 == 0)
    while not g.game_over and g.move_count < 450:
        is_a = (g.current_player == BLACK) == a_is_black
        temp = 1.0 if g.move_count < temp_moves else 0.0
        a = _pick(mcts_a if is_a else mcts_b, g, temp)
        if a is None:
            g.game_over = True
            g.loser = g.current_player
            break
        g.make_move(a)
    res = "白胜" if g.loser == BLACK else ("黑胜" if g.loser == WHITE else "和棋")
    moves = " ".join(_annotate_moves(g))
    return res, moves, a_is_black


def play_net_heur(net, config, device, sims, h_depth, h_budget, seed, temp_moves, show_moves):
    """net 与启发式(深度 h_depth)对弈,交替执黑。返回 (结果文本, 步序, net_执黑)。"""
    rng = np.random.default_rng(seed)
    g = ReverseGomoku(config)
    mcts = MCTS(net, device, sims=sims, config=config)
    net_is_black = (seed % 2 == 0)
    while not g.game_over and g.move_count < 450:
        if (g.current_player == BLACK) == net_is_black:
            temp = 1.0 if g.move_count < temp_moves else 0.0
            a = _pick(mcts, g, temp)
        else:
            a = h.choose_heuristic_move(g.board, g.current_player, g.pending,
                                        g.turn_count, g.white_turns, config, rng,
                                        depth=h_depth,
                                        time_budget=h_budget if h_budget > 0 else None)
        if a is None:
            g.game_over = True
            g.loser = g.current_player
            break
        g.make_move(a)
    res = "白胜" if g.loser == BLACK else ("黑胜" if g.loser == WHITE else "和棋")
    moves = " ".join(_annotate_moves(g))
    return res, moves, net_is_black


def _net_net_task(task):
    """并行 worker:一局 net_a vs net_b → (seed, 结果, 着法, a_执黑)。"""
    model_a, model_b, sims_a, sims_b, seed, temp_moves, cfg_dict, device_str = task
    torch.set_num_threads(2)
    device = torch.device(device_str)
    config = GameConfig(**cfg_dict)
    net_a = _load(model_a, device)
    net_b = _load(model_b, device)
    res, moves, a_black = play_net_net(net_a, net_b, config, device,
                                       sims_a, sims_b, seed, temp_moves, True)
    return seed, res, moves, a_black


def _net_heur_task(task):
    """并行 worker:一局 net vs 启发式 → (seed, 结果, 着法, net_执黑)。"""
    model_a, sims_a, h_depth, h_budget, seed, temp_moves, cfg_dict, device_str = task
    torch.set_num_threads(2)
    device = torch.device(device_str)
    config = GameConfig(**cfg_dict)
    net_a = _load(model_a, device)
    res, moves, net_black = play_net_heur(net_a, config, device, sims_a,
                                          h_depth, h_budget, seed, temp_moves, True)
    return seed, res, moves, net_black


def main():
    p = argparse.ArgumentParser(description="神经网络 vs 神经网络 / 启发式 AI")
    p.add_argument("--model-a", required=True, help="模型 A(或对启发式的网络)")
    p.add_argument("--model-b", default="", help="模型 B(空=对启发式)")
    p.add_argument("--heur-depth", type=int, default=4, help="启发式搜索深度")
    p.add_argument("--heur-budget", type=float, default=0.0,
                   help="启发式每步限时(秒;深度≥6 建议配,如 2.0)")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--sims-a", type=int, default=200)
    p.add_argument("--sims-b", type=int, default=200)
    p.add_argument("--workers", type=int, default=1,
                   help="并行进程数(>1 时按局并行;网络侧共享 GPU,启发式侧并行)")
    p.add_argument("--white-restrict", type=int, default=2)
    p.add_argument("--loss-start", type=int, default=8)
    p.add_argument("--device", default="", help="cuda/cpu,空=自动")
    p.add_argument("--show-moves", action="store_true")
    p.add_argument("--temp-moves", type=int, default=0,
                   help="前 N 步温度采样选步(制造对局多样性,0=纯 argmax 确定性)")
    args = p.parse_args()

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    device_str = str(device)
    config = GameConfig(loss_start_turns=args.loss_start,
                        white_restrict_turns=args.white_restrict)
    cfg_dict = config.__dict__.copy()
    name_a = os.path.basename(args.model_a)
    stats = {"黑胜": 0, "白胜": 0, "和棋": 0}
    side_wins = 0
    t0 = time.time()

    def handle(res, moves, side_black, label_head):
        nonlocal side_wins
        stats[res] += 1
        if (res == "黑胜") == side_black:
            side_wins += 1
        print(f"{label_head} → {res}", flush=True)
        if args.show_moves:
            print("    " + moves)

    def report(opp_name):
        print(f"\n{name_a} vs {opp_name} {args.games} 局: 网胜 {side_wins} | "
              f"{opp_name}胜 {args.games - side_wins - stats['和棋']} | "
              f"和棋 {stats['和棋']} | 用时 {time.time() - t0:.0f}s")

    if args.workers > 1 and args.games > 1:
        if args.model_b:
            name_b = os.path.basename(args.model_b)
            tasks = [(args.model_a, args.model_b, args.sims_a, args.sims_b,
                      i, args.temp_moves, cfg_dict, device_str)
                     for i in range(args.games)]
            with mp.Pool(args.workers) as pool:
                for i, (seed, res, moves, a_black) in enumerate(
                        pool.imap_unordered(_net_net_task, tasks), 1):
                    handle(res, moves, a_black,
                           f"[{i}/{args.games}] A({'黑' if a_black else '白'}) "
                           f"{name_a} vs {name_b}({'白' if a_black else '黑'})")
            report(name_b)
        else:
            name_b = f"heuristic-depth{args.heur_depth}({args.heur_budget}s)"
            tasks = [(args.model_a, args.sims_a, args.heur_depth,
                      args.heur_budget, i, args.temp_moves, cfg_dict, device_str)
                     for i in range(args.games)]
            with mp.Pool(args.workers) as pool:
                for i, (seed, res, moves, net_black) in enumerate(
                        pool.imap_unordered(_net_heur_task, tasks), 1):
                    handle(res, moves, net_black,
                           f"[{i}/{args.games}] Net({'黑' if net_black else '白'}) "
                           f"{name_a} vs {name_b}({'白' if net_black else '黑'})")
            report(name_b)
        return

    net_a = _load(args.model_a, device)
    if args.model_b:
        net_b = _load(args.model_b, device)
        name_b = os.path.basename(args.model_b)
        for i in range(args.games):
            res, moves, a_black = play_net_net(net_a, net_b, config, device,
                                               args.sims_a, args.sims_b, i,
                                               args.temp_moves, args.show_moves)
            handle(res, moves, a_black,
                   f"[{i + 1}/{args.games}] A({'黑' if a_black else '白'}) "
                   f"{name_a} vs {name_b}({'白' if a_black else '黑'})")
        report(name_b)
    else:
        name_b = f"heuristic-depth{args.heur_depth}({args.heur_budget}s)"
        for i in range(args.games):
            res, moves, net_black = play_net_heur(net_a, config, device,
                                                  args.sims_a, args.heur_depth,
                                                  args.heur_budget,
                                                  i, args.temp_moves, args.show_moves)
            handle(res, moves, net_black,
                   f"[{i + 1}/{args.games}] Net({'黑' if net_black else '白'}) "
                   f"{name_a} vs {name_b}({'白' if net_black else '黑'})")
        report(name_b)


if __name__ == "__main__":
    main()
