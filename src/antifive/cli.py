"""人机对战入口(逆五子棋)

规则提示:
- 每回合二选一:普通落子(空位放子)或移子(两步:占领对方棋子格 → 再把
  拿起的对方棋子沿直线放到空位)。移子时本回合仍由你继续走"安置"步。
- 连成五子者判负。

用法:
    python -m antifive.cli [--model models/best_model.pth] [--sims 模拟次数]
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from .mcts import MCTS
from .network import PolicyValueNet
from .reversegomoku import (
    BLACK, BOARD_SIZE, EMPTY, WHITE, ReverseGomoku,
)


def print_board(game: ReverseGomoku) -> None:
    symbols = {EMPTY: " .", BLACK: " X", WHITE: " O"}
    print("    " + " ".join(f"{i:2}" for i in range(BOARD_SIZE)))
    for r in range(BOARD_SIZE):
        print(f"{r:2} |" + "".join(symbols[game.board[r, c]] for c in range(BOARD_SIZE)))
    if game.pending >= 0:
        pr, pc = divmod(game.pending, BOARD_SIZE)
        print(f"     (你手中拿着 {pr},{pc} 处的对方棋子,请安置到空位)")


def get_human_move(game: ReverseGomoku) -> int:
    mask = game.legal_mask()
    if not mask.any():
        return None                  # 死局(所有落点均为自杀且未满盘):行棋方必败
    hint = ("请输入坐标(行 列),落子到空位或占领对方棋子格" if game.pending < 0
            else "请把手中的对方棋子安置到空位(输入 行 列)")
    while True:
        try:
            print(f"    {hint}:")
            s = input("    输入坐标(行 列): ").strip().split()
            if len(s) != 2:
                print(">>> 请输入两个数字,如: 7 7")
                continue
            r, c = map(int, s)
            if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
                print(">>> 坐标越界(0-14)")
                continue
            idx = r * BOARD_SIZE + c
            if mask[idx]:
                return idx
            print(">>> 该格不可用(需为空位/可占领的对方棋子/手中棋子的可达位置)")
        except ValueError:
            print(">>> 请输入数字坐标!")


def get_ai_move(mcts: MCTS, game: ReverseGomoku):
    print("    AI 思考中...")
    t0 = time.time()
    a = mcts.search(game)
    print(f"    AI 落子耗时 {time.time() - t0:.1f}s")
    return a


def play_game(mode: int, mcts_black, mcts_white) -> None:
    game = ReverseGomoku()
    print_board(game)
    while not game.game_over:
        color_name = "黑棋" if game.current_player == BLACK else "白棋"
        print(f"当前回合: {color_name} (回合 {game.turn_count + 1})")
        if mode == 3:
            mcts = mcts_black if game.current_player == BLACK else mcts_white
            a = get_ai_move(mcts, game)
        elif (mode == 1 and game.current_player == BLACK) or \
             (mode == 2 and game.current_player == WHITE):
            a = get_human_move(game)
        else:
            mcts = mcts_black if game.current_player == BLACK else mcts_white
            a = get_ai_move(mcts, game)
        if a is None:
            game.game_over = True
            game.loser = game.current_player     # 无子可走,行棋方必败
            print_board(game)
            break
        r, c = divmod(a, BOARD_SIZE)
        was_pending = game.pending >= 0
        game.make_move(a)
        if game.pending >= 0:
            print(f"    占领 ({r},{c}),请安置手中棋子")
        elif was_pending:
            print(f"    安置手中棋子 → ({r},{c})")
        else:
            print(f"    落子 ({r},{c})")
        print_board(game)
    if game.is_draw:
        print("结果: 和棋(棋盘已满)")
    else:
        winner = "白棋" if game.loser == BLACK else "黑棋"
        print(f"结果: {winner} 获胜(对方连成五子判负)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="", help="模型文件(空=仅随机先验的 MCTS)")
    p.add_argument("--sims", type=int, default=400, help="AI 每步模拟次数")
    p.add_argument("--c-puct", type=float, default=5.0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = None
    if args.model and os.path.exists(args.model):
        net = PolicyValueNet.load(args.model, device).to(device)
        print(f"已加载模型: {args.model}")
    else:
        print("未加载模型,AI 仅凭随机先验搜索(低强度)")

    mcts_black = MCTS(net, device, sims=args.sims, c_puct=args.c_puct)
    mcts_white = MCTS(net, device, sims=args.sims, c_puct=args.c_puct)
    while True:
        print("\n" + "=" * 24)
        print("逆五子棋 Antifive")
        print("1. 人机对战 (玩家执黑)")
        print("2. 人机对战 (玩家执白)")
        print("3. 观察 AI 对战 (AI vs AI)")
        print("4. 退出")
        choice = input("请选择 (1-4): ").strip()
        if choice == "4":
            break
        if choice in ("1", "2", "3"):
            play_game(int(choice), mcts_black, mcts_white)
        else:
            print("无效选择")


if __name__ == "__main__":
    main()
