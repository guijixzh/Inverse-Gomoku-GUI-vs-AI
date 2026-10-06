"""KataGo(GTP) vs 分级对手 棋力检验

把编译好的逆五子棋 katago 引擎(GTP)依次与 随机基线、启发式 d2、d4、d8 对战
(交替先后手),每层 --games 局;若 KataGo 对某一层全败(0 胜),则判定当前
模型棋力不足,不开启下一层(更强对手)测试,提前结束。

规则:white_restrict=2(与引擎一致),自杀着双方都允许(heuristic 不会主动走自杀)。

用法:
    python -m antifive.tools.katago_vs_heur --model models/katago/antifive15-init.bin \
        [--games 10] [--sims 400] [--heur-budget 2.0] [--engine <katago.exe>] \
        [--engine-cfg <gtp.cfg>]

引擎路径默认取自 config/engines.json(config/gtp_test.cfg 为本仓库附带配置)。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np

from ..heuristic import choose_heuristic_move
from ..paths import config_dir, resolve
from ..reversegomoku import (
    BLACK, GameConfig, ReverseGomoku, WHITE,
)


def _engine_defaults() -> tuple:
    """从 config/engines.json 读取 KataGo 可执行文件与配置路径。"""
    try:
        with open(config_dir() / "engines.json", encoding="utf-8") as f:
            for e in json.load(f):
                if e.get("kind") == "heuristic":
                    continue
                return resolve(e.get("engine_path", "")), resolve(e.get("config_path", ""))
    except Exception:
        pass
    return "", ""


_ENGINE_DEFAULT, _ENGINE_CFG_DEFAULT = _engine_defaults()

# KataGo 坐标: 列字母(A-O, 跳过I) + 行号(顶行=15)
LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def idx_to_gtp(idx: int) -> str:
    r, c = divmod(int(idx), 15)
    return f"{LETTERS[c]}{15 - r}"


def gtp_to_idx(coord: str) -> int:
    coord = coord.strip().upper()
    letter = coord[0]
    num = int(coord[1:])
    c = LETTERS.index(letter)
    r = 15 - num
    return r * 15 + c


class KatagoGTP:
    def __init__(self, engine, model, cfg, sims):
        self.proc = subprocess.Popen(
            [engine, "gtp", "-model", model, "-config", cfg, "-override-config", f"maxVisits={sims},ponderingEnabled=false"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace",
        )
        self.boardsize(15)

    def _cmd(self, line: str) -> str:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        while True:
            out = self.proc.stdout.readline()
            if out.startswith("= "):
                return out[2:].strip()
            if out.startswith("? "):
                raise RuntimeError(f"GTP error for {line}: {out[2:].strip()}")

    def boardsize(self, n: int):
        self._cmd(f"boardsize {n} {n}")
        self._cmd("clear_board")

    def play(self, color: str, coord: str):
        self._cmd(f"play {color} {coord}")

    def genmove(self, color: str) -> str:
        return self._cmd(f"genmove {color}")

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.terminate()
        except Exception:
            pass


def _random_move(board, player, pending, turn_count, white_turns, config, rng):
    """随机基线:等概率选一个合法着法(含自杀着,由引擎裁决)。"""
    mask = ReverseGomoku.legal_mask_for(board, player, pending,
                                        white_turns, config, turn_count)
    legal = np.nonzero(mask)[0]
    if not legal.size:
        return None
    return int(rng.choice(legal))


def _heur_move(depth, h_budget):
    """构造固定深度的启发式着法函数。"""
    def choose(board, player, pending, turn_count, white_turns, config, rng):
        return choose_heuristic_move(board, player, pending, turn_count,
                                     white_turns, config, rng,
                                     depth=depth,
                                     time_budget=h_budget if h_budget > 0 else None)
    return choose


def play_game(engine, model, cfg, sims, opp_move, rng, seed, white_restrict):
    """KataGo 与 opp_move(棋盘状态 → 着法 idx 的函数)对弈一局。"""
    g = ReverseGomoku(GameConfig(loss_start_turns=8, white_restrict_turns=white_restrict, mask_suicide=False))
    kg = KatagoGTP(engine, model, cfg, sims)
    kg_is_black = (seed % 2 == 0)
    try:
        while not g.game_over and g.move_count < 450:
            color = "B" if g.current_player == BLACK else "W"
            if (g.current_player == BLACK) == kg_is_black:
                coord = kg.genmove(color)
                if coord.lower() == "resign":
                    g.game_over = True
                    g.loser = g.current_player
                    break
                a = gtp_to_idx(coord)
            else:
                a = opp_move(g.board, g.current_player, g.pending,
                             g.turn_count, g.white_turns, g.config, rng)
                if a is None:
                    g.game_over = True
                    g.loser = g.current_player
                    break
                kg.play(color, idx_to_gtp(a))
            g.make_move(a)
    finally:
        kg.close()
    if g.is_draw:
        return "draw"
    return "W" if g.loser == BLACK else "B"


def _run_tier(args, rng, name, opp_move, tier_idx):
    """与单个对手对战 args.games 局,返回 (kg_wins, opp_wins, draws, total)。"""
    kg_wins = opp_wins = draws = 0
    for i in range(args.games):
        seed = args.seed + tier_idx * 1000 + i
        res = play_game(args.engine, args.model, args.engine_cfg, args.sims,
                        opp_move, rng, seed, args.white_restrict)
        kg_black = (seed % 2 == 0)
        if res == "draw":
            draws += 1
        elif (res == "B") == kg_black:
            kg_wins += 1
        else:
            opp_wins += 1
        verdict = "和棋" if res == "draw" else ("KataGo胜" if (res == "B") == kg_black else f"{name}胜")
        print(f"[{i + 1}/{args.games}] katago({'黑' if kg_black else '白'}) vs {name}: {verdict}",
              flush=True)
    return kg_wins, opp_wins, draws


def main():
    p = argparse.ArgumentParser(description="KataGo vs 分级对手棋力检验")
    p.add_argument("--model", required=True)
    p.add_argument("--engine", default=_ENGINE_DEFAULT,
                   help="katago 可执行文件(默认取 config/engines.json)")
    p.add_argument("--engine-cfg", default=_ENGINE_CFG_DEFAULT,
                   help="GTP 配置文件(默认取 config/engines.json)")
    p.add_argument("--games", type=int, default=10, help="每层对手对局数")
    p.add_argument("--sims", type=int, default=400)
    p.add_argument("--heur-budget", type=float, default=2.0,
                   help="启发式深度 8 层每步限时(秒),d2/d4 不设限")
    p.add_argument("--white-restrict", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    # 对手由弱到强:随机基线 → d2 → d4 → d8
    tiers = [
        ("随机基线", _random_move),
        ("启发式 d2", _heur_move(2, 0.0)),
        ("启发式 d4", _heur_move(4, 0.0)),
        ("启发式 d8", _heur_move(8, args.heur_budget)),
    ]
    print(f"KataGo({args.model}) 分级检验: 随机→d2→d4→d8 各 {args.games} 局"
          f"(sims={args.sims}, d8 限时 {args.heur_budget}s/步)")
    for idx, (name, opp_move) in enumerate(tiers):
        print(f"\n=== 第 {idx + 1} 层: vs {name} ===", flush=True)
        kg_wins, opp_wins, draws = _run_tier(args, rng, name, opp_move, idx)
        total = kg_wins + opp_wins + draws
        print(f"KataGo 胜 {kg_wins} | {name} 胜 {opp_wins} | 和棋 {draws} | 共 {total}")
        print(f"KataGo 胜率: {kg_wins / max(total, 1):.0%}")
        if kg_wins == 0:
            print(f"\nKataGo 对 {name} 全败(0 胜),棋力不足,停止后续更强对手测试")
            return
    print("\nKataGo 通过了全部层级检验(每层均有胜局)")


if __name__ == "__main__":
    main()
