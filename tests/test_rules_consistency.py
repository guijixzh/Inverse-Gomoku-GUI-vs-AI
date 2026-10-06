"""C++ 引擎 (rules_probe) 与 Python 参考引擎 (reversegomoku) 规则一致性验证。

对每一局随机对局:
  1. 用 Python 引擎生成合法着法序列(per-stage);
  2. 用 C++ rules_probe 回放同一序列,逐步比对 legal / whiteTurns / 终局胜负;
  3. 在抽样局面比对"合法着法集合"(忽略 C++ 的 pass)。

rules_probe 由 KataGomo(KataGo 分支)构建,本仓库不附带可执行文件。
设置环境变量 ANTIFIVE_RULES_PROBE 指向 rules_probe.exe 后运行,否则跳过。

用法:
    python -m pytest tests/test_rules_consistency.py
    python tests/test_rules_consistency.py [--games N] [--white-restrict 2] [--probe PATH]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
import pytest

from antifive.reversegomoku import (
    BLACK, EMPTY, GameConfig, ReverseGomoku, WHITE,
)

PROBE_DEFAULT = os.environ.get("ANTIFIVE_RULES_PROBE", "")


def idx_to_xy(idx: int) -> tuple:
    """Python 平铺索引 r*15+c → 探针坐标 (x=c, y=r)。"""
    r, c = divmod(int(idx), 15)
    return c, r


def parse_probe(text: str):
    """解析 rules_probe 输出 → (moves 列表, legal_set)。"""
    moves = []
    legal_set = set()
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "M":
            moves.append({
                "idx": int(parts[1]),
                "x": int(parts[2].split(",")[0]),
                "y": int(parts[2].split(",")[1]),
                "pla": parts[3].split("=")[1],
                "legal": int(parts[4].split("=")[1]),
                "whiteTurns": int(parts[6].split("=")[1]),
                "over": int(parts[7].split("=")[1]),
                "winner": parts[8].split("=")[1],
            })
        elif parts[0] == "S":
            d = dict()
            for p in parts[2:]:
                k, v = p.split("=", 1)
                d[k] = v
            moves[-1]["post"] = d
        elif parts[0] == "legal-set":
            for tok in parts[1:]:
                if tok == "pass":
                    continue
                x, y = tok.split(",")
                legal_set.add((int(x), int(y)))
    return moves, legal_set


def run_probe(probe: str, white_restrict: int, seq_xy) -> str:
    args = [probe, str(white_restrict)] + [f"{x},{y}" for x, y in seq_xy]
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    return r.stdout


def play_one_random_game(rng) -> tuple:
    """用 Python 引擎随机合法走一局,返回 (config, 步序列 idx, 每步 white_turns, 终局胜负)。"""
    cfg = GameConfig(loss_start_turns=8, white_restrict_turns=2, mask_suicide=False)
    g = ReverseGomoku(cfg)
    seq_idx = []
    white_turns_log = []
    while not g.game_over and g.move_count < 400:
        mask = g.legal_mask()
        legal = np.nonzero(mask)[0]
        if len(legal) == 0:
            break
        a = int(rng.choice(legal))
        seq_idx.append(a)
        white_turns_log.append(g.white_turns)
        g.make_move(a)
    winner = 0 if g.is_draw else (WHITE if g.loser == BLACK else (BLACK if g.loser == WHITE else 0))
    return cfg, seq_idx, white_turns_log, winner


def verify(probe: str, games: int, white_restrict: int,
           legal_set_samples: int) -> int:
    """跑一致性验证,返回不一致局数。"""
    rng = np.random.default_rng(12345)
    n_mismatch = 0
    n_games = 0
    n_legal_set = 0

    for gi in range(games):
        cfg, seq_idx, wt_log, py_winner = play_one_random_game(rng)
        seq_xy = [idx_to_xy(i) for i in seq_idx]
        text = run_probe(probe, white_restrict, seq_xy)
        moves, _ = parse_probe(text)

        if len(moves) != len(seq_idx):
            print(f"[game {gi}] length mismatch: python {len(seq_idx)} vs probe {len(moves)}")
            n_mismatch += 1
            continue

        ok = True
        for k, m in enumerate(moves):
            if m["legal"] != 1:
                print(f"[game {gi}] move {k} legal mismatch: probe={m['legal']}")
                ok = False
            if m["whiteTurns"] != wt_log[k]:
                print(f"[game {gi}] move {k} whiteTurns: probe={m['whiteTurns']} py={wt_log[k]}")
                ok = False
        probe_winner = moves[-1]["post"]["winner"] if "post" in moves[-1] else "."
        probe_over = moves[-1]["post"]["over"] if "post" in moves[-1] else "0"
        py_winner_str = {BLACK: "B", WHITE: "W", 0: "E"}[py_winner]
        if probe_over == "1" and probe_winner != py_winner_str:
            print(f"[game {gi}] final winner: probe={probe_winner} py={py_winner_str}")
            ok = False
        if probe_over == "0" and py_winner != 0:
            print(f"[game {gi}] python finished but probe not: py={py_winner_str}")
            ok = False

        if ok:
            n_games += 1
        else:
            n_mismatch += 1

        sample_step = max(1, len(seq_idx) // legal_set_samples)
        for step in range(0, len(seq_idx), sample_step):
            prefix = seq_xy[:step + 1]
            txt = run_probe(probe, white_restrict, prefix)
            _, cpp_set = parse_probe(txt)
            g = ReverseGomoku(cfg)
            for a in seq_idx[:step + 1]:
                g.make_move(a)
            py_mask = g.legal_mask()
            py_set = {idx_to_xy(i) for i in np.nonzero(py_mask)[0]}
            if py_set != cpp_set:
                print(f"[game {gi}] legal-set mismatch at step {step}: "
                      f"py-only={py_set - cpp_set} cpp-only={cpp_set - py_set}")
                ok = False
            n_legal_set += 1

    print(f"\n结果: {n_games} 局一致 | {n_mismatch} 局不一致 | 合法集合比对 {n_legal_set} 处")
    return n_mismatch


@pytest.mark.skipif(not PROBE_DEFAULT, reason="未设置 ANTIFIVE_RULES_PROBE,rules_probe 不可用")
def test_rules_consistency():
    """小规模一致性冒烟(完整验证请用命令行 --games 30)。"""
    assert verify(PROBE_DEFAULT, games=5, white_restrict=2,
                  legal_set_samples=3) == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=30)
    p.add_argument("--white-restrict", type=int, default=2)
    p.add_argument("--probe", default=PROBE_DEFAULT)
    p.add_argument("--legal-set-samples", type=int, default=6,
                   help="每局抽几个前缀局面比对合法集合")
    args = p.parse_args()
    if not args.probe:
        print("请用 --probe 或环境变量 ANTIFIVE_RULES_PROBE 指定 rules_probe 可执行文件")
        sys.exit(2)
    if verify(args.probe, args.games, args.white_restrict,
              args.legal_set_samples) != 0:
        print("FAIL: 存在不一致,请检查")
        sys.exit(1)
    print("OK: 规则一致性验证通过")


if __name__ == "__main__":
    main()
