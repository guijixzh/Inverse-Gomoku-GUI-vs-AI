"""启发式 AI 战术回归:固定局面选点 + 杀棋枚举与暴力枚举对拍。

这些测试锁定可验证的战术语义(一步杀/两步杀/必守/防御路线),
并保证优化(早退、缓存、根剪枝等)不改变正确性;不锁定"棋风参数"。
"""

from __future__ import annotations

import numpy as np
import pytest

from antifive.heuristic import analysis_moves, choose_heuristic_move
from antifive.reversegomoku import (
    BLACK, BOARD_SIZE, EMPTY, WHITE, GameConfig, ReverseGomoku,
)
from antifive.tactics import kill_captures, kill_placements

B, W = BLACK, WHITE
CFG = GameConfig(loss_start_turns=8, white_restrict_turns=2, mask_suicide=True)


def _board(stones: dict) -> np.ndarray:
    b = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    for (r, c), v in stones.items():
        b[r, c] = v
    return b


def _pick(board, player=BLACK, pending=-1, tc=20, wt=5, depth=2, seed=0):
    return choose_heuristic_move(board, player, pending, tc, wt, CFG,
                                 np.random.default_rng(seed), depth=depth)


# ------------------------------------------------------------------ 两步杀
def test_kill_captures_two_step():
    board = _board({(5, 1): W, (5, 2): W, (5, 3): W, (5, 4): W,
                    (8, 8): W, (0, 0): B, (0, 2): B, (1, 1): B})
    kc = set(kill_captures(board, B, -1, 20, 5, CFG))
    assert (8 * BOARD_SIZE + 8, 5 * BOARD_SIZE + 5) in kc
    assert _pick(board) == 8 * BOARD_SIZE + 8


# ------------------------------------------------------------------ 一步杀(持子)
def test_kill_placement_holding_stone():
    board = _board({(5, 1): W, (5, 2): W, (5, 3): W, (5, 4): W,
                    (8, 8): B, (0, 0): B, (0, 2): B, (1, 1): B})
    kp = kill_placements(board, B, 8 * BOARD_SIZE + 8, 20, 5, CFG)
    assert list(kp) == [5 * BOARD_SIZE + 5]
    assert _pick(board, pending=8 * BOARD_SIZE + 8) == 5 * BOARD_SIZE + 5


# ------------------------------------------------------------------ 必守(堵桥射线)
def test_defend_bridge_route():
    board = _board({(7, 5): B, (7, 6): B, (7, 7): B, (7, 8): B,
                    (7, 12): B, (0, 14): W})
    move = _pick(board, depth=3)
    assert move in (7 * BOARD_SIZE + 10, 7 * BOARD_SIZE + 11)


# ------------------------------------------------------------------ 双杀(率先执行)
def test_double_kill_executes_capture():
    board = _board({(5, 1): W, (5, 2): W, (5, 3): W, (5, 4): W, (5, 9): W,
                    (9, 1): W, (9, 2): W, (9, 3): W, (9, 4): W, (9, 9): W,
                    (0, 0): B, (0, 2): B, (1, 1): B})
    move = _pick(board, depth=2)
    assert move in (5 * BOARD_SIZE + 9, 9 * BOARD_SIZE + 9)


# ---------------------------------------------------------- 杀棋枚举暴力对拍
def _brute_captures(board, player, pending, tc, wt, config) -> set:
    if pending >= 0 or tc < config.loss_start_turns:
        return set()
    opp = W if player == B else B
    legal = np.nonzero(ReverseGomoku.legal_mask_for(
        board, player, pending, wt, config, tc))[0]
    out = set()
    for c in legal:
        ci = int(c)
        if board.reshape(-1)[ci] != opp:
            continue
        b2, p2, pend2, tc2, wt2, loser = ReverseGomoku.apply_step(
            board, ci, player, pending, tc, wt, config)
        if loser == player:
            continue
        pmask = ReverseGomoku.legal_mask_for(b2, p2, pend2, wt2, config, tc2)
        for t in np.nonzero(pmask)[0]:
            b3, _, _, _, _, loser3 = ReverseGomoku.apply_step(
                b2, int(t), p2, pend2, tc2, wt2, config)
            if loser3 == opp:
                out.add((ci, int(t)))
                break
    return out


def test_kill_captures_matches_bruteforce_random():
    rng = np.random.default_rng(11)
    checked = 0
    for game_i in range(4):
        g = ReverseGomoku(CFG)
        for _ in range(80):
            if g.game_over or g.move_count >= 120:
                break
            legal = np.nonzero(g.legal_mask())[0]
            if legal.size == 0:
                break
            g.make_move(int(rng.choice(legal)))
            if g.pending >= 0 or g.turn_count < CFG.loss_start_turns:
                continue
            fast = {(c, t) for c, t in kill_captures(
                g.board, g.current_player, g.pending, g.turn_count,
                g.white_turns, g.config)}
            slow = _brute_captures(g.board, g.current_player, g.pending,
                                   g.turn_count, g.white_turns, g.config)
            assert {c for c, _ in fast} == {c for c, _ in slow}
            checked += 1
    assert checked >= 8


# ---------------------------------------------------------- 随机局面合法性
@pytest.mark.parametrize("depth", [2, 3])
def test_choose_returns_legal_move_random(depth):
    rng = np.random.default_rng(5 + depth)
    for game_i in range(3):
        g = ReverseGomoku(CFG)
        for _ in range(30):
            if g.game_over:
                break
            legal = np.nonzero(g.legal_mask())[0]
            if legal.size == 0:
                break
            g.make_move(int(rng.choice(legal)))
        if g.game_over:
            continue
        mask = g.legal_mask()
        if not mask.any():
            continue
        move = _pick(g.board, g.current_player, g.pending, g.turn_count,
                     g.white_turns, depth=depth, seed=game_i)
        assert move is not None and mask[move]


def test_analysis_values_are_ordered():
    board = _board({(5, 1): W, (5, 2): W, (5, 3): W, (5, 4): W,
                    (8, 8): W, (0, 0): B, (0, 2): B, (1, 1): B})
    pos_v, moves = analysis_moves(board, B, -1, 20, 5, CFG,
                                  np.random.default_rng(0), depth=2, k=5)
    assert moves and moves[0][0] == 8 * BOARD_SIZE + 8
    vals = [v for _, v in moves]
    assert vals == sorted(vals, reverse=True)
    assert pos_v == vals[0]
    assert abs(pos_v) < 1e6


def test_empty_board_first_move_stable():
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    move = _pick(board, depth=2)
    assert move is not None and 0 <= move < BOARD_SIZE * BOARD_SIZE


# ---------------------------------------------------------- 根剪枝/参数/PVS
def _random_positions(n_games=3, moves=24, seed=17):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_games):
        g = ReverseGomoku(CFG)
        for _ in range(moves):
            if g.game_over:
                break
            legal = np.nonzero(g.legal_mask())[0]
            if legal.size == 0:
                break
            g.make_move(int(rng.choice(legal)))
        if not g.game_over:
            out.append(g)
    return out


def test_root_prune_matches_exact_values():
    for gi, g in enumerate(_random_positions()):
        _, moves = analysis_moves(g.board, g.current_player, g.pending,
                                  g.turn_count, g.white_turns, CFG,
                                  np.random.default_rng(gi), depth=3, k=8)
        exact = {m: v for m, v in moves}
        best = max(exact.values())
        move = choose_heuristic_move(
            g.board, g.current_player, g.pending, g.turn_count, g.white_turns,
            CFG, np.random.default_rng(gi), depth=3)
        assert move in exact
        assert exact[move] >= best - 0.1


def test_pvs_full_width_equivalence(monkeypatch):
    import antifive.heuristic as h
    monkeypatch.setattr(h, "_include_extras", lambda *a: None)
    monkeypatch.setattr(h, "ROOT_K", 300)
    monkeypatch.setattr(h, "INNER_K", 300)
    monkeypatch.setattr(h, "O_HIST", 0.0)
    for g in _random_positions(n_games=2, moves=8, seed=23):
        for depth in (2, 3):
            monkeypatch.setattr(h, "USE_PVS", True)
            ctx1 = h._SearchCtx(h._TT(), None)
            v1 = h._search(g.board, g.current_player, g.pending, g.turn_count,
                           g.white_turns, CFG, depth, -1e18, 1e18, ctx1)
            monkeypatch.setattr(h, "USE_PVS", False)
            ctx2 = h._SearchCtx(h._TT(), None)
            v2 = h._search(g.board, g.current_player, g.pending, g.turn_count,
                           g.white_turns, CFG, depth, -1e18, 1e18, ctx2)
            assert v1 == pytest.approx(v2, abs=1e-9)


def test_load_params_whitelist(tmp_path):
    import antifive.heuristic as h
    path = tmp_path / "params.json"
    path.write_text('{"E_KILLABLE4": 1.234, "NOT_A_PARAM": 9, '
                    '"ROOT_K": 5, "E_DOUBLE_KILL": "bad"}', encoding="utf-8")
    old = {k: getattr(h, k) for k in ("E_KILLABLE4", "ROOT_K", "E_DOUBLE_KILL")}
    applied = h.load_params(path, force=True)
    try:
        assert applied == {"E_KILLABLE4": 1.234, "ROOT_K": 5}
        assert h.E_KILLABLE4 == pytest.approx(1.234)
        assert h.ROOT_K == 5 and isinstance(h.ROOT_K, int)
        assert h.E_DOUBLE_KILL == old["E_DOUBLE_KILL"]
    finally:
        for k, v in old.items():
            setattr(h, k, v)
    assert h.load_params(tmp_path / "missing.json", force=True) == {}
