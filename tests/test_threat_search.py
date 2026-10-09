"""强制杀链搜索(threat_search)回归测试。

覆盖:kill_pairs 与暴力枚举对拍、route 语义(封堵格必须有效)、
防守交集引理的两个方向(可封堵=逃生 / 无路可封=必败)、
find_forced_kill 的证明线(全量枚举防守复核)、开关/预算回退、
以及集成后 choose_heuristic_move 的行为与 VCF 关闭时的一致性。
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from antifive import threat_search as ts
from antifive.heuristic import choose_heuristic_move
from antifive.reversegomoku import (
    BLACK, BOARD_SIZE, EMPTY, WHITE, GameConfig, ReverseGomoku, check_win_at,
)
from antifive.tactics import kill_captures, kill_pairs

B, W = BLACK, WHITE
CFG = GameConfig(loss_start_turns=8, white_restrict_turns=2, mask_suicide=True)


def _board(stones: dict) -> np.ndarray:
    b = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    for (r, c), v in stones.items():
        b[r, c] = v
    return b


def _other(color):
    return WHITE if color == BLACK else BLACK


def _brute_pairs(board, player, tc, wt, config) -> set:
    """暴力枚举全部真杀对:模拟占领 c + 安置 t,检查对方是否连五。"""
    if tc < config.loss_start_turns:
        return set()
    opp = _other(player)
    mask = ReverseGomoku.legal_mask_for(board, player, -1, wt, config, tc)
    flat = board.reshape(-1)
    out = set()
    for ci in np.nonzero(mask)[0]:
        ci = int(ci)
        if flat[ci] != opp:
            continue
        b2 = board.copy()
        b2[ci // BOARD_SIZE, ci % BOARD_SIZE] = player
        r0, c0 = divmod(ci, BOARD_SIZE)
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nr, nc = r0 + dr, c0 + dc
            while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE \
                    and b2[nr, nc] == EMPTY:
                b3 = b2.copy()
                b3[nr, nc] = opp
                if check_win_at(b3, nr, nc, opp):
                    out.add((ci, nr * BOARD_SIZE + nc))
                nr += dr
                nc += dc
    return out


def _random_games(n_games=4, moves=70, seed=31):
    rng = np.random.default_rng(seed)
    for _ in range(n_games):
        g = ReverseGomoku(CFG)
        for _ in range(moves):
            if g.game_over:
                break
            legal = np.nonzero(g.legal_mask())[0]
            if legal.size == 0:
                break
            g.make_move(int(rng.choice(legal)))
        if not g.game_over and g.pending < 0:
            yield g


# ---------------------------------------------------------------- 枚举正确性
def test_kill_pairs_matches_kill_captures():
    checked = 0
    for g in _random_games():
        pairs = kill_pairs(g.board, g.current_player, g.pending, g.turn_count,
                           g.white_turns, g.config)
        caps = kill_captures(g.board, g.current_player, g.pending,
                             g.turn_count, g.white_turns, g.config)
        assert [(c, t) for c, t, _ in pairs] == list(caps)
        checked += 1
    assert checked >= 3


def test_kill_pairs_sound_and_routes_block():
    fixtures = [
        _board({(5, 1): W, (5, 2): W, (5, 3): W, (5, 4): W, (5, 8): W,
                (0, 0): B, (0, 1): B}),
        _board({(5, 1): W, (5, 2): W, (5, 3): W, (5, 4): W, (4, 6): W,
                (14, 0): B, (14, 1): B, (14, 2): B}),
    ]
    tc, wt, player = 20, 5, B
    checked_pairs = 0
    for board in fixtures:
        pairs = kill_pairs(board, player, -1, tc, wt, CFG)
        assert pairs
        brute = _brute_pairs(board, player, tc, wt, CFG)
        assert {(c, t) for c, t, _ in pairs} <= brute   # 只保守,不虚报
        flat = board.reshape(-1)
        for c, t, route in pairs:
            assert flat[t] == EMPTY
            b_ok = board.copy()
            b_ok[c // BOARD_SIZE, c % BOARD_SIZE] = player   # 模拟占领 c
            mask_ok = ReverseGomoku.legal_mask_for(
                b_ok, player, c, wt, CFG, tc)
            assert mask_ok[t]                 # 不封堵时 T 可达
            for cell in route:
                assert flat[cell] == EMPTY and cell != t
                b3 = board.copy()
                b3[cell // BOARD_SIZE, cell % BOARD_SIZE] = player
                b3[c // BOARD_SIZE, c % BOARD_SIZE] = player
                mask3 = ReverseGomoku.legal_mask_for(
                    b3, player, c, wt, CFG, tc)
                assert not mask3[t]           # 封堵后安置到 T 非法
            checked_pairs += 1
    assert checked_pairs >= 2


def test_jump_through_matches_snapshot():
    from antifive.tools import heuristic_v3 as h3
    from antifive.tactics import jump_through
    boards = [g.board for g in _random_games(n_games=2, seed=5)]
    for b in boards:
        for r in range(0, BOARD_SIZE, 3):
            for c in range(0, BOARD_SIZE, 4):
                for color in (B, W):
                    assert jump_through(b, r, c, color) == \
                        h3._jump_through(b, r, c, color)


# ---------------------------------------------------------------- 防守方语义
def test_defender_escapes_by_blocking_route():
    board = _board({(5, 1): W, (5, 2): W, (5, 3): W, (5, 4): W, (5, 8): W,
                    (8, 6): B, (0, 0): B, (0, 1): B})
    ctx = ts._Ctx(None, 100000)
    assert ts.defender_survives(ctx, board, W, 20, 5, CFG, 0)


def test_defender_doomed_when_no_block_available():
    board = _board({(5, 1): W, (5, 2): W, (5, 3): W, (5, 4): W, (5, 8): W,
                    (14, 0): B, (14, 1): B, (14, 2): B})
    ctx = ts._Ctx(None, 100000)
    assert not ts.defender_survives(ctx, board, W, 20, 5, CFG, 2)


def test_defender_blocks_completion_cell_via_capture_place():
    # 贴身杀 route=():唯一封堵格是完成格 T;普通落子自杀,但可用
    # 占领+安置把我方子放到 T 封堵(此前的剪枝 bug 回归)
    board = _board({(5, 1): W, (5, 2): W, (5, 3): W, (5, 4): W, (4, 6): W,
                    (8, 5): B, (0, 0): B, (0, 1): B})
    pairs = kill_pairs(board, B, -1, 20, 5, CFG)
    assert pairs and pairs[0][2] == ()
    ctx = ts._Ctx(None, 100000)
    assert ts.defender_survives(ctx, board, W, 20, 5, CFG, 0)
    board2 = _board({(5, 1): W, (5, 2): W, (5, 3): W, (5, 4): W, (4, 6): W,
                     (14, 0): B, (14, 1): B, (14, 2): B})
    ctx2 = ts._Ctx(None, 100000)
    assert not ts.defender_survives(ctx2, board2, W, 20, 5, CFG, 0)


# ---------------------------------------------------------------- 证明线与集成
DOUBLE_FIXTURE = {(7, 4): W, (7, 5): W, (7, 6): W,
                  (4, 7): W, (5, 7): W, (6, 7): W,
                  (7, 0): W, (7, 11): W, (0, 7): W, (11, 7): W, (14, 14): W,
                  (0, 0): B, (0, 2): B, (2, 0): B}


def _all_defender_replies(board, defender, tc, wt):
    opp = _other(defender)
    mask = ReverseGomoku.legal_mask_for(board, defender, -1, wt, CFG, tc)
    flat = board.reshape(-1)
    out = []
    for p in np.nonzero(mask)[0]:
        pi = int(p)
        if flat[pi] != EMPTY:
            continue
        b2, _, _, tc2, wt2, loser = ReverseGomoku.apply_step(
            board, pi, defender, -1, tc, wt, CFG)
        out.append((b2, tc2, wt2, loser))
    for x in np.nonzero(mask)[0]:
        xi = int(x)
        if flat[xi] != opp:
            continue
        b1, p1, pend1, tc1, wt1, loser1 = ReverseGomoku.apply_step(
            board, xi, defender, -1, tc, wt, CFG)
        if loser1 == defender:
            continue
        pmask = ReverseGomoku.legal_mask_for(b1, p1, pend1, wt1, CFG, tc1)
        for p in np.nonzero(pmask)[0]:
            b2, _, _, tc2, wt2, loser2 = ReverseGomoku.apply_step(
                b1, int(p), p1, pend1, tc1, wt1, CFG)
            out.append((b2, tc2, wt2, loser2))
    return out


def test_find_forced_kill_double_fixture_proof():
    board = _board(DOUBLE_FIXTURE)
    assert not kill_captures(board, B, -1, 20, 5, CFG)   # 无立即杀
    move = ts.find_forced_kill(board, B, -1, 20, 5, CFG, max_depth=6)
    assert move is not None
    b1, p1, pend1, tc1, wt1, loser1 = ReverseGomoku.apply_step(
        board, move, B, -1, 20, 5, CFG)
    assert loser1 == 0
    place = ts.find_forced_kill(b1, B, pend1, tc1, wt1, CFG, max_depth=6)
    assert place is not None
    b2, _, _, tc2, wt2, loser2 = ReverseGomoku.apply_step(
        b1, place, p1, pend1, tc1, wt1, CFG)
    assert loser2 == 0
    # 证明线复核:全量枚举对手所有防守,每一种都被我方下一手立即杀
    for b3, tc3, wt3, loser in _all_defender_replies(b2, W, tc2, wt2):
        if loser == W:
            continue
        if loser == B:
            pytest.fail("防守竟然反杀成功")
        assert kill_captures(b3, B, -1, tc3, wt3, CFG), \
            "存在逃生的防守着法,证明不成立"


def test_find_forced_kill_none_on_quiet_and_disabled(monkeypatch):
    quiet = _board({(7, 7): B, (7, 8): W, (8, 8): B})
    assert ts.find_forced_kill(quiet, B, -1, 20, 5, CFG, max_depth=6) is None
    board = _board(DOUBLE_FIXTURE)
    monkeypatch.setattr(ts, "USE_VCF", False)
    assert ts.find_forced_kill(board, B, -1, 20, 5, CFG, max_depth=6) is None


def test_find_forced_kill_budget_fallback():
    board = _board(DOUBLE_FIXTURE)
    forced = ts.find_forced_kill(board, B, -1, 20, 5, CFG, max_depth=6)
    mv = ts.find_forced_kill(board, B, -1, 20, 5, CFG, max_depth=6,
                             deadline=time.perf_counter() - 1.0)
    assert mv in (None, forced)          # 超时只允许回退或已证实的结论
    assert ts.find_forced_kill(board, B, -1, 20, 5, CFG, max_depth=6,
                               node_cap=1) is None


def test_defensive_vcf_demotes_proven_loss(monkeypatch):
    """根候选走完若对手可证明强制杀,该着法降为必败分(防守校验单元)。"""
    from antifive import heuristic as h

    cfg = GameConfig(loss_start_turns=8, white_restrict_turns=2,
                     mask_suicide=True)
    board = np.zeros((15, 15), dtype=np.int8)
    board[7, 7] = W
    board[0, 0] = B
    board[14, 14] = W
    pending = 7 * 15 + 7                     # 白子刚被占领,轮黑安置
    legal = np.nonzero(ReverseGomoku.legal_mask_for(
        board, B, pending, 5, cfg, 20))[0]
    assert legal.size >= 3
    ranked = [(int(m), 1.0) for m in legal[:3]]

    monkeypatch.setattr(ts, "find_forced_kill", lambda *a, **k: None)
    out = h._filter_defensive_kills(list(ranked), board, B, pending, 20, 5,
                                    cfg, 0.05)
    assert all(v > -h.MATE_LIMIT for _, v in out)         # 无杀:不降分
    monkeypatch.setattr(ts, "find_forced_kill", lambda *a, **k: 1)
    out2 = h._filter_defensive_kills(list(ranked), board, B, pending, 20, 5,
                                     cfg, 0.05)
    assert all(v <= -h.MATE_LIMIT for _, v in out2)       # 有杀:降为必败


def test_vcf_defaults_and_budget_slice():
    """深杀修复:v5.1 默认深度 8/节点 60000;有对局限时时预算按 5% 放宽,
    夹在 [VCF_TIME_CAP, 0.6s]。"""
    from antifive.heuristic import _vcf_slice
    assert ts.VCF_MAX_DEPTH >= 8
    assert ts.VCF_NODE_CAP >= 60000
    assert _vcf_slice(None) == ts.VCF_TIME_CAP
    assert _vcf_slice(12.0) == 0.6
    assert abs(_vcf_slice(2.0) - 0.1) < 1e-9
    assert _vcf_slice(0.5) == ts.VCF_TIME_CAP


def test_choose_uses_vcf():
    board = _board(DOUBLE_FIXTURE)
    forced = ts.find_forced_kill(board, B, -1, 20, 5, CFG, max_depth=6)
    mask = ReverseGomoku.legal_mask_for(board, B, -1, 5, CFG, 20)
    for seed in range(3):
        move = choose_heuristic_move(board, B, -1, 20, 5, CFG,
                                     np.random.default_rng(seed), depth=2)
        assert move == forced and mask[move]


def test_choose_matches_analysis_when_vcf_off(monkeypatch):
    """VCF 关闭时 choose 不得偏离 analysis 的根评估(噪声带内)。

    引擎候选生成/根窗口/占领步符号已按复盘结论有意修复,不再与 v3
    快照逐着相等。"""
    from antifive import heuristic as h
    monkeypatch.setattr(ts, "USE_VCF", False)
    for g in list(_random_games(n_games=2, seed=13))[:2]:
        args = (g.board, g.current_player, g.pending, g.turn_count,
                g.white_turns, CFG)
        _, moves = h.analysis_moves(*args, np.random.default_rng(0),
                                    depth=2, k=6)
        top_v = moves[0][1]
        m1 = choose_heuristic_move(*args, np.random.default_rng(0), depth=2)
        assert dict(moves).get(int(m1), -1e18) >= top_v - h.O_NOISE - 1e-9
