"""自杀掩码回归测试(修复"占领自杀漏网"后)。

包含:确定性用例、随机全量比对、numpy<->torch 引擎一致性、自对弈冒烟。
需要 torch(缺失时整体跳过)。
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from antifive.mcts import MCTS
from antifive.network import PolicyValueNet
from antifive.paths import models_dir
from antifive.reversegomoku import (
    ACTION_SIZE, BLACK, BOARD_SIZE, EMPTY, GameConfig, ReverseGomoku, WHITE,
    _line5_map_for, check_win_at, fwd_run_lengths,
)
from antifive.training.reverse_gomoku_torch import BatchedReverseGomoku


# ---------------------------------------------------------------- 确定性用例
def test_deterministic_suicide_mask():
    cfg0 = GameConfig(loss_start_turns=0)                 # 窗口外
    cfg_win = GameConfig(loss_start_turns=8)              # 含窗口

    # A. 落子自杀(窗口外剔除)
    g = ReverseGomoku(cfg0)
    for a in [((7, 5), None), ((0, 0), None), ((7, 6), None), ((0, 1), None),
              ((7, 7), None), ((0, 2), None), ((7, 8), None), ((0, 3), None),
              ((1, 0), None), ((1, 1), None)]:
        g.make_move(a)
    m = g.legal_mask()
    assert not m[7 * 15 + 4] and not m[7 * 15 + 9]       # A1 落子自杀剔除
    assert m[0 * 15 + 5]                                  # A2 无关空位保留

    # B. 占领自杀(窗口外剔除)← bug 核心
    g = ReverseGomoku(cfg0)
    for a in [((2, 1), None), ((0, 0), None), ((2, 2), None), ((0, 1), None),
              ((2, 4), None), ((2, 3), None), ((2, 5), None), ((0, 2), None),
              ((1, 0), None), ((1, 1), None)]:
        g.make_move(a)
    m = g.legal_mask()
    assert not m[2 * 15 + 3]
    g_raw = ReverseGomoku(GameConfig(loss_start_turns=0, mask_suicide=False))
    for a in [((2, 1), None), ((0, 0), None), ((2, 2), None), ((0, 1), None),
              ((2, 4), None), ((2, 3), None), ((2, 5), None), ((0, 2), None),
              ((1, 0), None), ((1, 1), None)]:
        g_raw.make_move(a)
    assert g_raw.is_suicide_move(2 * 15 + 3)

    # C. 窗口内保留
    g = ReverseGomoku(cfg_win)
    for a in [((7, 5), None), ((0, 0), None), ((7, 6), None), ((0, 1), None),
              ((7, 7), None), ((0, 2), None)]:
        g.make_move(a)
    m = g.legal_mask()
    assert m[7 * 15 + 4] and m[7 * 15 + 9]

    # D. 受限白方窗口外同样剔除(白方块死局)
    b = np.full((15, 15), WHITE, dtype=np.int8)
    b[7, 7] = BLACK
    for r in range(15):
        b[r, 14] = EMPTY
        b[14, r] = EMPTY
    b[14, 0] = BLACK
    b[14, 7] = BLACK
    b[14, 14] = BLACK
    gd = ReverseGomoku()
    gd.board = b
    gd.turn_count = 10
    gd.current_player = WHITE
    gd.white_turns = 0
    assert gd.legal_mask().sum() == 0
    assert gd.is_stuck()
    gd.make_move(0)
    assert gd.game_over and gd.loser == WHITE

    # E. 黑方块死局(对称)
    b2 = np.full((15, 15), BLACK, dtype=np.int8)
    b2[7, 7] = WHITE
    for r in range(15):
        b2[r, 14] = EMPTY
        b2[14, r] = EMPTY
    b2[14, 0] = WHITE
    b2[14, 7] = WHITE
    b2[14, 14] = WHITE
    ge = ReverseGomoku()
    ge.board = b2
    ge.turn_count = 10
    ge.current_player = BLACK
    ge.white_turns = 5
    assert ge.is_stuck()

    # F. 满盘和棋(不判死局负)
    b3 = np.full((15, 15), BLACK, dtype=np.int8)
    for r in range(15):
        b3[r, r % 2] = WHITE
    gf = ReverseGomoku()
    gf.board = b3
    gf.turn_count = 10
    assert not gf.is_stuck()


def test_mcts_avoids_capture_suicide():
    cfg0 = GameConfig(loss_start_turns=0)
    device = torch.device("cpu")
    ckpt = models_dir() / "best_model.pth"
    if ckpt.is_file():
        net = PolicyValueNet.load(str(ckpt), device)
    else:
        net = PolicyValueNet(32, 2).to(device)
    mcts = MCTS(net, device, sims=150, config=cfg0)
    g = ReverseGomoku(cfg0)
    for a in [((2, 1), None), ((0, 0), None), ((2, 2), None), ((0, 1), None),
              ((2, 4), None), ((2, 3), None), ((2, 5), None), ((0, 2), None),
              ((1, 0), None), ((1, 1), None)]:
        g.make_move(a)
    a = mcts.search(g)            # (2,3) 为窗口外占领自杀
    assert a != 2 * 15 + 3


# ---------------------------------------------------------------- 随机全量比对
def test_random_mask_matches_suicide():
    rng = np.random.default_rng(42)
    cfg0 = GameConfig(loss_start_turns=0)
    cfg_win = GameConfig(loss_start_turns=8)
    cfg_raw = GameConfig(loss_start_turns=0, mask_suicide=False)
    n_states = n_suicide = n_legal = 0
    for trial in range(400):
        g = ReverseGomoku(cfg_raw)
        for _ in range(rng.integers(20, 80)):
            mm = g.legal_mask()
            L = np.nonzero(mm)[0]
            if len(L) == 0:
                break
            g.make_move(int(rng.choice(L)))
        if g.game_over or g.pending >= 0:
            continue
        n_states += 1
        m_raw = g.legal_mask()
        m_su = ReverseGomoku.legal_mask_for(
            g.board, g.current_player, g.pending, g.white_turns, cfg0, g.turn_count)
        for i in np.nonzero(m_raw)[0]:
            n_legal += 1
            s = g.is_suicide_move(i)
            if s:
                n_suicide += 1
            assert s == (not m_su[i]), f"trial {trial}: {g.idx_to_move(i)}"
    assert n_states > 0
    assert n_suicide > 30, f"自杀着仅 {n_suicide} 个,n= {n_legal}"

    # 窗口内: 不剔除
    n_win = 0
    for trial in range(150):
        g = ReverseGomoku(cfg_raw)
        for _ in range(rng.integers(3, 8)):
            mm = g.legal_mask()
            L = np.nonzero(mm)[0]
            if len(L) == 0:
                break
            g.make_move(int(rng.choice(L)))
        if g.game_over or g.pending >= 0 or g.turn_count >= 8:
            continue
        n_win += 1
        m_raw = g.legal_mask()
        m_win = ReverseGomoku.legal_mask_for(
            g.board, g.current_player, g.pending, g.white_turns, cfg_win,
            g.turn_count)
        assert np.array_equal(m_raw, m_win), f"trial {trial}"
    assert n_win > 20, f"窗口内样本仅 {n_win}"


# ---------------------------------------------------------------- 引擎一致性
def test_numpy_torch_engine_consistency():
    rng = np.random.default_rng(11)
    dev = torch.device("cpu")
    N = 8
    eng = BatchedReverseGomoku(N, dev)
    games = [ReverseGomoku() for _ in range(N)]
    steps = 0
    while True:
        m_np = np.stack([g.legal_mask() for g in games])
        m_ts = eng.legal_mask().cpu().numpy()
        assert np.array_equal(m_np, m_ts), f"step {steps} 掩码不一致"
        if eng.gameover.all() or steps > 500:
            break
        acts = np.zeros(N, dtype=np.int64)
        for i in range(N):
            if games[i].game_over:
                continue
            L = np.nonzero(m_np[i])[0]
            if len(L) == 0:
                games[i].make_move(0)
            else:
                a = int(rng.choice(L))
                acts[i] = a
                games[i].make_move(a)
        eng.apply_moves(torch.from_numpy(acts).to(dev))
        for i in range(N):
            expect = np.where(games[i].board == 2, -1, games[i].board)
            assert np.array_equal(eng.board[i].cpu().numpy(), expect)
            assert int(eng.pending[i]) == games[i].pending
            assert int(eng.turn_count[i]) == games[i].turn_count
            assert int(eng.white_turns[i]) == games[i].white_turns
            assert bool(eng.gameover[i]) == games[i].game_over
            assert bool(eng.draw[i]) == games[i].is_draw
            assert int(eng.loser[i]) == {0: 0, BLACK: 1, WHITE: -1}[games[i].loser]
        steps += 1


# ---------------------------------------------------------------- 自对弈冒烟
def test_selfplay_smoke():
    from antifive.training.selfplay import self_play_batch

    dev = torch.device("cpu")
    net32 = PolicyValueNet(32, 2).to(dev)
    exs = self_play_batch(net32, config=GameConfig(), n_games=4, sims=16, k=4,
                          temperature_moves=3, device=dev)
    v = np.array([e.value for e in exs])
    assert len(exs) > 0
    assert set(np.unique(v)) <= {-1.0, 0.0, 1.0}
