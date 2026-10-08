"""纯 numpy 引擎冒烟测试(不依赖 torch / pygame)。

覆盖:初始状态、落子与回合推进、移子(占领 + 安置)、安置路径约束、
白棋前 N 回合禁占领、连五判负、随机自对弈不变量、5 通道编码、
.afg 棋谱读写往返。

无 torch 环境也可完整运行,保证轻量用户与 CI 能验证核心规则。
"""

from __future__ import annotations

import numpy as np

from antifive import record
from antifive.reversegomoku import (
    ACTION_SIZE, BLACK, BOARD_SIZE, EMPTY, GameConfig, ReverseGomoku, WHITE,
    check_win_at,
)

def _place(r: int, c: int):
    """引擎的落子/占领参数形式:((r, c), None)。"""
    return ((r, c), None)


# ---------------------------------------------------------------- 基础状态
def test_initial_state():
    g = ReverseGomoku()
    assert g.board.shape == (BOARD_SIZE, BOARD_SIZE)
    assert np.all(g.board == EMPTY)
    assert g.current_player == BLACK
    assert not g.game_over and g.loser == 0 and not g.is_draw
    assert g.pending == -1 and g.turn_count == 0 and g.move_count == 0
    mask = g.legal_mask()
    assert mask.shape == (ACTION_SIZE,)
    assert int(mask.sum()) == BOARD_SIZE * BOARD_SIZE
    assert not g.is_stuck()


def test_place_alternates_and_records():
    g = ReverseGomoku()
    g.make_move(_place(7, 7))
    assert g.board[7, 7] == BLACK
    assert g.current_player == WHITE
    assert g.turn_count == 1 and g.move_count == 1
    g.make_move(0)                               # 也接受平铺 idx
    assert g.board[0, 0] == WHITE
    assert g.white_turns == 1
    assert g.current_player == BLACK
    assert g.turn_count == 2 and g.move_count == 2
    assert g.record() == [7 * BOARD_SIZE + 7, 0]


# ---------------------------------------------------------------- 移子规则
def test_capture_and_placement():
    g = ReverseGomoku()
    g.make_move(_place(1, 1))                    # 黑
    g.make_move(_place(0, 0))                    # 白
    g.make_move(_place(0, 0))                    # 黑占领:拿起白子
    assert g.board[0, 0] == BLACK
    assert g.pending == 0
    assert g.current_player == BLACK             # 同方继续安置步
    assert g.turn_count == 2                     # 占领不推进回合

    mask = g.legal_mask()
    assert not mask[0]                           # 原格已被己方占据
    assert mask[0 * BOARD_SIZE + 2]              # (0,2) 直线可达
    assert mask[1 * BOARD_SIZE + 0]              # (1,0) 直线可达
    assert not mask[2 * BOARD_SIZE + 2]          # (2,2) 对角线被 (1,1) 阻挡

    g.make_move(_place(0, 2))                    # 安置白子到 (0,2)
    assert g.board[0, 2] == WHITE
    assert g.pending == -1
    assert g.current_player == WHITE
    assert g.turn_count == 3 and g.white_turns == 1
    assert g.move_count == 4                     # 2 落子 + 占领 + 安置


def test_white_cannot_capture_early():
    g = ReverseGomoku()                          # white_restrict_turns=2
    g.make_move(_place(0, 0))                    # 黑
    g.make_move(_place(7, 7))                    # 白 1
    g.make_move(_place(1, 1))                    # 黑
    assert g.current_player == WHITE and g.white_turns == 1
    assert not g.legal_mask()[0]                 # 白第 1 回合禁止占领黑子 (0,0)

    g.make_move(_place(8, 8))                    # 白 2
    g.make_move(_place(2, 2))                    # 黑
    assert g.white_turns == 2
    assert g.legal_mask()[0]                     # 限制解除后可占领


# ---------------------------------------------------------------- 胜负判定
def test_check_win_at():
    b = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    b[7, 3:8] = BLACK
    assert check_win_at(b, 7, 3, BLACK)
    assert check_win_at(b, 7, 7, BLACK)
    assert not check_win_at(b, 7, 5, WHITE)
    b[7, 7] = EMPTY
    assert not check_win_at(b, 7, 3, BLACK)


def test_five_in_row_loses():
    cfg = GameConfig(loss_start_turns=0, mask_suicide=False)
    g = ReverseGomoku(cfg)
    for m in [_place(7, 3), _place(0, 0), _place(7, 4), _place(0, 1),
              _place(7, 5), _place(0, 2), _place(7, 6), _place(0, 3),
              _place(7, 7)]:
        assert not g.game_over
        g.make_move(m)
    assert g.game_over and not g.is_draw
    assert g.loser == BLACK                      # 连五者判负:黑棋输,白胜
    assert record.game_result(g) == "白胜"


# ---------------------------------------------------------------- 随机对局
def test_random_selfplay_invariants():
    rng = np.random.default_rng(7)
    for game_i in range(8):
        g = ReverseGomoku()
        for _ in range(600):
            if g.game_over:
                break
            legal = np.nonzero(g.legal_mask())[0]
            if legal.size == 0:
                g.make_move(0)                   # 无子可走:立即判负
                assert g.game_over
                break
            g.make_move(int(rng.choice(legal)))
        assert g.game_over, f"对局 {game_i} 未在 600 步内结束"
        if g.game_over and not g.is_draw:
            assert g.loser in (BLACK, WHITE)
        moves = g.record()
        assert len(moves) == g.move_count
        g2 = ReverseGomoku.replay(moves, g.config)
        assert np.array_equal(g2.board, g.board)
        if g2.game_over:
            assert g2.loser == g.loser and g2.is_draw == g.is_draw


# ---------------------------------------------------------------- 状态编码
def test_encode_state_channels():
    g = ReverseGomoku()
    g.make_move(_place(7, 7))
    g.make_move(_place(0, 0))
    enc = g.encode_state()
    assert enc.shape == (5, BOARD_SIZE, BOARD_SIZE)
    assert enc.dtype == np.float32
    assert enc[0, 7, 7] == 1 and enc[1, 0, 0] == 1
    assert enc[0].sum() == 1 and enc[1].sum() == 1
    assert enc[2].sum() == 0 and enc[3].sum() == 0   # 第 7 手前危险通道恒零
    assert enc[4].sum() == 0                         # 无待定移子


# ---------------------------------------------------------------- 棋谱读写
def test_record_dumps_loads_roundtrip():
    moves = [0, 112, 7 * BOARD_SIZE + 7, 42, 224, 113]
    cfg = GameConfig(white_restrict_turns=3, loss_start_turns=0,
                     mask_suicide=False)
    analysis = {1: {"wr": 50.0, "best": "H8",
                    "moves": [("J8", 48.5), ("K8", 47.2)]}}
    text = record.dumps(moves, cfg, "未完", analysis)
    rec = record.loads(text)
    assert rec.moves == moves
    assert rec.config == cfg
    assert rec.result == "未完"
    assert rec.analysis[1]["best"] == "H8"
    assert rec.analysis[1]["moves"][0] == ("J8", 48.5)


def test_save_load_game(tmp_path):
    g = ReverseGomoku()
    for m in [_place(6, 6), _place(0, 0), _place(6, 7), _place(0, 1),
              _place(6, 8), _place(0, 2)]:
        g.make_move(m)
    path = tmp_path / "game.afg"
    result = record.save_game(str(path), g)
    rec = record.load_game(str(path))
    assert rec.moves == g.record()
    assert rec.result == result == "未完"
    assert rec.config == g.config


# ---------------------------------------------------------------- 实时计算量
def test_mcts_progress_counter():
    from antifive.mcts import MCTS
    mcts = MCTS(None, None, sims=16)          # 无网络:均匀先验随机搜索
    move = mcts.search(ReverseGomoku())
    assert move is not None
    assert mcts.progress == 16
    assert mcts.progress_total == 16


def test_heuristic_progress_callback():
    from antifive.heuristic import choose_heuristic_move
    g = ReverseGomoku()
    g.make_move(112)                     # 走两手后避开开局库(首手直接查表不搜索)
    g.make_move(113)
    seen: list = []
    move = choose_heuristic_move(
        g.board, g.current_player, g.pending, g.turn_count, g.white_turns,
        g.config, np.random.default_rng(0), depth=2, time_budget=5.0,
        progress_cb=seen.append)
    assert move is not None
    assert seen and seen[-1] > 0
