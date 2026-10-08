"""开局库(openbook)回归测试:数据加载、合法性、D4 对称一致性与降级。"""

from __future__ import annotations

import numpy as np

from antifive import openbook
from antifive.reversegomoku import (
    BLACK, PLACE_SIZE, SYM_FLAT, WHITE, GameConfig, ReverseGomoku,
)


def test_book_loaded_and_first_move_legal():
    assert openbook.OPENING_BOOK, "缺少开局库数据模块 openbook_data.py"
    board = np.zeros((15, 15), dtype=np.int8)
    mv = openbook.lookup(board, BLACK, -1)
    assert mv is not None
    mask = ReverseGomoku.legal_mask_for(board, BLACK, -1, 0, GameConfig(), 0)
    assert mask[mv]


def _place(flat, mv, color):
    b = flat.copy()
    assert b[mv] == 0
    b[mv] = color
    return b


def test_book_symmetry_consistency():
    """棋盘经 D4 变换后,查表结果必须给出对称等价的着法(以规范化终局校验;
    对称棋盘上允许稳定子内等价的多个正确着法,故不逐坐标比较)。"""
    keys = [k for k in openbook.OPENING_BOOK if k[1] == WHITE]
    assert keys
    for key_hex, _ in keys:
        flat = np.frombuffer(bytes.fromhex(key_hex), dtype=np.int8).copy()
        ref = openbook.lookup(flat.reshape(15, 15), WHITE, -1)
        assert ref is not None
        target = openbook.canonical(_place(flat, ref, WHITE))[0]
        for t in range(8):
            tb = np.zeros(PLACE_SIZE, dtype=np.int8)
            tb[SYM_FLAT[t]] = flat
            got = openbook.lookup(tb.reshape(15, 15), WHITE, -1)
            assert got is not None, (t, ref)
            assert openbook.canonical(_place(tb, got, WHITE))[0] == target


def test_choose_uses_book_on_first_move():
    from antifive.heuristic import choose_heuristic_move
    g = ReverseGomoku()
    book = openbook.lookup(g.board, g.current_player, g.pending)
    assert book is not None
    mv = choose_heuristic_move(g.board, g.current_player, g.pending,
                               g.turn_count, g.white_turns, g.config,
                               np.random.default_rng(0), depth=2)
    assert mv == book


def test_book_env_switch(monkeypatch):
    from antifive import openbook as ob
    board = np.zeros((15, 15), dtype=np.int8)
    assert ob.lookup(board, BLACK, -1) is not None
    monkeypatch.setattr(ob, "_BOOK_ENABLED", False)
    assert ob.lookup(board, BLACK, -1) is None


def test_book_off_when_busy_or_pending():
    board = np.zeros((15, 15), dtype=np.int8)
    board[0, 0] = 1
    board[0, 1] = 2
    assert openbook.lookup(board, BLACK, -1) is None
    empty = np.zeros((15, 15), dtype=np.int8)
    assert openbook.lookup(empty, BLACK, 7) is None
