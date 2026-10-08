"""开局库(可选):首手结论来自 KataGo 20k visits 的离线分析。

只覆盖"己方第一手":黑方空盘首手(0 子局面)与白方对任意黑首手的应手
(1 子局面)。查找时对棋盘做 D4 八重对称规范化,命中后把着法映射回原
坐标;未命中(含子数过多/数据缺失/数据非法)一律返回 None,由常规搜索
照常决策。数据是包内纯数据模块,网页端(Pyodide)同样可用,无额外性能负担。

数据文件由 `tools/openbook_gen.py` 生成,键为 (规范化棋盘字节 hex, 行棋方)。
"""

from __future__ import annotations

import os

import numpy as np

from .reversegomoku import PLACE_SIZE, SYM_FLAT, SYM_INV

try:                                     # 数据缺失时静默降级(如裁剪打包)
    from .openbook_data import OPENING_BOOK
except Exception:                        # pragma: no cover - 防裁剪/损坏
    OPENING_BOOK = {}

BOOK_MAX_STONES = 1                      # 仅首手:0 子(黑先)/ 1 子(白应)
# 开关:ANTIFIVE_NO_BOOK=1 关闭开局库(调试/对拍用)
_BOOK_ENABLED = os.environ.get("ANTIFIVE_NO_BOOK", "").strip().lower() \
    not in ("1", "true", "yes", "on")


def canonical(flat: np.ndarray) -> tuple:
    """棋盘(D4 八重对称)→ (规范化字节, 使用的对称索引 t)。

    约定与生成器一致:transformed[SYM_FLAT[t][a]] = flat[a],取字典序最小。"""
    best, best_t = None, 0
    for t in range(8):
        tb = np.zeros(PLACE_SIZE, dtype=np.int8)
        tb[SYM_FLAT[t]] = flat
        b = tb.tobytes()
        if best is None or b < best:
            best, best_t = b, t
    return best, best_t


def lookup(board: np.ndarray, player: int, pending: int) -> int | None:
    """开局库着法 idx;未命中返回 None。"""
    if not _BOOK_ENABLED or not OPENING_BOOK or pending >= 0:
        return None
    flat = np.asarray(board, dtype=np.int8).reshape(-1)
    if int(np.count_nonzero(flat)) > BOOK_MAX_STONES:
        return None
    key, t_star = canonical(flat)
    mv = OPENING_BOOK.get((key.hex(), int(player)))
    if mv is None:
        return None
    # 规范化坐标 → 原坐标(SYM_FLAT[t] 为置换,逆变换用 SYM_INV)
    return int(SYM_FLAT[SYM_INV[t_star]][int(mv)])
