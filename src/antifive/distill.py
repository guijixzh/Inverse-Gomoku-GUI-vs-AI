"""KataGo 蒸馏推理(numpy):候选策略先验 + 局面价值校正。

模型权重由 `tools/train_distill.py` 生成到包内 `distill_data.py`(纯数据模块,
网页/Pyodide 可用)。数据缺失或 `ANTIFIVE_NO_DISTILL=1` 时全部退化为零,
引擎行为与未蒸馏完全一致,便于回退与对拍。

策略先验(D1)对每个候选着法计算线性打分:
    prior = w·标量特征 + Σ_axis w_模式[4 格 base-4 编码]
模式特征覆盖 4 条轴线两侧各 4 格(0 空 / 1 己方 / 2 对方 / 3 界外),
直线与斜线各一张共享权重表(256 项),可表达手写特征之外的局部棋形。
价值校正(D2)用局面的"我方/对方"对称特征做线性回归,预测 KataGo 胜率
(价值尺度,正=行棋方有利),在 evaluate 中按权重混入。
"""

from __future__ import annotations

import os

import numpy as np

from .reversegomoku import BOARD_SIZE, EMPTY, PLACE_SIZE

try:                                     # 模型缺失时静默降级(回退到未蒸馏)
    from .distill_data import (          # type: ignore
        SCALAR_DIM, W_SCALAR, B_PRIOR, W_STRAIGHT, W_DIAG, PRIOR_W,
        W_VALUE, B_VALUE, VALUE_W,
    )
    HAS_PRIOR = True
    HAS_VALUE = True
except Exception:                        # pragma: no cover - 数据缺失/损坏
    SCALAR_DIM = 0
    W_SCALAR = B_PRIOR = W_STRAIGHT = W_DIAG = None
    PRIOR_W = 0.0
    W_VALUE = B_VALUE = None
    VALUE_W = 0.0
    HAS_PRIOR = False
    HAS_VALUE = False

_DISABLED = os.environ.get("ANTIFIVE_NO_DISTILL", "").strip().lower() \
    in ("1", "true", "yes", "on")
ENABLED = (HAS_PRIOR or HAS_VALUE) and not _DISABLED


def _env_float(name: str, default: float) -> float:
    try:
        v = os.environ.get(name)
        return default if v is None or v == "" else float(v)
    except ValueError:
        return default


# 混合权重可用环境变量临时缩放,便于 A/B 选值(不改动权重文件)
PRIOR_W = _env_float("ANTIFIVE_PRIOR_W", PRIOR_W)
VALUE_W = _env_float("ANTIFIVE_VALUE_W", VALUE_W)

_PAD = 4
_STATE_OFF = 3                           # 棋盘外
_AXES = ((0, 1), (1, 0), (1, 1), (1, -1))
_PATTERN_ORDER = 4                       # 每侧 4 格


def pattern_tokens(board: np.ndarray, player: int, moves) -> np.ndarray:
    """(M,8) uint8:每步 4 轴 × 两侧的 base-4 编码(0 空/1 己/2 对方/3 界外)。"""
    flat = np.asarray(board, dtype=np.int8).reshape(-1)
    rel = np.zeros(PLACE_SIZE, dtype=np.int8)
    rel[flat == player] = 1
    rel[(flat != player) & (flat != EMPTY)] = 2
    size = BOARD_SIZE + 2 * _PAD
    pad = np.full((size, size), _STATE_OFF, dtype=np.int8)
    pad[_PAD:_PAD + BOARD_SIZE, _PAD:_PAD + BOARD_SIZE] = \
        rel.reshape(BOARD_SIZE, BOARD_SIZE)
    mv = np.asarray(moves, dtype=np.int64)
    r, c = np.divmod(mv, BOARD_SIZE)
    r = r + _PAD
    c = c + _PAD
    out = np.empty((len(mv), 8), dtype=np.uint8)
    for a, (dr, dc) in enumerate(_AXES):
        cp = np.zeros(len(mv), dtype=np.int16)
        cn = np.zeros(len(mv), dtype=np.int16)
        for i in range(_PATTERN_ORDER):
            k = 4 ** i
            cp += k * pad[r + (i + 1) * dr, c + (i + 1) * dc]
            cn += k * pad[r - (i + 1) * dr, c - (i + 1) * dc]
        out[:, 2 * a] = cp
        out[:, 2 * a + 1] = cn
    return out


def prior_scores(scalars: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """策略先验线性打分(未启用时返回全 0)。"""
    if not ENABLED or not HAS_PRIOR:
        return np.zeros(len(scalars), dtype=np.float64)
    s = np.asarray(scalars, dtype=np.float64) @ W_SCALAR + B_PRIOR
    s = s + W_STRAIGHT[tokens[:, 0:4]].sum(axis=1)
    s = s + W_DIAG[tokens[:, 4:8]].sum(axis=1)
    return s


def value_score(feats) -> float:
    """局面价值校正(未启用时返回 0.0)。feats 顺序见 heuristic._value_features。

    返回已乘混合权重 VALUE_W(训练端在离线回归输出上标定)。"""
    if not ENABLED or not HAS_VALUE:
        return 0.0
    raw = float(np.asarray(feats, dtype=np.float64) @ W_VALUE + B_VALUE)
    return VALUE_W * raw
