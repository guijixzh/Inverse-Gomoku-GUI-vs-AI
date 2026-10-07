"""A/B 用包装模块:调用主启发式但临时关闭 VCF(仅用于同进程对拍)。

match_ai 两侧同进程运行时,直接用 env 无法分侧控制;本模块在每次调用期间
把 threat_search.USE_VCF 置 False 再还原,使对手侧仍保持默认(启用)。
"""

from __future__ import annotations

import antifive.heuristic as _h
import antifive.threat_search as _ts


def choose_heuristic_move(*args, **kwargs):
    old = _ts.USE_VCF
    _ts.USE_VCF = False
    try:
        return _h.choose_heuristic_move(*args, **kwargs)
    finally:
        _ts.USE_VCF = old


def analysis_moves(*args, **kwargs):
    old = _ts.USE_VCF
    _ts.USE_VCF = False
    try:
        return _h.analysis_moves(*args, **kwargs)
    finally:
        _ts.USE_VCF = old
