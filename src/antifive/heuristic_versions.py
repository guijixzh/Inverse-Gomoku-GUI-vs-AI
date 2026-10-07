"""启发式引擎版本调度(桌面 GUI / 网页共用)

- new:当前强化版 `antifive.heuristic`(根 α+PVS+缓存+强制扩展,可选 VCF);
- old:`antifive.tools.heuristic_v2` 冻结快照(优化前的老算法,无 VCF)。

统一入口,避免 GUI / web_api 各自处理两套函数签名差异。
"""

from __future__ import annotations

VERSION_NEW = "new"
VERSION_OLD = "old"
VERSION_CHOICES = (VERSION_NEW, VERSION_OLD)
VERSION_LABELS = {VERSION_NEW: "新版", VERSION_OLD: "老版"}

_OLD_MODULE = None


def label(version: str) -> str:
    return VERSION_LABELS.get(version, VERSION_LABELS[VERSION_NEW])


def normalize(version: str | None) -> str:
    return version if version in VERSION_CHOICES else VERSION_NEW


def _old():
    global _OLD_MODULE
    if _OLD_MODULE is None:
        from .tools import heuristic_v2
        _OLD_MODULE = heuristic_v2
    return _OLD_MODULE


def choose(version: str, board, player, pending, turn_count, white_turns,
           config, rng, depth=None, time_budget=None, progress_cb=None,
           tt=None, use_vcf: bool | None = None):
    """统一着法选择:按版本分发;新版支持 use_vcf(None=模块默认)。"""
    if normalize(version) == VERSION_OLD:
        return _old().choose_heuristic_move(
            board, player, pending, turn_count, white_turns, config, rng,
            depth=depth, time_budget=time_budget, progress_cb=progress_cb)
    from .heuristic import choose_heuristic_move
    return choose_heuristic_move(
        board, player, pending, turn_count, white_turns, config, rng,
        depth=depth, time_budget=time_budget, progress_cb=progress_cb,
        tt=tt, use_vcf=use_vcf)


def analysis(version: str, board, player, pending, turn_count, white_turns,
             config, rng, depth=None, time_budget=None, k=8, progress_cb=None,
             tt=None, use_vcf: bool | None = None):
    """统一局面分析(老版无 tt/use_vcf 参数,自动降级)。"""
    if normalize(version) == VERSION_OLD:
        return _old().analysis_moves(
            board, player, pending, turn_count, white_turns, config, rng,
            depth=depth, time_budget=time_budget, k=k, progress_cb=progress_cb)
    from .heuristic import analysis_moves
    return analysis_moves(
        board, player, pending, turn_count, white_turns, config, rng,
        depth=depth, time_budget=time_budget, k=k, progress_cb=progress_cb,
        tt=tt, use_vcf=use_vcf)
