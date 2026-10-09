"""启发式引擎版本调度测试:初级/中级/高级三档的统一入口。"""

from __future__ import annotations

import numpy as np
import pytest

from antifive import heuristic_versions as hv
from antifive.reversegomoku import ReverseGomoku


def _args(g):
    return (g.board, g.current_player, g.pending, g.turn_count,
            g.white_turns, g.config)


def test_normalize_and_labels():
    assert hv.normalize("beginner") == hv.VERSION_BEGINNER
    assert hv.normalize("mid") == hv.VERSION_MID
    assert hv.normalize("advanced") == hv.VERSION_ADVANCED
    assert hv.normalize("old") == hv.VERSION_BEGINNER     # 兼容旧值
    assert hv.normalize("new") == hv.VERSION_ADVANCED     # 兼容旧值
    assert hv.normalize(None) == hv.VERSION_ADVANCED
    assert hv.normalize("bogus") == hv.VERSION_ADVANCED
    assert hv.label("beginner") == "初级"
    assert hv.label("mid") == "中级"
    assert hv.label("advanced") == "高级"
    assert hv.label("old") == "初级"
    assert hv.supports_vcf("beginner") is False
    assert hv.supports_vcf("mid") is True
    assert hv.supports_vcf("advanced") is True


@pytest.mark.parametrize("version,use_vcf", [
    ("beginner", True), ("mid", True), ("mid", False),
    ("advanced", True), ("advanced", False)])
def test_choose_and_analysis_dispatch(version, use_vcf):
    g = ReverseGomoku()
    rng = np.random.default_rng(0)
    move = hv.choose(version, *_args(g), rng, depth=1, time_budget=1.0,
                     tt=None, use_vcf=use_vcf)
    assert move is not None and g.legal_mask()[move]
    pv, moves = hv.analysis(version, *_args(g), np.random.default_rng(0),
                            depth=1, time_budget=1.0, k=3, tt=None,
                            use_vcf=use_vcf)
    assert moves and all(0 <= m < 225 for m, _ in moves)
    assert float(pv) == float(pv)   # 非 NaN(初级返回 np.float32)


def test_use_vcf_off_skips_vcf_and_matches_analysis(monkeypatch):
    """use_vcf=False 时不调用杀链搜索,且 choose 与 analysis 根评估一致。

    注:不再要求与 v3 快照逐着相等——候选生成/根窗口/占领步符号已按
    KataGo 复盘结论有意修复(见 test_choose_analysis_consistency)。"""
    from antifive import heuristic as h
    from antifive import threat_search

    def _boom(*a, **k):
        raise AssertionError("use_vcf=False 仍调用了 VCF")

    monkeypatch.setattr(threat_search, "find_forced_kill", _boom)
    g = ReverseGomoku()
    for depth in (2, 3):
        move = hv.choose("advanced", *_args(g), np.random.default_rng(7),
                         depth=depth, use_vcf=False)
        _, moves = h.analysis_moves(*_args(g), np.random.default_rng(7),
                                    depth=depth, use_vcf=False)
        assert dict(moves).get(int(move), -1e18) \
            >= moves[0][1] - h.O_NOISE - 1e-9


def test_choose_analysis_consistency():
    """根剪枝不得让 choose 偏离 analysis:选中着法的根价值须在最优值
    的噪声带内(占领/安置/普通着法混用局面,含中盘随机局面)。"""
    from antifive import heuristic as h
    from antifive.tools import heuristic_v4, heuristic_v5
    games = []
    rng0 = np.random.default_rng(20261008)
    for _ in range(3):
        g = ReverseGomoku()
        for _ in range(14):
            if g.game_over:
                break
            legal = np.nonzero(g.legal_mask())[0]
            if not legal.size:
                break
            g.make_move(int(rng0.choice(legal)))
        if not g.game_over:
            games.append(g)
    assert len(games) >= 2
    for g in games:
        for seed in (0, 3):
            _, moves = h.analysis_moves(*_args(g), np.random.default_rng(seed),
                                        depth=3, k=6, use_vcf=False)
            top_v = moves[0][1]
            for rp in (True, False):
                mv = h.choose_heuristic_move(
                    *_args(g), np.random.default_rng(seed), depth=3,
                    use_vcf=False, root_prune=rp)
                val = dict(moves).get(int(mv), -1e18)
                assert val >= top_v - h.O_NOISE - 1e-9, \
                    f"choose(root_prune={rp}) 偏离 analysis: {mv} v={val} top={top_v}"
    # 冻结快照仍可加载(供 match_ai --old-module A/B)
    assert heuristic_v4.choose_heuristic_move is not None
    assert heuristic_v5.choose_heuristic_move is not None


def test_gui_engine_accepts_three_versions():
    pytest.importorskip("pygame")
    from antifive.gui import _HeuristicEngine
    g = ReverseGomoku()
    cases = (("beginner", True, False), ("beginner", False, False),
             ("mid", True, True), ("mid", False, False),
             ("advanced", True, True), ("advanced", False, False))
    for version, vcf, expected_vcf in cases:
        eng = _HeuristicEngine(depth=1, time_budget=None,
                               version=version, use_vcf=vcf)
        assert eng.version == version
        assert eng.use_vcf == expected_vcf     # 初级强制无 VCF
        move = eng.choose(g)
        assert move is not None and g.legal_mask()[move]
