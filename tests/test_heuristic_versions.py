"""启发式引擎版本调度测试:新版(含 VCF 开关)与老版快照的统一入口。"""

from __future__ import annotations

import numpy as np
import pytest

from antifive import heuristic_versions as hv
from antifive.reversegomoku import ReverseGomoku


def _args(g):
    return (g.board, g.current_player, g.pending, g.turn_count,
            g.white_turns, g.config)


def test_normalize_and_labels():
    assert hv.normalize("old") == hv.VERSION_OLD
    assert hv.normalize("new") == hv.VERSION_NEW
    assert hv.normalize(None) == hv.VERSION_NEW
    assert hv.normalize("bogus") == hv.VERSION_NEW
    assert hv.label("old") == "老版"
    assert hv.label("new") == "新版"


@pytest.mark.parametrize("version,use_vcf", [
    ("new", True), ("new", False), ("old", True), ("old", False)])
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
    assert float(pv) == float(pv)   # 非 NaN(老版返回 np.float32)


def test_use_vcf_flag_off_matches_snapshot():
    from antifive.tools import heuristic_v3
    g = ReverseGomoku()
    move = hv.choose("new", *_args(g), np.random.default_rng(7), depth=2,
                     use_vcf=False)
    ref = heuristic_v3.choose_heuristic_move(*_args(g),
                                             np.random.default_rng(7), depth=2)
    assert move == ref


def test_gui_engine_accepts_version_and_vcf():
    pytest.importorskip("pygame")
    from antifive.gui import _HeuristicEngine
    g = ReverseGomoku()
    for version, vcf in (("new", True), ("new", False), ("old", True)):
        eng = _HeuristicEngine(depth=1, time_budget=None,
                               version=version, use_vcf=vcf)
        assert eng.version == version and eng.use_vcf == vcf
        move = eng.choose(g)
        assert move is not None and g.legal_mask()[move]
