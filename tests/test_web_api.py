"""web_api(网页版 JSON 接口)测试(不依赖 torch / pygame)。

覆盖:新局/落子/移子两步/悔棋/AI 应手/辅助射线/提示/导出导入往返/无子可走终局/
人类认输与认输棋谱往返。
"""

from __future__ import annotations

import json
from pathlib import Path

from antifive import web_api
from antifive.reversegomoku import BLACK, BOARD_SIZE, WHITE


def _call(**req):
    resp = json.loads(web_api.handle(json.dumps(req)))
    assert resp["ok"], resp.get("error")
    return resp


def test_new_and_state():
    resp = _call(cmd="new")
    st = resp["state"]
    assert st["board"] == [0] * (BOARD_SIZE * BOARD_SIZE)
    assert st["player"] == BLACK
    assert st["pending"] == -1
    assert len(st["legal"]) == BOARD_SIZE * BOARD_SIZE
    assert st["moves"] == [] and not st["game_over"] and not st["stuck"]
    assert st["danger_me"] == [0] * (BOARD_SIZE * BOARD_SIZE)
    assert st["config"]["white_restrict_turns"] == 2


def test_move_and_undo_roundtrip():
    _call(cmd="new")
    st = _call(cmd="move", idx=112)["state"]
    assert st["board"][112] == BLACK and st["player"] == WHITE
    assert st["move_count"] == 1 and st["turn_count"] == 1
    assert st["numbers"][112] == 1
    st = _call(cmd="move", idx=0)["state"]
    assert st["board"][0] == WHITE and st["white_turns"] == 1
    st = _call(cmd="undo")["state"]
    assert st["board"][0] == 0 and st["player"] == WHITE
    st = _call(cmd="undo")["state"]
    assert st["board"][112] == 0 and st["player"] == BLACK
    assert st["move_count"] == 0


def test_capture_placement_and_rays():
    _call(cmd="new")
    _call(cmd="move", idx=1 * BOARD_SIZE + 1)          # 黑 (1,1)
    _call(cmd="move", idx=0)                           # 白 (0,0)
    rays = _call(cmd="rays", idx=0)                    # 悬停白子:预览可安置格
    assert 0 * BOARD_SIZE + 2 in rays["rays"] and 1 * BOARD_SIZE + 0 in rays["rays"]
    assert 2 * BOARD_SIZE + 2 not in rays["rays"]      # 被黑 (1,1) 阻挡

    st = _call(cmd="move", idx=0)["state"]             # 黑占领 (0,0)
    assert st["pending"] == 0 and st["pending_color"] == WHITE
    assert st["board"][0] == BLACK and st["player"] == BLACK
    assert st["turn_count"] == 2                       # 占领不推进回合
    legal = set(st["legal"])
    assert 0 * BOARD_SIZE + 2 in legal and 1 * BOARD_SIZE + 0 in legal
    st = _call(cmd="move", idx=0 * BOARD_SIZE + 2)["state"]   # 安置白子
    assert st["board"][0 * BOARD_SIZE + 2] == WHITE
    assert st["pending"] == -1 and st["player"] == WHITE
    assert st["turn_count"] == 3 and st["white_turns"] == 1


def test_white_capture_restriction():
    _call(cmd="new")
    _call(cmd="move", idx=0)                           # 黑 (0,0)
    _call(cmd="move", idx=112)                         # 白 1
    _call(cmd="move", idx=1)                           # 黑 (0,1)
    st = _call(cmd="state")["state"]
    assert st["player"] == WHITE and st["white_turns"] == 1
    assert 0 not in st["legal"]                        # 白第 1 回合禁止占领
    _call(cmd="move", idx=113)                         # 白 2
    _call(cmd="move", idx=2)                           # 黑 (0,2)
    st = _call(cmd="state")["state"]
    assert st["white_turns"] == 2 and 0 in st["legal"]  # 限制解除


def test_ai_moves_and_progress():
    _call(cmd="new")
    _call(cmd="move", idx=112)           # 走两手避开开局库(首手直接查表不搜索)
    _call(cmd="move", idx=113)
    seen: list = []
    web_api.set_progress_cb(lambda n: seen.append(int(n)))
    try:
        resp = _call(cmd="ai", depth=2, budget=5.0)
    finally:
        web_api.set_progress_cb(None)
    st = resp["state"]
    assert resp["move"] is not None and st["board"][resp["move"]] != 0
    assert st["move_count"] == 3
    assert resp["nodes"] > 0 and seen


def test_ai_accepts_null_budget():
    """budget=None(不限时)走无时限搜索路径,仍应返回合法着法。"""
    _call(cmd="new")
    _call(cmd="move", idx=112)           # 避开开局库
    _call(cmd="move", idx=113)
    resp = _call(cmd="ai", depth=2, budget=None)
    st = resp["state"]
    assert resp["move"] is not None and st["board"][resp["move"]] != 0
    assert st["move_count"] == 3


def test_hint_candidates():
    _call(cmd="new")
    _call(cmd="move", idx=112)
    resp = _call(cmd="hint", depth=2, budget=5.0)
    assert resp["moves"] and resp["moves"][0]["idx"] in range(BOARD_SIZE * BOARD_SIZE)


def test_ai_engine_version_and_vcf_flags():
    for engine, vcf in (("new", True), ("new", False), ("old", True),
                        ("old", False)):
        _call(cmd="new")
        resp = _call(cmd="ai", depth=1, budget=2.0, engine=engine, vcf=vcf)
        assert resp["move"] is not None
        assert resp["state"]["board"][resp["move"]] != 0
    _call(cmd="new")
    _call(cmd="move", idx=112)
    resp = _call(cmd="hint", depth=1, budget=2.0, engine="old", vcf=False)
    assert resp["moves"]


def test_config_roundtrip_in_state():
    resp = _call(cmd="new", config={"white_restrict_turns": 0,
                                    "loss_start_turns": 0, "mask_suicide": False})
    st = resp["state"]
    assert st["config"] == {"white_restrict_turns": 0, "loss_start_turns": 0,
                            "mask_suicide": False}


def test_export_import_roundtrip():
    _call(cmd="new")
    for idx in (112, 0, 113, 1, 114, 2):
        _call(cmd="move", idx=idx)
    text = _call(cmd="export")["text"]
    assert "[AntiFive 1.0]" in text and "H7" in text
    st = _call(cmd="import", text=text)["state"]
    assert st["moves"] == [112, 0, 113, 1, 114, 2]
    assert st["move_count"] == 6


def test_import_sample_record():
    sample = Path(__file__).resolve().parents[1] / "data" / "samples" / "sample_black_win.afg"
    resp = _call(cmd="import", text=sample.read_text(encoding="utf-8"))
    st = resp["state"]
    assert st["game_over"] and not st["is_draw"]
    assert st["winner"] in (BLACK, WHITE)
    assert st["moves"]


def test_illegal_move_reports_error():
    _call(cmd="new")
    resp = json.loads(web_api.handle(json.dumps({"cmd": "move", "idx": 999})))
    assert not resp["ok"] and "error" in resp
    resp = json.loads(web_api.handle(json.dumps({"cmd": "move", "idx": 0})))
    assert resp["ok"]
    resp = json.loads(web_api.handle(json.dumps({"cmd": "move", "idx": 0})))
    assert not resp["ok"]


def test_stuck_command_declares_loss():
    """构造行棋方所有空位均为自杀着(落子即连五)的局面,验证 stuck 终局处理。"""
    from antifive.reversegomoku import BLACK as B, EMPTY, GameConfig

    _call(cmd="new", config={"loss_start_turns": 0, "mask_suicide": True})
    g = web_api._SESSION.game
    g.board[:] = B
    g.board[0, 0] = EMPTY
    g.board[0, 1] = EMPTY
    g.board[7, 7] = EMPTY
    g.current_player = B
    st = _call(cmd="state")["state"]
    assert st["stuck"]
    st = _call(cmd="stuck")["state"]
    assert st["game_over"] and st["loser"] == B and st["winner"] == WHITE


def test_step_kinds_and_goto():
    _call(cmd="new")
    _call(cmd="move", idx=1 * BOARD_SIZE + 1)          # 落子
    _call(cmd="move", idx=0)                           # 落子
    st = _call(cmd="move", idx=0)["state"]             # 占领
    assert st["kinds"] == ["place", "place", "capture"]
    st = _call(cmd="move", idx=2)["state"]             # 安置
    assert st["kinds"] == ["place", "place", "capture", "place2"]
    moves = st["moves"]
    st = _call(cmd="goto", moves=moves, pos=2)["state"]
    assert st["moves"] == moves[:2] and st["move_count"] == 2
    assert st["board"][0] == WHITE                      # 重建后白子回到 (0,0)
    st = _call(cmd="goto", moves=moves, pos=4)["state"]
    assert st["moves"] == moves


def test_error_does_not_corrupt_state():
    _call(cmd="new")
    _call(cmd="move", idx=10)
    resp = json.loads(web_api.handle(json.dumps({"cmd": "move", "idx": 10})))
    assert not resp["ok"]
    st = _call(cmd="state")["state"]
    assert st["moves"] == [10]


def test_resign_command_and_repeat_errors():
    _call(cmd="new")
    _call(cmd="move", idx=112)
    st = _call(cmd="resign", player=BLACK)["state"]
    assert st["game_over"] and not st["is_draw"]
    assert st["loser"] == BLACK and st["winner"] == WHITE and st["resigned"] == BLACK
    assert st["result"] == "黑认输"
    resp = json.loads(web_api.handle(json.dumps({"cmd": "resign"})))
    assert not resp["ok"] and "error" in resp


def test_resign_record_roundtrip_and_goto():
    _call(cmd="new")
    for idx in (112, 0, 113, 1):
        _call(cmd="move", idx=idx)
    _call(cmd="resign", player=WHITE)             # 白方认输
    text = _call(cmd="export")["text"]
    assert "[Result 白认输]" in text
    imported = _call(cmd="import", text=text)
    st = imported["state"]
    assert imported["result"] == "白认输"
    assert st["game_over"] and st["resigned"] == WHITE and st["result"] == "白认输"
    moves = st["moves"]
    # 中途跳转不补标认输;末手带原谱结果时补标终局
    st = _call(cmd="goto", moves=moves, pos=len(moves) - 1)["state"]
    assert not st["game_over"] and st["resigned"] == 0
    st = _call(cmd="goto", moves=moves, pos=len(moves), result="白认输")["state"]
    assert st["game_over"] and st["resigned"] == WHITE and st["result"] == "白认输"
