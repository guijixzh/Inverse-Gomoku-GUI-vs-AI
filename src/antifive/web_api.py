"""网页版(Pyodide)JSON 接口:规则引擎与启发式 AI 的无 GUI 封装。

Web Worker 中的 JS 侧只做两件事:

    mod = pyodide.pyimport("antifive.web_api")
    mod.set_progress_cb(fn)                  # 可选:AI 搜索节点数回调(每 2048 节点)
    resp = mod.handle(request_json)          # 同步调用,返回 JSON 字符串

请求/响应均为 JSON 字符串,所有状态保存在模块级单例(一个 worker 一局会话)。
命令一览:

- {"cmd": "new", "config": {...}}                      新局
- {"cmd": "state"}                                     当前状态
- {"cmd": "move", "idx": n, "allow_suicide": bool}     走一步(落子/占领/安置)
- {"cmd": "undo"}                                      悔一步
- {"cmd": "resign", "player": 1|2 可选}                人类认输(默认当前行棋方)
- {"cmd": "ai", "depth": 2|4|8, "budget": sec,
   "engine": "beginner"|"mid"|"advanced", "vcf": bool}
                                                       AI 走一步(默认高级+VCF;
                                                       兼容旧值 old/new)
- {"cmd": "rays", "idx": n}                            占领预览:可安置格掩码
- {"cmd": "hint", "depth": 2, "budget": 2.0,
   "engine": "beginner"|"mid"|"advanced", "vcf": bool} 提示:候选着法与价值
- {"cmd": "export", "moves": [..]}                     导出 .afg 文本
- {"cmd": "import", "text": "..."}                     读取 .afg 文本
- {"cmd": "goto", "moves": [..], "pos": k, "result": s}
                                                       复盘跳转:按变化线重建到第 k 手
                                                       (result 为原谱结果,末手补标认输)
- {"cmd": "stuck"}                                     无子可走时替胜方走出绝杀并终局

响应统一为 {"ok": true, ...} 或 {"ok": false, "error": "..."}。
状态字段见 _state()。
"""

from __future__ import annotations

import json
import time
from typing import Optional

import numpy as np

from . import heuristic_versions as hv
from . import record as record_mod
from .heuristic import DEFAULT_DEPTH, _TT
from .reversegomoku import (
    BLACK, BOARD_SIZE, WHITE, GameConfig, ReverseGomoku, danger_map_for,
)
from .tactics import kill_captures

_PROGRESS_CB = None


def set_progress_cb(cb) -> None:
    """注册 JS 进度回调 cb(nodes) 或传 None 注销(供 Web Worker 显示计算量)。"""
    global _PROGRESS_CB
    _PROGRESS_CB = cb


def _emit_progress(nodes: int) -> None:
    if _PROGRESS_CB is not None:
        try:
            _PROGRESS_CB(int(nodes))
        except Exception:
            pass


class _Session:
    def __init__(self) -> None:
        self.game = ReverseGomoku()
        self.rng = np.random.default_rng()
        self.last_nodes = 0
        self.last_elapsed = 0.0
        self.tt = _TT()                 # 跨步复用置换表(网页可用 ANTIFIVE_TT_BITS 调小)

    def progress(self, nodes: int) -> None:
        self.last_nodes = int(nodes)
        _emit_progress(nodes)


_SESSION = _Session()


def _config_from(raw) -> GameConfig:
    cfg = GameConfig()
    if isinstance(raw, dict):
        if "white_restrict_turns" in raw:
            cfg.white_restrict_turns = int(raw["white_restrict_turns"])
        if "loss_start_turns" in raw:
            cfg.loss_start_turns = int(raw["loss_start_turns"])
        if "mask_suicide" in raw:
            cfg.mask_suicide = bool(raw["mask_suicide"])
    return cfg


def _step_kinds(g: ReverseGomoku) -> list:
    """逐步动作类型:"place"=普通落子, "capture"=占领, "place2"=安置。
    占领步的判定:其下一步的 pending 等于本步 idx(最后一步则用当前 pending)。"""
    h = g.history
    kinds = []
    for i, entry in enumerate(h):
        if entry[2] >= 0:
            kinds.append("place2")
        elif (i + 1 < len(h) and h[i + 1][2] == entry[0]) or \
                (i == len(h) - 1 and g.pending == entry[0]):
            kinds.append("capture")
        else:
            kinds.append("place")
    return kinds


def _state(sess: _Session) -> dict:
    g = sess.game
    legal = np.nonzero(g.legal_mask())[0]
    nums = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.int32)
    for k, entry in enumerate(g.history, 1):
        nums[int(entry[0])] = k
    me = g.current_player
    opp = WHITE if me == BLACK else BLACK
    winner = 0
    if g.game_over and not g.is_draw:
        winner = WHITE if g.loser == BLACK else BLACK
    return {
        "board": g.board.reshape(-1).tolist(),
        "player": int(me),
        "pending": int(g.pending),
        "pending_color": int(opp) if g.pending >= 0 else 0,
        "turn_count": int(g.turn_count),
        "white_turns": int(g.white_turns),
        "move_count": int(g.move_count),
        "game_over": bool(g.game_over),
        "loser": int(g.loser),
        "is_draw": bool(g.is_draw),
        "resigned": int(getattr(g, "resigned", 0)),
        "winner": int(winner),
        "stuck": bool(legal.size == 0 and np.any(g.board == 0)),
        "result": record_mod.game_result(g),
        "legal": [int(i) for i in legal],
        "moves": [int(x[0]) for x in g.history],
        "kinds": _step_kinds(g),
        "numbers": nums.tolist(),
        "danger_me": danger_map_for(g.board, me).reshape(-1).astype(np.int8).tolist(),
        "danger_opp": danger_map_for(g.board, opp).reshape(-1).astype(np.int8).tolist(),
        "config": {
            "white_restrict_turns": g.config.white_restrict_turns,
            "loss_start_turns": g.config.loss_start_turns,
            "mask_suicide": bool(g.config.mask_suicide),
        },
    }


def _finish_stuck(sess: _Session) -> bool:
    """行棋方无子可走(全部着法均为自杀)时的终局处理,与 GUI _finish_by_kill 一致:
    先替胜方走出一步绝杀(占领+安置),让棋盘出现真实五连;无绝杀则直接判负。"""
    g = sess.game
    mover = g.current_player
    if g.pending >= 0 or g.turn_count < g.config.loss_start_turns:
        g.game_over, g.loser = True, mover
        return True
    winner = WHITE if mover == BLACK else BLACK
    try:
        kills = kill_captures(g.board, winner, g.pending, g.turn_count,
                              g.white_turns, g.config)
    except Exception:
        kills = []
    if kills:
        cap, place = int(kills[0][0]), int(kills[0][1])
        g.make_move(cap)
        if not g.game_over:
            g.make_move(place)
        if g.game_over and not g.is_draw:
            return True
    g.game_over, g.loser = True, mover
    return True


def _cmd_new(sess: _Session, req: dict) -> dict:
    cfg = _config_from(req.get("config"))
    sess.game = ReverseGomoku(cfg)
    sess.last_nodes = 0
    sess.last_elapsed = 0.0
    return {"state": _state(sess)}


def _cmd_move(sess: _Session, req: dict) -> dict:
    if "idx" not in req:
        raise ValueError("缺少 idx")
    if sess.game.game_over:
        raise ValueError("对局已结束")
    idx = int(req["idx"])
    mask = sess.game.legal_mask()
    if not mask[idx]:
        if mask.any():
            raise ValueError(f"非法着法 idx={idx}")
        # 无合法着法:行棋方必败
        _finish_stuck(sess)
        return {"state": _state(sess), "finished": True}
    sess.game.make_move(idx, allow_suicide=bool(req.get("allow_suicide", False)))
    conflict = False
    if not sess.game.game_over and sess.game.is_stuck():
        _finish_stuck(sess)
        conflict = True
    return {"state": _state(sess), "finished": conflict}


def _cmd_undo(sess: _Session, req: dict) -> dict:
    if not sess.game.history:
        raise ValueError("没有可悔的着法")
    sess.game.undo_move()
    return {"state": _state(sess)}


def _cmd_resign(sess: _Session, req: dict) -> dict:
    """人类认输:player 可选(BLACK/WHITE),缺省为当前行棋方;AI 不调用。"""
    g = sess.game
    if g.game_over:
        raise ValueError("对局已结束")
    g.resign(int(req.get("player", 0) or 0))
    return {"state": _state(sess)}


def _cmd_ai(sess: _Session, req: dict) -> dict:
    g = sess.game
    if g.game_over:
        raise ValueError("对局已结束")
    depth = int(req.get("depth", DEFAULT_DEPTH))
    budget_raw = req.get("budget")
    budget: Optional[float] = float(budget_raw) if budget_raw else None
    engine = hv.normalize(req.get("engine"))
    vcf_raw = req.get("vcf")
    use_vcf = None if vcf_raw is None else bool(vcf_raw)
    extra = hv.supports_vcf(engine)          # 初级无 tt/VCF
    sess.last_nodes = 0
    t0 = time.perf_counter()
    move = hv.choose(engine, g.board, g.current_player, g.pending,
                     g.turn_count, g.white_turns, g.config, sess.rng,
                     depth=depth, time_budget=budget,
                     progress_cb=sess.progress,
                     tt=sess.tt if extra else None,
                     use_vcf=use_vcf if extra else None)
    sess.last_elapsed = time.perf_counter() - t0
    if move is None:
        _finish_stuck(sess)
        return {"state": _state(sess), "move": None, "nodes": sess.last_nodes,
                "elapsed": sess.last_elapsed}
    idx = int(move)
    mask = g.legal_mask()
    if not mask[idx]:
        raise ValueError(f"AI 返回非法着法 idx={idx}")
    g.make_move(idx)
    return {"state": _state(sess), "move": idx, "nodes": sess.last_nodes,
            "elapsed": sess.last_elapsed}


def _cmd_rays(sess: _Session, req: dict) -> dict:
    g = sess.game
    idx = int(req.get("idx", -1))
    out = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=bool)
    if 0 <= idx < BOARD_SIZE * BOARD_SIZE and g.pending < 0:
        opp = WHITE if g.current_player == BLACK else BLACK
        if g.board.reshape(-1)[idx] == opp and g.legal_mask()[idx]:
            out = ReverseGomoku.legal_mask_for(
                g.board, g.current_player, idx, g.white_turns, g.config,
                g.turn_count)
    return {"rays": [int(i) for i in np.nonzero(out)[0]]}


def _cmd_hint(sess: _Session, req: dict) -> dict:
    g = sess.game
    if g.game_over:
        raise ValueError("对局已结束")
    depth = int(req.get("depth", 2))
    budget_raw = req.get("budget")
    budget: Optional[float] = float(budget_raw) if budget_raw else None
    engine = hv.normalize(req.get("engine"))
    vcf_raw = req.get("vcf")
    use_vcf = None if vcf_raw is None else bool(vcf_raw)
    extra = hv.supports_vcf(engine)          # 初级无 tt/VCF
    pos_v, moves = hv.analysis(
        engine, g.board, g.current_player, g.pending, g.turn_count,
        g.white_turns, g.config, sess.rng, depth=depth, time_budget=budget,
        k=5, tt=sess.tt if extra else None,
        use_vcf=use_vcf if extra else None)
    return {"pos_v": float(pos_v),
            "moves": [{"idx": int(m), "v": float(v)} for m, v in moves]}


def _cmd_export(sess: _Session, req: dict) -> dict:
    g = sess.game
    if "moves" in req and req["moves"] is not None:
        moves = [int(m) for m in req["moves"]]
        result = req.get("result") or record_mod.game_result(g)
        return {"text": record_mod.dumps(moves, g.config, result)}
    moves = g.record()
    if g.pending >= 0:
        if not g.game_over:
            raise ValueError("移子待定:请先完成安置再保存")
        moves = moves[:-1]
    return {"text": record_mod.dumps(moves, g.config, record_mod.game_result(g))}


def _cmd_import(sess: _Session, req: dict) -> dict:
    text = req.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("棋谱内容为空")
    rec = record_mod.loads(text)
    if not rec.moves:
        raise ValueError("棋谱中没有着法")
    g = ReverseGomoku(rec.config)
    for m in rec.moves:
        mask = g.legal_mask()
        if not mask[int(m)] and mask.any():
            raise ValueError(f"棋谱第 {len(g.history) + 1} 手非法")
        if not mask.any():
            break
        g.make_move(int(m))
    if not g.game_over:                    # 认输局无五连,按原谱结果补标终局
        loser = record_mod.resign_loser(rec.result)
        if loser:
            g.resign(loser)
    sess.game = g
    sess.last_nodes = 0
    sess.last_elapsed = 0.0
    return {"state": _state(sess), "result": rec.result}


def _cmd_state(sess: _Session, req: dict) -> dict:
    return {"state": _state(sess)}


def _cmd_goto(sess: _Session, req: dict) -> dict:
    """复盘跳转:用给定变化线前面 pos 手从零重建局面(与 GUI review_jump 一致)。
    result 可选(原谱结果):跳到末手且为认输局时补标认输终局。"""
    moves = [int(m) for m in req.get("moves", [])]
    pos = int(req.get("pos", len(moves)))
    pos = max(0, min(pos, len(moves)))
    g = ReverseGomoku(sess.game.config)
    for m in moves[:pos]:
        mask = g.legal_mask()
        if not mask.any():
            break
        if not mask[m]:
            raise ValueError(f"变化线第 {len(g.history) + 1} 手非法")
        g.make_move(m)
    if pos == len(moves):
        loser = record_mod.resign_loser(str(req.get("result") or ""))
        if loser and not g.game_over:
            g.resign(loser)                # 认输局末手:按原谱结果恢复终局
    sess.game = g
    sess.last_nodes = 0
    sess.last_elapsed = 0.0
    return {"state": _state(sess), "pos": pos}


_COMMANDS = {
    "new": _cmd_new,
    "move": _cmd_move,
    "undo": _cmd_undo,
    "resign": _cmd_resign,
    "ai": _cmd_ai,
    "rays": _cmd_rays,
    "hint": _cmd_hint,
    "export": _cmd_export,
    "import": _cmd_import,
    "goto": _cmd_goto,
    "state": _cmd_state,
}


def handle(request_json: str) -> str:
    """JSON 请求 → JSON 响应(同步)。异常转为 {"ok": false, "error": ...}。"""
    try:
        req = json.loads(request_json or "{}")
        cmd = req.get("cmd")
        if cmd == "stuck":
            _finish_stuck(_SESSION)
            resp = {"state": _state(_SESSION)}
        elif cmd in _COMMANDS:
            resp = _COMMANDS[cmd](_SESSION, req)
        else:
            return json.dumps({"ok": False, "error": f"未知命令: {cmd!r}"},
                              ensure_ascii=False)
        resp["ok"] = True
        return json.dumps(resp, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001 - 边界处统一转为错误响应
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"},
                          ensure_ascii=False)
