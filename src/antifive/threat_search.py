"""独立强制杀链搜索(威胁空间搜索,VCF 类)

在当前行棋方常规迭代加深之外,专门证明"存在一串强制着法,无论对手如何
防守,都将在有限步内完成对对手的连五判定(对手判负)"。只覆盖行棋方的
强制胜;找不到或超预算返回 None,由常规搜索接管。

核心引理(见 kill_pairs 的 route 定义):
- 杀对 (c, T, route):占领对手桥子 c,再经无阻射线把该子安置到完成格 T,
  使对手连五判负;route 为 c 与 T 之间严格中间的空格集合。
- 防守完备性:对手单回合只会新增/替换一颗石头;能同时封死全部当前杀对的
  着法,当且仅当其最终落点属于所有"封堵集"的交集,封堵集 = route ∪ {T}
  (占 route 挡安置路径;占 T 则安置目标非空——T 可用占领+安置放己方子,
  普通落子会自杀)。落在交集外⇒至少一个杀对存活⇒进攻方下一回合必杀。
  故防守枚举只需考虑"落在交集内"的着法。
- 双杀快速判定:交集为空(含贴身杀 route 为空)⇒对手无法封堵⇒强制胜。
- 威胁来源:无持子时只有"占领+安置"完整回合能创造对手可杀四(普通落子
  既不能给对手加子,也不能清出线路)。

任何"必胜"结论都基于完全枚举证明;超时/超节点一律返回未知(不主张胜利)。
开关与预算环境变量:ANTIFIVE_VCF=0 关闭;ANTIFIVE_VCF_DEPTH(默认 6)、
ANTIFIVE_VCF_BUDGET 秒(默认 0.05;等时对拍显示 0.3s 过贵,0.05s 净增益)、
ANTIFIVE_VCF_NODES(默认 20000)。
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict

import numpy as np

from .reversegomoku import (
    BLACK, BOARD_SIZE, EMPTY, MOVE_DIRS, MOVE_DISTS, PLACE_SIZE, RAYS, WHITE,
    ReverseGomoku,
)
from .tactics import jump_through, kill_captures, kill_pairs, kill_placements


def _env_flag(name: str, default: bool = True) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


USE_VCF = _env_flag("ANTIFIVE_VCF", True)
VCF_MAX_DEPTH = _env_int("ANTIFIVE_VCF_DEPTH", 6)
VCF_TIME_CAP = _env_float("ANTIFIVE_VCF_BUDGET", 0.05)
VCF_NODE_CAP = _env_int("ANTIFIVE_VCF_NODES", 20000)
MEMO_CAP = 50000
KILL_CACHE_CAP = 4096


def _other(color: int) -> int:
    return WHITE if color == BLACK else BLACK


class _Ctx:
    """一次威胁搜索的预算与缓存。"""

    __slots__ = ("deadline", "nodes", "node_cap", "aborted",
                 "memo_a", "memo_d", "kill_cache", "cb")

    def __init__(self, deadline: float | None, node_cap: int, cb=None):
        self.deadline = deadline
        self.nodes = 0
        self.node_cap = node_cap
        self.aborted = False
        self.memo_a: dict = {}          # attacker_wins_in 记忆
        self.memo_d: dict = {}          # defender_survives 记忆
        self.kill_cache: "OrderedDict[tuple, list]" = OrderedDict()
        self.cb = cb


def _tick(ctx: _Ctx) -> None:
    ctx.nodes += 1
    if ctx.cb is not None and (ctx.nodes & 2047) == 0:
        ctx.cb(ctx.nodes)
    if ctx.node_cap and ctx.nodes >= ctx.node_cap:
        ctx.aborted = True                       # 节点上限每步检查
    elif (ctx.nodes & 31) == 0 and ctx.deadline is not None \
            and time.perf_counter() > ctx.deadline:
        ctx.aborted = True                       # 时限每 32 节点检查一次


def has_kill(board, player: int, pending: int, turn_count: int,
             white_turns: int, config) -> bool:
    """行棋方当前是否有一手可执行的杀(安置杀/两步杀)。"""
    if turn_count < config.loss_start_turns:
        return False
    if pending >= 0:
        return kill_placements(board, player, pending, turn_count,
                               white_turns, config).size > 0
    return bool(kill_captures(board, player, pending, turn_count,
                              white_turns, config))


def _kill_pairs_cached(ctx: _Ctx, board, attacker: int, tc: int, wt: int,
                       config) -> list:
    """kill_pairs 的 LRU 缓存(键含棋盘与回合计数)。"""
    key = (board.tobytes(), tc, wt)
    v = ctx.kill_cache.get(key)
    if v is not None:
        ctx.kill_cache.move_to_end(key)
        return v
    pairs = kill_pairs(board, attacker, -1, tc, wt, config)
    if len(ctx.kill_cache) >= KILL_CACHE_CAP:
        ctx.kill_cache.popitem(last=False)
    ctx.kill_cache[key] = pairs
    return pairs


def _attacker_threat_moves(ctx: _Ctx, board, attacker: int, tc: int, wt: int,
                           config):
    """产出进攻方(无持子)的强制威胁着法完整回合。

    每项:(占领 idx, 回合结束棋盘, tc2, wt2, immediate_win)。
    仅保留"安置后对手存在可杀四"的着法;immediate_win 表示本回合直接杀。"""
    defender = _other(attacker)
    mask = ReverseGomoku.legal_mask_for(board, attacker, -1, wt, config, tc)
    legal = np.nonzero(mask)[0]
    if not legal.size:
        return
    flat = board.reshape(-1)
    for c in legal:
        ci = int(c)
        if flat[ci] != defender:
            continue
        b1, p1, pend1, tc1, wt1, loser1 = ReverseGomoku.apply_step(
            board, ci, attacker, -1, tc, wt, config)
        if loser1 == attacker:
            continue
        pmask = ReverseGomoku.legal_mask_for(b1, p1, pend1, wt1, config, tc1)
        for p in np.nonzero(pmask)[0]:
            pi = int(p)
            b2, _, _, tc2, wt2, loser2 = ReverseGomoku.apply_step(
                b1, pi, p1, pend1, tc1, wt1, config)
            if loser2 == attacker:
                continue
            if loser2 == defender:
                yield ci, b2, tc2, wt2, True
                continue
            pr, pc = divmod(pi, BOARD_SIZE)
            if jump_through(b2, pr, pc, defender) < 4:
                continue
            if not _kill_pairs_cached(ctx, b2, attacker, tc2, wt2, config):
                continue
            yield ci, b2, tc2, wt2, False


def _defender_reach(board, defender: int, tc: int, wt: int, config) -> dict:
    """防守方全部合法占领 x 的可达安置格映射:t → [x, ...]。"""
    attacker = _other(defender)
    mask = ReverseGomoku.legal_mask_for(board, defender, -1, wt, config, tc)
    legal = np.nonzero(mask)[0]
    flat = board.reshape(-1)
    reach: dict = {}
    for x in legal:
        xi = int(x)
        if flat[xi] != attacker:
            continue
        for d in range(MOVE_DIRS):
            for k in range(MOVE_DISTS):
                cell = int(RAYS[xi, d, k])
                if cell < 0 or flat[cell] != EMPTY:
                    break
                reach.setdefault(cell, []).append(xi)
    return reach


def _defender_replies(ctx: _Ctx, board, defender: int, target: int, reach: dict,
                      tc: int, wt: int, config):
    """防守方"最终落点 = target"的全部完整回合,产出 (棋盘, tc2, wt2, loser)。"""
    attacker = _other(defender)
    mask = ReverseGomoku.legal_mask_for(board, defender, -1, wt, config, tc)
    flat = board.reshape(-1)
    if flat[target] == EMPTY and mask[target]:
        b2, _, _, tc2, wt2, loser2 = ReverseGomoku.apply_step(
            board, target, defender, -1, tc, wt, config)
        yield b2, tc2, wt2, loser2
    for xi in reach.get(target, ()):
        b1, _, pend1, tc1, wt1, loser1 = ReverseGomoku.apply_step(
            board, xi, defender, -1, tc, wt, config)
        if loser1 == defender:
            continue
        if b1.reshape(-1)[target] != EMPTY:
            continue
        pmask = ReverseGomoku.legal_mask_for(b1, defender, pend1, wt1, config, tc1)
        if not pmask[target]:
            continue
        b2, _, _, tc2, wt2, loser2 = ReverseGomoku.apply_step(
            b1, target, defender, pend1, tc1, wt1, config)
        yield b2, tc2, wt2, loser2


def _memo_store(memo: dict, key, value) -> None:
    if len(memo) >= MEMO_CAP:
        memo.clear()
    memo[key] = value


def attacker_wins_in(ctx: _Ctx, board, attacker: int, tc: int, wt: int,
                     config, moves_left: int) -> bool:
    """进攻方(轮其行棋)能否在 moves_left 个完整回合内强制获胜。"""
    _tick(ctx)
    if ctx.aborted:
        return False
    key = (board.tobytes(), tc, wt, moves_left)
    v = ctx.memo_a.get(key)
    if v is not None:
        return v
    if has_kill(board, attacker, -1, tc, wt, config):
        _memo_store(ctx.memo_a, key, True)
        return True
    if moves_left <= 0:
        _memo_store(ctx.memo_a, key, False)
        return False
    defender = _other(attacker)
    for _, b2, tc2, wt2, win in _attacker_threat_moves(ctx, board, attacker,
                                                       tc, wt, config):
        if win or not defender_survives(ctx, b2, defender, tc2, wt2, config,
                                        moves_left - 1):
            _memo_store(ctx.memo_a, key, True)
            return True
        if ctx.aborted:
            return False
    _memo_store(ctx.memo_a, key, False)
    return False


def defender_survives(ctx: _Ctx, board, defender: int, tc: int, wt: int,
                      config, moves_left: int) -> bool:
    """进攻方已形成杀威胁,轮到防守方:其是否存在逃生应手。

    返回 False = 无论怎么防都会在 moves_left 个进攻回合内被杀。"""
    _tick(ctx)
    if ctx.aborted:
        return True                      # 未完成证明:保守认为可逃生
    key = (board.tobytes(), tc, wt, moves_left)
    v = ctx.memo_d.get(key)
    if v is not None:
        return v
    attacker = _other(defender)
    if has_kill(board, defender, -1, tc, wt, config):
        _memo_store(ctx.memo_d, key, True)      # 防守方先杀
        return True
    mask = ReverseGomoku.legal_mask_for(board, defender, -1, wt, config, tc)
    if not mask.any():
        _memo_store(ctx.memo_d, key, False)     # 无子可走,判负
        return False
    pairs = _kill_pairs_cached(ctx, board, attacker, tc, wt, config)
    if not pairs:
        _memo_store(ctx.memo_d, key, True)      # 无威胁(不应发生)
        return True
    inter = set(pairs[0][2])
    inter.add(pairs[0][1])                      # 封堵集 = route ∪ {T}
    for _, t, route in pairs[1:]:
        block = set(route)
        block.add(t)
        inter &= block
        if not inter:
            break
    if not inter:
        _memo_store(ctx.memo_d, key, False)     # 双杀:无法同时封堵
        return False
    reach = _defender_reach(board, defender, tc, wt, config)
    for target in inter:
        for b2, tc2, wt2, loser in _defender_replies(
                ctx, board, defender, int(target), reach, tc, wt, config):
            if loser == defender:
                continue                 # 防守自杀,不成立
            if loser == attacker:
                _memo_store(ctx.memo_d, key, True)
                return True              # 防守方反杀成功
            if np.count_nonzero(b2) == PLACE_SIZE:
                _memo_store(ctx.memo_d, key, True)
                return True              # 满盘和棋,非败
            if not attacker_wins_in(ctx, b2, attacker, tc2, wt2, config,
                                    moves_left):
                if ctx.aborted:
                    return True          # 未完成证明:保守认为可逃生
                _memo_store(ctx.memo_d, key, True)
                return True              # 找到逃生
    _memo_store(ctx.memo_d, key, False)
    return False


def find_forced_kill(board, player: int, pending: int, turn_count: int,
                     white_turns: int, config,
                     max_depth: int | None = None,
                     deadline: float | None = None,
                     node_cap: int | None = None,
                     progress_cb=None) -> int | None:
    """尝试证明行棋方存在强制杀链,返回起始着法 idx;否则 None。

    返回的 idx 语义为一个 step(安置步即安置格;否则为占领格,行棋方随后
    按同一证明继续安置)。deadline/node_cap 超限返回 None(未知,不误判)。"""
    if not USE_VCF or turn_count < config.loss_start_turns:
        return None
    if pending >= 0:
        kp = kill_placements(board, player, pending, turn_count,
                             white_turns, config)
        if kp.size:
            return int(kp[0])
    else:
        kc = kill_captures(board, player, pending, turn_count,
                           white_turns, config)
        if kc:
            return int(kc[0][0])
    depth = VCF_MAX_DEPTH if max_depth is None else max_depth
    if depth <= 0:
        return None
    ctx = _Ctx(deadline, VCF_NODE_CAP if node_cap is None else node_cap,
               cb=progress_cb)
    defender = _other(player)
    if pending >= 0:
        mask = ReverseGomoku.legal_mask_for(board, player, pending, white_turns,
                                            config, turn_count)
        for pi in np.nonzero(mask)[0]:
            p = int(pi)
            b2, _, _, tc2, wt2, loser = ReverseGomoku.apply_step(
                board, p, player, pending, turn_count, white_turns, config)
            if loser == player:
                continue
            if loser == defender:
                return p                 # 安置杀,直接终局
            pr, pc = divmod(p, BOARD_SIZE)
            if jump_through(b2, pr, pc, defender) < 4:
                continue
            pairs = _kill_pairs_cached(ctx, b2, player, tc2, wt2, config)
            if not pairs:
                continue
            if not defender_survives(ctx, b2, defender, tc2, wt2, config,
                                     depth - 1):
                return p
            if ctx.aborted:
                return None
        return None
    for ci, b2, tc2, wt2, win in _attacker_threat_moves(ctx, board, player,
                                                        turn_count, white_turns,
                                                        config):
        if win:
            return ci
        if not defender_survives(ctx, b2, defender, tc2, wt2, config, depth - 1):
            return ci
        if ctx.aborted:
            return None
    return None
