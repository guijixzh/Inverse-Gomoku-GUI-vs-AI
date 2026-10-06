"""启发式小 AI(强化版):极小化搜索 + 全局局面评估 + 选择性候选着法

参考经典非神经网络 AI 原理:
- 评估函数 evaluate():对双方连珠材料(3/4 连)、可杀性(4 连开放端外
  是否有可借杀棋子)、危险格杠杆、受困度全局打分——"己方可杀 4 连"是
  最大的负债(-0.95),"对方可杀 4 连"是最大的资产(+0.9);
- 极小化 + alpha-beta:深度 2 搜索(我方着法 → 对方最佳回应),从机制上
  杜绝"攻彼忘我"——任何走完留下己方可杀 4 连的着法,对方回应即杀,
  值 -WIN 直接淘汰;伪威胁(对方能堵端/反杀)也会被搜索戳穿;
- 选择性候选着法:只生成 杀棋 / 解杀(堵危险点、堵路线) / 构造连珠 /
  前 K 位置着,分支从 ~100 压到 ~12,杀棋优先排序加速 alpha-beta 剪枝;
- 静止评估:叶节点含"持子一步杀"等战术信息。

命令行:
    python -m antifive.heuristic --games 1000 --out records [--depth 1..8] [--workers N]
        --depth 1: 贪心全局评估(快,批量生成用)
        --depth 2: 深度 2 极小化(默认,战术可靠)
        --depth 3+: 更深的迭代加深搜索(单步更慢,对局更强;GUI 中每步默认限时 4 秒)
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

import numpy as np

from .. import record as record_mod
from ..reversegomoku import (
    BLACK, BOARD_SIZE, DIRECTIONS, EMPTY, GameConfig, MOVE_DIRS, MOVE_DISTS,
    RAYS, ReverseGomoku, WHITE, danger_map_for, fwd_run_lengths,
)
from ..tactics import kill_captures, kill_placements, kill_samples

WIN = 1e6                       # 必杀/必败

# ---- 全局评估系数(叶节点) ----
E_KILLABLE4 = 0.9               # 对方可杀 4 连(我方资产)
E_DEAD4 = 0.2                   # 对方死 4 连(可后续造借杀子)
E_OPEN3 = 0.35                  # 对方开放 3 连(可构造为 4)
E_CLOSED3 = 0.1
E_DANGER = 0.02                 # 对方连五危险格(其不能走的格)
E_SELF_KILLABLE4 = 0.95         # 我方可杀 4 连(最大负债)
E_SELF_DEAD4 = 0.15
E_SELF_OPEN3 = 0.2
E_SELF_CLOSED3 = 0.08
E_PENDING_KILL = 0.95           # 持子且一步可杀
E_ADJ = 0.02                    # 同色相邻对数:己方集中=负债,对方集中=材料潜力
E_CENTER_OPP = 0.70             # 对方棋子中心度:被推中间=空间大易连五=我方资产
E_CENTER_SELF = 0.80            # 己方棋子中心度:自己居中同样易连五=负债(零和,落子偏中环)
E_CLUSTER = 0.03                # 己方棋子"居中×聚集"联合负债:被推中聚团=大危险
                                # (分散推中间不罚,聚团推中间重罚——直接反制推中战术)

# ---- 候选排序分(搜索内部) ----
O_KILL = 1e5
O_DEFEND = 9e4
O_ROUTE = 8e4
O_BUILD4 = 150.0
O_BUILD3 = 40.0
O_BREAK4 = -250.0
O_BREAK3 = -80.0
O_OWN4 = -400.0
O_OWN3 = -50.0
O_FIGHT = 2.0
O_PUSH_OPP = 2.0               # 安置:把对方棋子推回对方棋群(对方越集中越易连五)
O_PUSH_CENTER = 5.0            # 安置:把对方棋子往中间/空间大处推(边角受边界限制不易连五)
O_PUSH_SELF = 1.0              # 安置:避免贴己方棋子
O_SPREAD = 1.0                 # 落子:远离己方棋子(打散自己)
O_MIDDLE = 0.0                 # 落子:不奖励中场落子(置 0 停用)。
                                # 棋理:己方棋子尽量在边角(安全,不给自己送连五材料),
                                # 推对方棋子往中间由 O_PUSH_CENTER/O_PUSH_OPP 负责。
O_NOISE = 0.02

ROOT_K = 8                      # 根节点候选数
INNER_K = 6                     # 内部节点候选数
DEFAULT_DEPTH = 2

# 中心度图:0..1,棋盘中心=1,边角=0(空间越大越易连五)
_CENTER = BOARD_SIZE // 2
CENTER_MAP = np.array(
    [[1.0 - max(abs(r - _CENTER), abs(c - _CENTER)) / _CENTER
      for c in range(BOARD_SIZE)] for r in range(BOARD_SIZE)],
    dtype=np.float32)


def _other(color: int) -> int:
    return WHITE if color == BLACK else BLACK


def _shift(board2d: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """out[r,c] = board2d[r-dr, c-dc],越界补 0。"""
    out = np.zeros_like(board2d)
    sr = np.clip(np.arange(BOARD_SIZE) - dr, 0, BOARD_SIZE - 1)
    sc = np.clip(np.arange(BOARD_SIZE) - dc, 0, BOARD_SIZE - 1)
    out[...] = board2d[sr][:, sc]
    if dr == 1:
        out[0, :] = 0
    if dr == -1:
        out[-1, :] = 0
    if dc == 1:
        out[:, 0] = 0
    if dc == -1:
        out[:, -1] = 0
    return out


def _line_maps(board: np.ndarray, color: int) -> tuple:
    """向量化候选排序图:
    line_through[t] :若在 t 落 color 子,经过 t 的最大连续数(空/占格均有效);
    nb_opp[t], nb_same[t]:t 的 8 邻域内对方/己方棋子数。"""
    opp = _other(color)
    runs = fwd_run_lengths(board == color)
    lt = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
    for d1, d2 in ((0, 1), (2, 3), (4, 5), (6, 7)):
        dr, dc = DIRECTIONS[d1]
        f = _shift(runs[d1], -dr, -dc)      # 越过 t 的前向连子
        b = _shift(runs[d2], -dr, -dc)      # 越过 t 的后向连子
        lt = np.maximum(lt, 1 + f + b)
    m_opp = (board == opp).astype(np.int32)
    m_same = (board == color).astype(np.int32)
    nb_opp = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
    nb_same = np.zeros_like(nb_opp)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nb_opp += _shift(m_opp, dr, dc)
            nb_same += _shift(m_same, dr, dc)
    return lt, nb_opp, nb_same


def _axis_run(board: np.ndarray, r: int, c: int, color: int, dr: int, dc: int) -> int:
    """若在 (r,c) 落 color 子,沿 (dr,dc) 单轴连续数。"""
    n = 1
    nr, nc = r + dr, c + dc
    while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == color:
        n += 1
        nr += dr
        nc += dc
    nr, nc = r - dr, c - dc
    while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == color:
        n += 1
        nr -= dr
        nc -= dc
    return n


def _run_at(board: np.ndarray, r: int, c: int, color: int) -> int:
    """若在 (r,c) 落 color 子,经过该格的最大连续数(单格快速版)。"""
    best = 0
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        n = 1
        nr, nc = r + dr, c + dc
        while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == color:
            n += 1
            nr += dr
            nc += dc
        nr, nc = r - dr, c - dc
        while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == color:
            n += 1
            nr -= dr
            nc -= dc
        best = max(best, n)
    return best


def _loses_to_kill(board: np.ndarray, player: int, white_turns: int,
                   turn_count: int, config: GameConfig) -> bool:
    """当前局面下,对方是否有借我方连珠的必杀(我方可杀 4 连存在)。"""
    return (turn_count >= config.loss_start_turns
            and _run_material(board, player, white_turns, turn_count,
                              config)["killable4"] > 0)


def _threats(board: np.ndarray, player: int, white_turns: int,
             turn_count: int, config: GameConfig) -> tuple:
    """己方被杀威胁分析,返回 (ends, route):
    - ends :己方 4 连开放端,对方可占领我方棋子后安置完成我方五连(危险点);
    - route:堵杀棋路线可用的中间格(威胁端 T 与对方借用的我方棋子之间的空格)。
    判负窗口外或对方无法移子时无威胁。"""
    ends, route = set(), set()
    if turn_count < config.loss_start_turns:
        return ends, route
    opp = _other(player)
    if opp == WHITE and white_turns < config.white_restrict_turns:
        return ends, route
    flat = board.reshape(-1)
    danger = danger_map_for(board, player)         # 己方落子即连五的格 = 己方 4 连开放端
    for r, c in zip(*np.nonzero(danger)):
        t = r * BOARD_SIZE + c
        for d in range(MOVE_DIRS):
            dr, dc = DIRECTIONS[d]
            mid = []
            for k in range(MOVE_DISTS):
                cell = int(RAYS[t, d, k])
                if cell < 0:
                    break
                v = flat[cell]
                if v == EMPTY:
                    mid.append(cell)
                    continue
                if v == player:
                    # 首个子即己方子且是 4 连本体(相邻同轴)不算威胁,
                    # 其余情况对方占领该子即可借杀
                    if k > 0 or _axis_run(board, r, c, player, dr, dc) < 4:
                        ends.add(t)
                        route.update(mid)
                break
    return ends, route


def _end_reachable(board: np.ndarray, T: int, color: int, run: set) -> bool:
    """开放端 T 可被借杀:存在 color 的棋子(不在该 4 连上)射线直通 T。"""
    flat = board.reshape(-1)
    for d in range(MOVE_DIRS):
        for k in range(MOVE_DISTS):
            cell = int(RAYS[T, d, k])
            if cell < 0:
                break
            v = flat[cell]
            if v == EMPTY:
                continue
            if v == color and cell not in run:
                return True
            break
    return False


def _run_material(board: np.ndarray, color: int, white_turns: int,
                  turn_count: int, config: GameConfig) -> dict:
    """color 方的连珠材料。返回 {killable4, dead4, open3, closed3, danger}。
    killable4:开放端外有可借杀棋子的 4 连(对方一步可杀);
    danger    :该色落子即连五的格列表(其不能走的格)。判负窗口外全部归零。"""
    m = {"killable4": 0, "dead4": 0, "open3": 0, "closed3": 0, "danger": []}
    if turn_count < config.loss_start_turns:
        return m
    killer = _other(color)
    killer_can_capture = not (killer == WHITE
                              and white_turns < config.white_restrict_turns)
    runs = fwd_run_lengths(board == color)
    flat = board.reshape(-1)
    for d in range(MOVE_DIRS):
        dr, dc = DIRECTIONS[d]
        L = runs[d]
        shifted = np.full_like(L, 0)
        sr = np.clip(np.arange(BOARD_SIZE) - dr, 0, BOARD_SIZE - 1)
        sc = np.clip(np.arange(BOARD_SIZE) - dc, 0, BOARD_SIZE - 1)
        shifted[...] = L[sr][:, sc]
        if dr == 1:
            shifted[0, :] = 0
        if dr == -1:
            shifted[-1, :] = 0
        if dc == 1:
            shifted[:, 0] = 0
        if dc == -1:
            shifted[:, -1] = 0
        starts = np.argwhere((L >= 2) & (L <= 4) & (shifted <= L))
        for r, c in starts:
            ln = int(L[r, c])
            if ln == 2:
                continue                          # 2 连不参与材料统计
            e1r, e1c = r - dr, c - dc
            e2r, e2c = r + ln * dr, c + ln * dc
            open1 = (0 <= e1r < BOARD_SIZE and 0 <= e1c < BOARD_SIZE
                     and flat[e1r * BOARD_SIZE + e1c] == EMPTY)
            open2 = (0 <= e2r < BOARD_SIZE and 0 <= e2c < BOARD_SIZE
                     and flat[e2r * BOARD_SIZE + e2c] == EMPTY)
            if ln == 4:
                run_cells = {(r + k * dr) * BOARD_SIZE + c + k * dc
                             for k in range(4)}
                killable = False
                for T, ok in ((e1r * BOARD_SIZE + e1c, open1),
                              (e2r * BOARD_SIZE + e2c, open2)):
                    if ok:
                        m["danger"].append(T)
                        if killer_can_capture and _end_reachable(board, T, color, run_cells):
                            killable = True
                if killable:
                    m["killable4"] += 1
                else:
                    m["dead4"] += 1
            elif ln == 3:
                if open1 or open2:
                    m["open3"] += 1
                else:
                    m["closed3"] += 1
    return m


def _adj_pairs(board: np.ndarray, color: int) -> int:
    """同色棋子的 8 邻域相邻对数(聚集度)。集中 = 易连五 = 负债/材料。"""
    m = (board == color).astype(np.int32)
    total = 0
    for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
        total += int((m * _shift(m, dr, dc)).sum())
    return total


def _cluster_center(board: np.ndarray, color: int) -> float:
    """"居中×聚集"联合度:每个 color 棋子 8 邻域同色邻居数 × 自身中心度之和。
    分散在中间不计数,聚团在中间计数大——推中战术的直接度量。"""
    m = (board == color).astype(np.int32)
    nb = np.zeros_like(m)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nb += _shift(m, dr, dc)
    return float((m * nb * CENTER_MAP).sum())


def evaluate(board: np.ndarray, player: int, pending: int, turn_count: int,
             white_turns: int, config: GameConfig) -> float:
    """全局局面评估,从 player 视角,值约 ∈ [-3, 3]。
    资产:对方 3/4 连材料与可杀性、对方受困度、对方聚集度;
    负债:己方连珠被借杀风险、己方聚集度(集中=易连五)。"""
    opp = _other(player)
    mine = _run_material(board, player, white_turns, turn_count, config)
    theirs = _run_material(board, opp, white_turns, turn_count, config)
    s = 0.0
    s += E_KILLABLE4 * theirs["killable4"] + E_DEAD4 * theirs["dead4"]
    s += E_OPEN3 * theirs["open3"] + E_CLOSED3 * theirs["closed3"]
    s += E_DANGER * min(len(theirs["danger"]), 20)
    s -= E_SELF_KILLABLE4 * mine["killable4"] + E_SELF_DEAD4 * mine["dead4"]
    s -= E_SELF_OPEN3 * mine["open3"] + E_SELF_CLOSED3 * mine["closed3"]
    s -= E_DANGER * min(len(mine["danger"]), 20)
    # 聚集度:对方棋子越集中越易连五(我方资产),己方越集中越是负债
    s += E_ADJ * (_adj_pairs(board, opp) - _adj_pairs(board, player))
    # 中心度:把对方棋子推中间=他们危险(我方资产);己方分散中场有控制力
    # (己方聚团仍由相邻对惩罚;边角棋子被动——边角受边界限制不易连五)
    # 中心度(零和视角对称):对方棋子被推中间=空间大易连五=我方资产;
    # 己方棋子居中同样易连五=轻微负债。搜索取负传播时不会抵消。
    s += E_CENTER_OPP * ((board == opp).astype(np.float32) * CENTER_MAP).sum()
    s -= E_CENTER_SELF * ((board == player).astype(np.float32) * CENTER_MAP).sum()
    # 己方"居中×聚集"联合负债:分散推中间不罚,聚团推中间重罚(反制推中战术)
    s += E_CLUSTER * (_cluster_center(board, opp) - _cluster_center(board, player))
    empty = int(np.count_nonzero(board == EMPTY))
    their_safe = empty - len(theirs["danger"])
    if their_safe <= 2:
        s += 0.8
    elif their_safe <= 5:
        s += 0.4
    elif their_safe <= 10:
        s += 0.2
    my_safe = empty - len(mine["danger"])
    if my_safe <= 2:
        s -= 0.8
    elif my_safe <= 5:
        s -= 0.4
    if pending >= 0 and turn_count >= config.loss_start_turns \
            and kill_placements(board, player, pending, turn_count,
                                white_turns, config).size:
        s += E_PENDING_KILL
    return s


# ---- 候选静态排序(廉价查表,仅用于排序/剪枝) ----
def _place_targets(board: np.ndarray, c: int) -> list:
    """占领 c 后的可达安置格(基于空位射线,不模拟)。"""
    flat = board.reshape(-1)
    out = []
    for d in range(MOVE_DIRS):
        for k in range(MOVE_DISTS):
            cell = int(RAYS[c, d, k])
            if cell < 0 or flat[cell] != EMPTY:
                break
            out.append(cell)
    return out


def _ord_placement_map(lt_opp: np.ndarray, nb_self: np.ndarray,
                       nb_opp: np.ndarray, t: int) -> float:
    """安置步排序分:构造对方连珠 > 把对方棋子推回对方棋群/往中间推。"""
    line = int(lt_opp.reshape(-1)[t])
    if line >= 5:
        return O_KILL
    if line == 4:
        return O_BUILD4
    if line == 3:
        return O_BUILD3
    # 推挤:对方棋子越集中越易连五,且中间空间大更易连五(边角受边界限制),
    # 故把手中对方棋子放回对方棋群并尽量往中间推,不贴己方棋子
    r, c = divmod(int(t), BOARD_SIZE)
    return (nb_opp.reshape(-1)[t] * O_PUSH_OPP
            + CENTER_MAP[r, c] * O_PUSH_CENTER
            - nb_self.reshape(-1)[t] * O_PUSH_SELF)


def _ord_normal_map(lt_self: np.ndarray, nb_self: np.ndarray,
                    nb_opp: np.ndarray, t: int) -> float:
    """普通落子排序分:阻止自己被连 + 靠近对方 + 打散自己(己方棋子偏边角,
    不主动中场——落子位置不奖励中心度,中心度只在评估层作为负债 E_CENTER_SELF)。"""
    line = int(lt_self.reshape(-1)[t])
    s = O_OWN4 if line >= 4 else (O_OWN3 if line == 3 else 0.0)
    s += nb_opp.reshape(-1)[t] * O_FIGHT - nb_self.reshape(-1)[t] * O_SPREAD
    return s


def _candidates(board: np.ndarray, player: int, pending: int, turn_count: int,
                white_turns: int, config: GameConfig, k: int) -> list:
    """选择性候选着法 [(排序分, move), ...] 降序。杀/解杀/堵路线 + 前 k 构造与位置着。"""
    opp = _other(player)
    mask = ReverseGomoku.legal_mask_for(board, player, pending, white_turns,
                                        config, turn_count)
    legal = np.nonzero(mask)[0]
    if not legal.size:
        return []
    flat = board.reshape(-1)
    lt_self, nb_opp, nb_self = _line_maps(board, player)
    lt_opp = _line_maps(board, opp)[0]
    out = []
    if pending >= 0:
        ends, route = _threats(board, player, white_turns, turn_count, config)
        kills = {int(x) for x in kill_placements(
            board, player, pending, turn_count, white_turns, config)}
        for t in kills:
            out.append((O_KILL, t))
        for t in legal:
            ti = int(t)
            if ti in kills:
                continue
            if ti in ends:
                out.append((O_DEFEND, ti))
            elif ti in route:
                out.append((O_ROUTE, ti))
        rest = [int(t) for t in legal
                if int(t) not in kills and int(t) not in ends and int(t) not in route]
        scored = sorted(
            ((_ord_placement_map(lt_opp, nb_self, nb_opp, t), t) for t in rest),
            reverse=True)
        out.extend(scored[:k])
        return sorted(out, reverse=True)
    ends, route = _threats(board, player, white_turns, turn_count, config)
    kc = kill_captures(board, player, pending, turn_count, white_turns,
                       config, legal)
    kill_cells = {c for c, _ in kc}
    for c in kill_cells:
        out.append((O_KILL, c))
    if ends or route:
        # 解杀占领:占领后可达危险点/路线
        for c in legal:
            ci = int(c)
            if flat[ci] != opp or ci in kill_cells:
                continue
            for t in _place_targets(board, ci):
                if t in ends:
                    out.append((O_DEFEND, ci))
                    break
                if t in route:
                    out.append((O_ROUTE, ci))
                    break
        # 堵路线普通着法
        for t in legal:
            ti = int(t)
            if flat[ti] == EMPTY and ti in route:
                out.append((O_ROUTE - 1, ti))
    scored = []
    for t in legal:
        ti = int(t)
        if ti in kill_cells:
            continue
        if flat[ti] == opp:
            r, c0 = divmod(ti, BOARD_SIZE)
            self_line = int(lt_self[r, c0])
            opp_line = int(lt_opp[r, c0])
            s = O_OWN4 if self_line >= 4 else (O_OWN3 if self_line == 3 else 0.0)
            s += O_BREAK4 if opp_line >= 4 else (O_BREAK3 if opp_line >= 3 else 0.0)
            best = -1e9
            for p in _place_targets(board, ti):
                v = _ord_placement_map(lt_opp, nb_self, nb_opp, p)
                if v > best:
                    best = v
            scored.append((s + best, ti))
        else:
            scored.append((_ord_normal_map(lt_self, nb_self, nb_opp, ti), ti))
    scored.sort(reverse=True)
    out.extend(scored[:k])
    return sorted(out, reverse=True)


def _search(board: np.ndarray, player: int, pending: int, turn_count: int,
            white_turns: int, config: GameConfig, depth: int,
            alpha: float, beta: float, k: int,
            deadline: float | None = None) -> float:
    """极小化搜索(alpha-beta)。返回当前行棋方视角的值;WIN=必杀/必败。
    deadline 非 None 时超时即返回当前近似值(根节点会丢弃超时深度)。"""
    if depth <= 0:
        return evaluate(board, player, pending, turn_count, white_turns, config)
    opp = _other(player)
    cands = _candidates(board, player, pending, turn_count, white_turns,
                        config, k)
    if not cands:
        if np.count_nonzero(board == EMPTY) == 0:
            return 0.0                    # 满盘和棋
        return -WIN                       # 无子可走,行棋方必败
    best = -1e18
    for idx, (_, m) in enumerate(cands):
        if deadline is not None and (idx & 3) == 0 and time.perf_counter() > deadline:
            return best                   # 超时:近似值,根节点会丢弃
        # 预过滤:该步是否延伸了行棋方的连子(只有这种步才可能新造己方可杀 4 连)
        r, c = divmod(int(m), BOARD_SIZE)
        pre = 0 if pending >= 0 else _run_at(board, r, c, player)
        b2, p2, pend2, tc2, wt2, loser = ReverseGomoku.apply_step(
            board, m, player, pending, turn_count, white_turns, config)
        if loser == player:
            continue                      # 自杀着(掩码已剔除,防御)
        if loser == opp:
            return WIN                    # 有杀着:立即胜
        if np.count_nonzero(b2 == EMPTY) == 0:
            v = 0.0
        elif pend2 >= 0:
            v = _search(b2, p2, pend2, tc2, wt2, config, depth - 1,
                        alpha, beta, k, deadline)   # 同一方继续(占领→安置)
        elif pre >= 3 and _loses_to_kill(b2, player, wt2, tc2, config):
            v = -WIN                      # 走完留己方可杀 4 连:对方必杀(攻彼忘我!)
        else:
            v = -_search(b2, p2, pend2, tc2, wt2, config, depth - 1,
                         -beta, -alpha, k, deadline)
        if v > best:
            best = v
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
    return best


def choose_heuristic_move(board, player, pending, turn_count, white_turns,
                          config, rng, depth: int | None = None,
                          time_budget: float | None = None) -> int | None:
    """返回着法 idx;无子可走(所有着法均自杀)返回 None。

    先验:存在一步/两步杀直接走(精确必胜,免搜索);
    否则迭代加深:从深度 1 逐层加深,time_budget(秒)内完成多少算多少,
    超时采用上一完整深度选出的着法。"""
    if depth is None:
        depth = DEFAULT_DEPTH
    opp = _other(player)
    if pending >= 0:
        kp = kill_placements(board, player, pending, turn_count,
                             white_turns, config)
        if kp.size:
            return int(rng.choice(kp))
    else:
        kc = kill_captures(board, player, pending, turn_count, white_turns, config)
        if kc:
            return int(rng.choice([c for c, _ in kc]))
    cands = _candidates(board, player, pending, turn_count, white_turns,
                        config, ROOT_K)
    if not cands:
        return None
    t0 = time.perf_counter()
    deadline = t0 + time_budget if time_budget else None
    best_m, best_v = None, -1e18
    for d in range(1, depth + 1):
        depth_best_m, depth_best_v = None, -1e18
        completed = True
        for _, m in cands:
            r, c = divmod(int(m), BOARD_SIZE)
            pre = 0 if pending >= 0 else _run_at(board, r, c, player)
            b2, p2, pend2, tc2, wt2, loser = ReverseGomoku.apply_step(
                board, m, player, pending, turn_count, white_turns, config)
            if loser == player:
                continue
            if loser == opp:
                v = WIN
            elif np.count_nonzero(b2 == EMPTY) == 0:
                v = 0.0
            elif pend2 >= 0:
                v = _search(b2, p2, pend2, tc2, wt2, config, d - 1,
                            -1e18, 1e18, INNER_K, deadline)
            elif pre >= 3 and _loses_to_kill(b2, player, wt2, tc2, config):
                v = -WIN
            else:
                v = -_search(b2, p2, pend2, tc2, wt2, config, d - 1,
                             -1e18, 1e18, INNER_K, deadline)
            v += rng.uniform(0, O_NOISE)
            if v > depth_best_v:
                depth_best_v, depth_best_m = v, m
            if deadline is not None and time.perf_counter() > deadline:
                completed = False
                break
        if completed and depth_best_m is not None:
            best_m, best_v = depth_best_m, depth_best_v
            if depth_best_v >= WIN - 1 or depth_best_v <= -WIN + 1:
                break                    # 已定胜负,再深无益
        if deadline is not None and time.perf_counter() > deadline:
            break
    if best_m is None:
        return int(cands[0][1])          # 兜底:静态分最高(仅极端超时)
    return best_m


def play_game(config: GameConfig, rng: np.random.Generator, max_steps: int = 450):
    """启发式自对弈一局,返回 (moves, result)。"""
    g = ReverseGomoku(config)
    while not g.game_over and g.move_count < max_steps:
        a = choose_heuristic_move(g.board, g.current_player, g.pending,
                                  g.turn_count, g.white_turns, config, rng)
        if a is None:
            g.game_over = True         # 无子可走,行棋方必败
            g.loser = g.current_player
            break
        g.make_move(a)
    if not g.game_over:
        g.game_over = True             # 超步数兜底,按和棋计
        g.is_draw = True
    return g.record(), record_mod.game_result(g)


def count_kill_positions(moves, config) -> int:
    """复盘统计棋谱中出现的杀棋局面数(判负窗口开启后)。"""
    g = ReverseGomoku(config)
    n = 0
    for m in moves:
        if g.turn_count >= config.loss_start_turns:
            n += len(kill_samples(g.board, g.current_player, g.pending,
                                  g.turn_count, g.white_turns, config))
        g.make_move(m)
    return n


def _play_task(args_task):
    """worker 任务:一局棋 → (moves, result)。"""
    seed, config = args_task
    return play_game(config, np.random.default_rng(seed))


def main():
    p = argparse.ArgumentParser(description="启发式小 AI 批量生成基础棋谱")
    p.add_argument("--games", type=int, default=1000, help="生成局数")
    p.add_argument("--out", default="records", help="输出根目录(按结果分类子目录)")
    p.add_argument("--prefix", default="game")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--white-restrict", type=int, default=2)
    p.add_argument("--loss-start", type=int, default=8)
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                   help=f"搜索深度(1=贪心全局评估,2=深度2(默认),3+ 更深的迭代加深)")
    p.add_argument("--workers", type=int, default=1, help="并行进程数(批量生成提速)")
    p.add_argument("--dry-run", action="store_true", help="只统计,不写文件")
    args = p.parse_args()

    if not (1 <= args.depth <= 8):
        p.error("--depth 仅支持 1-8")
    config = GameConfig(loss_start_turns=args.loss_start,
                        white_restrict_turns=args.white_restrict)
    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)
    stats = {"黑胜": 0, "白胜": 0, "和棋": 0, "其他": 0}
    n_kill_total = n_kill_games = 0
    file_idx = 0

    def handle(moves, result):
        nonlocal n_kill_total, n_kill_games, file_idx
        n_kill = count_kill_positions(moves, config)
        n_kill_total += n_kill
        n_kill_games += (n_kill > 0)
        cat = result if result in stats else "其他"
        stats[cat] += 1
        if args.dry_run:
            return
        file_idx += 1
        subs = [cat]
        if n_kill > 0:
            subs.append("有杀棋")
        for sub in subs:
            d = os.path.join(args.out, sub)
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{args.prefix}_{file_idx:05d}.afg")
            with open(path, "w", encoding="utf-8") as f:
                f.write(record_mod.dumps(moves, config, result))

    if args.workers > 1 and args.games > 1:
        tasks = [(args.seed + i * 1000, config) for i in range(args.games)]
        with mp.Pool(args.workers) as pool:
            for moves, result in pool.imap_unordered(_play_task, tasks, chunksize=4):
                handle(moves, result)
    else:
        rng = np.random.default_rng(args.seed)
        for _ in range(args.games):
            handle(*play_game(config, rng))
    if not args.dry_run:
        with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"生成: 启发式 AI (heuristic.py, 极小化搜索 depth={args.depth} "
                    f"+ 全局评估 + 威胁空间着法)\n"
                    f"规则: white_restrict={config.white_restrict_turns} "
                    f"loss_start={config.loss_start_turns}\n"
                    f"局数: {args.games}\n"
                    f"黑胜: {stats['黑胜']} | 白胜: {stats['白胜']} | "
                    f"和棋: {stats['和棋']} | 其他: {stats['其他']}\n"
                    f"含杀棋局: {n_kill_games} "
                    f"(杀棋局面 {n_kill_total}, 均 {n_kill_total / max(args.games, 1):.1f}/局)\n"
                    f"目录: 黑胜/ 白胜/ 和棋/ 按结果分类; "
                    f"有杀棋/ 为含杀棋局面的副本(训练价值最高)\n")
    print(f"共 {args.games} 局(depth={args.depth}) | 黑胜 {stats['黑胜']} | "
          f"白胜 {stats['白胜']} | 和棋 {stats['和棋']} | 含杀棋局 {n_kill_games} | "
          f"杀棋局面 {n_kill_total}(均 {n_kill_total / max(args.games, 1):.1f}/局)"
          + ("" if args.dry_run else f" → 已分类存入 {args.out}/"))


if __name__ == "__main__":
    main()
