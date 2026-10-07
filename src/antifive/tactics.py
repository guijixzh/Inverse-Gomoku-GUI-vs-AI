"""杀棋检测与杀棋监督样本(逆五子棋战术层)

逆五子棋的"杀棋"与普通五子棋相反——己方连五判负,所以杀棋是:
把对方棋子(手中)放到使其连五的位置,或占领对方棋子后安置连五:
- 安置杀(pending ≥ 0):把手中对方棋子放到完成对方连五的格 → 对方判负;
- 两步杀(无 pending):占领对方棋子 → 本回合再安置完成对方连五,必成。

这两类杀棋都是行棋方自己的连续着法(不依赖对方应手),可精确枚举:
杀棋样本以 one-hot 策略 + value=+1(行棋方视角,强制获胜)注入训练,
让网络早期就学会"能杀就杀"的战术,大幅加快前期收敛。

注意:判负窗口(loss_start_turns)内连五不判负,杀棋不存在,调用前先判断。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .reversegomoku import (
    BLACK, BOARD_SIZE, EMPTY, MOVE_DIRS, MOVE_DISTS, RAYS, WHITE,
    ReverseGomoku, _line5_map_for, check_win_at,
)


def _other(color: int) -> int:
    return WHITE if color == BLACK else BLACK


def _onehot(kills, legal: np.ndarray) -> np.ndarray:
    """在升序 legal 中给杀棋动作分配均匀概率。"""
    prob = np.zeros(len(legal), dtype=np.float32)
    pos = np.searchsorted(legal, np.asarray(kills, dtype=np.int64))
    prob[pos] = 1.0 / len(kills)
    return prob


def kill_placements(board, player, pending, turn_count, white_turns,
                    config, legal=None) -> np.ndarray:
    """安置杀:手中对方棋子可一步完成对方连五的放置格(升序数组,可能为空)。

    前提:己方正持对方棋子(pending >= 0)且判负窗口已开启。
    legal: 可选,已算好的合法着法数组(避免重复计算掩码)。"""
    if pending < 0 or turn_count < config.loss_start_turns:
        return np.zeros(0, dtype=np.int64)
    opp = _other(player)
    if legal is None:
        legal = np.nonzero(ReverseGomoku.legal_mask_for(
            board, player, pending, white_turns, config, turn_count))[0]
    l5 = _line5_map_for(board, opp).reshape(-1)
    return legal[l5[legal]]


def jump_through(board, r: int, c: int, color: int) -> int:
    """若在 (r,c) 落 color 子,经过该格的最大"子数"(允许一个空档,跳形感知)。
    覆盖:直四(=4)、填缺(=5)、断四延伸(XX_X→XX_XX, 1+1+2=4)、跳四(XX_X, 2+1+1=4)。"""
    best = 0
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        lo = hi = 0
        nr, nc = r - dr, c - dc
        while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == color:
            lo += 1
            nr -= dr
            nc -= dc
        nr, nc = r + dr, c + dc
        while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == color:
            hi += 1
            nr += dr
            nc += dc
        v = 1 + lo + hi
        # 跳形:连子(或直接相邻)外隔一个空格再接连子,只允许一个空档,取最大延伸侧
        extra = 0
        gr, gc = r - (lo + 1) * dr, c - (lo + 1) * dc
        if 0 <= gr < BOARD_SIZE and 0 <= gc < BOARD_SIZE and board[gr, gc] == EMPTY:
            lo2 = 0
            nr, nc = gr - dr, gc - dc
            while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE \
                    and board[nr, nc] == color:
                lo2 += 1
                nr -= dr
                nc -= dc
            extra = max(extra, lo2)
        gr, gc = r + (hi + 1) * dr, c + (hi + 1) * dc
        if 0 <= gr < BOARD_SIZE and 0 <= gc < BOARD_SIZE and board[gr, gc] == EMPTY:
            hi2 = 0
            nr, nc = gr + dr, gc + dc
            while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE \
                    and board[nr, nc] == color:
                hi2 += 1
                nr += dr
                nc += dc
            extra = max(extra, hi2)
        best = max(best, v + extra)
    return best


def _capture_paths(flat, ci: int):
    """产出占领 ci 后沿空位射线可达的 (目标格 t, route)。

    route = ci 与 t 之间严格中间格(元组):对手要封死该杀,必须有一子落在
    route 的某个格上(route 为空表示贴身杀,无法中途封堵)。"""
    for d in range(MOVE_DIRS):
        path: List[int] = []
        for k in range(MOVE_DISTS):
            cell = int(RAYS[ci, d, k])
            if cell < 0 or flat[cell] != EMPTY:
                break
            yield cell, tuple(path)
            path.append(cell)


def _kill_pairs_impl(board, player, pending, turn_count, white_turns, config,
                     legal=None, first_only: bool = False) -> List[Tuple]:
    """两步杀枚举核心:返回 [(占领 idx, 安置 idx, route), ...]。

    占领+安置都是行棋方自己的着法,必成。first_only=True 时每个占领格只取
    第一个杀点(与旧 kill_captures 语义/顺序完全一致)。
    优化:占领后合法安置 = 从占领格出发的空位射线,无需完整掩码计算;
    判杀在"对方子已离盘"的棋盘上进行(避免占领格参与连五的假杀)。"""
    if pending >= 0 or turn_count < config.loss_start_turns:
        return []
    opp = _other(player)
    if legal is None:
        legal = np.nonzero(ReverseGomoku.legal_mask_for(
            board, player, pending, white_turns, config, turn_count))[0]
    flat = board.reshape(-1)
    # 原棋盘上对方落子即连五的格(向量化一次);占领格 ci 离盘后,
    # 只有"五连依赖 ci"的杀点失效,按轴判断即可。
    l5_orig = _line5_map_for(board, opp).reshape(-1)
    dang = np.nonzero(l5_orig)[0]
    if not dang.size:
        return []                 # 对方无连五点:不存在任何安置杀,直接早退
    axis_info = {}                # t → [(dr, dc, lo, hi)]:在 t 落子即连五的轴及两侧延伸
    for t in dang:
        tr, tc = divmod(int(t), BOARD_SIZE)
        info = []
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            lo, hi = 1, 1
            nr, nc = tr - dr, tc - dc
            while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE \
                    and flat[nr * BOARD_SIZE + nc] == opp:
                lo += 1
                nr -= dr
                nc -= dc
            nr, nc = tr + dr, tc + dc
            while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE \
                    and flat[nr * BOARD_SIZE + nc] == opp:
                hi += 1
                nr += dr
                nc += dc
            if lo + hi - 1 >= 5:
                info.append((dr, dc, lo, hi))
        axis_info[t] = info
    out: List[Tuple] = []
    for c in legal:
        ci = int(c)
        if flat[ci] != opp:
            continue
        cix, ciy = divmod(ci, BOARD_SIZE)
        for t, route in _capture_paths(flat, ci):
            if not l5_orig[t]:
                continue
            # 该杀点是否在占领 ci 后仍成立:存在一条不经过 ci 的五连轴
            tr, tc = divmod(t, BOARD_SIZE)
            ok = False
            for dr, dc, lo, hi in axis_info[t]:
                if (cix - tr) * dc == (ciy - tc) * dr:
                    s = cix - tr if dr else ciy - tc
                    if -(lo - 1) <= s <= hi - 1:
                        continue          # 该轴五连依赖 ci
                ok = True
                break
            if ok:
                out.append((ci, t, route))
                if first_only:
                    break
    return out


def kill_captures(board, player, pending, turn_count, white_turns,
                  config, legal=None) -> List[Tuple[int, int]]:
    """两步杀:占领对方棋子后本回合存在一步安置杀。

    返回 [(占领idx, 安置idx), ...](占领+安置都是行棋方自己的着法,必成)。
    legal: 可选,已算好的合法着法数组(避免重复计算掩码)。"""
    return [(c, t) for c, t, _ in _kill_pairs_impl(
        board, player, pending, turn_count, white_turns, config,
        legal=legal, first_only=True)]


def kill_pairs(board, player, pending, turn_count, white_turns,
               config, legal=None) -> List[Tuple[int, int, Tuple[int, ...]]]:
    """全部两步杀对 (占领 idx, 安置 idx, route):route 为封堵该杀需落子的
    中间格集合(空元组=贴身杀不可封)。供强制杀链搜索使用,不提前 break。"""
    return _kill_pairs_impl(board, player, pending, turn_count, white_turns,
                            config, legal=legal, first_only=False)


@dataclass
class KillSample:
    """杀棋监督样本(value 恒 +1,行棋方视角)。字段与 selfplay.Example 对齐。"""
    state: np.ndarray   # (5,15,15) float32
    legal: np.ndarray   # (K,) int32
    prob: np.ndarray    # (K,) float32 one-hot
    player: int         # ±1(1=黑)


def kill_samples(board, player, pending, turn_count, white_turns,
                 config) -> List[KillSample]:
    """当前局面全部杀棋监督样本(行棋方视角,value 恒 +1)。

    - 无持子:每个两步杀产生 2 个样本——占领步(本局面)与占领后的安置步;
    - 已持子:安置杀样本(本局面)。"""
    if turn_count < config.loss_start_turns:
        return []
    sgn = 1 if player == BLACK else -1
    legal = np.nonzero(ReverseGomoku.legal_mask_for(
        board, player, pending, white_turns, config, turn_count))[0]
    out: List[KillSample] = []
    if pending >= 0:
        kills = kill_placements(board, player, pending, turn_count,
                                white_turns, config)
        if kills.size:
            out.append(KillSample(
                ReverseGomoku.encode_for(board, player, pending, turn_count, config),
                legal, _onehot(kills, legal), sgn))
        return out
    for c, t in kill_captures(board, player, pending, turn_count,
                              white_turns, config):
        out.append(KillSample(
            ReverseGomoku.encode_for(board, player, pending, turn_count, config),
            legal, _onehot([c], legal), sgn))
        b2, p2, pend2, tc2, wt2, _ = ReverseGomoku.apply_step(
            board, c, player, pending, turn_count, white_turns, config)
        pmask = np.nonzero(ReverseGomoku.legal_mask_for(
            b2, p2, pend2, wt2, config, tc2))[0]
        kills2 = kill_placements(b2, p2, pend2, tc2, wt2, config)
        if kills2.size:
            out.append(KillSample(
                ReverseGomoku.encode_for(b2, p2, pend2, tc2, config),
                pmask, _onehot(kills2, pmask), sgn))
    return out
