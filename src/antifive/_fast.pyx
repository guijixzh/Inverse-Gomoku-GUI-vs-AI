# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""热路径 kernel 的 Cython 实现(带纯 Python 回退)。

设计原则:**整数运算逐位等价**。每个函数都是原 numpy/Python 版本的直接翻译,
只在扩展存在时替换;不存在时调用方走原实现,行为不变。
"""
import numpy as np
cimport numpy as np
from libc.stdint cimport int8_t, int32_t, uint8_t

ctypedef fused anyint:
    np.int8_t
    np.int32_t
    np.uint8_t

cdef int DIRS[8][2]
# 与 reversegomoku.DIRECTIONS 完全一致:N, S, W, E, NW, NE, SW, SE
DIRS[0][0] = -1; DIRS[0][1] = 0
DIRS[1][0] = 1;  DIRS[1][1] = 0
DIRS[2][0] = 0;  DIRS[2][1] = -1
DIRS[3][0] = 0;  DIRS[3][1] = 1
DIRS[4][0] = -1; DIRS[4][1] = -1
DIRS[5][0] = -1; DIRS[5][1] = 1
DIRS[6][0] = 1;  DIRS[6][1] = -1
DIRS[7][0] = 1;  DIRS[7][1] = 1


def shift(anyint[:, ::1] b, int dr, int dc):
    """out[r,c] = b[r-dr, c-dc],越界补 0(等价 heuristic_v6._shift)。"""
    cdef Py_ssize_t h = b.shape[0], w = b.shape[1]
    cdef Py_ssize_t r, c, rr, cc
    cdef anyint[:, ::1] out = np.zeros((h, w), dtype=np.asarray(b).dtype)
    for r in range(h):
        rr = r - dr
        if rr < 0 or rr >= h:
            continue
        for c in range(w):
            cc = c - dc
            if 0 <= cc < w:
                out[r, c] = b[rr, cc]
    return np.asarray(out)


cdef void _runs(int8_t[:, ::1] b, int color, int[::1] runs):
    """runs[d*225 + r*15 + c] = 从 (r,c) 出发沿方向 d 的连续同色数(含自身)。"""
    cdef int d, r, c, k, cnt, rr, cc
    for d in range(8):
        for r in range(15):
            for c in range(15):
                if b[r, c] != color:
                    runs[d * 225 + r * 15 + c] = 0
                    continue
                cnt = 0
                k = 0
                while True:
                    rr = r + k * DIRS[d][0]
                    cc = c + k * DIRS[d][1]
                    if rr < 0 or rr > 14 or cc < 0 or cc > 14:
                        break
                    if b[rr, cc] != color:
                        break
                    cnt += 1
                    k += 1
                runs[d * 225 + r * 15 + c] = cnt


def line_maps(int8_t[:, ::1] board):
    """等价 heuristic_v6._line_maps_combo:返回 6 张 uint8(15,15) 数组。"""
    cdef int[::1] runs = np.zeros(8 * 225, dtype=np.int32)
    cdef int color, r, c, d1, d2, k, f, bb, run, lo, l2, rr, cc, cnt
    cdef list outs = []
    cdef uint8_t[:, ::1] lt
    cdef uint8_t[:, ::1] lt2
    cdef uint8_t[:, ::1] nb_b
    cdef uint8_t[:, ::1] nb_w
    lt_np_arr = np.ones((15, 15), dtype=np.uint8)
    lt2_np_arr = np.ones((15, 15), dtype=np.uint8)
    nb_b_np_arr = np.zeros((15, 15), dtype=np.uint8)
    nb_w_np_arr = np.zeros((15, 15), dtype=np.uint8)
    for color in (1, 2):
        lt_np_arr = np.ones((15, 15), dtype=np.uint8)
        lt2_np_arr = np.ones((15, 15), dtype=np.uint8)
        lt = lt_np_arr
        lt2 = lt2_np_arr
        _runs(board, color, runs)
        for r in range(15):
            for c in range(15):
                lo = 1
                l2 = 1
                for k in range(4):
                    d1 = 2 * k
                    d2 = 2 * k + 1
                    f = 0
                    rr = r + DIRS[d1][0]
                    cc = c + DIRS[d1][1]
                    if 0 <= rr < 15 and 0 <= cc < 15:
                        f = runs[d1 * 225 + rr * 15 + cc]
                    bb = 0
                    rr = r - DIRS[d1][0]
                    cc = c - DIRS[d1][1]
                    if 0 <= rr < 15 and 0 <= cc < 15:
                        bb = runs[d2 * 225 + rr * 15 + cc]
                    run = 1 + f + bb
                    if run > 255:
                        run = 255
                    if run >= lo:
                        if lo > l2:
                            l2 = lo
                        lo = run
                    else:
                        if run > l2:
                            l2 = run
                lt[r, c] = lo
                lt2[r, c] = l2
        outs.append((lt_np_arr, lt2_np_arr))
    nb_b = nb_b_np_arr
    nb_w = nb_w_np_arr
    for r in range(15):
        for c in range(15):
            cnt = 0
            for k in range(8):
                rr = r + DIRS[k][0]
                cc = c + DIRS[k][1]
                if 0 <= rr < 15 and 0 <= cc < 15 and board[rr, cc] == 1:
                    cnt += 1
            nb_b[r, c] = cnt
            cnt = 0
            for k in range(8):
                rr = r + DIRS[k][0]
                cc = c + DIRS[k][1]
                if 0 <= rr < 15 and 0 <= cc < 15 and board[rr, cc] == 2:
                    cnt += 1
            nb_w[r, c] = cnt
    return (outs[0][0], outs[1][0], nb_b_np_arr, nb_w_np_arr, outs[0][1], outs[1][1])


def neighbor_counts(int8_t[:, ::1] board):
    """等价 _line_maps_combo 里的 nb 部分 + positional_summary 的 adj 部分。
    返回 (nb_black, nb_white, adj_black, adj_white):
    nb_color[r,c] = 8 邻域内 color 子数(uint8);adj_color = 同色 8 邻域相邻对数(有向和)。"""
    cdef uint8_t[:, ::1] nb_b = np.zeros((15, 15), dtype=np.uint8)
    cdef uint8_t[:, ::1] nb_w = np.zeros((15, 15), dtype=np.uint8)
    cdef int r, c, k, rr, cc, cnt
    cdef long adj_b = 0, adj_w = 0
    for r in range(15):
        for c in range(15):
            cnt = 0
            for k in range(8):
                rr = r + DIRS[k][0]
                cc = c + DIRS[k][1]
                if 0 <= rr < 15 and 0 <= cc < 15 and board[rr, cc] == 1:
                    cnt += 1
            nb_b[r, c] = cnt
            cnt = 0
            for k in range(8):
                rr = r + DIRS[k][0]
                cc = c + DIRS[k][1]
                if 0 <= rr < 15 and 0 <= cc < 15 and board[rr, cc] == 2:
                    cnt += 1
            nb_w[r, c] = cnt
    # 相邻对数:4 个方向(下,右,右下,左下)各统计一次同色对
    cdef int dirs4[4][2]
    dirs4[0][0] = 1; dirs4[0][1] = 0
    dirs4[1][0] = 0; dirs4[1][1] = 1
    dirs4[2][0] = 1; dirs4[2][1] = 1
    dirs4[3][0] = 1; dirs4[3][1] = -1
    for k in range(4):
        for r in range(15):
            for c in range(15):
                rr = r + dirs4[k][0]
                cc = c + dirs4[k][1]
                if rr < 15 and 0 <= cc < 15:
                    if board[r, c] == 1 and board[rr, cc] == 1:
                        adj_b += 1
                    elif board[r, c] == 2 and board[rr, cc] == 2:
                        adj_w += 1
    return np.asarray(nb_b), np.asarray(nb_w), int(adj_b), int(adj_w)


cdef inline bint _reachable(int flat[225], int T, int color, object run, list rays_T):
    """等价 _end_reachable_list:存在 color 棋子(不在 run 内)射线直通 T。"""
    cdef int d, k, cell, v
    for d in range(8):
        rays_d = rays_T[d]
        for cell in rays_d:
            if cell < 0:
                break
            v = flat[cell]
            if v == 0:
                continue
            if v == color and cell not in run:
                return True
            break
    return False


cdef inline void _tally_four(int flat[225], list cells, dict m, list fours,
                             set dang, int color, int s, int e, bint l_open,
                             bint r_open, int gap, bint killer_can_capture,
                             list rays_list):
    end_Ts = []
    run_cells = set()
    cdef int i, T
    if gap >= 0:
        end_Ts.append(cells[gap])
    else:
        if l_open:
            end_Ts.append(cells[s - 1])
        if r_open:
            end_Ts.append(cells[e])
    if gap < 0:
        for i in range(s, e):
            run_cells.add(cells[i])
    else:
        for i in range(s, gap):
            run_cells.add(cells[i])
        for i in range(gap + 1, e):
            run_cells.add(cells[i])
    killable = False
    if killer_can_capture:
        for T in end_Ts:
            if _reachable(flat, T, color, run_cells, rays_list[T]):
                killable = True
                break
    if killable:
        m["killable4"] = m["killable4"] + 1
    else:
        m["dead4"] = m["dead4"] + 1
    fours.append((run_cells, end_Ts))
    for T in end_Ts:
        dang.add(T)


def board_features(int8_t[:, ::1] board, int white_turns, int turn_count,
                   int loss_start, int white_restrict, list lines4,
                   list rays_list):
    """等价 heuristic_v6._board_features(逐位一致,含 Python 容器的构造)。"""
    cdef int flat[225]
    cdef int r, c, i, j, k, n, s, e, L, total
    cdef int j2, l_open, r_open, v, is_black
    for r in range(15):
        for c in range(15):
            flat[r * 15 + c] = board[r, c]
    m_black = {"killable4": 0, "dead4": 0, "open3": 0, "jump3": 0, "closed3": 0}
    m_white = {"killable4": 0, "dead4": 0, "open3": 0, "jump3": 0, "closed3": 0}
    fours_black = []
    fours_white = []
    dang_black = set()
    dang_white = set()
    cdef bint cap_black = white_turns >= white_restrict
    cdef bint cap_white = True
    cdef bint cap
    m = None
    fours = None
    dang = None
    cells = None
    lines = None
    line_cells = None
    for lines in lines4:
        for line_cells in lines:
            n = len(line_cells)
            i = 0
            while i < n:
                j = i + 1
                v = flat[line_cells[i]]
                while j < n and flat[line_cells[j]] == v:
                    j += 1
                s = i
                e = j
                i = j
                if v == 0:
                    continue
                L = e - s
                is_black = 1 if v == 1 else 0
                if is_black:
                    m = m_black
                    fours = fours_black
                    dang = dang_black
                    cap = cap_black
                else:
                    m = m_white
                    fours = fours_white
                    dang = dang_white
                    cap = cap_white
                l_open = 1 if (s > 0 and flat[line_cells[s - 1]] == 0) else 0
                r_open = 1 if (e < n and flat[line_cells[e]] == 0) else 0
                if L == 4:
                    _tally_four(flat, line_cells, m, fours, dang, v, s, e,
                                l_open, r_open, -1, cap, rays_list)
                elif L == 3:
                    if l_open and r_open:
                        m["open3"] = m["open3"] + 1
                    else:
                        m["closed3"] = m["closed3"] + 1
                if e + 2 < n and flat[line_cells[e]] == 0 and flat[line_cells[e + 1]] == v:
                    j2 = e + 1
                    while j2 < n and flat[line_cells[j2]] == v:
                        j2 += 1
                    total = L + (j2 - e - 1)
                    if total >= 4:
                        _tally_four(flat, line_cells, m, fours, dang, v, s, j2,
                                    l_open, 1 if (j2 < n and flat[line_cells[j2]] == 0) else 0,
                                    e, cap, rays_list)
                    elif total == 3:
                        m["jump3"] = m["jump3"] + 1
    return (m_black, m_white, fours_black, fours_white, dang_black, dang_white)
