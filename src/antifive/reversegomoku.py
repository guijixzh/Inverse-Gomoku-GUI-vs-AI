"""逆五子棋游戏引擎(numpy 版,AI 训练加固重构)

规则说明
--------
1. 棋盘 15x15,黑先(常量 BLACK=1, WHITE=2, EMPTY=0)。
2. 每一手(step)二选一:
   - 落子:在空位放己方棋子(一步,本回合结束);
   - 移子:同一方连续两个步,共属一个回合:
     ① 占领(落子):把己方棋子放在对方棋子所在格,对方棋子被拿离棋盘;
     ② 安置:把拿起的对方棋子沿 8 个方向之一直线放到连续空位(路径不得受阻)。
   依据对局规则,移子不是"一步走完的整体动作",而是同一方连续的两个步,
   AI 训练(动作空间/合法掩码/MCTS/策略目标)一律按两步建模。
3. 平衡规则:白棋前 `white_restrict_turns` 个回合不能移子(可配置)。
4. 判负:连成五子的一方判负(逆五子棋,连五者输);
   判负检查从第 9 手起生效(连五最早在第 9 手才可能,此前检查纯属浪费,
   该门限是性能优化而非规则,设为 0 不改变对局结果)。
5. 和棋:棋盘无空位且无人连五。

状态字段
--------
- move_count   :已完成的步数
- turn_count   :已完成的回合数(移子占 2 步但只算 1 回合)
- white_turns  :白棋已完成的回合数(限制规则用)
- pending      :-1 表示无"手中棋子";否则为被拿起的对方棋子原位(平铺索引),
                 此时轮到同一位玩家走"安置"步

动作空间(共 225,纯落子空间)
----------------------------
- 每个动作 = "在某格落子",idx = r*15+c (0..224),语义随状态阶段变化:
  - 普通回合:空位 → 落子;对方棋子位且存在可安置路径 → 占领(移子第一步)
  - 安置回合:沿 pending 射线可达的空位 → 把手中的对方棋子放到该格
  方向/距离信息隐含在目标格坐标中,游戏树与"方向-距离"编码完全等价,
  但动作空间缩小 100 倍以上,网络输出/MCTS 节点/训练数据全部受益。
- 判负检查:占领步查己方连五(判己负),安置步查对方连五(判对方负)。
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

EMPTY, BLACK, WHITE = 0, 1, 2
BOARD_SIZE = 15
WIN_STREAK = 5
# 8 个半方向:(行增量, 列增量) N, S, E, W, NW, NE, SW, SE
DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1),
)
# 4 条轴线对应的两个半方向:(横, 纵, 主对角, 副对角)
AXES_DIRS: Tuple[Tuple[int, int], ...] = ((3, 2), (1, 0), (7, 4), (6, 5))

PLACE_SIZE = BOARD_SIZE * BOARD_SIZE                # 225
MOVE_DIRS = len(DIRECTIONS)                         # 8
MOVE_DISTS = BOARD_SIZE - 1                         # 14
ACTION_SIZE = PLACE_SIZE                            # 225(纯落子空间)

# 判负检查启动时机:连五最早出现在第 9 手(黑第 5 子),之前检查纯属浪费。
# 此阈值是性能优化而非规则——连五在更早的回合数学上不可能发生,设 0 不改变对局结果。
LOSS_CHECK_START_TURNS = 8
# 危险通道启动时机:连四(4 连)最早出现在第 7 手(黑第 4 子落下后),
# 危险图在此之前的局面恒为全零,无需计算。
DANGER_START_TURNS = 7

NO_PENDING = -1


@dataclass
class GameConfig:
    """规则配置(阈值可配置,供不同训练/对局实验使用)"""
    loss_start_turns: int = LOSS_CHECK_START_TURNS   # 判负检查启动回合数(性能优化,非规则:
                                                     # 连五最早在第 9 手才可能,设 0 不影响对局)
    white_restrict_turns: int = 2   # 白棋前 white_restrict_turns 个回合禁止移子(0=不限制)
    mask_suicide: bool = True       # 判负生效后剔除自杀着(落子/占领完成己方连五):
                                    # 此类着法必败,剔除可避免自对弈被失误主导


DEFAULT_CONFIG = GameConfig()


# --------------------------------------------------------------------------
# 静态几何表(与棋盘无关,进程内只构建一次)
# --------------------------------------------------------------------------
def _build_rays() -> np.ndarray:
    """rays[base, dir, k]: 从 base 沿 dir 方向第 k+1 格的平铺索引,越界为 -1。
    形状 (225, 8, 14)。"""
    rays = np.full((PLACE_SIZE, MOVE_DIRS, MOVE_DISTS), -1, dtype=np.int32)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            base = r * BOARD_SIZE + c
            for d, (dr, dc) in enumerate(DIRECTIONS):
                for k in range(1, BOARD_SIZE):
                    nr, nc = r + dr * k, c + dc * k
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        rays[base, d, k - 1] = nr * BOARD_SIZE + nc
                    else:
                        break
    return rays


def _build_lines():
    """按方向把棋盘组织成若干条"线",用于向量化连续子(游程)计算。

    返回:
      line_idx : list[8] of (Ld, 15) int32,线内单元按 +d 方向排序,-1 填充
      line_back: list[8] of (225,) int32,每个格子 -> line*15+pos(必然命中)
    """
    line_idx, line_back = [], []
    for d in range(MOVE_DIRS):
        dr, dc = DIRECTIONS[d]
        cells: List[List[Tuple[int, int]]] = []
        if dr == 0:  # 横线
            col_range = range(BOARD_SIZE) if dc == 1 else range(BOARD_SIZE - 1, -1, -1)
            cells = [[(r, c) for c in col_range] for r in range(BOARD_SIZE)]
        elif dc == 0:  # 竖线
            row_range = range(BOARD_SIZE) if dr == 1 else range(BOARD_SIZE - 1, -1, -1)
            cells = [[(r, c) for r in row_range] for c in range(BOARD_SIZE)]
        elif dr == dc:  # 主对角线方向 (1,1) 或 (-1,-1)
            row_range = range(BOARD_SIZE) if dr == 1 else range(BOARD_SIZE - 1, -1, -1)
            for k in range(-(BOARD_SIZE - 1), BOARD_SIZE):
                line = []
                for r in row_range:
                    c = r - k
                    if 0 <= c < BOARD_SIZE:
                        line.append((r, c))
                cells.append(line)
        else:  # 副对角线方向 (1,-1) 或 (-1,1)
            row_range = range(BOARD_SIZE) if dr == 1 else range(BOARD_SIZE - 1, -1, -1)
            for k in range(0, 2 * BOARD_SIZE - 1):
                line = []
                for r in row_range:
                    c = k - r
                    if 0 <= c < BOARD_SIZE:
                        line.append((r, c))
                cells.append(line)

        width = max(len(x) for x in cells)
        idx = np.full((len(cells), width), -1, dtype=np.int32)
        back = np.full((PLACE_SIZE,), -1, dtype=np.int32)
        for li, line in enumerate(cells):
            for pi, (r, c) in enumerate(line):
                idx[li, pi] = r * BOARD_SIZE + c
                back[r * BOARD_SIZE + c] = li * width + pi
        line_idx.append(idx)
        line_back.append(back)
    return line_idx, line_back


RAYS = _build_rays()
LINE_IDX, LINE_BACK = _build_lines()

# 方向向量 -> 方向索引(移子动作解码用)
DIR_INDEX = {(dr, dc): i for i, (dr, dc) in enumerate(DIRECTIONS)}
# 反方向索引(N↔S, E↔W, NW↔SE, NE↔SW)
OPP_DIR = tuple(DIR_INDEX[(-dr, -dc)] for dr, dc in DIRECTIONS)
# 安置步向量化用:平铺射线索引 (112,) → (方向, 距离)
_D_ARR = np.arange(MOVE_DIRS * MOVE_DISTS) // MOVE_DISTS
_K_ARR = np.arange(MOVE_DIRS * MOVE_DISTS) % MOVE_DISTS
D_ARR, K_ARR = _D_ARR, _K_ARR


def _other(color: int) -> int:
    return WHITE if color == BLACK else BLACK


# --------------------------------------------------------------------------
# 向量化连续子(游程)计算
# --------------------------------------------------------------------------
# LRU 缓存:同一棋盘掩码在一次搜索中被反复重算(合法掩码/危险图/候选/评估),
# 缓存命中避免重复的 8 方向 cumsum 计算。键 = 掩码字节(225B),DFS 局部性好。
# GUI 的对弈引擎(worker)/分析引擎(worker)与主线程 play 会并发访问,必须加锁。
_RUNS_CACHE: "OrderedDict[bytes, np.ndarray]" = OrderedDict()
_RUNS_CACHE_LOCK = threading.Lock()
RUNS_CACHE_MAX = 8192


def fwd_run_lengths(mask: np.ndarray) -> np.ndarray:
    """mask: (15,15) bool → (8,15,15) int32
    runs[d, r, c] = 从 (r,c) 出发沿方向 d 连续为真的格子数(含自身,起点为假则 0)。

    实现:cumsum - cummax 公式天然给出"以 i 结尾的后向游程",对整条线做一次;
    方向 d 的前向游程 ≡ 在反方向线(顺序取反)上的后向游程。"""
    key = np.ascontiguousarray(mask).tobytes()
    with _RUNS_CACHE_LOCK:
        cached = _RUNS_CACHE.get(key)
        if cached is not None:
            _RUNS_CACHE.move_to_end(key)
            return cached[0]
    flat = np.ascontiguousarray(mask, dtype=np.int8).reshape(-1)
    out = np.zeros((MOVE_DIRS, BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
    for d in range(MOVE_DIRS):
        op = OPP_DIR[d]
        li, lb = LINE_IDX[op], LINE_BACK[op]
        lm = np.where(li >= 0, flat[li], 0).astype(np.int32)   # (L,15) 0/1,沿 -d 排序
        s = lm.cumsum(axis=1)
        z = np.where(lm == 0, s, 0)
        cm = np.maximum.accumulate(z, axis=1)
        fwd = s - cm                                          # 反方向线上的后向游程
        out[d] = fwd.reshape(-1)[lb].reshape(BOARD_SIZE, BOARD_SIZE)
    out.flags.writeable = False
    with _RUNS_CACHE_LOCK:
        if len(_RUNS_CACHE) >= RUNS_CACHE_MAX:
            _RUNS_CACHE.popitem(last=False)
        _RUNS_CACHE[key] = (out, None)
    return out


def fwd_run_lengths_padded(mask: np.ndarray) -> np.ndarray:
    """fwd_run_lengths 的 (8,17,17) 零填充版(避免重复 np.pad,pad 开销 ~40μs)。"""
    key = np.ascontiguousarray(mask).tobytes()
    with _RUNS_CACHE_LOCK:
        cached = _RUNS_CACHE.get(key)
        if cached is not None:
            _RUNS_CACHE.move_to_end(key)
            out, padded = cached
            if padded is None:
                padded = np.pad(out, ((0, 0), (1, 1), (1, 1)))
                padded.flags.writeable = False
                _RUNS_CACHE[key] = (out, padded)
            return padded
    out = fwd_run_lengths(mask)
    padded = np.pad(out, ((0, 0), (1, 1), (1, 1)))
    padded.flags.writeable = False
    with _RUNS_CACHE_LOCK:
        if len(_RUNS_CACHE) >= RUNS_CACHE_MAX:
            _RUNS_CACHE.popitem(last=False)
        _RUNS_CACHE[key] = (out, padded)
    return padded


def _line5_map_for(board: np.ndarray, color: int) -> np.ndarray:
    """(15,15) bool:在 (r,c) 落 color 子即连五的格子(不筛空格,含对方棋子位)。
    占领对方棋子格完成己方连五即用此判定(与 danger_map 同一公式,只是不滤空)。"""
    runs = fwd_run_lengths(board == color)
    out = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    runs_pad = fwd_run_lengths_padded(board == color)
    for d1, d2 in AXES_DIRS:
        dr1, dc1 = DIRECTIONS[d1]
        dr2, dc2 = DIRECTIONS[d2]
        s1 = runs_pad[d1, 1 + dr1:1 + dr1 + BOARD_SIZE, 1 + dc1:1 + dc1 + BOARD_SIZE]
        s2 = runs_pad[d2, 1 + dr2:1 + dr2 + BOARD_SIZE, 1 + dc2:1 + dc2 + BOARD_SIZE]
        out |= (s1 + s2 >= WIN_STREAK - 1)
    return out


def danger_map_for(board: np.ndarray, color: int) -> np.ndarray:
    """board: (15,15) int → (15,15) bool:若 color 在空位 (r,c) 落子会连五则 True。"""
    return _line5_map_for(board, color) & (board == EMPTY)


def check_win_at(board: np.ndarray, r: int, c: int, color: int) -> bool:
    """增量式判负:(r,c) 处已落 color 子,检查 4 轴线是否连五。O(1) 常数时间。"""
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        count = 1
        nr, nc = r + dr, c + dc
        while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == color:
            count += 1
            nr += dr
            nc += dc
        nr, nc = r - dr, c - dc
        while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == color:
            count += 1
            nr -= dr
            nc -= dc
        if count >= WIN_STREAK:
            return True
    return False


# --------------------------------------------------------------------------
# 对称性(数据增强)
# --------------------------------------------------------------------------
def _symmetry_maps():
    """8 重棋盘对称变换。SYM_FLAT[t][in_idx] = 输出格子;SYM_DIR[t][d] = 方向置换。"""
    def trans(cell: int, t: int) -> int:
        r, c = divmod(int(cell), BOARD_SIZE)
        if t == 0: nr, nc = r, c
        elif t == 1: nr, nc = c, BOARD_SIZE - 1 - r          # 旋转 90°
        elif t == 2: nr, nc = BOARD_SIZE - 1 - r, BOARD_SIZE - 1 - c
        elif t == 3: nr, nc = BOARD_SIZE - 1 - c, r          # 旋转 270°
        elif t == 4: nr, nc = r, BOARD_SIZE - 1 - c          # 水平镜像
        elif t == 5: nr, nc = BOARD_SIZE - 1 - r, c          # 垂直镜像
        elif t == 6: nr, nc = c, r                           # 主对角线
        else: nr, nc = BOARD_SIZE - 1 - c, BOARD_SIZE - 1 - r  # 副对角线
        return nr * BOARD_SIZE + nc

    sym_flat = np.array([[trans(b, t) for b in range(PLACE_SIZE)] for t in range(8)],
                        dtype=np.int32)
    dir_imgs = (
        ((dr, dc) for dr, dc in DIRECTIONS),                    # t0 恒等
        ((dc, -dr) for dr, dc in DIRECTIONS),                   # t1 旋转90°
        ((-dr, -dc) for dr, dc in DIRECTIONS),                  # t2 180°
        ((-dc, dr) for dr, dc in DIRECTIONS),                   # t3 270°
        ((dr, -dc) for dr, dc in DIRECTIONS),                   # t4 水平镜像
        ((-dr, dc) for dr, dc in DIRECTIONS),                   # t5 垂直镜像
        ((dc, dr) for dr, dc in DIRECTIONS),                    # t6 转置
        ((-dc, -dr) for dr, dc in DIRECTIONS),                  # t7 副对角线
    )
    sym_dir = np.zeros((8, MOVE_DIRS), dtype=np.int32)
    for t, imgs in enumerate(dir_imgs):
        for d, v in enumerate(imgs):
            sym_dir[t, d] = DIR_INDEX[v]
    return sym_flat, sym_dir


SYM_FLAT, SYM_DIR = _symmetry_maps()
# 每个对称的逆(旋转 90/270 互逆,镜像与恒等自逆)
SYM_INV = (0, 3, 2, 1, 4, 5, 6, 7)
_SYM_ACTION_CACHE: List[Optional[np.ndarray]] = [None] * 8


def symmetry_action_map(t: int) -> np.ndarray:
    """动作 idx 在对称 t 下的映射(纯落子空间:即棋盘格置换)。

    约定:棋盘内容经变换 t(旋转/镜像)后,原动作 a 对应新棋盘上的动作
    symmetry_action_map[t][a];相应地,状态应按下式同步变换(像素取逆映射):
        state'[:, SYM_FLAT[SYM_INV[t]]] = state"""
    return SYM_FLAT[t]


# --------------------------------------------------------------------------
# 游戏引擎
# --------------------------------------------------------------------------
class ReverseGomoku:
    """逆五子棋引擎。坐标用 BLACK/WHITE/EMPTY(1/2/0) 表示。"""

    def __init__(self, config: GameConfig = DEFAULT_CONFIG):
        self.config = config
        self.reset()

    # ---- 基础状态 ----
    def reset(self) -> None:
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = BLACK
        self.game_over = False
        self.loser = 0                 # 0=未分胜负(或和棋), BLACK/WHITE = 连五判负方
        self.is_draw = False
        self.move_count = 0            # 步数(移子占 2 步)
        self.turn_count = 0            # 回合数
        self.white_turns = 0           # 白棋已完成的回合数
        self.pending = NO_PENDING      # 手中棋子原位, -1 = 无
        self.history: List[Tuple[int, int, int, int, int]] = []  # (idx, player, pending, turn_count, white_turns)

    @property
    def gameover(self) -> bool:
        return self.game_over

    # ---- 动作空间 ----
    @staticmethod
    def get_action_size() -> int:
        return ACTION_SIZE

    @staticmethod
    def idx_to_move(idx: int) -> Tuple[Tuple[int, int], Optional[Tuple[int, int]]]:
        """idx → ((r,c), None)。纯落子空间:安置步的 idx 即目标格。"""
        return (divmod(idx, BOARD_SIZE), None)

    @staticmethod
    def move_to_idx(move) -> int:
        ((r, c), _target) = move
        return r * BOARD_SIZE + c

    # ---- 规则 ----
    @staticmethod
    def _white_restricted(player: int, white_turns: int, config: GameConfig) -> bool:
        """白棋前 white_restrict_turns 个回合禁止移子(占领步)。"""
        n = config.white_restrict_turns
        return n > 0 and player == WHITE and white_turns < n

    # ---- 合法动作 ----
    @staticmethod
    def legal_mask_for(board: np.ndarray, player: int, pending: int,
                       white_turns: int, config: GameConfig,
                       turn_count: int = 0) -> np.ndarray:
        """(ACTION_SIZE,) bool 合法动作掩码。静态版,任意状态可调。
        turn_count 用于判负窗口:窗口内连五不判负,自杀着不剔除。"""
        mask = np.zeros(ACTION_SIZE, dtype=bool)
        empty_flat = (board == EMPTY).reshape(-1)

        if pending >= 0:
            # 安置步:把手中的对方棋子沿直线放到连续空位(从 pending 起射线)
            run_pad = fwd_run_lengths_padded(board == EMPTY)
            pr, pc = divmod(int(pending), BOARD_SIZE)
            targets = RAYS[int(pending)].reshape(MOVE_DIRS * MOVE_DISTS)  # (112,)
            target_ok = (targets >= 0) & empty_flat[targets]
            run_at = np.empty(MOVE_DIRS, dtype=np.int32)
            for d in range(MOVE_DIRS):
                dr, dc = DIRECTIONS[d]
                run_at[d] = run_pad[d, 1 + dr + pr, 1 + dc + pc]
            run_ok = run_at[D_ARR] >= K_ARR + 1
            mask[targets[target_ok & run_ok]] = True
            return mask

        # 普通回合:落子(空位) + 占领(对方棋子位且存在可安置路径)
        mask[:PLACE_SIZE] = empty_flat
        if not ReverseGomoku._white_restricted(player, white_turns, config):
            opp = _other(player)
            opp_flat = (board.reshape(-1) == opp)
            run_pad = fwd_run_lengths_padded(board == EMPTY)
            has_target = np.zeros(PLACE_SIZE, dtype=bool)
            for d in range(MOVE_DIRS):
                dr, dc = DIRECTIONS[d]
                sh = run_pad[d, 1 + dr:1 + dr + BOARD_SIZE,
                             1 + dc:1 + dc + BOARD_SIZE].reshape(-1)  # 从邻位起连续空位长
                has_target |= (sh >= 1)
            mask[:PLACE_SIZE] |= (opp_flat & has_target)

        # 自杀着剔除(判负生效后):落子/占领完成己方连五必败,一律剔除
        # (占领在上一行已加入掩码,此处统一过筛;白棋限制期同样生效)
        if config.mask_suicide and turn_count >= config.loss_start_turns:
            line5 = _line5_map_for(board, player).reshape(-1)
            mask[:PLACE_SIZE] &= ~line5
        return mask

    def legal_mask(self) -> np.ndarray:
        return self.legal_mask_for(self.board, self.current_player, self.pending,
                                   self.white_turns, self.config, self.turn_count)

    def is_stuck(self) -> bool:
        """当前行棋方是否无子可走(判负窗口后所有着法均为自杀)且未满盘。"""
        return (not self.legal_mask().any()) and np.any(self.board == EMPTY)

    def get_legal_moves(self) -> List[Tuple[Tuple[int, int], Optional[Tuple[int, int]]]]:
        return [self.idx_to_move(i) for i in np.nonzero(self.legal_mask())[0]]

    # ---- 走子(纯函数,供 MCTS 节点使用) ----
    @staticmethod
    def apply_step(board: np.ndarray, idx: int, player: int, pending: int,
                   turn_count: int, white_turns: int,
                   config: GameConfig) -> Tuple[np.ndarray, int, int, int, int, int]:
        """推进一个步,返回 (新棋盘, 新玩家, 新pending, 新回合数, 新白棋回合数, loser)。
        loser ∈ {0, BLACK, WHITE},0=无人连五。idx 必须对当前状态合法。"""
        b = board.copy()
        opp = _other(player)
        loser = 0
        r, c = divmod(int(idx), BOARD_SIZE)
        if pending < 0:
            if b[r, c] == opp:
                # 移子第一步(占领):对方棋子离盘,本回合继续
                b[r, c] = player
                pending2 = int(idx)
                tc2, wt2 = turn_count, white_turns
                if turn_count >= config.loss_start_turns and check_win_at(b, r, c, player):
                    loser = player
            else:
                # 普通落子:本回合结束
                b[r, c] = player
                pending2 = NO_PENDING
                tc2, wt2 = turn_count + 1, white_turns + (1 if player == WHITE else 0)
                if turn_count >= config.loss_start_turns and check_win_at(b, r, c, player):
                    loser = player
            player2 = player if pending2 >= 0 else opp
        else:
            # 移子第二步(安置):手中对方棋子放到 idx 格,本回合结束
            b[r, c] = opp
            pending2 = NO_PENDING
            player2 = opp
            tc2, wt2 = turn_count + 1, white_turns + (1 if player == WHITE else 0)
            if turn_count >= config.loss_start_turns and check_win_at(b, r, c, opp):
                loser = opp
        return b, player2, pending2, tc2, wt2, loser

    def make_move(self, move, allow_suicide: bool = False) -> None:
        """执行一个步(接受 idx 或 ((r,c), target) 形式)。
        allow_suicide=True 时按引擎规则校验(自杀手合法,走出即完成己方五连
        自然判负)——供 GUI 接受 KataGo 引擎着法;人类着法仍走默认掩码。"""
        if not isinstance(move, (int, np.integer)):
            move = self.move_to_idx(move)
        idx = int(move)
        if allow_suicide:
            from dataclasses import replace as _dc_replace
            cfg_open = _dc_replace(self.config, mask_suicide=False)
            mask = self.legal_mask_for(self.board, self.current_player,
                                       self.pending, self.white_turns,
                                       cfg_open, self.turn_count)
        else:
            mask = self.legal_mask()
        if not mask.any() and np.any(self.board == EMPTY):
            # 无子可走(判负窗口后所有着法均为自杀):行棋方必败,立即终局
            self.game_over = True
            self.loser = self.current_player
            return
        assert mask[idx], f"非法动作 idx={idx} (state: player={self.current_player} pending={self.pending})"
        b2, p2, pend2, tc2, wt2, loser = self.apply_step(
            self.board, idx, self.current_player, self.pending,
            self.turn_count, self.white_turns, self.config)
        self.history.append((idx, self.current_player, self.pending,
                             self.turn_count, self.white_turns))
        self.board = b2
        self.current_player = p2
        self.pending = pend2
        self.turn_count = tc2
        self.white_turns = wt2
        if loser:
            self.game_over = True
            self.loser = loser
        elif np.all(self.board != EMPTY):
            self.game_over = True
            self.is_draw = True
            self.loser = 0
        self.move_count += 1

    def undo_move(self) -> bool:
        """回滚一个步。只在非终局状态调用。"""
        if not self.history:
            return False
        idx, player_pre, pend_pre, tc_pre, wt_pre = self.history.pop()
        r, c = divmod(idx, BOARD_SIZE)
        if self.pending == idx:
            # 占领步:还原被拿起的对方棋子
            self.board[r, c] = _other(player_pre)
        else:
            # 普通落子或安置步:清空该格
            self.board[r, c] = EMPTY
        self.current_player = player_pre
        self.pending = pend_pre
        self.turn_count = tc_pre
        self.white_turns = wt_pre
        self.game_over = False
        self.loser = 0
        self.is_draw = False
        self.move_count -= 1
        return True

    def is_suicide_move(self, move) -> bool:
        """执行该步是否会使己方连五判负(真实模拟,精确)。"""
        me = self.current_player
        self.make_move(move)
        suicide = self.game_over and self.loser == me and not self.is_draw
        self.undo_move()
        return suicide

    # ---- 危险平面 ----
    def get_danger_map(self, color: int) -> np.ndarray:
        """(15,15) float32,1.0 = 该色在此落子会连五(向量化)。"""
        return danger_map_for(self.board, color).astype(np.float32)

    def get_quick_action_mask(self) -> dict:
        """动作 → 是否自杀(执行后己方连五判负)。精确模拟,不再依赖危险平面近似。"""
        return {m: self.is_suicide_move(m) for m in self.get_legal_moves()}

    # ---- 神经网络输入 ----
    @staticmethod
    def encode_for(board: np.ndarray, player: int, pending: int,
                   turn_count: int, config: GameConfig) -> np.ndarray:
        """标准 5 通道编码 (5,15,15) float32:
        [己方子, 对方子, 己方危险, 对方危险, 移子待定标记]。
        危险通道从第 DANGER_START_TURNS 手起计算(连四最早在第 7 手出现,
        此前恒为全零,跳过以省算力);待定标记仅在 pending 步为 1。"""
        opp = _other(player)
        state = np.zeros((5, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        state[0] = (board == player)
        state[1] = (board == opp)
        if turn_count >= DANGER_START_TURNS:
            state[2] = danger_map_for(board, player).astype(np.float32)
            state[3] = danger_map_for(board, opp).astype(np.float32)
        if pending >= 0:
            r, c = divmod(int(pending), BOARD_SIZE)
            state[4, r, c] = 1.0
        return state

    def encode_state(self) -> np.ndarray:
        return self.encode_for(self.board, self.current_player, self.pending,
                               self.turn_count, self.config)

    def encode_board(self) -> np.ndarray:  # 旧接口,语义已升级为 5 通道
        return self.encode_state()

    # ---- 对局记录 ----
    def record(self) -> List[int]:
        """本局全部步的 idx 序列(含移子的两步)。"""
        return [x[0] for x in self.history]

    @classmethod
    def replay(cls, moves, config: GameConfig = DEFAULT_CONFIG) -> "ReverseGomoku":
        g = cls(config)
        for i in moves:
            g.make_move(i)
        return g

    # ---- 复制 ----
    def copy(self) -> "ReverseGomoku":
        g = ReverseGomoku(self.config)
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.game_over = self.game_over
        g.loser = self.loser
        g.is_draw = self.is_draw
        g.move_count = self.move_count
        g.turn_count = self.turn_count
        g.white_turns = self.white_turns
        g.pending = self.pending
        g.history = list(self.history)
        return g

    # ---- 辅助 ----
    def print_board(self) -> None:
        symbols = {EMPTY: " .", BLACK: " X", WHITE: " O"}
        print("    " + " ".join(f"{i:2}" for i in range(BOARD_SIZE)))
        for r in range(BOARD_SIZE):
            print(f"{r:2} |" + "".join(symbols[self.board[r, c]] for c in range(BOARD_SIZE)))
