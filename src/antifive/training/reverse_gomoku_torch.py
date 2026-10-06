"""逆五子棋批量 GPU 引擎(torch 版,与 numpy 版同规则同动作空间)

与 reversegomoku.py 完全同构,区别:
- 棋盘编码为 ±1(N 局批量):1=黑, -1=白, 0=空;
- 全部操作为张量批处理,供批量自对弈(MCTS 叶子批量评估)使用;
- 动作空间为 225 纯落子空间(语义随阶段变化):
    普通回合 → 空位落子 / 占领对方棋子格(置 pending);
    安置回合 → 把手中对方棋子放到 pending 射线可达的空格。
- 静态方法(legal_mask_for / encode_for / danger_maps / _fwd_runs)可对任意批量
  棋盘快照调用,供 MCTS 叶子节点批量扩展使用。

对外主要接口:
    BatchedReverseGomoku(n_games, device, config)
        .reset() / .legal_mask() / .encode() / .apply_moves(actions)
        .board / .current_player(±1) / .pending / .gameover / .loser(±1) / .draw
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..reversegomoku import (
    ACTION_SIZE, AXES_DIRS, BLACK, BOARD_SIZE, DANGER_START_TURNS, DIRECTIONS,
    GameConfig, DEFAULT_CONFIG, LINE_IDX, LINE_BACK, MOVE_DISTS, MOVE_DIRS,
    OPP_DIR, PLACE_SIZE, RAYS, WHITE, WIN_STREAK, NO_PENDING,
)

_TABLE_CACHE: dict = {}


def _tables(device):
    """按设备缓存静态几何表(构建一次)。"""
    if device not in _TABLE_CACHE:
        _TABLE_CACHE[device] = (
            torch.tensor(RAYS, dtype=torch.int64, device=device),
            [torch.tensor(x, dtype=torch.int64, device=device) for x in LINE_IDX],
            [torch.tensor(x, dtype=torch.int64, device=device) for x in LINE_BACK],
        )
    return _TABLE_CACHE[device]


class BatchedReverseGomoku:
    def __init__(self, n_games: int, device=None, config: GameConfig = DEFAULT_CONFIG):
        self.n = n_games
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.board = torch.zeros((self.n, BOARD_SIZE, BOARD_SIZE), dtype=torch.int8, device=self.device)
        self.current_player = torch.full((self.n,), 1, dtype=torch.int8, device=self.device)  # +1=黑
        self.pending = torch.full((self.n,), NO_PENDING, dtype=torch.int16, device=self.device)
        self.move_count = torch.zeros((self.n,), dtype=torch.int16, device=self.device)
        self.turn_count = torch.zeros((self.n,), dtype=torch.int16, device=self.device)
        self.white_turns = torch.zeros((self.n,), dtype=torch.int16, device=self.device)
        self.gameover = torch.zeros((self.n,), dtype=torch.bool, device=self.device)
        self.draw = torch.zeros((self.n,), dtype=torch.bool, device=self.device)
        self.loser = torch.zeros((self.n,), dtype=torch.int8, device=self.device)  # 0 无, +1 黑负, -1 白负

    # ------------------------------------------------------------------
    # 向量化连续子(游程)
    # ------------------------------------------------------------------
    @staticmethod
    def _fwd_runs(mask: torch.Tensor, device) -> torch.Tensor:
        """mask: (N,15,15) bool → (N,8,15,15) int16: runs[d,r,c] = 沿 d 从 (r,c) 连续为真的长度。
        cumsum-cummax 公式给出后向游程;方向 d 的前向游程 = 反方向线上的后向游程。"""
        N = mask.shape[0]
        flat = mask.reshape(N, PLACE_SIZE)
        _, li_t, lb_t = _tables(device)
        out = torch.zeros((N, 8, BOARD_SIZE, BOARD_SIZE), dtype=torch.int16, device=device)
        for d in range(8):
            op = OPP_DIR[d]
            li, lb = li_t[op], lb_t[op]                      # (L,15), (225,)
            lm = flat[:, li]                                 # (N,L,15) 含 -1 填充
            lm = torch.where(li >= 0, lm, torch.zeros((), dtype=torch.bool, device=device))
            lm = lm.to(torch.int16)
            s = lm.cumsum(dim=2)
            z = torch.where(lm == 0, s, torch.zeros_like(s))
            cm = torch.cummax(z, dim=2).values
            fwd = s - cm
            out[:, d] = fwd.reshape(N, -1)[:, lb].reshape(N, BOARD_SIZE, BOARD_SIZE)
        return out

    @staticmethod
    def _wins_at(positions: torch.Tensor, colors: torch.Tensor,
                 runs_b: torch.Tensor, runs_w: torch.Tensor) -> torch.Tensor:
        """positions (K,) flat, colors (K,) ±1 → (K,) bool:该位置该色是否连五。
        runs 必须是放置棋子后的棋盘游程。"""
        rp_b = F.pad(runs_b.float(), (1, 1, 1, 1, 0, 0))
        rp_w = F.pad(runs_w.float(), (1, 1, 1, 1, 0, 0))
        pos_r = positions // BOARD_SIZE
        pos_c = positions % BOARD_SIZE
        br = torch.arange(positions.shape[0], device=positions.device)
        rp = torch.where(colors[:, None, None, None] == 1, rp_b, rp_w)
        win = torch.zeros_like(positions, dtype=torch.bool)
        for d1, d2 in AXES_DIRS:
            dr1, dc1 = DIRECTIONS[d1]
            dr2, dc2 = DIRECTIONS[d2]
            v1 = rp[br, d1, 1 + dr1 + pos_r, 1 + dc1 + pos_c]
            v2 = rp[br, d2, 1 + dr2 + pos_r, 1 + dc2 + pos_c]
            win |= ((v1 + v2) >= WIN_STREAK - 1)
        return win

    # ------------------------------------------------------------------
    # 危险平面 / 编码 / 合法掩码(静态,任意批量快照)
    # ------------------------------------------------------------------
    @classmethod
    def _line5_map(cls, board: torch.Tensor, player: torch.Tensor, device) -> torch.Tensor:
        """board (N,15,15) ±1, player (N,) ±1 → (N,15,15) bool:
        该格若落 player 子即连五(不筛空格,含对方棋子位=占领自杀判定)。"""
        N = board.shape[0]
        runs = cls._fwd_runs(board == player[:, None, None], device)
        rp = F.pad(runs.float(), (1, 1, 1, 1, 0, 0))
        out = torch.zeros((N, BOARD_SIZE, BOARD_SIZE), dtype=torch.bool, device=device)
        for d1, d2 in AXES_DIRS:
            dr1, dc1 = DIRECTIONS[d1]
            dr2, dc2 = DIRECTIONS[d2]
            s1 = rp[:, d1, 1 + dr1:1 + dr1 + BOARD_SIZE, 1 + dc1:1 + dc1 + BOARD_SIZE]
            s2 = rp[:, d2, 1 + dr2:1 + dr2 + BOARD_SIZE, 1 + dc2:1 + dc2 + BOARD_SIZE]
            out |= (s1 + s2 >= WIN_STREAK - 1)
        return out

    @classmethod
    def danger_maps(cls, board: torch.Tensor, device) -> torch.Tensor:
        """board (N,15,15) ±1 → (N,2,15,15) float32 [黑危险, 白危险],1.0=落子会连五。"""
        empty = (board == 0)
        d_b = cls._line5_map(board, torch.ones(board.shape[0], dtype=torch.int8, device=device), device)
        d_w = cls._line5_map(board, torch.full((board.shape[0],), -1, dtype=torch.int8, device=device), device)
        return torch.stack([(d_b & empty).float(), (d_w & empty).float()], dim=1)

    @classmethod
    def encode_for(cls, board: torch.Tensor, player: torch.Tensor, pending: torch.Tensor,
                   turn_count: torch.Tensor, config: GameConfig, device) -> torch.Tensor:
        """→ (N,5,15,15) float32 [己方子, 对方子, 己方危险, 对方危险, 移子待定标记]"""
        N = board.shape[0]
        state = torch.zeros((N, 5, BOARD_SIZE, BOARD_SIZE), dtype=torch.float32, device=device)
        state[:, 0] = (board == player[:, None, None])
        state[:, 1] = (board == -player[:, None, None])
        active = turn_count >= DANGER_START_TURNS
        if active.any():
            danger = cls.danger_maps(board, device)          # (N,2,15,15) [黑,白]
            d_me = torch.where(player[:, None, None] == 1, danger[:, 0], danger[:, 1])
            d_op = torch.where(player[:, None, None] == 1, danger[:, 1], danger[:, 0])
            state[:, 2] = torch.where(active[:, None, None], d_me, torch.zeros_like(d_me))
            state[:, 3] = torch.where(active[:, None, None], d_op, torch.zeros_like(d_op))
        pend_ok = pending >= 0
        if pend_ok.any():
            idx_p = torch.nonzero(pend_ok).squeeze(1)
            pr = (pending[idx_p] // BOARD_SIZE).to(torch.int64)
            pc = (pending[idx_p] % BOARD_SIZE).to(torch.int64)
            state[idx_p, 4, pr, pc] = 1.0
        return state

    @staticmethod
    def _white_restricted(player: torch.Tensor, white_turns: torch.Tensor,
                          config: GameConfig) -> torch.Tensor:
        n = config.white_restrict_turns
        if n <= 0:
            return torch.zeros_like(player, dtype=torch.bool)
        return (player == -1) & (white_turns < n)

    @classmethod
    def legal_mask_for(cls, board: torch.Tensor, player: torch.Tensor, pending: torch.Tensor,
                       white_turns: torch.Tensor, config: GameConfig, device,
                       turn_count: torch.Tensor = None) -> torch.Tensor:
        """→ (N, ACTION_SIZE) bool 合法动作掩码(纯落子空间)。
        turn_count 用于判负窗口:窗口内连五不判负,自杀着不剔除。"""
        N = board.shape[0]
        flat = board.reshape(N, PLACE_SIZE)
        mask = torch.zeros((N, ACTION_SIZE), dtype=torch.bool, device=device)

        # ---- 安置步(pending>=0):手中对方棋子沿 pending 射线放到连续空位 ----
        pend_ok = pending >= 0
        if pend_ok.any():
            sub = torch.nonzero(pend_ok).squeeze(1)
            K = sub.numel()
            pbase = pending[sub].clamp(min=0).to(torch.int64)
            run_empty = cls._fwd_runs(board[sub] == 0, device)
            rp = F.pad(run_empty.float(), (1, 1, 1, 1, 0, 0))
            pr, pc = pbase // BOARD_SIZE, pbase % BOARD_SIZE
            targets = _tables(device)[0][pbase]              # (K,8,14) 或 -1
            targets = targets.reshape(K, MOVE_DIRS * MOVE_DISTS)
            target_ok = (targets >= 0) & (flat[sub[:, None], targets.clamp(min=0)] == 0)
            run_ok = torch.zeros((K, MOVE_DIRS * MOVE_DISTS), dtype=torch.bool, device=device)
            br = torch.arange(K, device=device)
            for t in range(MOVE_DIRS * MOVE_DISTS):
                d, dist_idx = divmod(t, MOVE_DISTS)
                dr, dc = DIRECTIONS[d]
                run_ok[:, t] = rp[br, d, 1 + dr + pr, 1 + dc + pc] >= dist_idx + 1
            legal = target_ok & run_ok                       # (K,112)
            for k in range(K):
                mask[sub[k], targets[k, legal[k]]] = True    # 只写合法格(避免重复索引覆盖)

        # ---- 普通回合:落子(空位) + 占领(对方棋子位且存在可安置路径) ----
        norm = ~pend_ok
        if norm.any():
            sub = torch.nonzero(norm).squeeze(1)
            mask[sub, :PLACE_SIZE] = (flat[sub] == 0)
            rest = cls._white_restricted(player[sub], white_turns[sub], config)
            sub2 = sub[~rest]
            if sub2.numel():
                K = sub2.numel()
                opp = -player[sub2]
                run_empty = cls._fwd_runs(board[sub2] == 0, device)
                rp = F.pad(run_empty.float(), (1, 1, 1, 1, 0, 0))
                has_target = torch.zeros((K, PLACE_SIZE), dtype=torch.bool, device=device)
                for d in range(MOVE_DIRS):
                    dr, dc = DIRECTIONS[d]
                    sh = rp[:, d, 1 + dr:1 + dr + BOARD_SIZE, 1 + dc:1 + dc + BOARD_SIZE]
                    has_target |= (sh.reshape(K, PLACE_SIZE) >= 1)
                opp_flat = (flat[sub2] == opp[:, None])
                mask[sub2, :PLACE_SIZE] |= (opp_flat & has_target)
            # 自杀着剔除(判负生效后):落子/占领完成己方连五必败,一律剔除
            # (占领已在上方加入掩码,此处统一过筛;白棋限制期同样生效)
            if config.mask_suicide:
                active = (turn_count[sub] >= config.loss_start_turns) \
                    if turn_count is not None else torch.ones(sub.numel(), dtype=torch.bool, device=device)
                if active.any():
                    line5 = cls._line5_map(board[sub], player[sub], device)
                    mask[sub, :PLACE_SIZE] &= ~(line5.reshape(sub.numel(), PLACE_SIZE) & active[:, None])
        return mask

    def legal_mask(self) -> torch.Tensor:
        return self.legal_mask_for(self.board, self.current_player, self.pending,
                                   self.white_turns, self.config, self.device,
                                   self.turn_count)

    def encode(self) -> torch.Tensor:
        return self.encode_for(self.board, self.current_player, self.pending,
                               self.turn_count, self.config, self.device)

    # ------------------------------------------------------------------
    # 批量执行一个步(仅对未结束的游戏生效)
    # ------------------------------------------------------------------
    def apply_moves(self, actions: torch.Tensor) -> None:
        """actions (N,) int64:每个游戏的合法动作 idx(0..224,纯落子空间)。
        未结束的游戏必须提供合法动作。"""
        live = ~self.gameover
        b_idx = torch.nonzero(live).squeeze(1)
        if b_idx.numel() == 0:
            return
        # 无子可走检测:判负窗口后所有着法均为自杀 → 行棋方必败,立即终局
        mask = self.legal_mask_for(self.board[b_idx], self.current_player[b_idx],
                                   self.pending[b_idx], self.white_turns[b_idx],
                                   self.config, self.device, self.turn_count[b_idx])
        not_full = (self.board[b_idx] == 0).any(dim=(1, 2))
        stuck = ~mask.any(dim=1) & not_full
        if stuck.any():
            s_idx = b_idx[stuck]
            self.gameover[s_idx] = True
            self.loser[s_idx] = self.current_player[s_idx]
            b_idx = b_idx[~stuck]
            if b_idx.numel() == 0:
                return
        act = actions[b_idx]
        me = self.current_player[b_idx]                      # ±1
        opp = -me
        flat = self.board.reshape(self.n, PLACE_SIZE)

        place = self.pending[b_idx] < 0                     # 普通回合(落子或占领)
        pend = ~place                                       # 安置回合
        was_opp = torch.zeros_like(place)
        p_idx = b_idx[place]
        was_opp[place] = flat[p_idx, act[place]] == opp[place]
        b1 = place & was_opp                                # 占领步(移子第一步)
        end_turn = (place & ~b1) | pend                     # 落子或安置步结束本回合

        # 落子/占领:己方棋子放到 act;安置步:手中对方棋子放到 act
        flat[p_idx, act[place]] = me[place]
        q_idx = b_idx[pend]
        flat[q_idx, act[pend]] = opp[pend]

        # 判负(增量式):落子/占领查己方在 act 连五,安置步查对方在 act 连五
        me_win = torch.zeros_like(place)
        op_win = torch.zeros_like(place)
        if self.config.loss_start_turns > 0:
            chk = self.turn_count[b_idx] >= self.config.loss_start_turns
            runs_b = self._fwd_runs(self.board[b_idx] == 1, self.device)
            runs_w = self._fwd_runs(self.board[b_idx] == -1, self.device)
            me_win = self._wins_at(act, me, runs_b, runs_w) & chk & place
            op_win = self._wins_at(act, opp, runs_b, runs_w) & chk & pend

        over = me_win | op_win
        self.gameover[b_idx] |= over
        self.loser[b_idx] = torch.where(me_win, me, self.loser[b_idx])
        self.loser[b_idx] = torch.where(op_win & ~me_win, opp, self.loser[b_idx])
        if not bool(over.all()):
            # 和棋:棋盘无空位(只有落子/安置步会增加子数)
            full = (flat[b_idx] == 0).sum(dim=1) == 0
            d = full & ~over
            self.gameover[b_idx] |= d
            self.draw[b_idx] |= d

        self.pending[b_idx] = torch.where(b1, act, torch.full_like(act, NO_PENDING)).to(self.pending.dtype)
        self.turn_count[b_idx] += end_turn
        self.white_turns[b_idx] += end_turn & (me == -1)
        self.current_player[b_idx] = torch.where(end_turn, opp, me)
        self.move_count[b_idx] += 1

    # ---- 便捷:单局状态导出为 numpy 引擎语义(供回放/调试) ----
    def game_state_np(self, i: int):
        board = self.board[i].cpu().numpy()
        np_board = np.zeros_like(board)
        np_board[board == 1] = BLACK
        np_board[board == -1] = WHITE
        return np_board, (BLACK if self.current_player[i] == 1 else WHITE), \
               int(self.pending[i]), int(self.turn_count[i]), int(self.white_turns[i])
