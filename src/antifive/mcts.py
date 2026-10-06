"""AlphaZero MCTS(单局搜索,供人机对战与模型评估)

按"步"搜索:移子被建模为同一方连续两步(占领+安置),搜索树节点即步状态。
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    import torch

    _no_grad = torch.no_grad
except ImportError:                     # 无 torch:退化为均匀先验随机搜索
    torch = None

    def _no_grad(func):
        return func

from .reversegomoku import (
    ACTION_SIZE, BLACK, BOARD_SIZE, EMPTY, GameConfig, DEFAULT_CONFIG,
    ReverseGomoku,
)


class MCTSNode:
    __slots__ = ("board", "player", "pending", "turn_count", "white_turns",
                 "terminal", "loser", "draw", "legal", "P", "N", "W", "children")

    def __init__(self, board, player, pending, turn_count, white_turns,
                 terminal: bool = False, loser: int = 0, draw: bool = False):
        self.board = board                       # (15,15) int8
        self.player = player                     # BLACK/WHITE
        self.pending = pending                   # -1 或手中棋子原位
        self.turn_count = turn_count
        self.white_turns = white_turns
        self.terminal = terminal
        self.loser = loser
        self.draw = draw
        self.legal: Optional[np.ndarray] = None  # (K,) int32 合法动作
        self.P: Optional[np.ndarray] = None      # (K,) 先验概率
        self.N = np.zeros(0, dtype=np.int32)
        self.W = np.zeros(0, dtype=np.float32)
        self.children: Dict[int, "MCTSNode"] = {}

    def is_expanded(self) -> bool:
        return self.legal is not None

    def print_debug(self) -> None:
        symbols = {0: " .", 1: " X", 2: " O"}
        for r in range(BOARD_SIZE):
            print("    " + "".join(symbols[int(self.board[r, c])] for c in range(BOARD_SIZE)))


class MCTS:
    def __init__(self, network, device, sims: int = 400, c_puct: float = 5.0,
                 dirichlet_alpha: float = 0.3, noise_eps: float = 0.25,
                 config: GameConfig = DEFAULT_CONFIG, time_budget: float = 0.0):
        self.network = network
        self.device = device
        self.sims = sims
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_eps = noise_eps
        self.config = config
        self.time_budget = time_budget          # 每步思考时间上限(秒);0=不限时,只看 sims
        self.progress = 0                       # 当前搜索已完成的模拟次数(GUI 实时显示)
        self.progress_total = sims

    # ---- 节点 ----
    @staticmethod
    def node_from_game(game: ReverseGomoku) -> MCTSNode:
        return MCTSNode(game.board.copy(), game.current_player, game.pending,
                        game.turn_count, game.white_turns,
                        terminal=game.game_over, loser=game.loser, draw=game.is_draw)

    # ---- 批量叶子评估:编码 + 合法掩码 + 网络前向 + 扩展 ----
    @_no_grad
    def _evaluate(self, nodes: List[MCTSNode]) -> np.ndarray:
        n = len(nodes)
        masks = np.zeros((n, ACTION_SIZE), dtype=bool)
        states = np.zeros((n, 5, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for k, nd in enumerate(nodes):
            masks[k] = ReverseGomoku.legal_mask_for(
                nd.board, nd.player, nd.pending, nd.white_turns, self.config,
                nd.turn_count)
            states[k] = ReverseGomoku.encode_for(
                nd.board, nd.player, nd.pending, nd.turn_count, self.config)
        if self.network is None:
            # 无网络:均匀先验 + 零价值(纯随机搜索)
            values = np.zeros(n)
            for k, nd in enumerate(nodes):
                if not masks[k].any():
                    nd.terminal = True
                    nd.loser = nd.player
                    nd.draw = False
                    values[k] = -1.0
                    continue
                legal = np.nonzero(masks[k])[0].astype(np.int32)
                p = np.full(len(legal), 1.0 / max(len(legal), 1), dtype=np.float32)
                nd.legal = legal
                nd.P = p
                nd.N = np.zeros(len(legal), dtype=np.int32)
                nd.W = np.zeros(len(legal), dtype=np.float32)
                nd.children = {}
            return values
        b_t = torch.from_numpy(states).to(self.device)
        m_t = torch.from_numpy(masks).to(self.device)
        probs, values = self.network.policy_value(b_t, m_t)
        probs_np = probs.cpu().numpy()
        for k, nd in enumerate(nodes):
            if not masks[k].any():
                # 无子可走(所有着法均为自杀):行棋方必败,视为终局,不扩展
                nd.terminal = True
                nd.loser = nd.player
                nd.draw = False
                values[k] = -1.0
                continue
            legal = np.nonzero(masks[k])[0].astype(np.int32)
            p = probs_np[k][legal].astype(np.float32)
            p = p / max(float(p.sum()), 1e-12)
            nd.legal = legal
            nd.P = p
            nd.N = np.zeros(len(legal), dtype=np.int32)
            nd.W = np.zeros(len(legal), dtype=np.float32)
            nd.children = {}
        return values.cpu().numpy()

    # ---- 搜索核心 ----
    def _select(self, node: MCTSNode, path: list) -> MCTSNode:
        """按 PUCT 下降至未扩展叶子或终局,路径记录 (node, action_idx)。"""
        while node.is_expanded() and not node.terminal:
            total = int(node.N.sum()) + 1
            score = node.W / np.maximum(node.N, 1) + \
                self.c_puct * node.P * math.sqrt(total) / (1 + node.N)
            j = int(np.argmax(score))
            a = int(node.legal[j])
            path.append((node, j))
            child = node.children.get(a)
            if child is None:
                b, p2, pend2, tc2, wt2, loser = ReverseGomoku.apply_step(
                    node.board, a, node.player, node.pending,
                    node.turn_count, node.white_turns, self.config)
                draw = (loser == 0) and bool(np.all(b != EMPTY))
                child = MCTSNode(b, p2, pend2, tc2, wt2,
                                 terminal=(loser != 0 or draw), loser=loser, draw=draw)
                node.children[a] = child
            node = child
        return node

    def _run_sims(self, root: MCTSNode, add_noise: bool,
                  time_budget: float = None) -> None:
        if add_noise and root.is_expanded():
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.legal))
            root.P = (1 - self.noise_eps) * root.P + self.noise_eps * noise
        if time_budget is None:
            time_budget = self.time_budget
        deadline = (time.perf_counter() + time_budget
                    if time_budget and time_budget > 0 else None)
        self.progress_total = self.sims
        self.progress = 0
        check_every = 4                      # 每 4 次模拟查一次表,减少 time() 调用开销
        for i in range(self.sims):
            self.progress = i + 1
            path: list = []
            leaf = self._select(root, path)
            if leaf.terminal:
                v = self._terminal_value(leaf)
            else:
                v = float(self._evaluate([leaf])[0])
            self._backprop(path, v)
            if deadline is not None and (i + 1) % check_every == 0 \
                    and time.perf_counter() > deadline:
                break

    @staticmethod
    def _terminal_value(node: MCTSNode) -> float:
        if node.draw or node.loser == 0:
            return 0.0
        return 1.0 if node.loser != node.player else -1.0

    @staticmethod
    def _backprop(path: list, v: float) -> None:
        """回传 v(叶子行棋方视角),每跳先取负再加:Q 以 node 行棋方视角存储。"""
        for node, j in reversed(path):
            v = -v
            node.N[j] += 1
            node.W[j] += v

    # ---- 对外接口 ----
    def search(self, game: ReverseGomoku, add_noise: bool = False,
               time_budget: float = None):
        """搜索并返回访问次数最多的动作 idx;无子可走(死局)时返回 None。
        time_budget:每步思考时间上限(秒);None=用实例默认,0=不限时。"""
        root = self.node_from_game(game)
        self._run_sims(root, add_noise, time_budget=time_budget)
        if root.legal is None or root.N.size == 0:
            return None
        return int(root.legal[int(np.argmax(root.N))])

    def get_action_probs(self, game: ReverseGomoku, temperature: float = 1.0,
                         add_noise: bool = False, time_budget: float = None):
        """返回 (legal_actions, probs)。temperature=0 → argmax 独热。无子可走时返回 (None, None)。"""
        root = self.node_from_game(game)
        self._run_sims(root, add_noise, time_budget=time_budget)
        if root.legal is None or root.N.size == 0:
            return None, None
        counts = root.N.astype(np.float64)
        if temperature == 0:
            p = np.zeros_like(counts)
            p[int(np.argmax(counts))] = 1.0
        else:
            p = counts ** (1.0 / temperature)
            p = p / p.sum()
        return root.legal, p
