"""批量并行自对弈(AlphaZero 训练数据生成)

- 使用 BatchedReverseGomoku 批量推进 N 局游戏;
- 每局一棵 MCTS 树,每轮模拟各下降一步,叶子批量(编码+掩码+网络前向)评估;
- 每步记录一个训练样本(状态/合法动作/访问分布/价值),终局后按胜负回填价值;
- 判负窗口开启后,每步额外生成杀棋监督样本(tactics.py:安置杀/两步杀,
  one-hot 策略 + value=+1),让网络早期学会"能杀就杀"的战术。

移子按两步建模:占领步与安置步各是一个样本、各是一次树搜索决策。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from ..mcts import MCTS, MCTSNode
from ..reversegomoku import (
    ACTION_SIZE, BLACK, BOARD_SIZE, EMPTY, GameConfig, DEFAULT_CONFIG,
    ReverseGomoku, WHITE,
)
from .reverse_gomoku_torch import BatchedReverseGomoku
from ..tactics import kill_samples


@dataclass
class Example:
    """单步训练样本。value 在终局后回填(从行棋方视角 ±1/0)。"""
    state: np.ndarray      # (5,15,15) float32
    legal: np.ndarray      # (K,) int32 合法动作
    prob: np.ndarray       # (K,) float32 访问分布
    player: int            # 行棋方 ±1(1=黑)
    value: float = 0.0


def _select(node: MCTSNode, path: list, c_puct: float, config: GameConfig,
            vloss: float = 1.0) -> MCTSNode:
    """按 PUCT 下降至未扩展叶子或终局,路径记录 (node, action_idx)。
    vloss>0 时为虚拟损失:选中边立即 +N/-W,同轮后续下降会避开该路径,
    真实回传时对冲掉,保证每轮多路并行下降互不重叠。"""
    while node.is_expanded() and not node.terminal:
        total = int(node.N.sum()) + 1
        score = node.W / np.maximum(node.N, 1) + \
            c_puct * node.P * np.sqrt(total) / (1 + node.N)
        j = int(np.argmax(score))
        a = int(node.legal[j])
        node.N[j] += vloss
        node.W[j] -= vloss
        path.append((node, j))
        child = node.children.get(a)
        if child is None:
            b, p2, pend2, tc2, wt2, loser = ReverseGomoku.apply_step(
                node.board, a, node.player, node.pending,
                node.turn_count, node.white_turns, config)
            draw = (loser == 0) and bool(np.all(b != EMPTY))
            child = MCTSNode(b, p2, pend2, tc2, wt2,
                             terminal=(loser != 0 or draw), loser=loser, draw=draw)
            node.children[a] = child
        node = child
    return node


def _terminal_value(node: MCTSNode) -> float:
    if node.draw or node.loser == 0:
        return 0.0
    return 1.0 if node.loser != node.player else -1.0


def _backprop(path: list, v: float, vloss: float = 1.0) -> None:
    """真实回传 v(叶子行棋方视角),同时对冲下降时的虚拟损失。

    路径第一跳的节点行棋方是叶子行棋方的对手,因此每跳先取负再加:
    Q(node, a) 始终以 node 行棋方视角存储。"""
    for node, j in reversed(path):
        v = -v
        node.N[j] += 1 - vloss
        node.W[j] += v + vloss


@torch.no_grad()
def _batch_eval(nodes: List[MCTSNode], network, config: GameConfig, device) -> np.ndarray:
    """叶子批量评估:转换 ±1 棋盘 → 批量合法掩码/编码(引擎静态方法)→ 网络前向 → 扩展节点。
    返回各叶子价值(从该叶子行棋方视角)。"""
    n = len(nodes)
    boards = np.stack([nd.board for nd in nodes])                     # (n,15,15) int8 0/1/2
    players = np.array([1 if nd.player == BLACK else -1 for nd in nodes], dtype=np.int8)
    pends = np.array([nd.pending for nd in nodes], dtype=np.int16)
    tcs = np.array([nd.turn_count for nd in nodes], dtype=np.int16)
    wts = np.array([nd.white_turns for nd in nodes], dtype=np.int16)
    b_t = torch.from_numpy(boards).to(device)
    b_t = torch.where(b_t == 2, torch.tensor(-1, dtype=torch.int8, device=device), b_t)
    pl_t = torch.from_numpy(players).to(device)
    pe_t = torch.from_numpy(pends).to(device)
    tc_t = torch.from_numpy(tcs).to(device)
    wt_t = torch.from_numpy(wts).to(device)
    mask = BatchedReverseGomoku.legal_mask_for(b_t, pl_t, pe_t, wt_t, config, device, tc_t)
    mask_np = mask.cpu().numpy()
    state = BatchedReverseGomoku.encode_for(b_t, pl_t, pe_t, tc_t, config, device)
    probs, values = network.policy_value(state, mask)
    probs_np = probs.cpu().numpy()
    for k, nd in enumerate(nodes):
        if not mask_np[k].any():
            # 无子可走(所有着法均为自杀):行棋方必败,视为终局,不扩展
            nd.terminal = True
            nd.loser = nd.player
            nd.draw = False
            values[k] = -1.0
            continue
        legal = np.nonzero(mask_np[k])[0].astype(np.int32)
        p = probs_np[k][legal].astype(np.float32)
        p = p / max(float(p.sum()), 1e-12)
        nd.legal = legal
        nd.P = p
        nd.N = np.zeros(len(legal), dtype=np.int32)
        nd.W = np.zeros(len(legal), dtype=np.float32)
        nd.children = {}
    return values.cpu().numpy()


def _sim_round(engine: BatchedReverseGomoku, roots: List[MCTSNode],
               network, config: GameConfig, device, c_puct: float,
               k: int = 1, vloss: float = 1.0) -> None:
    """一轮模拟:每个进行中的游戏并行下降 k 条路径(虚拟损失防重复),
    所有叶子合并为一个大批量评估。k=1,vloss=0 即经典单叶轮次。"""
    batch: List[tuple] = []                      # (game_idx, path, leaf)
    for i in range(engine.n):
        if engine.gameover[i]:
            continue
        for _ in range(k):
            path: list = []
            leaf = _select(roots[i], path, c_puct, config, vloss)
            if leaf.terminal:
                _backprop(path, _terminal_value(leaf), vloss)
            else:
                batch.append((i, path, leaf))
    if batch:
        values = _batch_eval([lf[2] for lf in batch], network, config, device)
        for (_, path, leaf), v in zip(batch, values):
            _backprop(path, float(v), vloss)


def self_play_batch(network, config: GameConfig = DEFAULT_CONFIG, n_games: int = 16,
                    sims: int = 200, c_puct: float = 5.0,
                    dirichlet_alpha: float = 0.3, noise_eps: float = 0.25,
                    temperature_moves: int = 12, device=None,
                    max_steps: int = 450, k: int = 1, vloss: float = 1.0) -> List[Example]:
    """批量自对弈,返回所有局全部步的训练样本(终局后 value 已回填)。
    k: 每轮每局并行下降的路径数(虚拟损失多叶批量,增大 GPU 批大小);总模拟数不变。"""
    assert k >= 1
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = BatchedReverseGomoku(n_games, device, config)
    roots: List[MCTSNode] = []
    for i in range(n_games):
        b, pl, pe, tc, wt = engine.game_state_np(i)
        roots.append(MCTSNode(b, pl, pe, tc, wt, terminal=bool(engine.gameover[i]),
                              loser=0, draw=bool(engine.draw[i])))
    examples: List[List[Example]] = [[] for _ in range(n_games)]
    tactic_examples: List[List[Example]] = [[] for _ in range(n_games)]
    finalized = [False] * n_games

    # 开局:批量扩展根并加狄利克雷噪声
    if not bool(engine.gameover.all()):
        _batch_eval(roots, network, config, device)
        for i in range(n_games):
            if roots[i].is_expanded():
                noise = np.random.dirichlet([dirichlet_alpha] * len(roots[i].legal))
                roots[i].P = (1 - noise_eps) * roots[i].P + noise_eps * noise

    round_k = [k] * (sims // k) + ([sims % k] if sims % k else [])
    actions = torch.zeros(n_games, dtype=torch.int64, device=device)
    for step in range(max_steps):
        if bool(engine.gameover.all()):
            break
        for rk in round_k:
            _sim_round(engine, roots, network, config, device, c_puct, rk, vloss)
        temperature = 1.0 if step < temperature_moves else 0.0
        for i in range(n_games):
            if engine.gameover[i]:
                continue
            root = roots[i]
            if root.terminal or root.legal is None or root.N.size == 0:
                # 死局(无子可走,所有着法均为自杀):行棋方必败;满盘为和棋
                engine.gameover[i] = True
                if root.draw:
                    engine.draw[i] = True
                else:
                    engine.loser[i] = torch.tensor(
                        1 if root.player == BLACK else -1,
                        dtype=torch.int8, device=device)
                continue
            counts = root.N.astype(np.float64)
            if temperature > 0:
                p = counts ** (1.0 / temperature)
                p = p / p.sum()
                a = int(np.random.choice(root.legal, p=p))
            else:
                p = np.zeros_like(counts)
                j = int(np.argmax(counts))
                p[j] = 1.0
                a = int(root.legal[j])
            actions[i] = a
            st = ReverseGomoku.encode_for(root.board, root.player, root.pending,
                                          root.turn_count, config)
            examples[i].append(Example(st, root.legal.copy(), p.astype(np.float32),
                                       1 if root.player == BLACK else -1))
            # 杀棋监督:判负窗口开启后才可能有杀棋,直接生成 one-hot 战术样本
            if root.turn_count >= config.loss_start_turns:
                for ks in kill_samples(root.board, root.player, root.pending,
                                       root.turn_count, root.white_turns, config):
                    tactic_examples[i].append(
                        Example(ks.state, ks.legal, ks.prob, ks.player, 1.0))
        engine.apply_moves(actions)
        for i in range(n_games):
            if engine.gameover[i] and not finalized[i]:
                _finalize(examples[i], engine, i)
                finalized[i] = True
            elif not engine.gameover[i]:
                roots[i] = roots[i].children[int(actions[i])]   # 根复用
    for i in range(n_games):                                    # 超步数兜底(理论到不了)
        if not finalized[i]:
            _finalize(examples[i], engine, i)
    return ([ex for lst in examples for ex in lst] +
            [ex for lst in tactic_examples for ex in lst])


def _finalize(examples: List[Example], engine: BatchedReverseGomoku, i: int) -> None:
    """终局后按胜负回填所有样本价值(行棋方视角)。"""
    winner = 0 if bool(engine.draw[i]) else int(-engine.loser[i])   # ±1,0=和棋
    for ex in examples:
        ex.value = float(1.0 if ex.player == winner else (-1.0 if winner != 0 else 0.0))


def self_play_vs_heur(network, config: GameConfig = DEFAULT_CONFIG, n_games: int = 8,
                      sims: int = 400, c_puct: float = 5.0,
                      heur_depths=(2, 4), temperature_moves: int = 12,
                      h_budget: float = 0.0, device=None, seed: int = 0,
                      max_steps: int = 450) -> List[Example]:
    """网络(MCTS) vs 启发式实战对局,收集训练数据。

    替代"弱网互殴"自对弈——胜负由强对手(启发式)裁决,价值标签干净;
    每局交替先后手;启发式深度按 heur_depths[gi % len] 轮换(混合强弱对手,
    让网络也能赢到少数对局拿到正标签)。

    收集三类样本:
      - 网络侧:每步 MCTS 访问分布(前 temperature_moves 步温度采样);
      - 启发式侧:专家着法 one-hot 样本(行为克隆,直接学强手);
      - 杀棋监督:每步 kill_samples(判负窗口后,强化"能杀就杀")。
    价值按终局胜负从行棋方视角回填。"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import heuristic as h
    rng = np.random.default_rng(seed)
    all_exs: List[Example] = []
    for gi in range(n_games):
        g = ReverseGomoku(config)
        h_depth = heur_depths[gi % len(heur_depths)]
        net_is_black = (gi % 2 == 0)
        mcts = MCTS(network, device, sims=sims, c_puct=c_puct, config=config)
        exs: List[Example] = []
        steps = 0
        while not g.game_over and g.move_count < max_steps:
            player = g.current_player
            sgn = 1 if player == BLACK else -1
            if (player == BLACK) == net_is_black:
                temp = 1.0 if steps < temperature_moves else 0.0
                legal, p = mcts.get_action_probs(g, temperature=temp)
                if legal is None:                    # 死局:无子可走必败
                    g.game_over = True
                    g.loser = player
                    break
                a = int(rng.choice(legal, p=p)) if temp > 0 \
                    else int(legal[int(np.argmax(p))])
            else:
                a = h.choose_heuristic_move(g.board, player, g.pending,
                                            g.turn_count, g.white_turns,
                                            config, rng, depth=h_depth,
                                            time_budget=h_budget if h_budget > 0 else None)
                if a is None:
                    g.game_over = True
                    g.loser = player
                    break
                mask = g.legal_mask()
                legal = np.nonzero(mask)[0].astype(np.int32)
                p = np.zeros(len(legal), dtype=np.float32)
                p[np.searchsorted(legal, a)] = 1.0   # 专家着法 one-hot
            st = ReverseGomoku.encode_for(g.board, player, g.pending,
                                          g.turn_count, config)
            exs.append(Example(st, legal.copy(), np.asarray(p, dtype=np.float32),
                               sgn, 0.0))
            if g.turn_count >= config.loss_start_turns:
                for ks in kill_samples(g.board, player, g.pending,
                                       g.turn_count, g.white_turns, config):
                    exs.append(Example(ks.state, ks.legal, ks.prob, ks.player, 1.0))
            g.make_move(a)
            steps += 1
        winner = 0 if g.is_draw else (1 if g.loser == WHITE else -1)  # ±1,0=和棋
        for ex in exs:
            ex.value = float(1.0 if ex.player == winner else (-1.0 if winner != 0 else 0.0))
        all_exs.extend(exs)
    return all_exs
