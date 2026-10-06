"""AlphaZero 训练主循环:批量自对弈 → 经验回放 → 训练 → 评估 → 存档

注意:本自研网络的训练总体失败,最终强度不足,最多稍弱于启发式引擎,仅供复现研究。

自对弈样本内嵌杀棋监督(tactics.py:安置杀/两步杀,one-hot 策略 + value=+1),
网络早期即学会"能杀就杀"的战术;可用启发式 AI 棋谱预热:
    python -m antifive.heuristic --games 1000 --out records
    python -m antifive.training.pretrain_bc --records records --out checkpoints/warm_model.pth

用法示例:
    
    python -m antifive.training.train --resume models/warm_model.pth
"""

from __future__ import annotations

import argparse
import os
import random
import time
from collections import deque

import numpy as np
import torch

from ..mcts import MCTS
from ..network import PolicyValueNet
from ..reversegomoku import (
    BLACK, BOARD_SIZE, GameConfig, ReverseGomoku, SYM_FLAT, SYM_INV,
    symmetry_action_map,
)
from .selfplay import Example, self_play_batch


def parse_args():
    p = argparse.ArgumentParser(description="逆五子棋 AlphaZero 训练")
    p.add_argument("--iters", type=int, default=50, help="总迭代数")
    p.add_argument("--games", type=int, default=16, help="每迭代自对弈并行局数")
    p.add_argument("--sims", type=int, default=200, help="每步 MCTS 模拟次数")
    p.add_argument("--batch-k", type=int, default=4, help="每轮每局并行下降路径数(虚拟损失多叶批量)")
    p.add_argument("--epochs", type=int, default=2, help="每迭代训练轮数")
    p.add_argument("--batch", type=int, default=256, help="训练批量")
    p.add_argument("--lr", type=float, default=1e-3, help="学习率")
    p.add_argument("--channels", type=int, default=128, help="网络通道数")
    p.add_argument("--res", type=int, default=4, help="残差块数")
    p.add_argument("--buffer", type=int, default=10000, help="经验回放容量")
    p.add_argument("--c-puct", type=float, default=5.0)
    p.add_argument("--temp-moves", type=int, default=12, help="前 N 步使用温度采样")
    p.add_argument("--dirichlet-alpha", type=float, default=0.3)
    p.add_argument("--noise-eps", type=float, default=0.25)
    p.add_argument("--workers", type=int, default=0, help="自对弈 worker 进程数(0=主进程内串行自对弈)")
    p.add_argument("--worker-games", type=int, default=0, help="每 worker 并行局数(0=同 --games)")
    p.add_argument("--target-examples", type=int, default=0, help="每迭代目标样本量(0=自动)")
    p.add_argument("--collect-timeout", type=float, default=300.0, help="并行模式单次收集超时(秒)")
    p.add_argument("--max-wait", type=float, default=0.0, help="并行模式单迭代最大等待(秒,0=collect-timeout×12)")
    p.add_argument("--eval-every", type=int, default=5, help="评估间隔迭代数")
    p.add_argument("--eval-games", type=int, default=4)
    p.add_argument("--eval-sims", type=int, default=60)
    p.add_argument("--white-restrict", type=int, default=2, help="白棋禁移子回合数")
    p.add_argument("--loss-start", type=int, default=8, help="判负生效回合数")
    p.add_argument("--device", type=str, default="", help="cuda/cpu,空=自动")
    p.add_argument("--save-dir", type=str, default="checkpoints", help="存档目录")
    p.add_argument("--resume", type=str, default="", help="恢复的模型文件")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _game_config(args) -> GameConfig:
    return GameConfig(loss_start_turns=args.loss_start,
                      white_restrict_turns=args.white_restrict)


def sparse_ce_loss(logits: torch.Tensor, legal_list, prob_list, device) -> torch.Tensor:
    """稀疏策略交叉熵:仅在合法动作上有目标概率。"""
    B = logits.shape[0]
    ks = torch.tensor([len(x) for x in legal_list], device=device)
    b_rep = torch.repeat_interleave(torch.arange(B, device=device), ks)
    a_idx = torch.cat([torch.from_numpy(x).to(device) for x in legal_list])
    p = torch.cat([torch.from_numpy(x).to(device) for x in prob_list])
    lp = torch.log_softmax(logits, dim=1)
    return -(lp[b_rep, a_idx] * p).sum() / B


def augment_sample(ex: Example, t: int) -> Example:
    """随机对称增强:状态像素按逆变换置换,策略动作按正变换置换,价值不变。"""
    state = ex.state.reshape(5, -1)[:, SYM_FLAT[SYM_INV[t]]].reshape(5, 15, 15)
    return Example(state, symmetry_action_map(t)[ex.legal],
                   ex.prob, ex.player, ex.value)


def train_batch(net: PolicyValueNet, opt, buffer, args, device):
    total_p = total_v = 0.0
    n_batch = 0
    idxs = np.arange(len(buffer))
    np.random.shuffle(idxs)
    for s in range(0, len(buffer), args.batch):
        batch = idxs[s:s + args.batch]
        states = []
        legal = []
        probs = []
        values = []
        for i in batch:
            ex = augment_sample(buffer[i], random.randrange(8))
            states.append(ex.state)
            legal.append(ex.legal)
            probs.append(ex.prob)
            values.append(ex.value)
        st = torch.from_numpy(np.stack(states)).to(device)
        vt = torch.tensor(values, dtype=torch.float32, device=device)
        logits, v = net(st)
        loss_p = sparse_ce_loss(logits, legal, probs, device)
        loss_v = torch.mean((v - vt) ** 2)
        loss = loss_p + loss_v
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_p += float(loss_p.detach())
        total_v += float(loss_v.detach())
        n_batch += 1
    return total_p / max(n_batch, 1), total_v / max(n_batch, 1)


def evaluate(net_a: PolicyValueNet, net_b: PolicyValueNet, config: GameConfig,
             device, games: int = 4, sims: int = 60) -> float:
    """net_a 与 net_b 对弈,交替执黑(规则不对称)。返回 net_a 视角胜率 ∈ [-1,1]。"""
    score = 0.0
    for g in range(games):
        game = ReverseGomoku(config)
        mcts_a = MCTS(net_a, device, sims=sims, config=config)
        mcts_b = MCTS(net_b, device, sims=sims, config=config)
        a_is_black = (g % 2 == 0)
        while not game.game_over:
            is_a_turn = (game.current_player == BLACK) == a_is_black
            a = (mcts_a if is_a_turn else mcts_b).search(game)
            if a is None:
                game.game_over = True
                game.loser = game.current_player      # 无子可走,行棋方必败
                break
            game.make_move(a)
        if game.is_draw:
            continue
        black_won = game.loser != BLACK
        if black_won == a_is_black:
            score += 1.0
        else:
            score -= 1.0
    return score / games


def _atomic_save(net: PolicyValueNet, path: str, step: int = 0) -> None:
    """先写临时文件再原子替换,避免 worker 读到写了一半的模型。"""
    tmp = path + ".tmp"
    net.save(tmp, step=step)
    os.replace(tmp, path)


def _eval_and_save(net, args, config, device, best_path, log: str) -> str:
    if not (args.eval_every > 0 and _STATE["it"] % args.eval_every == 0
            and len(_STATE["buffer"]) >= args.batch):
        return log
    best = None
    if os.path.exists(best_path):
        best = PolicyValueNet.load(best_path, device)
    if best is None:
        _atomic_save(net, best_path)
        log += " | 初始存档 best"
    else:
        score = evaluate(net, best, config, device, args.eval_games, args.eval_sims)
        log += f" | 评估 vs best: {score:+.2f}"
        if score > 0:
            _atomic_save(net, best_path)
            log += " → 新 best"
    return log


_STATE = {}


def main():
    args = parse_args()
    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.save_dir, exist_ok=True)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    config = _game_config(args)
    print(f"设备: {device} | 规则: {config}")
    print(f"动作空间: {ReverseGomoku.get_action_size()} | 网络: {args.channels}ch x {args.res}res")

    net = PolicyValueNet(args.channels, args.res).to(device)
    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt["state_dict"])
        step = ckpt.get("step", 0)
        print(f"已恢复 {args.resume} (step={step})")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    buffer: deque = deque(maxlen=args.buffer)
    best_path = os.path.join(args.save_dir, "best_model.pth")
    latest_path = os.path.join(args.save_dir, "latest_model.pth")
    _STATE.update(it=0, buffer=buffer)

    if args.workers > 0:
        _atomic_save(net, latest_path)          # 先写初始权重,worker 才能加载
        from parallel_selfplay import ParallelSelfPlay
        psp = ParallelSelfPlay(
            model_path=latest_path, device_str=str(device),
            channels=args.channels, res_blocks=args.res, config=config,
            workers=args.workers,
            worker_games=args.worker_games or args.games,
            sims=args.sims, k=args.batch_k, c_puct=args.c_puct,
            temperature_moves=args.temp_moves,
            alpha=args.dirichlet_alpha, noise_eps=args.noise_eps,
            seed=args.seed)
        if args.target_examples > 0:
            psp.target_examples = args.target_examples
        try:
            for it in range(1, args.iters + 1):
                _STATE["it"] = it
                t0 = time.time()
                examples = psp.collect(timeout=args.collect_timeout,
                                       max_wait=args.max_wait or None)
                if not examples:
                    print("!!! 未收集到样本:worker 全部失活或超过最大等待。"
                          "请检查 worker_error.log、显存、以及是否仍有旧训练进程", flush=True)
                    break
                buffer.extend(examples)
                t_sp = time.time() - t0
                t0 = time.time()
                loss_p, loss_v = train_batch(net, opt, buffer, args, device)
                t_tr = time.time() - t0
                step += 1
                log = (f"[iter {it}] 收集 {t_sp:.1f}s ({len(examples)} 样本) | "
                       f"训练 {t_tr:.1f}s | loss_p={loss_p:.4f} loss_v={loss_v:.4f} | "
                       f"buffer={len(buffer)}")
                log = _eval_and_save(net, args, config, device, best_path, log)
                _atomic_save(net, latest_path, step)
                print(log, flush=True)
        finally:
            psp.close()
    else:
        for it in range(1, args.iters + 1):
            _STATE["it"] = it
            t0 = time.time()
            examples = self_play_batch(
                net, config=config, n_games=args.games, sims=args.sims,
                c_puct=args.c_puct, dirichlet_alpha=args.dirichlet_alpha,
                noise_eps=args.noise_eps, temperature_moves=args.temp_moves,
                device=device, k=args.batch_k)
            buffer.extend(examples)
            t_sp = time.time() - t0

            t0 = time.time()
            loss_p, loss_v = train_batch(net, opt, buffer, args, device)
            t_tr = time.time() - t0
            step += 1

            log = (f"[iter {it}] 自对弈 {t_sp:.1f}s ({len(examples)} 样本) | "
                   f"训练 {t_tr:.1f}s | loss_p={loss_p:.4f} loss_v={loss_v:.4f} | "
                   f"buffer={len(buffer)}")
            log = _eval_and_save(net, args, config, device, best_path, log)
            _atomic_save(net, latest_path, step)
            print(log, flush=True)


if __name__ == "__main__":
    main()
