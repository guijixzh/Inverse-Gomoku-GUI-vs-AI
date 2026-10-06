"""网络 vs 启发式 实战训练(替代弱网自对弈,解决价值信号噪声)

针对精修退化诊断的调整:
1. 数据来源:网络(MCTS)与启发式对弈,胜负由强对手裁决 → 价值标签干净;
   启发式侧着法作为 one-hot 专家样本(行为克隆,直接学强手);
   每步注入杀棋监督样本 → 强化"能杀就赢"。
2. 杀棋常驻:杀棋样本池每迭代按 kill_ratio 并入训练集,并周期性全量精修,
   修"有杀不杀"(--kill-ratio 加压,--kill-refresh 定期全量)。
3. 课程训练:先与默认AI(d2)对弈,评估胜率达到 parity_threshold 后自动
   切换 depth8(限时 --stage2-budget)做对手,层层递进。
4. --sims 建议 ≥400:搜索更准,减少烂对局。

用法:
    python -m antifive.training.train_vs_heur --resume checkpoints/warm_model.pth \
        --kill-epochs 3 --records records --iters 100 --games 8 --sims 400 \
        --stage1-depth 2 --stage2-depth 8 --stage2-budget 2.0 \
        --parity-threshold 0.5 --eval-every 2 --name vsheur
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import time
from collections import deque

import numpy as np
import torch

from .. import heuristic as h
from ..mcts import MCTS
from ..network import PolicyValueNet
from .. import record as record_mod
from ..reversegomoku import BLACK, WHITE, GameConfig, ReverseGomoku
from .selfplay import Example, self_play_vs_heur
from ..tactics import kill_samples
from .train import train_batch


def collect_kill_examples(records_dir: str, max_records: int = 0) -> list:
    """递归扫描棋谱,只收集杀棋监督样本(one-hot 杀着, value=+1)。"""
    paths = sorted(glob.glob(os.path.join(records_dir, "**", "*.afg"),
                             recursive=True))
    exs = []
    n = 0
    for path in paths:
        if max_records and n >= max_records:
            break
        try:
            rec = record_mod.load_game(path)
        except Exception:
            continue
        g = ReverseGomoku(rec.config)
        for m in rec.moves:
            if g.turn_count >= g.config.loss_start_turns:
                for ks in kill_samples(g.board, g.current_player, g.pending,
                                       g.turn_count, g.white_turns, g.config):
                    exs.append(Example(ks.state, ks.legal, ks.prob, ks.player, 1.0))
            g.make_move(m)
        n += 1
    print(f"杀棋样本收集: {len(exs)} 个(用棋谱 {n}/{len(paths)} 个文件)")
    return exs


def eval_vs_heur(net, config, device, sims, h_depth, games, seed=0, h_budget=0.0):
    """net vs 启发式(深度 h_depth),交替先后手,返回 (净胜局, 和棋, 总局)。"""
    rng = np.random.default_rng(seed)
    wins = draws = 0
    for gi in range(games):
        g = ReverseGomoku(config)
        mcts = MCTS(net, device, sims=sims, config=config)
        net_black = (gi % 2 == 0)
        while not g.game_over and g.move_count < 450:
            if (g.current_player == BLACK) == net_black:
                a = mcts.search(g)
            else:
                a = h.choose_heuristic_move(g.board, g.current_player, g.pending,
                                            g.turn_count, g.white_turns, config,
                                            rng, depth=h_depth,
                                            time_budget=h_budget if h_budget > 0 else None)
            if a is None:
                g.game_over = True
                g.loser = g.current_player
                break
            g.make_move(a)
        if g.is_draw:
            draws += 1
        elif (g.loser == WHITE) == net_black:
            wins += 1
    return wins, draws, games


def main():
    p = argparse.ArgumentParser(description="网络 vs 启发式 实战训练")
    p.add_argument("--resume", required=True, help="起始模型(建议 warm_model.pth)")
    p.add_argument("--records", default="", help="棋谱目录(杀棋预热用)")
    p.add_argument("--kill-epochs", type=int, default=0, help="开训前杀棋样本训练轮数")
    p.add_argument("--kill-max-records", type=int, default=4000,
                   help="杀棋预热最多采用棋谱数(0=全部)")
    p.add_argument("--iters", type=int, default=12)
    p.add_argument("--games", type=int, default=8, help="每迭代对局数")
    p.add_argument("--sims", type=int, default=400, help="网络每步 MCTS 模拟数")
    p.add_argument("--heur-depth", default="2,4",
                   help="启发式对手深度(逗号分隔,轮换用;被课程阶段覆盖时忽略)")
    p.add_argument("--stage1-depth", type=int, default=2, help="阶段1对手深度(默认AI)")
    p.add_argument("--stage2-depth", type=int, default=8, help="阶段2对手深度(持平后)")
    p.add_argument("--stage2-budget", type=float, default=2.0,
                   help="阶段2启发式每步限时(秒;depth8 必须限时)")
    p.add_argument("--parity-threshold", type=float, default=0.5,
                   help="阶段1评估胜率达到该值即切阶段2(与默认AI持平)")
    p.add_argument("--stage1-max-iters", type=int, default=40,
                   help="最多在第几迭代强制切阶段2(0=只等持平)")
    p.add_argument("--kill-ratio", type=float, default=2.0,
                   help="每迭代杀棋样本与对局样本比例(加压杀棋)")
    p.add_argument("--kill-refresh", type=int, default=10,
                   help="每 N 迭代全量杀棋精修 1 轮(0=关)")
    p.add_argument("--c-puct", type=float, default=5.0)
    p.add_argument("--temp-moves", type=int, default=12)
    p.add_argument("--epochs", type=int, default=2, help="每迭代训练轮数")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--buffer", type=int, default=10000)
    p.add_argument("--eval-every", type=int, default=2, help="评估间隔迭代数(0=关)")
    p.add_argument("--eval-games", type=int, default=6)
    p.add_argument("--eval-sims", type=int, default=200)
    p.add_argument("--white-restrict", type=int, default=2)
    p.add_argument("--loss-start", type=int, default=8)
    p.add_argument("--name", default="vsheur", help="存档文件名前缀")
    p.add_argument("--save-dir", default="checkpoints")
    p.add_argument("--device", default="", help="cuda/cpu,空=自动")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    config = GameConfig(loss_start_turns=args.loss_start,
                        white_restrict_turns=args.white_restrict)
    os.makedirs(args.save_dir, exist_ok=True)
    latest = os.path.join(args.save_dir, f"{args.name}_latest_model.pth")
    best = os.path.join(args.save_dir, f"{args.name}_best_model.pth")

    net = PolicyValueNet.load(args.resume, device)
    net.train()
    print(f"设备: {device} | 起始: {args.resume} | 课程: d{args.stage1_depth}→"
          f"d{args.stage2_depth}({args.stage2_budget}s/步) | sims={args.sims} "
          f"games={args.games} kill_ratio={args.kill_ratio}")

    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    buffer: deque = deque(maxlen=args.buffer)

    # ---- 杀棋样本池:常驻,每迭代训练集都并入,防止"能杀就杀"被冲走 ----
    kill_pool = collect_kill_examples(args.records, args.kill_max_records) \
        if args.records else []
    if kill_pool:
        t0 = time.time()
        for ep in range(1, args.kill_epochs + 1):
            loss_p, loss_v = train_batch(net, opt, kill_pool, args, device)
            print(f"[kill {ep}/{args.kill_epochs}] loss_p={loss_p:.4f} "
                  f"loss_v={loss_v:.4f} ({len(kill_pool)} 样本, "
                  f"{time.time() - t0:.0f}s)", flush=True)
    net.save(best, step=0)
    print(f"杀棋样本池 {len(kill_pool)} 常驻 | 初始 best: {best}")

    best_wr = -1.0
    stage = 1
    h_depth = args.stage1_depth
    h_budget = 0.0
    for it in range(1, args.iters + 1):
        t0 = time.time()
        net.eval()
        exs = self_play_vs_heur(net, config=config, n_games=args.games,
                                sims=args.sims, c_puct=args.c_puct,
                                heur_depths=(h_depth,),
                                temperature_moves=args.temp_moves,
                                h_budget=h_budget,
                                device=device, seed=args.seed + it)
        buffer.extend(exs)
        t_sp = time.time() - t0
        t0 = time.time()
        net.train()
        train_set = list(buffer)
        n_kill = 0
        if kill_pool:
            # 按 kill_ratio 抽杀棋样本常驻训练(至少 3000),持续强化"能杀就赢"
            rng = np.random.default_rng(args.seed + it)
            n_kill = min(len(kill_pool),
                         max(int(len(buffer) * args.kill_ratio), 3000))
            idx = rng.permutation(len(kill_pool))[:n_kill]
            train_set += [kill_pool[i] for i in idx]
        loss_p, loss_v = train_batch(net, opt, train_set, args, device)
        t_tr = time.time() - t0
        log = (f"[iter {it}/{args.iters}] 阶段{stage}(d{h_depth}"
               f"{f' {h_budget}s' if h_budget else ''}) 对弈 {t_sp:.0f}s "
               f"({len(exs)} 样本) | 训练 {t_tr:.0f}s | loss_p={loss_p:.4f} "
               f"loss_v={loss_v:.4f} | train={len(train_set)}"
               f"(game={len(buffer)} kill={n_kill})")
        if args.kill_refresh > 0 and it % args.kill_refresh == 0 and kill_pool:
            t0 = time.time()
            net.train()
            kp_, kv_ = train_batch(net, opt, kill_pool, args, device)
            log += f" | 杀棋精修 lp={kp_:.3f} lv={kv_:.3f} ({time.time() - t0:.0f}s)"
        if args.eval_every > 0 and it % args.eval_every == 0:
            net.eval()
            wins, draws, total = eval_vs_heur(net, config, device,
                                              args.eval_sims, h_depth,
                                              args.eval_games, args.seed + it,
                                              h_budget=h_budget)
            wr = wins / max(total, 1)
            log += f" | vs-heur(d{h_depth}): {wins}/{total} 胜率{wr:.0%}"
            if wr > best_wr:
                best_wr = wr
                net.save(best, step=it)
                log += " → 新 best"
            # 课程切换:阶段1与默认AI持平(或超时)后,换深度8对手
            if stage == 1 and (wr >= args.parity_threshold
                               or (args.stage1_max_iters
                                   and it >= args.stage1_max_iters)):
                stage = 2
                h_depth = args.stage2_depth
                h_budget = args.stage2_budget
                log += (f" -> 与d{args.stage1_depth}持平,切换对手 "
                        f"d{args.stage2_depth}({h_budget}s/步)")
        net.eval()
        net.save(latest, step=it)
        net.train()
        print(log, flush=True)
    print(f"完成 → latest: {latest} | best: {best} (最佳胜率 {best_wr:.0%})")


if __name__ == "__main__":
    main()
