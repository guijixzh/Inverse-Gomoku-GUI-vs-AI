"""行为克隆 + 杀棋监督预热训练

从启发式 AI(或少量人类)生成的 .afg 棋谱学习基础策略与价值:
- 策略:棋谱实际着法 one-hot + 杀棋样本 one-hot(tactics.py);
- 价值:棋谱结果(黑胜/白胜/和棋)回填;杀棋样本恒为 +1。

预热后网络不再是"随机先验",AlphaZero 自对弈从合理策略出发,
大幅缩短前期收敛时间(治冷启动慢的问题)。

用法:
    python -m antifive.heuristic --games 1000 --out records   # 棋谱按结果/有杀棋分类存放
    python -m antifive.training.pretrain_bc --records records --out checkpoints/warm_model.pth
    python -m antifive.training.train --resume models/warm_model.pth

说明:records 可含子目录(黑胜/ 白胜/ 和棋/ 有杀棋/),递归扫描;
「有杀棋」为精选副本,同局重复计入 = 训练权重加倍(有意为之)。
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

from .. import record as record_mod
from ..network import PolicyValueNet
from ..reversegomoku import BLACK, ReverseGomoku, WHITE
from .selfplay import Example
from ..tactics import kill_samples
from .train import train_batch


def collect_examples(records_dir: str, use_tactics: bool = True,
                     max_records: int = 0) -> list:
    """递归读取棋谱目录(含子目录分类)→ 行为克隆样本 + 杀棋监督样本。"""
    paths = sorted(glob.glob(os.path.join(records_dir, "**", "*.afg"),
                             recursive=True))
    exs = []
    n_tactic = 0
    n_used = 0
    for path in paths:
        if max_records and n_used >= max_records:
            break
        try:
            rec = record_mod.load_game(path)
        except Exception:
            continue
        g = ReverseGomoku(rec.config)
        winner = {"黑胜": BLACK, "白胜": WHITE}.get(rec.result, 0)
        w_sgn = 1 if winner == BLACK else (-1 if winner == WHITE else 0)
        ok = True
        for m in rec.moves:
            mask = g.legal_mask()
            legal = np.nonzero(mask)[0]
            pos = np.searchsorted(legal, m)
            if pos >= len(legal) or legal[pos] != m:
                ok = False             # 棋谱与当前规则不符,整局丢弃
                break
            prob = np.zeros(len(legal), dtype=np.float32)
            prob[pos] = 1.0
            sgn = 1 if g.current_player == BLACK else -1
            val = 1.0 if sgn == w_sgn else (-1.0 if w_sgn else 0.0)
            if use_tactics and g.turn_count >= g.config.loss_start_turns:
                for ks in kill_samples(g.board, g.current_player, g.pending,
                                       g.turn_count, g.white_turns, g.config):
                    exs.append(Example(ks.state, ks.legal, ks.prob, ks.player, 1.0))
                    n_tactic += 1
            exs.append(Example(g.encode_state(), legal, prob, sgn, val))
            g.make_move(m)
        if ok:
            n_used += 1
    print(f"棋谱 {len(paths)} 个文件,采用 {n_used} 局 | 样本 {len(exs)}"
          f"(含杀棋监督 {n_tactic})")
    return exs


def main():
    p = argparse.ArgumentParser(description="行为克隆 + 杀棋监督预热")
    p.add_argument("--records", default="records", help="棋谱目录(.afg)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--res", type=int, default=4)
    p.add_argument("--out", default="checkpoints/warm_model.pth")
    p.add_argument("--max-records", type=int, default=0, help="最多采用局数(0=全部)")
    p.add_argument("--no-tactics", action="store_true", help="关闭杀棋监督样本")
    p.add_argument("--device", default="", help="cuda/cpu,空=自动")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.seed:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    exs = collect_examples(args.records, use_tactics=not args.no_tactics,
                           max_records=args.max_records)
    if not exs:
        print(f"!! 未在 {args.records} 找到可用棋谱,请先运行 "
              "python -m antifive.heuristic --games 1000 --out records")
        return
    net = PolicyValueNet(args.channels, args.res).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    for ep in range(1, args.epochs + 1):
        loss_p, loss_v = train_batch(net, opt, exs, args, device)
        print(f"[epoch {ep}/{args.epochs}] loss_p={loss_p:.4f} "
              f"loss_v={loss_v:.4f} ({len(exs)} 样本)")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    net.save(args.out, step=args.epochs)
    print(f"已保存预热模型: {args.out} → python -m antifive.training.train --resume {args.out}")


if __name__ == "__main__":
    main()
