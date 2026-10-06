"""诊断:三个网络在"杀棋局面"上的策略/价值感知

杀棋局面取自真实棋谱(records 有杀棋/),用 tactics.kill_samples 标注"应当下的杀着",
检查网络策略 top-k 是否包含杀着、价值是否接近 +1。
用法:
    python -m antifive.tools.diag_net_kill --records "data/samples/*.afg" \
        --models models/warm_model.pth,models/best_model.pth
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import torch

from .. import record as record_mod
from ..network import PolicyValueNet
from ..reversegomoku import GameConfig, ReverseGomoku
from ..tactics import kill_samples


def collect_kill_positions(records_dir: str, max_pos: int = 300):
    paths = sorted(glob.glob(records_dir, recursive=True))
    out = []
    for p in paths:
        if len(out) >= max_pos:
            break
        try:
            rec = record_mod.load_game(p)
        except Exception:
            continue
        g = ReverseGomoku(rec.config)
        for m in rec.moves:
            if g.turn_count >= g.config.loss_start_turns:
                ks = kill_samples(g.board, g.current_player, g.pending,
                                  g.turn_count, g.white_turns, g.config)
                if ks:
                    out.append(ks)
                    if len(out) >= max_pos:
                        break
            g.make_move(m)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True,
                   help="逗号分隔的模型文件列表")
    p.add_argument("--records", default="records/有杀棋/*.afg")
    p.add_argument("--max-pos", type=int, default=300)
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = [m.split("\\")[-1] for m in args.models.split(",")]
    nets = [PolicyValueNet.load(m, device).eval() for m in args.models.split(",")]

    kill_groups = collect_kill_positions(args.records, args.max_pos)
    print(f"杀棋局面 {len(kill_groups)} 组(每组含 1+ 杀着)")
    print(f"{'模型':<18} {'top1命中':>8} {'top5命中':>8} {'top-k命中率':>10} "
          f"{'杀着均rank':>10} {'判杀值均值':>10}")
    for net, name in zip(nets, names):
        hit1 = hitk = 0
        ranks = []
        vals = []
        for ks_list in kill_groups:
            ks = ks_list[0]  # 每组取第一个杀着样本
            st = torch.from_numpy(ks.state[None]).to(device)
            with torch.no_grad():
                logits, v = net(st)
                v = float(v[0])
            legal = np.nonzero(ks.legal)[0] if ks.legal.dtype == bool else ks.legal
            # 杀着 = prob 非零位置(one-hot 目标)
            kill_acts = np.nonzero(ks.prob)[0]
            legal_logits = logits[0].detach().cpu().numpy()[legal]
            order = np.argsort(-legal_logits)
            ordered_legal = legal[order]
            rank = np.searchsorted(ordered_legal, kill_acts[0]) + 1
            ranks.append(rank)
            vals.append(v)
            if rank == 1:
                hit1 += 1
            if rank <= args.topk:
                hitk += 1
        n = len(kill_groups)
        print(f"{name:<18} {hit1:>6}/{n:<6} {hitk:>6}/{n:<6} {hitk / max(n, 1):>10.1%} "
              f"{np.mean(ranks):>10.1f} {np.mean(vals):>10.3f}")


if __name__ == "__main__":
    main()
