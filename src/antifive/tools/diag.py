# 行为诊断:对训练好的模型做"危机意识"检查
import argparse

import numpy as np
import torch

from ..mcts import MCTS
from ..network import PolicyValueNet
from ..paths import models_dir
from ..reversegomoku import *


def build_black_4row():
    g = ReverseGomoku()  # loss_start_turns=8
    for a in [((7,5),None),((0,0),None),((7,6),None),((0,1),None),((7,7),None),((0,2),None),((7,8),None),((0,3),None),
              ((14,14),None),((13,13),None),((14,13),None),((13,14),None),((12,12),None),((11,11),None),((12,11),None),((11,12),None),
              ((10,10),None),((9,9),None),((10,9),None),((9,10),None)]:
        g.make_move(a)
    assert g.current_player == BLACK and g.turn_count >= 8
    return g


def build_white_4row_win():
    # 白 4 连 (5,1)..(5,4) + 无关白子 (8,8);黑回合(黑无连五),
    # 可通过 占领(8,8)→安置(5,5) 让白连五获胜
    g = ReverseGomoku(GameConfig(loss_start_turns=0))
    for a in [((0,0),None),((5,1),None),((0,1),None),((5,2),None),((0,2),None),((5,3),None),((0,3),None),((5,4),None),
              ((1,0),None),((8,8),None)]:
        g.make_move(a)
    assert g.current_player == BLACK and not g.game_over
    return g


def main():
    p = argparse.ArgumentParser(description="训练模型行为诊断(危机意识检查)")
    p.add_argument("--model", default=str(models_dir() / "best_model.pth"),
                   help="待诊断的 .pth 模型(默认 models/best_model.pth)")
    p.add_argument("--device", default="", help="cuda/cpu,空=自动")
    args = p.parse_args()
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    net = PolicyValueNet.load(args.model, device)
    print("模型加载 ok")

    # ---- 测试1: 一步自连五 ----
    g = build_black_4row()
    suicide = {7 * 15 + 4, 7 * 15 + 9}
    st = torch.from_numpy(g.encode_state()[None]).to(device)
    mk = torch.from_numpy(g.legal_mask()[None]).to(device)
    p, v = net.policy_value(st, mk)
    pn = p[0].cpu().numpy()
    s_mass = sum(pn[i] for i in suicide)
    print(f"[测试1] 自杀着先验质量: 合计 {s_mass * 100:.2f}% (均匀基准 ~0.9%), 局面价值 {float(v[0]):+.2f}")
    m = MCTS(net, device, sims=200, config=GameConfig())
    a = m.search(g)
    tag = " <== 自杀着!" if a in suicide else " (正常)"
    print(f"        MCTS 选择: {g.idx_to_move(a)[0]}{tag}")

    # ---- 测试2: 制胜战术(移子完成对方连五) ----
    g2 = build_white_4row_win()
    occ = 8 * 15 + 8
    st2 = torch.from_numpy(g2.encode_state()[None]).to(device)
    mk2 = torch.from_numpy(g2.legal_mask()[None]).to(device)
    p2, v2 = net.policy_value(st2, mk2)
    pn2 = p2[0].cpu().numpy()
    print(f"[测试2] 制胜战术先验: 占领(8,8) 概率 {pn2[occ] * 100:.2f}%, 局面价值 {float(v2[0]):+.2f}")
    m2 = MCTS(net, device, sims=200, config=GameConfig(loss_start_turns=0))
    a2 = m2.search(g2)
    print(f"        MCTS 第一手: {g2.idx_to_move(a2)[0]} {'<== 占领(8,8)!' if a2 == occ else ''}")
    if a2 == occ:
        g2.make_move(a2)
        a3 = m2.search(g2)  # 安置步
        print(f"        安置步选择: {g2.idx_to_move(a3)[0]}")

    # ---- 测试4: 预防性防守(活三→活四是否回避) ----
    # 黑活三 (7,5)(7,6)(7,7) + 散子 (7,12):若黑走 (7,4)/(7,8) 形成活四,
    # 白可 占领(7,12)→安置(7,9)/(7,8) 移子杀黑(路径 (7,11)(7,10)(7,9) 全空)
    g4 = ReverseGomoku(GameConfig(loss_start_turns=0))
    for a in [((7,5),None),((0,0),None),((7,6),None),((0,1),None),((7,7),None),((0,2),None),
              ((14,0),None),((0,3),None),((14,14),None),((1,4),None),
              ((13,0),None),((1,0),None),((13,14),None),((1,1),None),((7,12),None),((1,2),None)]:
        g4.make_move(a)
    assert g.current_player == BLACK and not g.game_over
    ext = {7 * 15 + 4, 7 * 15 + 8}   # (7,4),(7,8):形成活四的危险着
    st4 = torch.from_numpy(g4.encode_state()[None]).to(device)
    mk4 = torch.from_numpy(g4.legal_mask()[None]).to(device)
    p4, v4 = net.policy_value(st4, mk4)
    pn4 = p4[0].cpu().numpy()
    e_mass = sum(pn4[i] for i in ext)
    print(f"[测试4] 预防性防守: 活三延伸着先验 {e_mass * 100:.2f}%"
          f"(均匀基准 {(len(ext) / g4.legal_mask().sum()) * 100:.2f}%), 局面价值 {float(v4[0]):+.2f}")
    m4 = MCTS(net, device, sims=200, config=GameConfig(loss_start_turns=0))
    a4 = m4.search(g4)
    tag = " <== 危险延伸!" if a4 in ext else " (安全)"
    print(f"        MCTS 选择: {g4.idx_to_move(a4)[0]}{tag}")

    # ---- 测试5: 反应性解杀(堵杀棋点) ----
    # 黑活四 (7,5)..(7,8) + 散子 (7,12),黑回合;白子 (3,9) 可移到 (7,9) 堵延伸格,
    # 从而同时封死白两条杀棋路线(路线均经过 (7,9))。正确着:占领(3,9)→安置(7,9)
    g5 = ReverseGomoku(GameConfig(loss_start_turns=0))
    for a in [((7,5),None),((0,0),None),((7,6),None),((0,1),None),((7,7),None),((0,2),None),
              ((7,8),None),((0,3),None),((7,12),None),((3,9),None)]:
        g5.make_move(a)
    assert g5.current_player == BLACK and not g5.game_over
    defend_b1 = 3 * 15 + 9
    st5 = torch.from_numpy(g5.encode_state()[None]).to(device)
    mk5 = torch.from_numpy(g5.legal_mask()[None]).to(device)
    p5, v5 = net.policy_value(st5, mk5)
    pn5 = p5[0].cpu().numpy()
    print(f"[测试5] 解杀-堵杀棋点: 占领(3,9) 先验 {pn5[defend_b1] * 100:.2f}%, 局面价值 {float(v5[0]):+.2f}")
    m5 = MCTS(net, device, sims=200, config=GameConfig(loss_start_turns=0))
    a5 = m5.search(g5)
    tag = " <== 解杀!" if a5 == defend_b1 else ""
    print(f"        MCTS 选择: {g5.idx_to_move(a5)[0]}{tag}")

    # ---- 测试6: 反应性解杀(堵杀棋路线) ----
    # 同上,但白子 (3,10) 可移到 (7,10)——堵在散子 (7,12) 到延伸格的射线上
    g6 = ReverseGomoku(GameConfig(loss_start_turns=0))
    for a in [((7,5),None),((0,0),None),((7,6),None),((0,1),None),((7,7),None),((0,2),None),
              ((7,8),None),((0,3),None),((7,12),None),((3,10),None)]:
        g6.make_move(a)
    assert g6.current_player == BLACK and not g6.game_over
    defend6_b1 = 3 * 15 + 10
    st6 = torch.from_numpy(g6.encode_state()[None]).to(device)
    mk6 = torch.from_numpy(g6.legal_mask()[None]).to(device)
    p6, v6 = net.policy_value(st6, mk6)
    pn6 = p6[0].cpu().numpy()
    print(f"[测试6] 解杀-堵路线: 占领(3,10) 先验 {pn6[defend6_b1] * 100:.2f}%, 局面价值 {float(v6[0]):+.2f}")
    m6 = MCTS(net, device, sims=200, config=GameConfig(loss_start_turns=0))
    a6 = m6.search(g6)
    tag = " <== 解杀!" if a6 == defend6_b1 else ""
    print(f"        MCTS 选择: {g6.idx_to_move(a6)[0]}{tag}")

    # ---- 测试7: 自对弈统计(结局分布/杀棋方式) ----
    print("[测试7] 自对弈统计(6 局, sims=80)...")
    stats = {"moves": [], "by_self5": 0, "by_kill": 0, "draw": 0}
    for _ in range(6):
        game = ReverseGomoku()
        mcts = MCTS(net, device, sims=80, config=GameConfig())
        was_b2 = False
        while not game.game_over:
            was_b2 = game.pending >= 0
            game.make_move(mcts.search(game))
        stats["moves"].append(game.move_count)
        if game.is_draw:
            stats["draw"] += 1
        elif was_b2:            # 败着是"安置"步 → 对方移子完成的杀棋
            stats["by_kill"] += 1
        else:                   # 败着是落子/占领 → 自己连五
            stats["by_self5"] += 1
    print(f"        平均步数 {np.mean(stats['moves']):.0f}: 自杀负 {stats['by_self5']} 局, "
          f"移子杀 {stats['by_kill']} 局, 和棋 {stats['draw']} 局")

    # ---- 测试3: 危险格先验统计(随机局面抽样) ----
    rng = np.random.default_rng(7)
    total_mass = 0.0
    total_danger_cells = 0
    n_states = 0
    for _ in range(40):
        g3 = ReverseGomoku()
        for _ in range(rng.integers(8, 60)):
            mm = g3.legal_mask()
            L = np.nonzero(mm)[0]
            if len(L) == 0:
                break
            g3.make_move(int(rng.choice(L)))
        if g3.game_over:
            continue
        st3 = torch.from_numpy(g3.encode_state()[None]).to(device)
        mk3 = torch.from_numpy(g3.legal_mask()[None]).to(device)
        p3, _ = net.policy_value(st3, mk3)
        pn3 = p3[0].cpu().numpy()
        me = g3.current_player
        danger = g3.get_danger_map(me) == 1.0
        mask = g3.legal_mask()
        danger_legal = danger.reshape(-1) & mask
        if danger_legal.sum() == 0:
            continue
        n_states += 1
        total_mass += pn3[danger_legal].sum()
        total_danger_cells += danger_legal.sum()
    if n_states:
        uniform = total_danger_cells / mask.sum() if mask.sum() else 0
        print(f"[测试3] 危险格平均先验: {total_mass / n_states * 100:.2f}% / 局面"
              f"(该局危险格 {total_danger_cells / n_states:.0f} 个, 均匀基准 {uniform * 100:.2f}%)")


if __name__ == "__main__":
    main()
