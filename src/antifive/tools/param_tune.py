"""启发式 AI 参数自博弈优化(局面制 + 深度分层 + 确定性适应度)

旧版"8 局自博弈胜率"适应度噪声极大(每候选用不同随机种子 → 代内 best 虚高、
稳定复核回落,基线 0.75 纯属抽样运气)。本版本重构为:

A. 确定性适应度:候选参数在**固定局面池**上做深度 `--depth` 的确定性搜索
   (无根噪声、独立置换表),取每个局面的搜索值;与深度 `--ref-depth` 参考
   搜索值(快照权重,建池时一次算好缓存)做符号一致性打分。同一批局面、
   同一确定性搜索 → 候选间完全可比,没有随机种子偏差;
B. 局面制评估:局面池一次性从 棋谱(afg_save) + 快棋自对弈 生成并缓存为
   npz(含杀棋/必守/持子等战术多样局面),参考值标签同存;候选评估不重算
   参考搜索;
C. 生效深度:候选评估默认 `--depth 4`(权重真正参与决策的深度),参考标签
   默认 `--ref-depth 5`;最终三方胜率验证(match_two)也在 `--depth` 进行。

冠军门槛:候选须在 train 局面池上超过当前冠军、且在 held-out verify 池上
不倒退才被采纳(防过拟合局面池)。参考权重启动时快照,主进程/worker 均不漂移。

用法:
    python -m antifive.tools.param_tune --group tactical --depth 4 --iters 20 --workers 8
    python -m antifive.tools.param_tune --verify               # 用 params.json 做三方胜率验证
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import time

import numpy as np

from .. import heuristic as h
from .. import record as record_mod
from ..paths import data_dir
from ..reversegomoku import BLACK, GameConfig, ReverseGomoku

# 参数组: name -> {param: (下限, 上限)}
# 默认值运行时从 heuristic 模块读取(getattr),保证与手改同步。
_PARAM_RANGES = {
    # ---- 战术/材料评估 ----
    "E_KILLABLE4":     (0.3, 2.0),      # 对方可杀 4 连
    "E_DEAD4":         (0.05, 0.8),     # 对方死 4 连
    "E_OPEN3":         (0.1, 1.0),      # 对方开放 3 连
    "E_JUMP3":         (0.05, 0.8),     # 对方跳三(一空档三连)
    "E_CLOSED3":       (0.02, 0.5),
    "E_DOUBLE_KILL":   (0.4, 2.5),      # 双杀:两处可杀四
    "E_KILL_PLUS3":    (0.1, 1.2),      # 可杀四 + 三连并存
    "E_DANGER":        (0.0, 0.1),      # 对方危险格杠杆
    "E_SELF_KILLABLE4": (0.3, 2.0),     # 我方可杀 4 连(负债)
    "E_SELF_DEAD4":    (0.03, 0.6),
    "E_SELF_OPEN3":    (0.05, 0.8),
    "E_SELF_JUMP3":    (0.02, 0.5),
    "E_SELF_CLOSED3":  (0.02, 0.4),
    "E_PENDING_KILL":  (0.5, 1.0),
    # ---- 位置/推中 ----
    "E_ADJ":           (0.0, 0.08),     # 聚集度
    "E_CENTER_OPP":    (0.1, 0.6),      # 对方棋子中心度
    "E_CENTER_SELF":   (0.0, 0.15),
    "E_CLUSTER":       (0.0, 0.12),     # 居中×聚集联合
    "O_FIGHT":         (0.5, 4.0),
    "O_PUSH_OPP":      (0.5, 5.0),
    "O_PUSH_CENTER":   (0.5, 8.0),
    "O_PUSH_SELF":     (0.0, 3.0),
    "O_SPREAD":        (0.0, 3.0),
    "O_MIDDLE":        (0.0, 3.0),
    # ---- 候选排序分 ----
    "O_BUILD4":        (40.0, 400.0),
    "O_BUILD3":        (10.0, 120.0),
    "O_BREAK4":        (-600.0, -60.0),
    "O_BREAK3":        (-200.0, -20.0),
    "O_OWN4":          (-800.0, -100.0),
    "O_OWN3":          (-150.0, -10.0),
    # ---- 搜索规模(整数,单独成组调优) ----
    "ROOT_K":          (2, 12),
    "INNER_K":         (2, 10),
}

_INT_KEYS = {"ROOT_K", "INNER_K"}

GROUPS = {
    "tactical": ["E_KILLABLE4", "E_DEAD4", "E_OPEN3", "E_JUMP3", "E_CLOSED3",
                 "E_DOUBLE_KILL", "E_KILL_PLUS3", "E_DANGER",
                 "E_SELF_KILLABLE4", "E_SELF_DEAD4", "E_SELF_OPEN3",
                 "E_SELF_JUMP3", "E_SELF_CLOSED3", "E_PENDING_KILL"],
    "positional": ["E_ADJ", "E_CENTER_OPP", "E_CENTER_SELF", "E_CLUSTER",
                   "O_FIGHT", "O_PUSH_OPP", "O_PUSH_CENTER", "O_PUSH_SELF",
                   "O_SPREAD", "O_MIDDLE"],
    "ordering": ["O_BUILD4", "O_BUILD3", "O_BREAK4", "O_BREAK3",
                 "O_OWN4", "O_OWN3"],
    "width": ["ROOT_K", "INNER_K"],
    "all": None,                      # 运行时展开(含 width)
}

VERIFY_SEED = 424242            # 跨代可比的最优参数评估种子
# 参考标签"决定性"阈值:低于它视为中性局面(0.5 分);建池时过滤掉过浅的噪声局面
REF_MIN = 0.4
TRAIN_RATIO = 0.8               # 局面池训练/验证划分


def group_keys(group: str) -> list:
    if group == "all":
        return [k for k in _PARAM_RANGES]
    return GROUPS[group]


def ranges_for(keys: list) -> dict:
    return {k: _PARAM_RANGES[k] for k in keys}


def defaults_for(keys: list) -> dict:
    """从 heuristic 模块读取当前值(与手改/params.json 同步)。"""
    return {k: (int(getattr(h, k)) if k in _INT_KEYS
                else float(getattr(h, k))) for k in keys}


# 强推中参数:模拟"把对方棋子往中间聚团、自己散边角"的用户战术
PUSHER_PARAMS = {
    "E_ADJ": 0.04, "E_CENTER_OPP": 2.0, "E_CENTER_SELF": 0.4,
    "E_CLUSTER": 0.06,
    "O_PUSH_OPP": 6.0, "O_PUSH_CENTER": 14.0, "O_PUSH_SELF": 0.2,
    "O_SPREAD": 3.0, "O_MIDDLE": 0.0, "O_FIGHT": 1.0,
}


def set_params(d: dict) -> None:
    for k, v in d.items():
        setattr(h, k, int(round(v)) if k in _INT_KEYS else v)


# --------------------------------------------------------------------------
# 局面池构建(一次性,缓存 npz)
# --------------------------------------------------------------------------
def _sample_moves(moves, cfg: GameConfig, out: list, lo: int = 24,
                  hi: int = 200, step: int = 4) -> None:
    """沿一条着法序列回放并采样中盘状态(不改变对局走势)。"""
    g = ReverseGomoku(cfg)
    for m in moves:
        if g.game_over:
            break
        mc = g.move_count
        if lo <= mc <= hi and g.turn_count >= cfg.loss_start_turns \
                and (mc - lo) % step == 0:
            out.append((g.board.copy(), g.current_player, g.pending,
                        g.turn_count, g.white_turns))
        if not g.legal_mask().any():
            break
        g.make_move(int(m))


def _collect_raw(cfg: GameConfig, rng) -> list:
    """从棋谱 + 快棋自对弈收集原始中盘状态。"""
    out: list = []
    for path in sorted(glob.glob(os.path.join("afg_save", "**", "*.afg"),
                                 recursive=True)):
        try:
            rec = record_mod.load_game(path)
        except Exception:
            continue
        _sample_moves(rec.moves, rec.config, out)
    for gi in range(40):                     # 快棋自对弈补充多样性
        moves, _ = h.play_game(cfg, np.random.default_rng(90000 + gi),
                               depth=2, max_steps=250)
        _sample_moves(moves, cfg, out)
    return out


def _build_pool(pool_size: int, cand_depth: int, ref_depth: int,
                ref_weights: dict, cfg, seed: int) -> dict:
    """构建局面池:原始状态 → 参考值标签 → 过滤"浅层迷惑/深层看清"局面 → 划分。
    只保留 |浅层值| 小 且 |深层值| 大 的战术局面——这些是权重调参的有效
    教学信号(浅层搜索纠结,深层搜索看清,权重把浅层推向正确方向)。"""
    rng = np.random.default_rng(seed)
    set_params(ref_weights)
    raw = _collect_raw(cfg, rng)
    seen, uniq = set(), []
    for s in raw:                            # 按棋盘字节去重
        k = s[0].tobytes()
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    raw = uniq
    ctx = h._SearchCtx(h._TT(), None)
    # 浅层值(候选深度,默认权重)与深层参考值
    shallow_v = np.zeros(len(raw), dtype=np.float32)
    ref_v = np.zeros(len(raw), dtype=np.float32)
    for i, (b, pl, pe, tc, wt) in enumerate(raw):
        shallow_v[i] = h._search(b, pl, pe, tc, wt, cfg, cand_depth,
                                 -1e18, 1e18, ctx)
        ref_v[i] = h._search(b, pl, pe, tc, wt, cfg, ref_depth,
                             -1e18, 1e18, ctx)
    # 过滤:深层有明确倾向 且 浅层迷惑(信号局面);浅层上限逐步放宽。
    # 若信号局面不足(新引擎浅层已能看清),退回"深层有倾向"的普通局面补足。
    solid = np.nonzero(np.abs(ref_v) >= REF_MIN)[0]
    shallow_max = 1.5
    confused = np.nonzero((np.abs(ref_v) >= REF_MIN)
                          & (np.abs(shallow_v) < shallow_max))[0]
    while len(confused) < pool_size and shallow_max < 3.0:
        shallow_max += 0.5
        confused = np.nonzero((np.abs(ref_v) >= REF_MIN)
                              & (np.abs(shallow_v) < shallow_max))[0]
    if len(confused) >= pool_size:
        idx = confused
    else:
        rest = np.setdiff1d(solid, confused)
        idx = np.concatenate([confused, rest]) if rest.size else confused
    if not idx.size:
        raise SystemExit(
            f"局面池为空:原始 {len(raw)} 个局面中无 |ref| ≥ {REF_MIN} 的局面")
    if len(idx) > pool_size:
        idx = rng.choice(idx, size=pool_size, replace=False)
    idx = np.sort(idx)
    rng.shuffle(idx)                          # 随机后按比例划分
    n_train = int(len(idx) * TRAIN_RATIO)
    split = idx[:n_train], idx[n_train:]
    pool = {"train": _pack([raw[i] for i in split[0]], ref_v[split[0]]),
            "verify": _pack([raw[i] for i in split[1]], ref_v[split[1]])}
    return pool


def _pack(states: list, refs: np.ndarray) -> tuple:
    """状态列表 → (boards, players, pends, tcs, wts, refs) 数组包。"""
    return (np.stack([s[0] for s in states]).astype(np.int8),
            np.array([s[1] for s in states], dtype=np.int8),
            np.array([s[2] for s in states], dtype=np.int16),
            np.array([s[3] for s in states], dtype=np.int16),
            np.array([s[4] for s in states], dtype=np.int16),
            refs.astype(np.float32))


def _save_pool(path: str, pool: dict, cand_depth: int, ref_depth: int) -> None:
    tb, tp, te, tt, tw, trv = pool["train"]
    vb, vp, ve, vt, vw, vrv = pool["verify"]
    np.savez(path, tb=tb, tp=tp, te=te, tt=tt, tw=tw, trv=trv,
             vb=vb, vp=vp, ve=ve, vt=vt, vw=vw, vrv=vrv,
             cand_depth=np.array([cand_depth]), ref_depth=np.array([ref_depth]))


def _load_pool(path: str) -> dict:
    z = np.load(path)
    return {"train": (z["tb"], z["tp"], z["te"], z["tt"], z["tw"], z["trv"]),
            "verify": (z["vb"], z["vp"], z["ve"], z["vt"], z["vw"], z["vrv"]),
            "cand_depth": int(z["cand_depth"][0]),
            "ref_depth": int(z["ref_depth"][0])}


def load_or_build_pool(args, ref_weights: dict) -> dict:
    if not args.rebuild and os.path.exists(args.pool_file):
        try:
            pool = _load_pool(args.pool_file)
            if pool["cand_depth"] != args.depth or pool["ref_depth"] != args.ref_depth:
                raise ValueError(
                    f"缓存深度({pool['cand_depth']}/{pool['ref_depth']})"
                    f"与当前({args.depth}/{args.ref_depth})不符,重建")
            n = len(pool["train"][5]) + len(pool["verify"][5])
            print(f"载入局面池缓存: {args.pool_file} ({n} 局)")
            return pool
        except Exception as e:
            print(f"缓存无效,重建: {e}")
    cfg = GameConfig()
    t0 = time.perf_counter()
    pool = _build_pool(args.pool_size, args.depth, args.ref_depth,
                       ref_weights, cfg, args.seed)
    _save_pool(args.pool_file, pool, args.depth, args.ref_depth)
    n = len(pool["train"][5]) + len(pool["verify"][5])
    print(f"局面池构建完成: {n} 局 "
          f"(train {len(pool['train'][5])} / verify {len(pool['verify'][5])}) "
          f"| 浅层深度 {args.depth} / 参考深度 {args.ref_depth} "
          f"| 用时 {time.perf_counter() - t0:.0f}s")
    return pool


# --------------------------------------------------------------------------
# 确定性适应度:候选深度搜索值 vs 参考标签的符号一致性
# --------------------------------------------------------------------------
def _candidate_vals(params: dict, pool_states, depth: int, ctx=None) -> np.ndarray:
    """候选参数在局面池上的确定性搜索值(当前行棋方视角)。
    pool_states: (boards, players, pends, tcs, wts, refs)。"""
    set_params(params)
    if ctx is None:
        ctx = h._SearchCtx(h._TT(), None)
    cfg = GameConfig()
    boards, players, pends, tcs, wts, _ = pool_states
    vals = np.empty(len(boards), dtype=np.float32)
    for i in range(len(boards)):
        vals[i] = h._search(boards[i], int(players[i]), int(pends[i]),
                            int(tcs[i]), int(wts[i]), cfg, depth,
                            -1e18, 1e18, ctx)
    return vals


def _value_fitness(vals: np.ndarray, ref_v: np.ndarray,
                   neutral: float = REF_MIN) -> float:
    """符号一致性:参考值中性 → 0.5;否则同号 1.0 / 异号 0.0。"""
    s = 0.0
    for v, rv in zip(vals, ref_v):
        if abs(float(rv)) < neutral:
            s += 0.5
        else:
            s += 1.0 if v * rv > 0 else 0.0
    return s / max(len(vals), 1)


def _worker_task(args) -> np.ndarray:
    params, pool_states, depth = args
    return _candidate_vals(params, pool_states, depth)


# --------------------------------------------------------------------------
# 胜率验证(深度 --depth 的三方自博弈)
# --------------------------------------------------------------------------
def play_one(params: dict, opp_params: dict, games: int, depth: int,
             seed: int, gi: int, time_budget: float = None) -> float:
    """一局:params(执 gi%2 侧) vs opp_params。返回 1/0.5/0。"""
    rng = np.random.default_rng(seed + gi * 7919)
    g = ReverseGomoku()
    test_is_black = gi % 2 == 0
    while not g.game_over and g.move_count < 450:
        is_me = (g.current_player == BLACK) == test_is_black
        set_params(params if is_me else opp_params)
        a = h.choose_heuristic_move(g.board, g.current_player, g.pending,
                                    g.turn_count, g.white_turns, g.config,
                                    rng, depth=depth, time_budget=time_budget)
        if a is None:
            g.game_over = True
            g.loser = g.current_player
            break
        g.make_move(a)
    if g.is_draw:
        return 0.5
    test_won = (g.loser != BLACK) == test_is_black
    return 1.0 if test_won else 0.0


def match_two(p_a: dict, p_b: dict, games: int, depth: int,
              seed: int, time_budget: float = None) -> tuple:
    """A vs B 自博弈,交替执黑白。返回 (A胜, 和, B胜)。"""
    rng = np.random.default_rng(seed)
    w = d = 0
    for gi in range(games):
        g = ReverseGomoku()
        a_is_black = gi % 2 == 0
        while not g.game_over and g.move_count < 450:
            is_a = (g.current_player == BLACK) == a_is_black
            set_params(p_a if is_a else p_b)
            a = h.choose_heuristic_move(g.board, g.current_player, g.pending,
                                        g.turn_count, g.white_turns,
                                        g.config, rng, depth=depth,
                                        time_budget=time_budget)
            if a is None:
                g.game_over = True
                g.loser = g.current_player
                break
            g.make_move(a)
        if g.is_draw:
            d += 1
        else:
            a_won = (g.loser != BLACK) == a_is_black
            w += a_won
    return w, d, games - w - d


def verify(best: dict, args, ref: dict) -> None:
    keys = group_keys(args.group)
    print("\n=== 三方验证(depth=%d, %d 局/组) ===" % (args.depth, args.games))
    w, d, l = match_two(ref, PUSHER_PARAMS, args.games, args.depth, 31)
    print(f"当前参数 vs 强推中 : {w}胜/{d}和/{l}负")
    w, d, l = match_two(best, PUSHER_PARAMS, args.games, args.depth, 31)
    print(f"最优参数 vs 强推中 : {w}胜/{d}和/{l}负")
    w, d, l = match_two(best, ref, args.games, args.depth, 37)
    print(f"最优参数 vs 当前   : {w}胜/{d}和/{l}负")
    print("\n=== 最优参数 patch(替换 heuristic.py 常量) ===")
    for k, v in best.items():
        print(f"{k} = {v:.3f}")


# --------------------------------------------------------------------------
# (1+λ) 进化策略 + 冠军门槛
# --------------------------------------------------------------------------
def optimize(args, pool: dict, ref: dict) -> dict:
    keys = group_keys(args.group)
    ranges = ranges_for(keys)
    rng = np.random.default_rng(args.seed)
    train_states = pool["train"]
    train_ref = pool["train"][5]
    verify_states = pool["verify"]
    verify_ref = pool["verify"][5]

    champion = ref.copy()
    champion_train = _value_fitness(
        _candidate_vals(champion, train_states, args.depth), train_ref)
    champion_verify = _value_fitness(
        _candidate_vals(champion, verify_states, args.depth), verify_ref)
    print(f"冠军基线: train={champion_train:.3f} verify={champion_verify:.3f}")

    sigma = 0.6
    t0 = time.perf_counter()
    with mp.Pool(args.workers) as mp_pool:
        for it in range(1, args.iters + 1):
            cands = []
            for _ in range(args.lam):
                c = champion.copy()
                for k in rng.choice(keys, size=int(
                        rng.integers(2, min(4, len(keys) + 1))),
                        replace=False):
                    lo, hi = ranges[k]
                    v = float(np.clip(c[k] * math.exp(rng.normal(0, sigma)),
                                      lo, hi))
                    c[k] = int(round(v)) if k in _INT_KEYS else v
                cands.append(c)
            cands.append(champion.copy())          # 父代保底
            tasks = [(c, train_states, args.depth) for c in cands]
            vals_list = mp_pool.map(_worker_task, tasks)
            fits = [_value_fitness(v, train_ref) for v in vals_list]
            bi = int(np.argmax(fits))
            cand_train = fits[bi]
            # 冠军门槛:train 超当前冠军,且 verify 不倒退(防过拟合局面池)
            cand_verify = _value_fitness(
                _candidate_vals(cands[bi], verify_states, args.depth,
                                h._SearchCtx(h._TT(), None)),
                verify_ref)
            accepted = (cand_train > champion_train + 1e-4
                        and cand_verify >= champion_verify - 1e-3)
            if accepted:
                champion = cands[bi].copy()
                champion_train, champion_verify = cand_train, cand_verify
            sigma = min(1.0, sigma * 1.08) if accepted \
                else max(0.12, sigma * 0.75)
            dt = time.perf_counter() - t0
            print(f"[iter {it}/{args.iters}] 代内best(train)={cand_train:.3f} "
                  f"{'★采纳' if accepted else '     '} | "
                  f"冠军 train={champion_train:.3f} verify={champion_verify:.3f} "
                  f"sigma={sigma:.2f} ({dt:.0f}s)", flush=True)
            print("   " + " ".join(f"{k}={champion[k]:.3f}" for k in keys),
                  flush=True)
            if champion_verify >= 0.95:
                print("已收敛(verify ≥ 0.95),提前结束")
                break
    return champion


def main():
    p = argparse.ArgumentParser(description="启发式 AI 参数优化(局面制适应度)")
    p.add_argument("--group", default="tactical",
                   choices=("tactical", "positional", "ordering", "width", "all"),
                   help="优化参数组(默认 tactical=战术/材料评估参数)")
    p.add_argument("--depth", type=int, default=4,
                   help="候选评估搜索深度(权重真正生效的深度,默认 4)")
    p.add_argument("--ref-depth", type=int, default=5,
                   help="参考标签搜索深度(默认 5,建池时一次算好)")
    p.add_argument("--pool-size", type=int, default=100,
                   help="局面池大小(train+verify,默认 100)")
    p.add_argument("--pool-file", default=str(data_dir() / "eval_pool.npz"),
                   help="局面池缓存文件(含参考标签,默认 data/eval_pool.npz)")
    p.add_argument("--rebuild", action="store_true", help="强制重建局面池")
    p.add_argument("--iters", type=int, default=20, help="进化代数")
    p.add_argument("--lam", type=int, default=5, help="每代子代数")
    p.add_argument("--workers", type=int, default=4, help="并行进程数")
    p.add_argument("--games", type=int, default=8, help="最终三方验证每局数")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verify", action="store_true",
                   help="只做三方胜率验证(用 params.json 对比当前默认)")
    p.add_argument("--out", default=str(data_dir() / "params.json"),
                   help="最优参数保存文件(默认 data/params.json)")
    args = p.parse_args()

    if not (1 <= args.depth <= 8) or not (1 <= args.ref_depth <= 8):
        p.error("--depth/--ref-depth 1-8")
    keys = group_keys(args.group)
    code_ref = defaults_for(keys)            # 代码常量默认(尚未加载 params.json)
    h.ensure_params()                        # 再加载 data/params.json
    ref = defaults_for(keys)                 # 当前生效参数(优化基准,永不漂移)

    if args.verify:
        with open(args.out, encoding="utf-8") as f:
            best = json.load(f)
        if ref != code_ref:
            print("检测到 data/params.json,验证对象 = 该文件参数 vs 代码常量默认")
        verify(best, args, code_ref)         # 对照基线:代码常量默认
        set_params(ref)
        return

    pool = load_or_build_pool(args, ref)
    best = optimize(args, pool, ref)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)
    print(f"已保存: {args.out}")
    set_params(ref)
    verify(best, args, ref)


if __name__ == "__main__":
    main()
