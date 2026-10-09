"""KataGo 蒸馏训练:策略先验(D1) + 价值校正(D2)

读取 `build_distill_data.py` 产出的 `checkpoints/distill_pool.npz`,按对局
划分 train/val,训练:

- D1 列表式 softmax 排序:标量特征(14)+ 直线/斜线 4 格 base-4 模式权重
  (各 256),目标为 KataGo 候选访问分布;用于根候选先验混合;
- D2 价值线性回归(我方/对方对称特征 20 维)→ KataGo 根胜率价值,用于
  evaluate 校正。

写出包内 `antifive/distill_data.py`(纯数据模块,网页/Pyodide 可用)与
`checkpoints/distill_report.json`,并打印离线指标(top-1/top-3 命中、NLL、
价值符号准确率/MSE)。权重可用环境变量 `ANTIFIVE_PRIOR_W`/`ANTIFIVE_VALUE_W`
在推理端临时缩放,便于 A/B 选混合权重。

用法:
    python -m antifive.tools.train_distill
    python -m antifive.tools.train_distill --val-ratio 0.15 --prior-w 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from antifive import heuristic as h                      # noqa: E402
from antifive.reversegomoku import BLACK, GameConfig, ReverseGomoku  # noqa: E402

P_SCALAR = h._PRIOR_DIM
P_PAT = 256
P_DIM = P_SCALAR + 1 + 2 * P_PAT


def _load_pool(path: Path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _split_by_game(pool: dict, val_ratio: float, seed: int):
    games = np.unique(pool["game_ids"])
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    n_val = max(1, int(len(games) * val_ratio))
    val_games = set(int(g) for g in games[:n_val])
    idx = np.arange(len(pool["boards"]))
    val = np.array([i for i in idx if int(pool["game_ids"][i]) in val_games])
    train = np.array([i for i in idx if int(pool["game_ids"][i]) not in val_games])
    return train, val


def _legal_moves(pool: dict, i: int) -> np.ndarray:
    cfg = GameConfig()
    mask = ReverseGomoku.legal_mask_for(
        pool["boards"][i], int(pool["players"][i]), int(pool["pends"][i]),
        int(pool["wts"][i]), cfg, int(pool["tcs"][i]))
    return np.nonzero(mask)[0]


def _d1_dataset(pool: dict, indices) -> list:
    cfg = GameConfig()
    out = []
    for i in indices:
        legal = _legal_moves(pool, i)
        if not legal.size:
            continue
        sc, tk = h._prior_features(
            pool["boards"][i], int(pool["players"][i]), int(pool["pends"][i]),
            int(pool["tcs"][i]), int(pool["wts"][i]), cfg, legal)
        y = np.zeros(len(legal), dtype=np.float64)
        for j in range(pool["lmoves"].shape[1]):
            m = int(pool["lmoves"][i][j])
            if m < 0 or pool["lvisits"][i][j] <= 0:
                continue
            pos = int(np.searchsorted(legal, m))
            if pos < len(legal) and legal[pos] == m:
                y[pos] = float(pool["lvisits"][i][j])
        if y.sum() <= 0:
            continue
        out.append((sc, tk, y / y.sum()))
    return out


def _d1_scores(params, sc, tk):
    w = params[:P_SCALAR]
    b = params[P_SCALAR]
    ws = params[P_SCALAR + 1:P_SCALAR + 1 + P_PAT]
    wd = params[P_SCALAR + 1 + P_PAT:]
    s = sc @ w + b
    s = s + ws[tk[:, 0:4]].sum(axis=1) + wd[tk[:, 4:8]].sum(axis=1)
    return s


def _d1_loss_grad(params, data, reg=1e-4):
    g = np.zeros_like(params)
    total = 0.0
    for sc, tk, y in data:
        s = _d1_scores(params, sc, tk)
        s = s - s.max()
        p = np.exp(s)
        p /= p.sum()
        total += float(-np.sum(y * np.log(p + 1e-12)))
        d = p - y
        g[:P_SCALAR] += sc.T @ d
        g[P_SCALAR] += d.sum()
        np.add.at(g[P_SCALAR + 1:P_SCALAR + 1 + P_PAT],
                  tk[:, 0:4].reshape(-1), np.repeat(d, 4))
        np.add.at(g[P_SCALAR + 1 + P_PAT:],
                  tk[:, 4:8].reshape(-1), np.repeat(d, 4))
    total += reg * float(params[:P_SCALAR] @ params[:P_SCALAR]
                         + params[P_SCALAR + 1:] @ params[P_SCALAR + 1:])
    g[:P_SCALAR] += 2 * reg * params[:P_SCALAR]
    g[P_SCALAR + 1:] += 2 * reg * params[P_SCALAR + 1:]
    return total, g


def _d1_metrics(params, data):
    top1 = top3 = 0
    nll = 0.0
    for sc, tk, y in data:
        s = _d1_scores(params, sc, tk)
        order = np.argsort(s)[::-1]
        tgt = int(np.argmax(y))
        top1 += int(order[0] == tgt)
        top3 += int(tgt in order[:3])
        z = s - s.max()
        p = np.exp(z)
        p /= p.sum()
        nll += float(-np.log(p[tgt] + 1e-12))
    n = max(len(data), 1)
    return top1 / n, top3 / n, nll / n


def _d2_dataset(pool: dict, indices):
    cfg = GameConfig()
    X, Y = [], []
    for i in indices:
        board = pool["boards"][i]
        player = int(pool["players"][i])
        tc, wt = int(pool["tcs"][i]), int(pool["wts"][i])
        mb, mw, _, _, db, dw = h._board_features(board, wt, tc, cfg)
        danger_cnt = len(dw) if player == BLACK else len(db)
        danger_mine = len(db) if player == BLACK else len(dw)
        empty = int(np.count_nonzero(board == 0))
        adj_b, adj_w, cl_b, cl_w, ct_b, ct_w = h._positional_summary(board)
        f = h._value_features(player, mb, mw, danger_cnt, danger_mine, empty,
                              adj_b, adj_w, cl_b, cl_w, ct_b, ct_w)
        wr = 0.5
        for j in range(pool["lmoves"].shape[1]):
            if pool["lvisits"][i][j] > 0:
                wr = float(pool["lwr"][i][j])
                break
        X.append(f)
        Y.append((wr - 0.5) * 4.0)
    return np.asarray(X), np.asarray(Y)


def _ridge_fit(X, Y, lam=1.0):
    XtX = X.T @ X
    reg = lam * np.eye(X.shape[1])
    reg[0, 0] = 0.0                          # 特征 0 恒为 1(偏置),不正则
    W = np.linalg.solve(XtX + reg, X.T @ Y)
    return W


def _write_module(path: Path, W_SCALAR, B_PRIOR, W_STRAIGHT, W_DIAG,
                  PRIOR_W, W_VALUE, VALUE_W) -> None:
    def arr(a, fmt="{:.6g}"):
        return "[" + ", ".join(fmt.format(float(x)) for x in a) + "]"

    lines = [
        '"""KataGo 蒸馏权重(由 tools/train_distill.py 生成,勿手改)。',
        "",
        "D1 策略先验:prior = W_SCALAR·scalars + B_PRIOR + Σ W_STRAIGHT/W_DIAG[模式码];",
        "D2 价值校正:value = VALUE_W * (W_VALUE·feats + B_VALUE)。",
        '"""',
        "",
        "import numpy as np",
        "",
        f"SCALAR_DIM = {P_SCALAR}",
        f"PRIOR_W = {PRIOR_W:.6g}",
        f"VALUE_W = {VALUE_W:.6g}",
        f"W_SCALAR = np.array({arr(W_SCALAR)}, dtype=np.float64)",
        f"B_PRIOR = {B_PRIOR:.6g}",
        f"W_STRAIGHT = np.array({arr(W_STRAIGHT)}, dtype=np.float64)",
        f"W_DIAG = np.array({arr(W_DIAG)}, dtype=np.float64)",
        f"W_VALUE = np.array({arr(W_VALUE)}, dtype=np.float64)",
        "B_VALUE = 0.0",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main() -> None:
    p = argparse.ArgumentParser(description="KataGo 蒸馏训练(D1 先验 + D2 价值)")
    p.add_argument("--pool", default="checkpoints/distill_pool.npz")
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                        / "distill_data.py"))
    p.add_argument("--report", default="checkpoints/distill_report.json")
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--prior-w", type=float, default=20.0)
    p.add_argument("--value-w", type=float, default=1.0)
    p.add_argument("--maxiter", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    pool = _load_pool(Path(args.pool))
    n = len(pool["boards"])
    train_idx, val_idx = _split_by_game(pool, args.val_ratio, args.seed)
    print(f"局面 {n}:train {len(train_idx)} / val {len(val_idx)}", flush=True)

    t0 = time.perf_counter()
    train = _d1_dataset(pool, train_idx)
    val = _d1_dataset(pool, val_idx)
    print(f"D1 数据集:{len(train)}/{len(val)} ({time.perf_counter() - t0:.0f}s)",
          flush=True)
    x0 = np.zeros(P_DIM)
    res = minimize(_d1_loss_grad, x0, args=(train,), jac=True,
                   method="L-BFGS-B",
                   options={"maxiter": args.maxiter, "maxfun": args.maxiter})
    print(f"D1 训练完成:loss={res.fun:.4f} nit={res.nit} "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)
    tr1, tr3, trn = _d1_metrics(res.x, train)
    va1, va3, van = _d1_metrics(res.x, val)
    print(f"D1 top1: train {tr1:.1%} val {va1:.1%} | top3: train {tr3:.1%} "
          f"val {va3:.1%} | NLL val {van:.3f}", flush=True)

    Xtr, Ytr = _d2_dataset(pool, train_idx)
    Xva, Yva = _d2_dataset(pool, val_idx)
    Wv = _ridge_fit(Xtr, Ytr)
    pred = Xva @ Wv
    mse = float(np.mean((pred - Yva) ** 2))
    sign = float(np.mean(np.sign(pred) == np.sign(Yva)))
    print(f"D2 价值:val MSE={mse:.4f} 符号准确率={sign:.1%}", flush=True)

    _write_module(Path(args.out), res.x[:P_SCALAR], res.x[P_SCALAR],
                  res.x[P_SCALAR + 1:P_SCALAR + 1 + P_PAT],
                  res.x[P_SCALAR + 1 + P_PAT:], args.prior_w, Wv, args.value_w)
    report = {"positions": int(n), "train": int(len(train_idx)),
              "val": int(len(val_idx)),
              "d1_top1_train": tr1, "d1_top1_val": va1,
              "d1_top3_train": tr3, "d1_top3_val": va3, "d1_nll_val": van,
              "d2_mse_val": mse, "d2_sign_val": sign,
              "prior_w": args.prior_w, "value_w": args.value_w}
    Path(args.report).write_text(json.dumps(report, indent=1),
                                 encoding="utf-8")
    print(f"已写出 {args.out} 与 {args.report}")


if __name__ == "__main__":
    main()
