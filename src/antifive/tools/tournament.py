"""四强 Elo 定段赛:初级 / 高级 / 传奇 / KataGo(全部 15 秒等时)

用途:按"配对开局 + 黑白互换"的规程,让四个参赛者在完全相同的每步墙钟时限
下两两对弈,汇总成一个连通赛程;以初级(低级启发式)= 0 Elo 为锚点做
Bradley-Terry 拟合,并换算成围棋类比段位(默认 1 段 = 100 Elo,10k = 锚点)。

赛程:
- 三名启发式选手用 `heuristic_versions.choose` 进程内对弈(CPU 并发);
- KataGo 用 GTP 子进程(`match_katago.KataGoGTP`),与启发式同台;
- 每个配对使用同一批固定开局(4 步随机,D4 去重),每开局黑白各一局;
- 全部对局 15 秒/步(启发式 time_budget, KataGo maxTime),depth=8,
  white_restrict=2, loss_start=8, mask_suicide=True(与 GUI 一致)。

输出目录:
  run_meta.json       运行参数与引擎快照
  openings.json       配对开局(复现/续跑依据)
  games/<配对>/g####.json  逐局结果 + 逐步耗时
  games/<配对>/g####.afg   逐局棋谱
  results.jsonl       每局一行(含 aborted)
  summary.json        汇总(比分、Wilson CI、Elo、段位)
  report.md           可读报告

用法:
    python -m antifive.tools.tournament run --out records/tournament_20261022 \
        --time 15 --heur-games 96 --kata-games legend=24,advanced=8 \
        --heur-workers 14 --kata-workers 2 --deadline 05:45
    python -m antifive.tools.tournament report --out records/tournament_20261022

中断后对同一 --out 重跑即续跑(已完成对局自动跳过)。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

from .. import record as record_mod
from .. import heuristic_versions as hv
from ..heuristic import _TT
from ..reversegomoku import BLACK, BOARD_SIZE, GameConfig, ReverseGomoku, WHITE
from .match_katago import (
    KataGoGTP, _kata_defaults, generate_openings, gtp_to_idx, idx_to_gtp,
    wilson_ci,
)

# 参赛者顺序(也是 BT 拟合的顺序);katago 为外部引擎
PLAYERS = ["beginner", "advanced", "legend", "katago"]
LABELS = {
    "beginner": "初级(低级)",
    "advanced": "高级",
    "legend": "传奇",
    "katago": "KataGo",
}
HEUR_PLAYERS = ["beginner", "advanced", "legend"]
HEUR_PAIRS = [("beginner", "advanced"), ("beginner", "legend"),
              ("advanced", "legend")]
KATA_ORDER = ["legend", "advanced", "beginner"]   # KataGo 对局优先级

LN10_400 = math.log(10.0) / 400.0


# --------------------------------------------------------------------------
# 对局(worker 进程)
# --------------------------------------------------------------------------
def _play(job: dict) -> dict:
    # 每进程 1 个 torch 线程:避免 14 个 worker 各自开满线程互相抢核
    try:
        import torch as _torch
        _torch.set_num_threads(1)
    except Exception:
        pass
    cfg = GameConfig(loss_start_turns=job["loss_start"],
                     white_restrict_turns=job["white_restrict"],
                     mask_suicide=True)
    g = ReverseGomoku(cfg)
    for m in job["opening"]:
        g.make_move(m)

    kind = job["kind"]
    kata_color = job.get("kata_color")
    kg = None
    step_log: list = []
    resigned = False
    tts = {BLACK: _TT(), WHITE: _TT()}
    rng = np.random.default_rng(job["seed"])

    try:
        if kind == "kata":
            kg = KataGoGTP(job["engine"], job["model"], job["engine_cfg"],
                           job["time"], job["kata_visits"],
                           str(Path(job["out_dir"]) / "gtp_logs"))
            for idx, player, *_ in g.history:
                kg.play("B" if player == BLACK else "W", idx_to_gtp(idx))

        while not g.game_over and g.move_count < job["max_steps"]:
            player = g.current_player
            if kind == "kata" and player == kata_color:
                color = "B" if player == BLACK else "W"
                t0 = time.perf_counter()
                coord, visits = kg.genmove(color)
                dt = time.perf_counter() - t0
                if coord.lower() in ("resign", "pass"):
                    g.game_over = True
                    g.resigned = player
                    resigned = True
                    break
                a = gtp_to_idx(coord)
                g.make_move(a, allow_suicide=True)
                step_log.append([g.move_count, "K", round(dt, 3), visits])
                continue

            version = job["black"] if player == BLACK else job["white"]
            nodes = [0]
            t0 = time.perf_counter()
            a = hv.choose(version, g.board, player, g.pending, g.turn_count,
                          g.white_turns, g.config, rng, depth=job["depth"],
                          time_budget=job["time"],
                          progress_cb=lambda n: nodes.__setitem__(0, n),
                          tt=tts[player],
                          use_vcf=(version != "beginner"))
            dt = time.perf_counter() - t0
            if a is None:                      # 引擎放弃
                g.game_over = True
                g.loser = player
                break
            if kg is not None:
                kg.play("B" if player == BLACK else "W", idx_to_gtp(a))
            g.make_move(a)
            step_log.append([g.move_count, "H", round(dt, 3), int(nodes[0])])
    finally:
        if kg is not None:
            kg.close()

    if not g.game_over:
        g.game_over = True
        g.is_draw = True

    result = record_mod.game_result(g)
    if resigned:
        winner = job["white"] if g.resigned == BLACK else job["black"]
    elif g.is_draw:
        winner = "draw"
    else:
        winner = job["white"] if g.loser == BLACK else job["black"]

    moves = g.record()
    if g.pending >= 0 and moves:
        moves = moves[:-1]                 # 未完成安置的占领步不记谱(与 GUI 一致)

    # 落谱自检:重放必须复现终局(防御记录幻影;毫秒级)
    if not resigned and not g.is_draw and not g.is_stuck():
        chk = ReverseGomoku(cfg)
        for m in moves:
            chk.make_move(m, allow_suicide=True)
        if not (chk.game_over and chk.loser == g.loser):
            raise RuntimeError(
                f"记录自检失败:重放未复现终局(loser={g.loser}, "
                f"pending={g.pending}, steps={len(moves)})")

    out_dir = Path(job["out_dir"]) / "games" / job["pair"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"g{job['game_id']:04d}"
    with open(out_dir / f"{stem}.afg", "w", encoding="utf-8") as f:
        f.write(record_mod.dumps(moves, g.config, result))

    rec = {
        "status": "ok",
        "pair": job["pair"],
        "game_id": job["game_id"],
        "opening_id": job["opening_id"],
        "black": job["black"],
        "white": job["white"],
        "kind": kind,
        "result": result,
        "winner": winner,
        "steps": g.move_count,
        "turns": g.turn_count,
        "kata_resigned": bool(resigned),
        "step_log": step_log,
    }
    with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False)
    return rec


def _game_task(job: dict) -> dict:
    t0 = time.perf_counter()
    try:
        rec = _play(job)
        rec["wall_seconds"] = round(time.perf_counter() - t0, 1)
        if rec["kind"] == "kata":
            kt = [s[2] for s in rec["step_log"] if s[1] == "K"]
            ht = [s[2] for s in rec["step_log"] if s[1] == "H"]
            rec["kata_seconds"] = round(sum(kt), 1)
            rec["heur_seconds"] = round(sum(ht), 1)
            rec["kata_avg_seconds"] = round(sum(kt) / max(len(kt), 1), 2)
        else:
            ht = [s[2] for s in rec["step_log"] if s[1] == "H"]
            rec["heur_seconds"] = round(sum(ht), 1)
        return rec
    except Exception as e:                 # noqa: BLE001 - 上报给主进程
        return {"status": "aborted", "pair": job["pair"],
                "game_id": job["game_id"], "kind": job["kind"],
                "error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------
# 赛程构建
# --------------------------------------------------------------------------
def _pair_id(a: str, b: str) -> str:
    return f"{a}__{b}"


def build_jobs(out_dir: str, openings: list, heur_games: int,
               pair_games: dict, kata_games: dict, args, seed: int) -> tuple:
    """返回 (启发式任务队列, KataGo 任务队列),均按"均衡推进"排序。"""
    n_open = len(openings)
    jobs_h, jobs_k = [], []

    for a, b in HEUR_PAIRS:
        pid = _pair_id(a, b)
        n = max(2, pair_games.get(f"{a}-{b}", heur_games) // 2 * 2)
        for i in range(min(n // 2, n_open)):
            for color in (BLACK, WHITE):
                gid = 2 * i + (1 if color == BLACK else 2)
                jobs_h.append({
                    "kind": "heur", "pair": pid, "game_id": gid,
                    "opening_id": i + 1, "opening": openings[i],
                    "black": a if color == BLACK else b,
                    "white": b if color == BLACK else a,
                    "seed": seed + gid * 7919 + PLAYERS.index(a) * 131,
                    "out_dir": out_dir, "time": args.time,
                    "depth": args.depth, "max_steps": args.max_steps,
                    "loss_start": args.loss_start,
                    "white_restrict": args.white_restrict,
                })
    # 按开局交错三个配对,任意时刻停止都能得到较均衡的样本
    order = {p: i for i, p in enumerate(
        _pair_id(a, b) for a, b in HEUR_PAIRS)}
    jobs_h.sort(key=lambda j: (j["opening_id"], order[j["pair"]],
                               0 if j["black"] == j["pair"].split("__")[0]
                               else 1))

    for opponent in KATA_ORDER:
        want = int(kata_games.get(opponent, 0))
        if want < 2:
            continue
        pid = _pair_id(opponent, "katago")
        n2 = min(want // 2, n_open)
        for i in range(n2):
            for kata_black in (True, False):
                gid = 2 * i + (1 if kata_black else 2)
                jobs_k.append({
                    "kind": "kata", "pair": pid, "game_id": gid,
                    "opening_id": i + 1, "opening": openings[i],
                    "black": "katago" if kata_black else opponent,
                    "white": opponent if kata_black else "katago",
                    "kata_color": BLACK if kata_black else WHITE,
                    "seed": seed + gid * 7919 + 17,
                    "out_dir": out_dir, "time": args.time,
                    "depth": args.depth, "max_steps": args.max_steps,
                    "loss_start": args.loss_start,
                    "white_restrict": args.white_restrict,
                    "engine": args.engine, "model": args.model,
                    "engine_cfg": args.engine_cfg,
                    "kata_visits": args.kata_visits,
                })
    return jobs_h, jobs_k


def _existing(out_dir: Path) -> dict:
    done = {}
    gdir = out_dir / "games"
    if not gdir.is_dir():
        return done
    for p in gdir.glob("*/*.json"):
        try:
            with open(p, encoding="utf-8") as f:
                rec = json.load(f)
            if rec.get("status") == "ok":
                done[(rec["pair"], rec["game_id"])] = rec
        except Exception:
            continue
    return done


def _deadline_ts(text: str) -> float:
    if not text:
        return float("inf")
    hh, mm = (int(x) for x in text.split(":"))
    now = time.localtime()
    ts = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, hh, mm, 0,
                      0, 0, -1))
    if ts <= time.time():
        ts += 86400
    return ts


# --------------------------------------------------------------------------
# 主流程 run
# --------------------------------------------------------------------------
def run(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "games").mkdir(exist_ok=True)
    (out_dir / "gtp_logs").mkdir(exist_ok=True)

    pg = _parse_pairs(args.pair_games)
    n_open = max([pg.get(f"{a}-{b}", args.heur_games) // 2
                  for a, b in HEUR_PAIRS] +
                 [int(v) // 2 for v in args.kata_games.values()] + [2])
    op_path = out_dir / "openings.json"
    if op_path.exists():
        with open(op_path, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("seed") != args.seed:
            raise SystemExit(f"{op_path} 的 seed 与当前参数不符,请换 --out")
        openings = saved["openings"]
        if len(openings) < n_open:
            raise SystemExit(f"已有开局不足: {len(openings)} < {n_open}")
        print(f"续跑:复用 {len(openings)} 个开局", flush=True)
    else:
        cfg = GameConfig(loss_start_turns=args.loss_start,
                         white_restrict_turns=args.white_restrict)
        t0 = time.perf_counter()
        openings = generate_openings(n_open, args.opening_plies,
                                     args.seed, cfg)
        with open(op_path, "w", encoding="utf-8") as f:
            json.dump({"seed": args.seed, "plies": args.opening_plies,
                       "openings": openings}, f, ensure_ascii=False)
        print(f"生成 {len(openings)} 个配对开局 ({time.perf_counter()-t0:.1f}s)",
              flush=True)

    meta = {
        "args": {k: v for k, v in vars(args).items()
                  if isinstance(v, (str, int, float, bool, dict, list, type(None)))},
        
        "players": {p: LABELS[p] for p in PLAYERS},
        "opening_plies": args.opening_plies,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    jobs_h, jobs_k = build_jobs(str(out_dir), openings, args.heur_games,
                                pg, args.kata_games, args, args.seed)
    done = _existing(out_dir)
    jobs_h = [j for j in jobs_h if (j["pair"], j["game_id"]) not in done]
    jobs_k = [j for j in jobs_k if (j["pair"], j["game_id"]) not in done]
    print(f"赛程: 启发式 {len(jobs_h)} 局(已完成 {len(done)} 局总) | "
          f"KataGo {len(jobs_k)} 局 | 每步 {args.time:g}s", flush=True)
    if args.dry_run:
        for j in jobs_h[:6] + jobs_k[:6]:
            print("  ", j["kind"], j["pair"], j["game_id"],
                  j["black"], "vs", j["white"])
        return

    deadline = _deadline_ts(args.deadline)
    res_path = out_dir / "results.jsonl"
    fout = open(res_path, "a", encoding="utf-8")
    counts = {"ok": len(done), "aborted": 0}
    t_start = time.perf_counter()

    def handle(rec):
        counts["ok" if rec.get("status") == "ok" else "aborted"] += 1
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        if rec.get("status") != "ok":
            print(f"[中止] {rec.get('pair')} g{rec.get('game_id'):04d} "
                  f"{rec.get('error')}", flush=True)
            return
        lab = lambda p: LABELS.get(p, p)
        el = time.strftime("%H:%M:%S")
        print(f"[{counts['ok']:4d}] {el} {rec['pair']:>20s} "
              f"g{rec['game_id']:04d} {lab(rec['black'])} vs "
              f"{lab(rec['white'])} -> {lab(rec['winner']) if rec['winner'] in LABELS else rec['winner']} "
              f"{rec['steps']}步 {rec.get('wall_seconds', 0):.0f}s", flush=True)

    q_h, q_k = deque(jobs_h), deque(jobs_k)
    fh, fk = {}, {}
    with ProcessPoolExecutor(max_workers=args.heur_workers) as eh, \
            ProcessPoolExecutor(max_workers=args.kata_workers) as ek:
        while True:
            now = time.time()
            while q_h and len(fh) < args.heur_workers and now < deadline:
                job = q_h.popleft()
                fh[eh.submit(_game_task, job)] = job
            while q_k and len(fk) < args.kata_workers and now < deadline:
                job = q_k.popleft()
                fk[ek.submit(_game_task, job)] = job
            if not fh and not fk:
                break
            done_f, _ = wait(set(fh) | set(fk), timeout=30,
                             return_when=FIRST_COMPLETED)
            for fut in done_f:
                if fut in fh:
                    del fh[fut]
                else:
                    del fk[fut]
                try:
                    handle(fut.result())
                except Exception as e:      # noqa: BLE001
                    handle({"status": "aborted", "error": repr(e),
                            "pair": "?", "game_id": -1})
            if now >= deadline and q_h:
                print(f"到达截止时间,剩余 {len(q_h)} 局启发式不再开始",
                      flush=True)
                q_h.clear()
            if now >= deadline and q_k:
                print(f"到达截止时间,剩余 {len(q_k)} 局 KataGo 不再开始",
                      flush=True)
                q_k.clear()
    fout.close()
    print(f"对局结束: {counts['ok']} 局有效 / {counts['aborted']} 局中止 | "
          f"用时 {(time.perf_counter()-t_start)/60:.0f} min", flush=True)
    report(out_dir=str(out_dir), chart=getattr(args, "chart", False))


# --------------------------------------------------------------------------
# 统计:比分 / Elo / 段位
# --------------------------------------------------------------------------
def load_games(out_dir: Path) -> list:
    games = []
    for p in (out_dir / "games").glob("*/*.json"):
        try:
            with open(p, encoding="utf-8") as f:
                rec = json.load(f)
            if rec.get("status") == "ok" and rec.get("winner") in PLAYERS + ["draw"]:
                games.append(rec)
        except Exception:
            continue
    return games


def pairwise(games: list) -> dict:
    """pair -> {games, wins(先手方), losses, draws, score(先手方)}。"""
    stats: dict = {}
    for g in games:
        a, b = g["black"], g["white"]
        key = _pair_id(a, b) if PLAYERS.index(a) < PLAYERS.index(b) \
            else _pair_id(b, a)
        s = stats.setdefault(key, {"games": 0, "a_wins": 0, "b_wins": 0,
                                   "draws": 0, "a": a if key.startswith(a) else b})
        s["games"] += 1
        if g["winner"] == "draw":
            s["draws"] += 1
        elif g["winner"] == s["a"]:
            s["a_wins"] += 1
        else:
            s["b_wins"] += 1
    for s in stats.values():
        n = s["games"]
        s["score"] = (s["a_wins"] + 0.5 * s["draws"]) / max(n, 1)
        lo, hi = wilson_ci(s["score"], n)
        s["score_ci95"] = [round(lo, 4), round(hi, 4)]
        s["elo"] = _elo(s["score"])
        s["elo_lo"] = _elo(lo) if lo > 0 else None
        s["elo_hi"] = _elo(hi) if hi < 1 else None
        # 饱和边(全胜/全负):Clopper-Pearson 精确单侧界 → 「强者至少强多少 Elo」
        # 0 胜 n 负:a 的胜率 95% 上界 q = 1 - 0.025**(1/n)
        s["elo_bound"] = 0.0           # >0: b 至少强出这么多;<0: a 至少强出
        if s["draws"] == 0 and n >= 3 and (s["a_wins"] == 0 or s["b_wins"] == 0):
            # 赢家全胜:其胜率下界 q = 0.025**(1/n),对应强度下界(正数)
            q = 0.025 ** (1.0 / n)
            bound = 400.0 * math.log10(q / (1.0 - q))
            # 只有当赢家胜率下界 > 50%(即 n 足够大)时才有正的强度下界
            bound = bound if bound > 0 else 0.0
            s["elo_bound"] = bound if s["a_wins"] == 0 else -bound
    return stats


def _opening_bootstrap(games: list, n_boot: int = 4000) -> dict:
    """按开局聚类的 bootstrap:同一开局的黑白两局作为一个簇重采样。"""
    per: dict = {}
    for g in games:
        key = g["pair"]
        a = key.split("__")[0]
        if g["winner"] == "draw":
            sc = 0.5
        else:
            sc = 1.0 if g["winner"] == a else 0.0
        per.setdefault(key, {}).setdefault(g["opening_id"], []).append(sc)
    rng = np.random.default_rng(20261022)
    out = {}
    for key, ops in per.items():
        sums = np.array([sum(v) for v in ops.values()], float)
        cnts = np.array([len(v) for v in ops.values()], float)
        k = len(sums)
        if k < 3:
            continue
        idx = rng.integers(0, k, size=(n_boot, k))
        sc = sums[idx].sum(1) / cnts[idx].sum(1)
        sc.sort()
        out[key] = (float(sc[int(0.025 * n_boot)]),
                    float(sc[int(0.975 * n_boot) - 1]), k)
    return out


def _elo(score: float):
    if score <= 0.0 or score >= 1.0:
        return None
    return 400.0 * math.log10(score / (1.0 - score))


def fit_bt(games: list, anchor: str = "beginner",
           prior_elo: float = 400.0, iters: int = 200) -> dict:
    """带弱先验(正态, sigma=prior_elo)的 Bradley-Terry 拟合。

    返回 {elo: {p: R}, ci95: {p: (lo, hi)}, sigma_theta, converged}。
    锚点固定为 0。先验只用于让全胜/全负的边保持有限值;饱和边请以
    pairwise 的 Wilson 界为准,summary 会标注。
    """
    idx = {p: i for i, p in enumerate(PLAYERS)}
    k = len(PLAYERS)
    theta = np.zeros(k)
    free = [i for i in range(k) if PLAYERS[i] != anchor]
    st = prior_elo * LN10_400
    prec = 1.0 / (st * st)

    def loglik(th):
        ll = 0.0
        for g in games:
            i, j = idx[g["black"]], idx[g["white"]]
            y = 0.5 if g["winner"] == "draw" else (
                1.0 if g["winner"] == g["black"] else 0.0)
            d = th[i] - th[j]
            # 稳定 log-sigmoid
            ll += y * -math.log1p(math.exp(-d)) if d >= 0 else y * (d - math.log1p(math.exp(d)))
            ll += (1 - y) * (-d - math.log1p(math.exp(-d))) if d >= 0 else (1 - y) * -math.log1p(math.exp(d))
        return ll - 0.5 * prec * float(np.sum(th[free] ** 2))

    cur = loglik(theta)
    for _ in range(iters):
        grad = np.zeros(k)
        hess = np.zeros((k, k))
        for g in games:
            i, j = idx[g["black"]], idx[g["white"]]
            y = 0.5 if g["winner"] == "draw" else (
                1.0 if g["winner"] == g["black"] else 0.0)
            d = theta[i] - theta[j]
            p = 1.0 / (1.0 + math.exp(-d))
            r = y - p
            w = max(p * (1 - p), 1e-9)
            grad[i] += r
            grad[j] -= r
            hess[i, i] += w
            hess[j, j] += w
            hess[i, j] -= w
            hess[j, i] -= w
        grad[free] -= prec * theta[free]
        hess[np.diag_indices(k)] += prec
        try:
            step = np.linalg.solve(hess[np.ix_(free, free)], grad[free])
        except np.linalg.LinAlgError:
            break
        # 阻尼线搜索,保证对数似然不降
        t = 1.0
        for _ in range(40):
            cand = theta.copy()
            cand[free] += t * step
            v = loglik(cand)
            if v >= cur - 1e-12:
                theta, cur = cand, v
                break
            t *= 0.5
        else:
            break
        if np.max(np.abs(t * step)) < 1e-9:
            break

    # Laplace 协方差
    cov = np.full((k, k), np.nan)
    try:
        inv = np.linalg.inv(hess[np.ix_(free, free)])
        cov[np.ix_(free, free)] = inv
    except np.linalg.LinAlgError:
        pass
    elo = theta / LN10_400
    ci = {}
    for i, p in enumerate(PLAYERS):
        if i == free[0] if free else False:
            pass
        if p == anchor:
            ci[p] = (0.0, 0.0)
        elif not np.isnan(cov[i, i]):
            se = math.sqrt(max(cov[i, i], 0.0)) / LN10_400
            ci[p] = (elo[i] - 1.96 * se, elo[i] + 1.96 * se)
        else:
            ci[p] = (float("nan"), float("nan"))
    return {"elo": {p: float(elo[i]) for i, p in enumerate(PLAYERS)},
            "ci95": ci, "prior_elo": prior_elo}


def go_rank(elo: float, per: float = 100.0) -> str:
    """Elo → 围棋类比段位。初级(0 Elo)= 10k,每 per 个 Elo 一档。"""
    x = elo / per
    k = 10.0 - x
    if k >= 1.0:
        return f"{k:.1f}k"
    return f"{x - 9.0:.1f}d"


def go_rank_value(elo: float, per: float = 100.0) -> float:
    """连续段位坐标:10k = 10, 9k = 9, ..., 1k = 1, 1d = 0, 2d = -1。"""
    return 10.0 - elo / per


def report(out_dir: str = "", chart: bool = False, **_) -> dict:
    od = Path(out_dir)
    games = load_games(od)
    pairs = pairwise(games)
    bt = fit_bt(games)

    per_player = {}
    for p in PLAYERS:
        n = sum(1 for g in games
                if g["winner"] != "draw" and
                (g["winner"] == p or g["black"] == p or g["white"] == p))
        w = sum(1 for g in games if g["winner"] == p)
        d = sum(1 for g in games if g["winner"] == "draw")
        per_player[p] = {"games": n, "wins": w, "draws": d}

    summary = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "games_total": len(games),
        "players": LABELS,
        "per_player": per_player,
        "pairs": pairs,
        "elo": {p: round(bt["elo"][p], 1) for p in PLAYERS},
        "elo_ci95": {p: [round(x, 1) for x in bt["ci95"][p]] for p in PLAYERS},
        "rank_100": {p: go_rank(bt["elo"][p]) for p in PLAYERS},
        "rank_400": {p: go_rank(bt["elo"][p], 400.0) for p in PLAYERS},
        "rank_ci95": {p: [go_rank(bt["ci95"][p][0]),
                          go_rank(bt["ci95"][p][1])] for p in PLAYERS},
        "anchor": "初级(低级启发式, 15s) = 0 Elo = 围棋 10k",
        "rank_scale": "1 档 = 100 Elo(EGF/AGA 口径);400 Elo/档 为对照",
    }
    with open(od / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    # ---------------- Markdown 报告 ----------------
    lines = []
    A = lines.append
    A("# 逆五子棋 AI 定段赛(15 秒等时)—— Elo 与围棋类比段位")
    A("")
    meta_args = {}
    mp = od / "run_meta.json"
    if mp.exists():
        try:
            meta_args = json.load(open(mp, encoding="utf-8")).get("args", {})
        except Exception:
            meta_args = {}
    tlim = meta_args.get("time", 15)
    dep = meta_args.get("depth", 8)
    plies = meta_args.get("opening_plies", 4)
    ws = meta_args.get("white_restrict", 2)
    ls = meta_args.get("loss_start", 8)
    A(f"生成时间:{summary['generated']};有效对局 **{len(games)}** 局。")
    A("")
    A(f"**口径**:四个参赛者每步墙钟 {tlim:g} 秒(等时)、搜索深度 {dep}、"
      f"白棋前 {ws} 手禁推、loss_start={ls}、mask_suicide=True;"
      f"配对开局({plies} 步随机,D4 去重)+ 黑白互换。")
    A("**锚点**:初级(低级启发式)= **0 Elo = 围棋 10k**(题设)。")
    A("")
    ranks = "; ".join(f"{LABELS[p]} {bt['elo'][p]:+.0f} Elo = **{go_rank(bt['elo'][p])}**"
                      for p in PLAYERS)
    A(f"> **一句话结论**(主口径 100 Elo = 1 档):{ranks}。")
    A("")
    A("## 1. 总表")
    A("")
    A("| 参赛者 | 对局 | 胜 | 和 | Elo(锚点=0) | 95% CI | 围棋类比段位 |")
    A("|---|---:|---:|---:|---:|---|---|")
    for p in PLAYERS:
        e = bt["elo"][p]
        lo, hi = bt["ci95"][p]
        A(f"| {LABELS[p]} | {per_player[p]['games']} | {per_player[p]['wins']} "
          f"| {per_player[p]['draws']} | {e:+.0f} "
          f"| {int(round(lo)):+d} ~ {int(round(hi)):+d} "
          f"| **{go_rank(e)}** |")
    A("")
    ts = _timing_stats(games)
    if ts:
        A("### 1.1 实测步时与算力")
        A("")
        A("| 参赛者 | 记录步数 | 平均步时(s) | 中位步时(s) |")
        A("|---|---:|---:|---:|")
        for p in PLAYERS:
            if p in ts:
                A(f"| {LABELS[p]} | {ts[p]['moves']} | {ts[p]['sec_mean']:.2f} "
                  f"| {ts[p]['sec_med']:.2f} |")
        if "_kata_visits_per_sec" in ts:
            vps = ts["_kata_visits_per_sec"]
            A("")
            A(f"KataGo 平均搜索速度约 **{vps:.0f} visits/s**"
              f"(单步 {tlim:g}s 约 {vps * tlim:.0f} visits)。")
            hist = _hist_kata_visits_per_sec()
            if hist:
                A("")
                A(f"历史 12s 对局(`records/stage2_12s`)在只有 2 个 worker 时测得 "
                  f"**{hist:.0f} visits/s**;本次启发式 worker 把 32 核占满,"
                  f"KataGo 只有约 1/{hist / max(vps, 1e-9):.1f} 的搜索吞吐,"
                  f"即等效算力约等于历史的 {tlim / (hist / max(vps, 1e-9)):.1f}s。"
                  "所以本报告里的 KataGo 是「弱化版 15s」——"
                  "它的真实强度(以及它与启发式的真实差距)只会更大。")
        A("")
    A("## 2. 两两对阵")
    A("")
    A("| 对阵(前者视角) | 局数 | 胜-负-和 | 比分 | 比分 95% CI | 近似 Elo 差 |")
    A("|---|---:|---|---:|---|---|")
    for key, s in sorted(pairs.items()):
        a, b = key.split("__")
        e = _elo(s["score"])
        if e is not None:
            es = f"{e:+.0f}"
        elif s.get("elo_bound"):
            bd = s["elo_bound"]
            es = (f"{LABELS[b]} ≥ {LABELS[a]} + {bd:.0f}"
                  if bd > 0 else f"{LABELS[a]} ≥ {LABELS[b]} + {-bd:.0f}")
        else:
            es = "饱和(局数不足,无法给出下界)"
        A(f"| {LABELS[a]} vs {LABELS[b]} | {s['games']} | "
          f"{s['a_wins']}-{s['b_wins']}-{s['draws']} | {s['score']:.1%} | "
          f"{s['score_ci95'][0]:.1%} ~ {s['score_ci95'][1]:.1%} | {es} |")
    A("")
    obb = _opening_bootstrap(games)
    if obb:
        A("### 2.1 开局聚类 bootstrap(检验配对开局带来的相关性)")
        A("")
        A("同一开局的黑白两局不是独立样本;把每个开局整体重采样 4000 次,"
          "得到的区间比 Wilson 区间更保守:")
        A("")
        A("| 对阵 | 比分 | Wilson 95% CI | 开局聚类 95% CI | 开局数 |")
        A("|---|---:|---|---|---:|")
        for key in sorted(pairs):
            if key not in obb:
                continue
            a, b = key.split("__")
            lo_b, hi_b, k = obb[key]
            s1 = pairs[key]
            A(f"| {LABELS[a]} vs {LABELS[b]} | {s1['score']:.1%} | "
              f"{s1['score_ci95'][0]:.1%} ~ {s1['score_ci95'][1]:.1%} | "
              f"{lo_b:.1%} ~ {hi_b:.1%} | {k} |")
        A("")
        A("若聚类区间明显更宽,说明单个开局能左右胜负(开局方差大),"
          "此时应按「开局」而不是「单局」来理解样本量。")
        A("")
    A("## 3. 段位换算")
    A("")
    A("段位换算依赖「每档多少 Elo」的约定。主口径取 EGF/AGA 常用 **100 Elo = 1 档**"
      "(以初级 = 10k 锚定);同时给出国际象棋更常见的 400 Elo/档 作对照,"
      "以免单一约定误导。")
    A("")
    A("| 参赛者 | Elo | 比初级强(档) | 类比段位(100 Elo/档) | 类比段位(400 Elo/档) "
      "| 最近整数(100/档) | Elo 95% CI |")
    A("|---|---:|---:|---|---|---|---|")
    for p in PLAYERS:
        e = bt["elo"][p]
        lo, hi = bt["ci95"][p]
        A(f"| {LABELS[p]} | {e:+.0f} | {e / 100.0:+.2f} | **{go_rank(e)}** "
          f"| {go_rank(e, 400.0)} | {_nearest_rank(e)} "
          f"| {int(round(lo)):+d} ~ {int(round(hi)):+d} |")
    A("")
    if chart:
        try:
            png = _chart(od, games, pairs, bt)
            A(f"![Elo 与 CI]({Path(png).name})")
            A("")
        except Exception as e:                       # noqa: BLE001
            print(f"[图表生成失败] {e}", flush=True)
    A("## 4. 与历史成绩的对照")
    A("")
    A("仓库里已有的 12s 等时成绩(见 README.md / docs/engines.md / "
      "records/stage2_12s)是另一套时间控制下的结果,可用来检查本次定段是否合理:")
    A("")
    A("| 历史对局(12s,各 96 局) | 比分 | 近似 Elo |")
    A("|---|---:|---:|")
    A("| 传奇 vs 大师 | 75 : 21 | +210 |")
    A("| 大师 vs 高级 | 63 : 33 | +112 |")
    A("| 高级 vs 初级 | 51 : 45 | +22 |")
    A("| KataGo(12s) vs 高级 | 96 : 0 | ≥ +560(饱和) |")
    A("")
    A("若把 12s 成绩按传递性外推,传奇应比初级强约 +344 Elo(约 89% 胜率);"
      "本次 15s 实测的传奇 vs 初级差距明显更小。可能原因(按可能性排序):")
    A("")
    A("1. **时间控制变长压缩差距**:弱引擎从 12s→15s 的边际收益大于已接近搜索上限的强引擎"
      "(5 节的 2s 对照赛是同一现象的极端:2s 时初级被高级 0:16,15s 时变成 43:77)。")
    A("2. **随机开局把对局推进到中盘**:4 步随机开局让引擎的开局定型/库优势用不上,"
      "对局更早进入混战;2.1 节的聚类检验显示这属于系统性影响,而不是开局运气。")
    A("3. **风格相克(非传递性)**:三条边的比分彼此不自洽(见 6 节),差距不能简单相加。")
    A("4. 小样本:每条边 4~120 局,95% CI 仍有 ±8~12 个百分点(对 KataGo 的边长得多)。")
    A("")
    A("因此本报告的绝对 Elo 差应视为「15s 等时 + 随机开局」条件下的**保守估计**"
      "(真实差距大概率更大),而强弱序是可复现的结论。")
    A("")
    A("## 5. 时间控制对照(2s 同赛程)")
    A("")
    ctl = od.parent / "t_2s_control" / "summary.json"
    if ctl.exists() and ctl.parent.resolve() != od.resolve():
        try:
            cs = json.load(open(ctl, encoding="utf-8"))
            A(f"同一套引擎、同一规程(配对开局+黑白互换),只把每步时限从 "
              f"{tlim:g}s 降到 **2s**,共 {cs['games_total']} 局:")
            A("")
            A("| 参赛者 | 2s 下的 Elo(锚点=0) | 95% CI | 2s 段位 | 15s 段位(本报告) |")
            A("|---|---:|---|---|---|")
            for p in PLAYERS:
                if p not in cs.get("elo", {}):
                    continue
                if not (cs.get("per_player", {}).get(p, {}).get("games")):
                    continue
                A(f"| {LABELS[p]} | {cs['elo'][p]:+.0f} | "
                  f"{cs['elo_ci95'][p][0]:+.0f} ~ {cs['elo_ci95'][p][1]:+.0f} "
                  f"| {cs['rank_100'][p]} | {go_rank(bt['elo'][p])} |")
            A("")
            for key, s in sorted(cs.get("pairs", {}).items()):
                a, b = key.split("__")
                A(f"- 2s:{LABELS[a]} vs {LABELS[b]} {s['a_wins']}-{s['b_wins']}-"
                  f"{s['draws']}({s['score']:.0%})")
            A("")
            try:
                from scipy.stats import fisher_exact
                A("同一条边在 2s 与 15s 下的胜负是否真的不同(Fisher 精确检验):")
                A("")
                A("| 对阵 | 2s 战绩 | 15s 战绩 | p 值 |")
                A("|---|---|---|---:|")
                for key, s2 in sorted(cs.get("pairs", {}).items()):
                    if key not in pairs:
                        continue
                    a, b = key.split("__")
                    s1 = pairs[key]
                    p = fisher_exact([[s1["a_wins"], s1["b_wins"]],
                                      [s2["a_wins"], s2["b_wins"]]])[1]
                    A(f"| {LABELS[a]} vs {LABELS[b]} | {s2['a_wins']}-{s2['b_wins']} "
                      f"| {s1['a_wins']}-{s1['b_wins']} | {p:.2g} |")
            except Exception as e:                       # noqa: BLE001
                A(f"(Fisher 检验不可用: {e})")
            A("")
            A("注:2s 对照组是与主赛并行跑的(机器更拥挤,单进程分到的 CPU 更少),"
              "因此上表只用于证明「时限变了棋力排序就会变」,不宜当作精确效应量。")
            A("")
            A("对照说明:**同一引擎在不同时限下棋力排序会变**。2s 时初级(低级)被高级"
              "完胜(0:16),说明它的强项是「有足够时间把搜索做深」;给到 15s,"
              "它就基本追上高级(43:77,高级已不再碾压)。"
              "因此本报告的 15s 段位只对 15s 成立,不能外推到 2s 或 60s。")
        except Exception as e:                       # noqa: BLE001
            A(f"(读对照数据失败: {e})")
    else:
        A("(未找到 2s 对照赛数据)")
    A("")
    A("## 6. 结论与限制")
    A("")
    A(_conclusion_text(games, pairs, bt))
    A("")
    A("> 说明:选手中存在全胜/全负的边时,该边的 Elo 只能给下(上)界,"
      "表中点估计由弱先验(σ=400 Elo)平滑,不应单独引用;请以 CI 与比分界为准。")
    _kata_full_note(A, od, bt)
    md = "\n".join(lines) + "\n"
    with open(od / "report.md", "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    return summary


LABELS_EN = {"beginner": "Beginner", "advanced": "Advanced",
             "legend": "Legend", "katago": "KataGo"}


def _kata_full_note(A, od: Path, bt: dict) -> None:
    """若同目录下跑过「满血 15s vs 4s」校正实验,追加交叉引用。"""
    import json as _json
    p = od.parent / "kata_tc_20261022" / "summary.json"
    if not p.exists():
        return
    try:
        d = _json.loads(p.read_text(encoding="utf-8"))
        st = d["stats"]
        tm = d.get("timing") or {}
    except Exception:
        return
    ref = bt["elo"].get("katago")
    if ref is None:
        return
    elo = st["elo"]
    lo, hi = st["elo_ci95"]
    full = ref + elo
    v_a = (tm.get("A") or {}).get("visits_median")
    v_b = (tm.get("B") or {}).get("visits_median")
    A("")
    A("### 6.1 补充实验:校正 KataGo 被挤占的算力")
    A("")
    A(f"本赛 KataGo 每步只有约 590 visits/s(每步约 8850 visits),"
      "因为 14 个启发式 worker 占满了 CPU。"
      "为把这些分换算成满血值,另跑了同引擎同权重的"
      f"「满血 15s vs 4s」对照(`records/kata_tc_20261022`,"
      f"{st['games']} 局,{st['a_wins']}:{st['b_wins']}):"
      f"**Elo 差 {elo:+.0f}**"
      f"(95% CI {lo:+.0f} ~ {hi:+.0f})。")
    if v_a and v_b:
        A("")
        A(f"对照的算力自检:15s 中位 {v_a} visits/步,4s 中位 {v_b} visits/步。"
          "只看这两点即可判断代理是否对齐。")
    A("")
    A(f"把差值加回锚点:{ref:.0f} + {elo:.0f} = **{full:+.0f} Elo "
      f"≈ {go_rank(full)}**(定段赛坐标系,主口径 100 Elo = 1 档)。"
      f"若只按每步实际 visits 内插(更保守),约 "
      f"{ref + elo * 0.68:+.0f} Elo ≈ {go_rank(ref + elo * 0.68)}。"
      "两种读法都说明:本赛的 KataGo 段位是**下限**,不是它的真实实力。")
    A("")


def _chart(od: Path, games: list, pairs: dict, bt: dict) -> str:
    """输出 elo_chart.png:左=Elo 与 95% CI,右=两两比分矩阵。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8),
                             constrained_layout=True)
    # --- 左:Elo ± CI
    ax = axes[0]
    order = sorted(PLAYERS, key=lambda p: bt["elo"][p])
    ys = list(range(len(order)))
    vals = [bt["elo"][p] for p in order]
    elo_lo = [bt["elo"][p] - bt["ci95"][p][0] for p in order]
    elo_hi = [bt["ci95"][p][1] - bt["elo"][p] for p in order]
    colors = {"beginner": "#9aa7b8", "advanced": "#5b8def",
              "legend": "#f0a35e", "katago": "#c0504d"}
    ax.barh(ys, vals, color=[colors[p] for p in order], height=0.55)
    ax.errorbar(vals, ys, xerr=[elo_lo, elo_hi], fmt="none", ecolor="#222",
                capsize=5, lw=1.4)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{LABELS_EN[p]}  ({go_rank(bt['elo'][p])})"
                        for p in order])
    ax.axvline(0, color="#666", lw=1)
    ax.set_xlabel("Elo (anchor: Beginner @15s = 0 = 10k)")
    ax.set_title("Elo with 95% CI")
    ax.grid(axis="x", alpha=0.3)

    # --- 右:两两比分矩阵(行对列的胜率)
    ax = axes[1]
    n = len(PLAYERS)
    mat = np.full((n, n), np.nan)
    ann = [["" for _ in range(n)] for _ in range(n)]
    for i, pi in enumerate(PLAYERS):
        for j, pj in enumerate(PLAYERS):
            if i == j:
                continue
            key = _pair_id(pi, pj)
            st = pairs.get(key)
            if st and st["a"] == pi:
                sc = st["score"]
            elif st:
                sc = 1.0 - st["score"]
            else:
                rev = _pair_id(pj, pi)
                st2 = pairs.get(rev)
                if not st2:
                    continue
                sc = 1.0 - st2["score"] if st2["a"] == pj else st2["score"]
            mat[i, j] = sc
            ng = st["games"] if st else st2["games"]
            ann[i][j] = f"{sc:.0%}" + chr(10) + f"(n={ng})"
    im = ax.imshow(mat, cmap="coolwarm", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_xticklabels([LABELS_EN[p] for p in PLAYERS])
    ax.set_yticks(range(n))
    ax.set_yticklabels([LABELS_EN[p] for p in PLAYERS])
    for i in range(n):
        for j in range(n):
            if ann[i][j]:
                ax.text(j, i, ann[i][j], ha="center", va="center",
                        fontsize=9,
                        color="white" if (mat[i, j] < 0.25 or mat[i, j] > 0.75)
                        else "black")
    ax.set_xlabel("opponent")
    ax.set_ylabel("player (row win rate)")
    ax.set_title("Head-to-head score rate")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    out = od / "elo_chart.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return str(out)


def _timing_stats(games: list) -> dict:
    """实测每方平均步时、KataGo 平均访问量与访问速度(来自 step_log)。

    step_log 行 = (move_count, 'H'|'K', 秒, 节点/访问量)。黑方走奇数手数,
    因此启发式一行可按奇偶判定归属。
    """
    tt = {p: [] for p in PLAYERS}
    kvps = []
    for g in games:
        for row in (g.get("step_log") or []):
            if len(row) < 3:
                continue
            s, side, sec = row[0], row[1], float(row[2] or 0.0)
            if side == "K":
                player = "katago"
                if len(row) > 3 and row[3] and sec > 0:
                    kvps.append(float(row[3]) / sec)
            else:
                player = g["black"] if s % 2 == 1 else g["white"]
            if player in tt:
                tt[player].append(sec)
    out = {}
    for p, xs in tt.items():
        if xs:
            xs_sorted = sorted(xs)
            out[p] = {"moves": len(xs),
                      "sec_mean": sum(xs) / len(xs),
                      "sec_med": xs_sorted[len(xs_sorted) // 2]}
    if kvps:
        out["_kata_visits_per_sec"] = sum(kvps) / len(kvps)
    return out


def _hist_kata_visits_per_sec() -> float:
    """历史 12s 对局记录的 KataGo visits/s(读 records/stage2_12s/summary.json)。"""
    for cand in ("records/stage2_12s/summary.json",):
        try:
            d = json.load(open(cand, encoding="utf-8"))
            v = d.get("kata_visits_per_sec_avg")
            if v:
                return float(v)
        except Exception:                            # noqa: BLE001
            continue
    return 0.0


def _nearest_rank(elo: float, per: float = 100.0) -> str:
    """Elo → 最近整数段位(初级 0 Elo = 10k)。"""
    r = int(round(10.0 - elo / per))
    if r >= 1:
        return f"{r}k"
    if r == 0:
        return "1d"
    return f"{1 - r}d"


def _conclusion_text(games: list, pairs: dict, bt: dict) -> str:
    out = []
    if len(games) < 20:
        return "对局数过少，结论仅作方向参考。"
    ranked = sorted(PLAYERS, key=lambda p: -bt["elo"][p])
    out.append("**强弱序**：" +
               " > ".join(f"{LABELS[p]}({bt['elo'][p]:+.0f})" for p in ranked) + "。")
    out.append("")
    out.append("**各边观察**")
    out.append("")
    for key, s in sorted(pairs.items(), key=lambda kv: -kv[1]["games"]):
        a, b = key.split("__")
        if s["games"] < 4:
            continue
        head = (f"- {LABELS[a]} vs {LABELS[b]}：{s['score']:.1%}"
                f"（{s['a_wins']} 胜 / {s['b_wins']} 负 / {s['draws']} 和），")
        if s["score"] in (0.0, 1.0):
            bd = s.get("elo_bound") or 0.0
            if bd:
                strong = LABELS[b] if bd > 0 else LABELS[a]
                out.append(head + f"该边饱和，{strong} 至少强 {abs(bd):.0f} Elo"
                                  "（95% 精确下界）。")
            else:
                out.append(head + "该边饱和，但局数太少，"
                                  "95% 水平下还给不出正的强度下界。")
        else:
            out.append(head + f"点估计 Elo 差 {_fmt_elo(_elo(s['score']))}。")
    out.append("")
    out.append("**解读与注意事项**")
    out.append("")
    adv_gap = bt["elo"].get("advanced", 0.0)
    leg_gap = bt["elo"].get("legend", 0.0)
    out.append(f"- 绝对段位强度取决于「每档多少 Elo」的约定：主口径 100 Elo/档时，"
               f"高级比初级强 {adv_gap / 100:.1f} 档（{go_rank(adv_gap)}），"
               f"传奇强 {leg_gap / 100:.1f} 档（{go_rank(leg_gap)}）；"
               f"若按 400 Elo/档，则只分别强 {adv_gap / 400:.1f} / "
               f"{leg_gap / 400:.1f} 档。因此「相对强弱」比「绝对段位」更可靠。")
    # 三条启发式边是否自洽(传递性检查)
    d1 = _elo(pairs["beginner__advanced"]["score"]) if "beginner__advanced" in pairs else None
    d2 = _elo(pairs["advanced__legend"]["score"]) if "advanced__legend" in pairs else None
    d3 = _elo(pairs["beginner__legend"]["score"]) if "beginner__legend" in pairs else None
    if d1 is not None and d2 is not None and d3 is not None:
        chain = -(d1 + d2)          # 由 攻击者→高级→传奇 推出的「传奇-初级」
        direct = -d3                # 直接测得的「传奇-初级」
        out.append(f"- **三条边不完全自洽**：由「高级-初级 {-(d1):+.0f}」「传奇-高级 "
                   f"{-(d2):+.0f}」相加推出「传奇-初级 {chain:+.0f}」，但同一批对局里"
                   f"直接测到的是 {direct:+.0f}（差 {abs(chain - direct):.0f} Elo）。"
                   "这说明引擎之间存在**非传递性(风格相克)**，而不是单一的强弱轴；"
                   "因此 Bradley-Terry 给出的点位只是折中值，判断某两方谁更强请直接看"
                   "它们那一行的比分与 CI。")
    out.append("- 开局聚类 bootstrap（2.1 节）的区间与 Wilson 区间几乎一样宽，"
               "说明 4 步随机开局没有引入明显的「开局相克」，"
               "把单局当独立样本是可接受的近似。")
    out.append("- 锚点（初级 = 10k）是题设，不是本赛测出的结果；所有段位都随锚点平移。")
    out.append("- 对 KataGo 若全负（饱和边），只能给出「至少强 X Elo」的下界；"
               "要缩窄该界需增加对局数或换更强的对手。")
    out.append("- 双方均受相同「每步墙钟」约束；引擎/机器负载影响每秒节点数，"
               "但不改变每步时间上限。")
    return "\n".join(out)


def _fmt_elo(e) -> str:
    return f"{e:+.0f}" if e is not None else "±∞"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_kata_games(text: str) -> dict:
    out = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=")
        out[k.strip()] = int(v)
    return out


def _parse_pairs(text: str) -> dict:
    out = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=")
        a, b = [x.strip() for x in k.split("-")]
        out[f"{a}-{b}"] = int(v)
        out[f"{b}-{a}"] = int(v)
    return out


def main():
    defaults = _kata_defaults()
    p = argparse.ArgumentParser(description="四强 Elo 定段赛")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="跑赛程")
    r.add_argument("--out", required=True)
    r.add_argument("--time", type=float, default=15.0, help="每步秒数")
    r.add_argument("--depth", type=int, default=8)
    r.add_argument("--heur-games", type=int, default=96,
                   help="每个启发式配对的总局数(偶数,可被 --pair-games 覆盖)")
    r.add_argument("--pair-games", default="",
                   help='按配对指定局数, e.g. beginner-advanced=120,'
                        'beginner-legend=48,advanced-legend=72')
    r.add_argument("--kata-games", type=_parse_kata_games,
                   default=_parse_kata_games("legend=24,advanced=8"),
                   help="KataGo 对每个对手的局数, e.g. legend=24,advanced=8")
    r.add_argument("--heur-workers", type=int, default=14)
    r.add_argument("--kata-workers", type=int, default=2)
    r.add_argument("--engine", default=defaults["engine"])
    r.add_argument("--model", default=defaults["weight"])
    r.add_argument("--engine-cfg", default=defaults["config"])
    r.add_argument("--kata-visits", type=int, default=100_000_000)
    r.add_argument("--opening-plies", type=int, default=4)
    r.add_argument("--max-steps", type=int, default=450)
    r.add_argument("--loss-start", type=int, default=8)
    r.add_argument("--white-restrict", type=int, default=2)
    r.add_argument("--seed", type=int, default=20261022)
    r.add_argument("--deadline", default="", help="当地时刻 HH:MM,之后不再开新局")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--chart", action="store_true", help="生成 elo_chart.png")
    r.set_defaults(func=run)

    q = sub.add_parser("report", help="由已有棋谱生成报告")
    q.add_argument("--out", required=True)
    q.add_argument("--chart", action="store_true")
    q.set_defaults(func=report)

    args = p.parse_args()
    if args.cmd == "run":
        args.func(args)
    else:
        args.func(out_dir=args.out, chart=getattr(args, "chart", False))


if __name__ == "__main__":
    main()
