"""KataGo 时间控制对比:满血 15s 与 4s 的直接 Elo 差。

背景:定段赛(``records/tournament_20261022``)里 KataGo 的 15s 被 14 个启发式
worker 挤到约 590 visits/s(历史 2 worker 无争抢时 2268 visits/s),等效算力只有
约 3.9s。``tournament.py`` 报告的 KataGo +588 Elo 因此是保守下界。本工具在机器
空闲时让同一个 KataGo 用两种时间预算直接对局,量出「满血 15s」比「4s」强多少
Elo,从而把 +588 换算成满血值。

用法::

    python -m antifive.tools.kata_tc_match run \
        --out records/kata_tc_20261022 --a-time 15 --b-time 4 \
        --n-openings 8 --workers 2 --deadline 08:45
    python -m antifive.tools.kata_tc_match report \
        --out records/kata_tc_20261022 --chart

约定:``A`` = 长时间预算方(满血),``B`` = 短时间预算方。Elo 差为正表示 A 更强。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import record as record_mod
from ..reversegomoku import BLACK, WHITE, GameConfig, ReverseGomoku
from .match_katago import KataGoGTP, _kata_defaults, gtp_to_idx, idx_to_gtp

LN10_400 = math.log(10.0) / 400.0
KATA_VPS_IDLE = 2268.0          # 历史无争抢吞吐(records/stage2_12s)


# --------------------------------------------------------------------------
# 单局(worker 进程)
# --------------------------------------------------------------------------
def _play(job: dict) -> dict:
    cfg = GameConfig(loss_start_turns=job["loss_start"],
                     white_restrict_turns=job["white_restrict"],
                     mask_suicide=True)
    g = ReverseGomoku(cfg)
    for m in job["opening"]:
        g.make_move(m)

    log_dir = Path(job["out_dir"]) / "gtp_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ka = kb = None
    step_log: list = []
    resigned = False
    try:
        ka = KataGoGTP(job["engine"], job["model"], job["engine_cfg"],
                       job["a_time"], job["kata_visits"],
                       str(log_dir))
        kb = KataGoGTP(job["engine"], job["model"], job["engine_cfg"],
                       job["b_time"], job["kata_visits"],
                       str(log_dir))
        for idx, player, *_ in g.history:
            coord = idx_to_gtp(idx)
            color = "B" if player == BLACK else "W"
            ka.play(color, coord)
            kb.play(color, coord)

        while not g.game_over and g.move_count < job["max_steps"]:
            player = g.current_player
            color = "B" if player == BLACK else "W"
            side = "A" if player == job["a_color"] else "B"
            kg = ka if side == "A" else kb
            t0 = time.perf_counter()
            coord, visits = kg.genmove(color)
            dt = time.perf_counter() - t0
            if coord.lower() in ("resign", "pass"):
                g.game_over = True
                g.resigned = player
                resigned = True
                break
            a = gtp_to_idx(coord)
            other = kb if side == "A" else ka
            other.play(color, idx_to_gtp(a))
            g.make_move(a, allow_suicide=True)
            step_log.append([g.move_count, side, round(dt, 3), visits])
    finally:
        for kg in (ka, kb):
            if kg is not None:
                kg.close()

    if not g.game_over:
        g.game_over = True
        g.is_draw = True

    result = record_mod.game_result(g)
    if resigned:
        winner = "B" if g.resigned == job["a_color"] else "A"
    elif g.is_draw:
        winner = "draw"
    else:
        winner = "A" if g.loser != job["a_color"] else "B"

    moves = g.record()
    if g.pending >= 0 and moves:
        moves = moves[:-1]

    # 落谱自检:重放必须复现终局
    if not resigned and not g.is_draw and not g.is_stuck():
        chk = ReverseGomoku(cfg)
        for m in moves:
            chk.make_move(m, allow_suicide=True)
        if not (chk.game_over and chk.loser == g.loser):
            raise RuntimeError(
                f"记录自检失败:重放未复现终局(loser={g.loser}, steps={len(moves)})")

    out_dir = Path(job["out_dir"]) / "games"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{job['pair']}_g{job['game_id']:04d}"
    with open(out_dir / f"{stem}.afg", "w", encoding="utf-8") as f:
        f.write(record_mod.dumps(moves, g.config, result))

    rec = {
        "status": "ok",
        "pair": job["pair"],
        "game_id": job["game_id"],
        "opening_id": job["opening_id"],
        "a_color": "black" if job["a_color"] == BLACK else "white",
        "a_time": job["a_time"],
        "b_time": job["b_time"],
        "result": result,
        "winner": winner,
        "steps": g.move_count,
        "resigned": bool(resigned),
        "step_log": step_log,
    }
    with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False)
    return rec


def _task(job: dict) -> dict:
    t0 = time.perf_counter()
    try:
        rec = _play(job)
        rec["wall_seconds"] = round(time.perf_counter() - t0, 1)
        for side in ("A", "B"):
            ts = [s[2] for s in rec["step_log"] if s[1] == side]
            vs = [s[3] for s in rec["step_log"] if s[1] == side and s[3]]
            rec[f"{side}_seconds"] = round(sum(ts), 1)
            rec[f"{side}_avg_seconds"] = round(sum(ts) / max(len(ts), 1), 2)
            rec[f"{side}_visits_avg"] = (
                round(sum(vs) / len(vs), 1) if vs else None)
        return rec
    except Exception as e:                 # noqa: BLE001 - 上报给主进程
        return {"status": "aborted", "pair": job["pair"],
                "game_id": job["game_id"],
                "error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------------------
# 赛程
# --------------------------------------------------------------------------
def _deadline_ts(text: str) -> float:
    if not text:
        return float("inf")
    hh, mm = text.split(":")[:2]
    now = datetime.now()
    ts = now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
    if ts <= now:
        return float("inf")
    return ts.timestamp()


def _load_openings(path: str, n: int) -> tuple:
    p = Path(path)
    if p.exists():
        d = json.loads(p.read_text("utf-8"))
        opens = d["openings"][:n]
        return opens, int(d.get("plies", len(opens[0]) if opens else 4))
    raise SystemExit(f"开局文件不存在: {path}")


def build_jobs(out_dir: str, openings: list, a_time: float, b_time: float,
               args) -> list:
    jobs = []
    for i, opening in enumerate(openings):
        for a_black in (True, False):
            gid = 2 * i + (1 if a_black else 2)
            jobs.append({
                "pair": f"a{int(a_time)}s__b{int(b_time)}s",
                "game_id": gid, "opening_id": i + 1, "opening": opening,
                "a_color": BLACK if a_black else WHITE,
                "a_time": a_time, "b_time": b_time,
                "out_dir": out_dir,
                "engine": args.engine, "model": args.model,
                "engine_cfg": args.engine_cfg,
                "kata_visits": args.kata_visits,
                "max_steps": args.max_steps,
                "loss_start": args.loss_start,
                "white_restrict": args.white_restrict,
            })
    return jobs


def _existing(out_dir: Path) -> dict:
    done = {}
    for p in sorted((out_dir / "games").glob("*.json")):
        try:
            d = json.loads(p.read_text("utf-8"))
        except Exception:                   # noqa: BLE001 - 半截文件重跑
            continue
        if d.get("status") == "ok":
            done[(d["opening_id"], d["a_color"])] = d
    return done


# --------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------
def _wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - s) / d), min(1.0, (c + s) / d)


def _elo_from_p(p: float) -> float:
    q = min(max(p, 1e-6), 1 - 1e-6)
    return 400.0 * math.log10(q / (1.0 - q))


def stats(games: list) -> dict:
    n = a = b = dr = 0
    for g in games:
        n += 1
        if g["winner"] == "A":
            a += 1
        elif g["winner"] == "B":
            b += 1
        else:
            dr += 1
    dec = a + b
    p = a / dec if dec else 0.5
    lo, hi = _wilson(a, dec) if dec else (0.0, 1.0)
    out = {"games": n, "a_wins": a, "b_wins": b, "draws": dr,
           "score_a": round(p, 4),
           "score_ci95": [round(lo, 4), round(hi, 4)],
           "elo": round(_elo_from_p(p), 1),
           "elo_ci95": [round(_elo_from_p(lo), 1), round(_elo_from_p(hi), 1)]}
    # 饱和(全胜/全负)时给 CP 精确界:p_L 是 A 胜率的 95% 单侧下界。
    # n 小时 p_L < 0.5,Elo 界为负 —— 此时"全胜"其实还证不出任何正优势。
    if dec and (a == 0 or b == 0):
        p_l = 0.025 ** (1.0 / dec)
        out["elo_bound"] = round(_elo_from_p(p_l if b == 0 else 1.0 - p_l), 1)
    out["elo_saturated"] = bool(dec and (a == 0 or b == 0))
    # 饱和时点估计无意义,给一个 Laplace 平滑值供参考(偏保守)
    if dec:
        out["elo_laplace"] = round(
            _elo_from_p((a + 0.5) / (dec + 1.0)), 1)
    return out


def timing(games: list) -> dict:
    out = {}
    for side in ("A", "B"):
        ts, nodes, vps = [], [], []
        for g in games:
            for step in g["step_log"]:
                if step[1] != side:
                    continue
                ts.append(step[2])
                if step[3]:
                    nodes.append(step[3])
                    if step[2] > 0:
                        vps.append(step[3] / step[2])
        # 用中位数:个别着法的 analyze 访问量会异常大(树复用/forced move),
        # 平均值会被单个离群值带偏
        out[side] = {
            "moves": len(ts),
            "avg_seconds": round(sum(ts) / max(len(ts), 1), 2),
            "median_seconds": round(float(np.median(ts)), 2) if ts else None,
            "visits_median": round(float(np.median(nodes))) if nodes else None,
            "vps_median": round(float(np.median(vps))) if vps else None,
            "vps_max": round(max(vps)) if vps else None,
        }
    return out


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def run(args) -> int:
    out_dir = Path(args.out)
    (out_dir / "games").mkdir(parents=True, exist_ok=True)
    openings, plies = _load_openings(args.openings, args.n_openings)
    jobs = build_jobs(str(out_dir), openings, args.a_time, args.b_time, args)

    done = _existing(out_dir)
    todo = [j for j in jobs
            if (j["opening_id"], "black" if j["a_color"] == BLACK else "white")
            not in done]
    meta = {
        "out": str(out_dir), "a_time": args.a_time, "b_time": args.b_time,
        "openings": str(args.openings), "opening_plies": plies,
        "n_openings": len(openings), "games_total": len(jobs),
        "engine": args.engine, "model": args.model,
        "engine_cfg": args.engine_cfg, "kata_visits": args.kata_visits,
        "workers": args.workers, "deadline": args.deadline,
        "loss_start": args.loss_start, "white_restrict": args.white_restrict,
        "max_steps": args.max_steps, "started": datetime.now().isoformat(" "),
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    print(f"开局 {len(openings)} 个 x2 手合 = {len(jobs)} 局;"
          f" 已完成 {len(done)}, 待下 {len(todo)};"
          f" A={args.a_time}s(满血) B={args.b_time}s;"
          f" workers={args.workers}", flush=True)

    deadline = _deadline_ts(args.deadline)
    t_start = time.time()
    results = list(done.values())
    n_ok = n_bad = 0

    def _note(rec: dict) -> None:
        nonlocal n_ok, n_bad
        if rec.get("status") != "ok":
            n_bad += 1
            print(f"  ! 失败 {rec.get('pair')} g{rec.get('game_id')}: "
                  f"{rec.get('error')}", flush=True)
            return
        n_ok += 1
        results.append(rec)
        st = stats(results)
        elo_txt = (f"{st['elo_bound']:+.0f}bound" if st.get("elo_saturated")
                   else f"{st['elo']:+.0f}")
        print(f"  [{n_ok:3d}/{len(todo)}] g{rec['game_id']:03d} "
              f"(开局{rec['opening_id']}, A={rec['a_color']}) "
              f"{rec['winner']:>4} {rec['steps']:3d}步 "
              f"{rec['wall_seconds']:6.0f}s | A:{st['a_wins']} "
              f"B:{st['b_wins']} 胜率 {st['score_a']:.3f} "
              f"Elo {elo_txt}", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        queue = list(todo)
        while True:
            while queue and len(futs) < args.workers and time.time() < deadline:
                j = queue.pop(0)
                futs[ex.submit(_task, j)] = j
            if not futs:
                break
            fut = next(as_completed(list(futs)))
            futs.pop(fut, None)
            try:
                rec = fut.result()
            except Exception as e:          # noqa: BLE001 - 进程崩溃
                n_bad += 1
                print(f"  ! worker 异常: {type(e).__name__}: {e}", flush=True)
            else:
                _note(rec)

    st = stats(results)
    sm = {"generated": datetime.now().isoformat(" "),
          "elapsed_seconds": round(time.time() - t_start, 1),
          "meta": meta, "stats": st, "timing": timing(results)}
    (out_dir / "summary.json").write_text(
        json.dumps(sm, ensure_ascii=False, indent=2), "utf-8")
    print(f"\n完成 {n_ok} 局, 失败 {n_bad} 局, 用时 "
          f"{(time.time() - t_start) / 60:.1f} min", flush=True)
    print(f"A({args.a_time}s) {st['a_wins']} : {st['b_wins']} B({args.b_time}s)"
          f"  A 胜率 {st['score_a']:.3f}  Elo {st['elo']:+.0f}"
          f" CI95 {st['elo_ci95'][0]:+.0f} ~ {st['elo_ci95'][1]:+.0f}",
          flush=True)
    return 0


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def load(out_dir: Path) -> list:
    games = []
    for p in sorted((out_dir / "games").glob("*.json")):
        d = json.loads(p.read_text("utf-8"))
        if d.get("status") == "ok":
            games.append(d)
    return games


def report(out_dir: Path, chart: bool = False) -> int:
    games = load(out_dir)
    if not games:
        raise SystemExit("没有可用对局")
    meta = json.loads((out_dir / "run_meta.json").read_text("utf-8"))
    st = stats(games)
    tm = timing(games)
    a_t, b_t = meta["a_time"], meta["b_time"]

    # 定段赛锚点(KataGo 在挤占下 +588,等效 b_time 左右)
    tz = Path("records/tournament_20261022/summary.json")
    kata_ref = None
    if tz.exists():
        try:
            kata_ref = json.loads(tz.read_text("utf-8"))["elo"]["katago"]
        except Exception:                   # noqa: BLE001
            kata_ref = None
    # 饱和时点估计被钳到 ±2400,没有意义:改用 CP 界(若界为正)当"至少多少"
    sat = bool(st.get("elo_saturated"))
    bnd = st.get("elo_bound")
    delta_pt = (bnd if (sat and bnd and bnd > 0) else st["elo"])
    delta_lo = delta_pt if sat else st["elo_ci95"][0]
    delta_hi = delta_pt if sat else st["elo_ci95"][1]
    full_elo = (kata_ref + delta_pt) if kata_ref is not None else None
    full_lo = (kata_ref + delta_lo) if kata_ref is not None else None
    full_hi = (kata_ref + delta_hi) if kata_ref is not None else None

    # 定段赛里 KataGo 的真实算力(~8850 visits/步)通常落在 A、B 之间:
    # 按「Elo 与 log(visits) 成正比」内插,得到只修正到定段赛算力的增量
    V_TOUR = 8850.0
    va, vb = tm["A"]["visits_median"], tm["B"]["visits_median"]
    frac = None
    if va and vb and va > V_TOUR > vb > 0:
        frac = math.log(va / V_TOUR) / math.log(va / vb)
    tour_elo = (kata_ref + abs(delta_pt) * frac
                if (frac is not None and kata_ref is not None) else None)

    def rank(e: float) -> str:
        r = int(round(10.0 - e / 100.0))
        if r >= 1:
            return f"{r}k"
        if r == 0:
            return "1d"
        return f"{1 - r}d"

    lines = []
    A = lines.append
    A(f"# KataGo 时间控制对比:满血 {a_t:g}s vs {b_t:g}s")
    A("")
    A(f"- 生成时间:{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    A(f"- 对局:{st['games']} 局({meta['n_openings']} 个 {meta['opening_plies']} 步随机开局"
      f" × 黑白各一),与定段赛同一批开局")
    A(f"- 引擎:`{meta['engine']}`")
    A(f"- 权重:`{meta['model']}`")
    A(f"- 时限:A(满血)={a_t:g}s/步,B={b_t:g}s/步;ponder 关闭,双方只在轮到自己时思考")
    A(f"- 并发:{meta['workers']} 局同时进行(GPU 争抢会同时压低双方,见 3 节)")
    A("")
    A("## 1. 战绩")
    A("")
    A("| | 胜 | 负 | 和 | 胜率 | Elo 差(A − B) | 95% CI |")
    A("|---|---:|---:|---:|---:|---:|---|")
    ci = st["elo_ci95"]
    A(f"| A = 满血 {a_t:g}s | {st['a_wins']} | {st['b_wins']} | {st['draws']} | "
      f"{st['score_a'] * 100:.1f}% | **{st['elo']:+.0f}** | "
      f"{ci[0]:+.0f} ~ {ci[1]:+.0f} |")
    A("")
    if st.get("elo_saturated") and st.get("elo_bound", 0) > 0:
        A(f"> 该边饱和({st['a_wins']} 比 {st['b_wins']}),点估计的 Elo 只在形式上有意义;"
          f"95% 水平下真实差距 **至少 {st['elo_bound']:+.0f} Elo**。")
        A("")
    elif st.get("elo_saturated"):
        A(f"> 该边饱和({st['a_wins']} 比 {st['b_wins']}),但局数太少,"
          f"95% 水平下还给不出正的强度下界(下界 {st['elo_bound']:+.0f} Elo),"
          f"只能说「占优」,不能说「至少强多少」。")
        A("")
    gap = (abs(st["elo_bound"]) if st.get("elo_saturated")
           and st["elo_bound"] > 0 else abs(st["elo"]))
    if st.get("elo_saturated"):
        A(f"(饱和情况下 Laplace 平滑点估计约 {st['elo_laplace']:+.0f} Elo,"
          f"仅供参考,不要当点估计用。)")
        A("")
    A(f"结论:**满血 {a_t:g}s 比 {b_t:g}s 强{'至少' if st.get('elo_saturated') else '约'}"
      f" {gap:.0f} Elo**"
      f"(换算成胜率,{b_t:g}s 版赢一局的概率"
      f"{'不到' if st.get('elo_saturated') else '约'} "
      f"{100 / (1 + 10 ** (gap / 400)):.0f}%)。")
    A("")
    A("## 2. 换算到定段赛坐标")
    A("")
    if kata_ref is not None:
        A(f"定段赛里 KataGo 的 15s 只有约 590 visits/s(每步 ~8850 visits),"
          f"本表 B = {b_t:g}s 的每步访问量见 3 节,量级相当,"
          f"当时测得 **+{kata_ref:.0f} Elo**(相对初级 = 10k)。"
          f"把本节测到的差距加上去:")
        A("")
        A("| KataGo 配置 | Elo(锚点:初级 15s = 0 = 10k) | 类比段位 |")
        A("|---|---:|---|")
        A(f"| {b_t:g}s(= 定段赛里被挤占的 15s) | +{kata_ref:.0f} | {rank(kata_ref)} |")
        if sat and bnd and bnd > 0:
            A(f"| **满血 {a_t:g}s** | **≥ +{full_elo:.0f}**(95% 下界) "
              f"| **{rank(full_elo)} 或更强** |")
            A("")
            A(f"即满血 {a_t:g}s 的 KataGo 至少是 **{rank(full_elo)}**,"
              f"而不是定段赛报告里那个保守的 {rank(kata_ref)}"
              f"(上界给不出:这条边全胜)。")
        else:
            A(f"| **满血 {a_t:g}s** | **+{full_elo:.0f}**"
              f"(区间 +{full_lo:.0f} ~ +{full_hi:.0f}) "
              f"| **{rank(full_elo)}**(区间 {rank(full_lo)} ~ {rank(full_hi)}) |")
            A("")
            A(f"即满血 {a_t:g}s 的 KataGo 在定段赛坐标系里约 **{rank(full_elo)}**,"
              f"而不是定段赛报告里那个保守的 {rank(kata_ref)}。")
        if tour_elo is not None:
            A("")
            A(f"更保守的换算:定段赛里 KataGo 每步只有约 {V_TOUR:.0f} visits,"
              f"介于本表 B({vb} visits)与 A({va} visits)之间。"
              f"按 Elo ∝ log(visits) 内插,只需补上增量的 {frac * 100:.0f}%"
              f"(约 +{abs(delta_pt) * frac:.0f} Elo),得到 "
              f"**+{tour_elo:.0f} Elo ≈ {rank(tour_elo)}**。"
              "这个值比上一行小,是因为 B 用的 4s 比定段赛当时的实际算力更弱。")
        A("")
        A(f"> 两行分别读作:「若拿 4s 当作定段赛的实际算力」与「按定段赛实际 "
          f"{V_TOUR:.0f} visits 内插」。真实答案落在两者之间,"
          "偏向上面的内插值。")
    else:
        A("未找到 `records/tournament_20261022/summary.json`,跳过换算;"
          "把本节的 Elo 差直接加到定段赛的 KataGo Elo 上即可。")
    A("")
    A("## 3. 算力自检(这一节决定结论能不能信)")
    A("")
    A("| 配置 | 平均步时 | 中位步时 | 中位访问量/步 | 中位 visits/s |")
    A("|---|---:|---:|---:|---:|")
    for side, label in (("A", f"满血 {a_t:g}s"), ("B", f"{b_t:g}s")):
        d = tm[side]
        A(f"| {label} | {d['avg_seconds']}s | {d['median_seconds']}s | "
          f"{d['visits_median'] or '-'} | {d['vps_median'] or '-'} |")
    A("")
    vp = [tm[s]["vps_median"] for s in ("A", "B") if tm[s]["vps_median"]]
    if vp:
        peak = max(vp)
        ratio = peak / KATA_VPS_IDLE
        A(f"- 历史无争抢吞吐是 {KATA_VPS_IDLE:.0f} visits/s;本次最高 {peak} visits/s,"
          f"相当于 {ratio * 100:.0f}%。")
        if ratio < 0.8:
            A(f"- **并发争抢明显**(< 80%):两边的 15s/4s 都被打了折扣,"
              f"所以本报告测到的 Elo 差是**偏小**的下界(真实满血差距更大)。")
        else:
            A("- 争抢可忽略(≥ 80% 满血吞吐),两边都接近单进程满血水平,"
              "因此这个 Elo 差可以直接用。")
    A("")
    A("## 4. 限制")
    A("")
    A(f"- 单边 {st['games']} 局,95% CI 仍有 ±{abs(ci[1] - ci[0]) / 2:.0f} Elo 量级;"
      "要收窄到 ±50 以内需要 ~4 倍局数。")
    bv = tm["B"]["visits_median"] or 0
    A(f"- 用 {b_t:g}s 作为「定段赛里被挤占的 15s」的代理:定段赛里 KataGo 每步约 "
      f"8850 visits,而这里满血 {b_t:g}s 每步约 {bv} visits"
      f"(≈ {bv / 8850:.2f} 倍),所以代理是「同一算力量级」而非严格等算力;"
      "要精确对齐应改用 maxVisits=8850 的配置。")
    A("- 只在 15×15 逆五子棋 + 当前权重上成立;换权重或换规则(如 loss_start)不保证。")
    A("")

    md = "\n".join(lines) + "\n"
    if chart:
        md = md.replace("## 2. 换算到定段赛坐标",
                        "![时间控制对比](kata_tc_chart.png)\n\n## 2. 换算到定段赛坐标")
    (out_dir / "report.md").write_text(md, "utf-8")
    # 同步刷新 summary.json,避免驱动(旧统计口径)与报告不一致
    (out_dir / "summary.json").write_text(json.dumps(
        {"generated": datetime.now().isoformat(" "), "meta": meta,
         "stats": st, "timing": tm}, ensure_ascii=False, indent=2), "utf-8")

    if chart:
        _chart(out_dir, st, tm, a_t, b_t, kata_ref, full_elo, full_lo, full_hi)
    print(f"已写出 {out_dir / 'report.md'}")
    print(f"A({a_t:g}s) {st['a_wins']}:{st['b_wins']} B({b_t:g}s)  "
          f"Elo {st['elo']:+.0f} [{ci[0]:+.0f}, {ci[1]:+.0f}]")
    if full_elo is not None:
        print(f"满血 {a_t:g}s 在定段赛坐标里: +{full_elo:.0f} Elo → {rank(full_elo)}")
    return 0


def _chart(out_dir: Path, st: dict, tm: dict, a_t: float, b_t: float,
           kata_ref, full_elo, full_lo, full_hi) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.2),
                             constrained_layout=True)
    ax = axes[0]
    if kata_ref is not None:
        vals = [kata_ref, full_elo]
        lo = [0, max(0.0, full_elo - full_lo)]
        hi = [0, max(0.0, full_hi - full_elo)]
        labels = [f"KataGo {b_t:g}s\n(= starved {a_t:g}s)", f"KataGo {a_t:g}s\n(full power)"]
        colors = ["#9aa7b8", "#c0504d"]
        ys = [1, 0]
        ax.barh(ys, vals, color=colors, height=0.5)
        ax.errorbar(vals, ys, xerr=[lo, hi], fmt="none", ecolor="black",
                    capsize=4, lw=1.2)
        ax.set_yticks(ys, labels)
        for y, v in zip(ys, vals):
            ax.text(v + 25, y, f"+{v:.0f}", va="center", fontsize=9)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("Elo (anchor: beginner heuristics @15s = 0)")
        ax.set_title(f"KataGo full {a_t:g}s vs {b_t:g}s "
                     f"({full_elo - kata_ref:+.0f} Elo)")
        ax.set_xlim(0, max(vals) * 1.25)
    ax = axes[1]
    ax.axhline(0.5, ls="--", color="#aaaaaa", lw=0.9, label="50% (equal)")
    ax.scatter([0.5], [st["score_a"]], color="#c0504d", zorder=3)
    ax.errorbar([0.5], [st["score_a"]],
                yerr=[[st["score_a"] - st["score_ci95"][0]],
                      [st["score_ci95"][1] - st["score_a"]]],
                fmt="none", ecolor="#c0504d", capsize=5)
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(0, 1)
    ax.set_xticks([0.5], ["full vs short"])
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylabel(f"score of full {a_t:g}s")
    ax.set_title(f"{st['a_wins']} : {st['b_wins']} (n={st['games']})")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.savefig(out_dir / "kata_tc_chart.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="kata_tc_match",
                                description="KataGo 时间控制对比")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="跑对局")
    r.add_argument("--out", required=True)
    r.add_argument("--a-time", type=float, default=15.0, help="满血方每步秒数")
    r.add_argument("--b-time", type=float, default=4.0, help="短时限方每步秒数")
    r.add_argument("--openings", default="records/tournament_20261022/openings.json")
    r.add_argument("--n-openings", type=int, default=8)
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--engine", default="")
    r.add_argument("--model", default="")
    r.add_argument("--engine-cfg", default="")
    r.add_argument("--kata-visits", type=int, default=100_000_000)
    r.add_argument("--max-steps", type=int, default=450)
    r.add_argument("--loss-start", type=int, default=8)
    r.add_argument("--white-restrict", type=int, default=2)
    r.add_argument("--deadline", default="")
    r.add_argument("--chart", action="store_true")

    q = sub.add_parser("report", help="出报告")
    q.add_argument("--out", required=True)
    q.add_argument("--chart", action="store_true")

    args = p.parse_args(argv)
    if args.cmd == "run":
        d = _kata_defaults()
        args.engine = args.engine or d["engine"]
        args.model = args.model or d["weight"]
        args.engine_cfg = args.engine_cfg or d["config"]
        for name, val in (("engine", args.engine), ("model", args.model),
                          ("config", args.engine_cfg)):
            if not val or not os.path.exists(val):
                raise SystemExit(f"KataGo {name} 不存在: {val!r}")
        return run(args)
    return report(Path(args.out), chart=args.chart)


if __name__ == "__main__":
    raise SystemExit(main())
