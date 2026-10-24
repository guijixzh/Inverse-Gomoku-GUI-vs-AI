# -*- coding: utf-8 -*-
"""败局验尸:网页端输掉的棋,它最后那一手是「已无解」还是「还能守」。

规则背景(逆五子棋):安置步可以把对方棋子摆到某格,若因此凑成对方五连,
则对方判负 —— 所以赢家的最后一手常常是「给对方安置出五连」的杀着。
因此要看的不是最后一手,而是**网页端自己的最后一手**:在那之前它还有没有不输的着法。

做法:
  1. 取 `records/web_vs_py_15s` 里 winner == 'py' 的对局(网页端输);
  2. 逐步重放,用当前行棋方定位「网页端最后一次走子」的下标 i;
  3. 还原到 i 之前(此时轮到网页端),用 Python analysis(advanced, depth 6,
     无时限, k=24)看它的候选着法:
     - 所有候选都 <= -MATE/2  → 这盘在它这手之前就已经必败(前面输掉的);
     - 存在安全着法           → 这手是「可避免的失误」,并记下它选的着法排第几;
"""
import argparse
import glob
import json
import random
import time
from pathlib import Path

from ..reversegomoku import BLACK, GameConfig, ReverseGomoku, WHITE
from .. import heuristic_versions as hv

MATE = 1_000_000.0
COLOR = {"black": BLACK, "white": WHITE}


def load_games(run_dir: Path, loser: str = "web"):
    """loser='web' 取网页端输的对局;loser='py' 取 Python 端输的对局(对照组)。"""
    winner = {"web": "py", "py": "web"}[loser]
    out = []
    for f in sorted(glob.glob(str(run_dir / "games" / "*" / "*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if d.get("winner") != winner or len(d.get("moves") or []) < 8:
            continue
        out.append(d)
    return out


def web_color_of(d):
    if "web_color" in d:
        return COLOR[d["web_color"]]
    return COLOR["white" if d["py_color"] == "black" else "black"]


def py_color_of(d):
    return COLOR[d["py_color"]]


def loser_color_of(d, loser):
    return web_color_of(d) if loser == "web" else py_color_of(d)


def last_web_move_index(d, cfg, loser="web"):
    web_color = loser_color_of(d, loser)
    g = ReverseGomoku(cfg)
    idx = -1
    for i, mv in enumerate(d["moves"]):
        if g.current_player == web_color:
            idx = i
        g.make_move(mv, allow_suicide=True)
        if g.game_over:
            break
    return idx


def analyse_py(job):
    d, cfg_kw, depth, k, loser = job
    cfg = GameConfig(**cfg_kw)
    i = last_web_move_index(d, cfg, loser)
    g = ReverseGomoku(cfg)
    for mv in d["moves"][:i]:
        g.make_move(mv, allow_suicide=True)
    side = g.current_player
    t0 = time.time()
    v, mv = hv.analysis("advanced", g.board.copy(), side, g.pending,
                        g.turn_count, g.white_turns, cfg, random.Random(12345),
                        depth=depth, time_budget=None, k=k, use_vcf=True)
    return {"game_id": d["game_id"], "web_move_index": i,
            "web_actual_move": int(d["moves"][i]),
            "side_to_move": "black" if side == BLACK else "white",
            "py_best_v": float(v),
            "py_moves": [(int(a), float(b)) for a, b in mv],
            "py_sec": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="records/web_vs_py_15s")
    ap.add_argument("--out", default="records/web_vs_py_loss_postmortem")
    ap.add_argument("--py-depth", type=int, default=6)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--loser", default="web", choices=("web", "py"))
    args = ap.parse_args()

    cfg_kw = {"loss_start_turns": 8, "white_restrict_turns": 2, "mask_suicide": True}
    games = load_games(Path(args.run), args.loser)
    print(f"{'网页端' if args.loser == 'web' else 'Python 端'}败局 {len(games)} 盘", flush=True)
    jobs = [(d, cfg_kw, args.py_depth, args.k, args.loser) for d in games]

    import multiprocessing as mp
    with mp.get_context("spawn").Pool(args.workers) as pool:
        res = pool.map(analyse_py, jobs, chunksize=1)

    rows = []
    forced = avoidable = 0
    for r in res:
        best = r["py_best_v"]
        all_lost = best <= -MATE / 2
        vals = dict(r["py_moves"])
        actual = r["web_actual_move"]
        rank = (list(vals).index(actual) if actual in vals else None)
        regret = (best - vals[actual]) if actual in vals else None
        forced += 1 if all_lost else 0
        avoidable += 0 if all_lost else 1
        rows.append({**{k: r[k] for k in ("game_id", "web_move_index", "web_actual_move",
                                          "side_to_move", "py_best_v", "py_sec")},
                     "all_moves_lose": all_lost,
                     "actual_rank_in_py": rank,
                     "py_regret_of_actual": regret,
                     "py_moves": r["py_moves"]})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "postmortem.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    secs = sorted(r["py_sec"] for r in rows)
    summary = {
        "games": len(rows),
        "already_lost_before_last_move": forced,
        "avoidable_blunder": avoidable,
        "share_avoidable": round(avoidable / max(1, len(rows)), 4),
        "py_sec_median": round(secs[len(secs) // 2], 2) if secs else None,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
