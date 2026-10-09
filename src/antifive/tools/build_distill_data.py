"""KataGo 蒸馏数据构建(策略/价值标签,20k visits,可并行续跑)

为每个采样局面记录 KataGo 的候选着法分布(visits/winrate,top-20)与根胜率,
供 `tools/train_distill.py` 训练蒸馏先验(D1)与价值校正(D2)。

对局来源:
- 当前启发式引擎自对弈(交替 0.5s/2s 每步限时,带开局库);
- KataGo(2000 visits 落子) vs 启发式,提供强手压迫下的防守局面。

按"局"并行:每个 worker 独占一个 katago 进程,完成一局后写
`<parts>/gXXXXX.npz`;再次运行自动跳过已完成局(断点续跑)。
`--minutes` 到点后不再开新局,退出并合并已有结果。

用法:
    python -m antifive.tools.build_distill_data --games-selfplay 200 \
        --games-katago 12 --workers 4 --visits 20000 --minutes 15
    python -m antifive.tools.build_distill_data --merge-only
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ..heuristic import _TT, choose_heuristic_move
from ..paths import config_dir, resolve
from ..reversegomoku import BLACK, WHITE, GameConfig, ReverseGomoku

LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
MAX_CAND = 20


def idx_to_gtp(i: int) -> str:
    r, c = divmod(int(i), 15)
    return f"{LETTERS[c]}{15 - r}"


def gtp_to_idx(coord: str) -> int:
    coord = coord.strip().upper()
    if coord in ("PASS", "RESIGN"):
        return -1                          # 逆五里 pass 即认输,对局结束
    c = LETTERS.index(coord[0])
    return (15 - int(coord[1:])) * 15 + c


def _engine_defaults() -> dict:
    for name in ("engines.local.json", "engines.json"):
        try:
            with open(config_dir() / name, encoding="utf-8") as f:
                for e in json.load(f):
                    if e.get("kind") == "heuristic":
                        continue
                    return {"engine": resolve(e.get("engine_path", "")),
                            "weight": resolve(e.get("weight_path", "")),
                            "config": resolve(e.get("config_path", ""))}
        except Exception:
            continue
    return {"engine": "", "weight": "", "config": ""}


class Engine:
    def __init__(self, exe: str, weight: str, cfg: str, visits: int):
        self.visits = visits
        self.proc = subprocess.Popen(
            [exe, "gtp", "-config", cfg, "-model", weight,
             "-override-config", f"maxVisits={visits},ponderingEnabled=false"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
            encoding="utf-8", errors="replace")

    def _read(self) -> str:
        out = self.proc.stdout.readline()
        if out == "":
            raise RuntimeError("katago 进程退出")
        return out.rstrip("\n")

    def simple(self, line: str) -> str:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        while True:
            out = self._read()
            if out.startswith("= "):
                return out[2:]
            if out.startswith("? "):
                raise RuntimeError(out)

    def set_visits(self, n: int) -> None:
        self.simple(f"kata-set-param maxVisits {int(n)}")

    def genmove(self, color: str, visits: int) -> str:
        self.set_visits(visits)
        self.proc.stdin.write(
            f"kata-genmove_analyze {color} 50 maxmoves 1\n")
        self.proc.stdin.flush()
        while True:
            out = self._read()
            if out.startswith("play "):
                return out[len("play "):]
            if out.startswith("? "):
                raise RuntimeError(out)

    def analyze(self, color: str) -> tuple:
        """返回 (best_coord, [(move, visits, winrate), ...] top-20)。"""
        self.set_visits(self.visits)
        self.proc.stdin.write(
            f"kata-genmove_analyze {color} 50 maxmoves {MAX_CAND}\n")
        self.proc.stdin.flush()
        cands = []
        while True:
            out = self._read()
            if out.startswith("play "):
                return out[len("play "):], cands
            if out.startswith("? "):
                raise RuntimeError(out)
            if "info move" in out:
                cands = _parse_info(out)

    def close(self) -> None:
        try:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
            time.sleep(0.1)
        except Exception:
            pass
        self.proc.terminate()


def _parse_info(line: str) -> list:
    toks = line.split()
    out, cur, i = [], None, 0
    while i < len(toks):
        t = toks[i]
        if t == "info":
            if cur is not None:
                out.append(cur)
            cur = {}
            i += 1
            continue
        if cur is None:
            i += 1
            continue
        if t == "pv":
            j = i + 1
            while j < len(toks) and toks[j] != "info":
                j += 1
            i = j
            continue
        if i + 1 < len(toks):
            v = toks[i + 1]
            try:
                cur[t] = int(v) if t in ("visits", "order") else float(v)
            except ValueError:
                cur[t] = v
            i += 2
            continue
        i += 1
    if cur is not None:
        out.append(cur)
    return out


def _sample_ply(ply: int, start: int, step: int, n: int, cap: int) -> bool:
    return ply >= start and (ply - start) % step == 0 and n < cap


def _play_game(eng: Engine, spec: dict, cfg: GameConfig) -> list:
    """下一局并按需采样;返回样本列表(每项含局面与 KataGo 标签)。"""
    kind = spec["kind"]
    rng = np.random.default_rng(spec["seed"])
    g = ReverseGomoku(cfg)
    tt = _TT()
    eng.simple("clear_board")
    samples = []
    heur_black = spec.get("heur_black", True)
    while not g.game_over and g.move_count < 450:
        color = "B" if g.current_player == BLACK else "W"
        is_katago_turn = (kind == "katago" and
                          (g.current_player == BLACK) == (not heur_black))
        if _sample_ply(g.move_count, spec["start"], spec["step"],
                       len(samples), spec["cap"]):
            wr, cands = eng.analyze(color)
            eng.simple("undo")
            samples.append(_pack_sample(g, wr, cands, spec))
        if is_katago_turn:
            coord = eng.genmove(color, spec["play_visits"])
            mv = gtp_to_idx(coord)
            if mv < 0:                 # pass/resign:KataGo 认输,本局结束
                break
        else:
            mv = choose_heuristic_move(
                g.board, g.current_player, g.pending, g.turn_count,
                g.white_turns, g.config, rng, depth=8,
                time_budget=spec["budget"], tt=tt, use_vcf=True)
            if mv is None:
                break
            eng.simple(f"play {color} {idx_to_gtp(mv)}")
        if not g.legal_mask()[mv]:
            break
        g.make_move(int(mv))
    return samples


def _pack_sample(g, wr_coord, cands, spec) -> dict:
    lmoves = np.full(MAX_CAND, -1, dtype=np.int16)
    lvisits = np.zeros(MAX_CAND, dtype=np.int32)
    lwr = np.zeros(MAX_CAND, dtype=np.float32)
    n = 0
    for c in sorted(cands, key=lambda d: -int(d.get("visits", 0))):
        if "move" not in c or n >= MAX_CAND:
            continue
        mi = gtp_to_idx(c["move"])
        if mi < 0:                         # pass/resign 候选跳过
            continue
        lmoves[n] = mi
        lvisits[n] = int(c.get("visits", 0))
        lwr[n] = float(c.get("winrate", 0.5))
        n += 1
    return {
        "board": g.board.copy(),
        "player": np.int8(g.current_player),
        "pending": np.int16(g.pending),
        "tc": np.int16(g.turn_count),
        "wt": np.int16(g.white_turns),
        "game_id": np.int32(spec["idx"]),
        "source": np.int8(0 if spec["kind"] == "selfplay" else 1),
        "lmoves": lmoves,
        "lvisits": lvisits,
        "lwr": lwr,
    }


def _worker(wargs: dict) -> int:
    exe, weight, cfgpath = wargs["engine"], wargs["weight"], wargs["config"]
    visits = wargs["visits"]
    cfg = GameConfig()
    done = 0
    eng = Engine(exe, weight, cfgpath, visits)
    eng.simple("boardsize 15 15")
    try:
        for spec in wargs["specs"]:
            if time.time() > wargs["deadline"]:
                break
            path = Path(wargs["parts"]) / f"g{spec['idx']:05d}.npz"
            if path.exists():
                continue
            t0 = time.perf_counter()
            samples = _play_game(eng, spec, cfg)
            _save_part(path, samples)
            done += 1
            print(f"[w{wargs['wid']}] game {spec['idx']} "
                  f"({spec['kind']}, {len(samples)} samples, "
                  f"{time.perf_counter() - t0:.0f}s)", flush=True)
    finally:
        eng.close()
    return done


def _save_part(path: Path, samples: list) -> None:
    if not samples:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, n=np.array([0]))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        boards=np.stack([s["board"] for s in samples]).astype(np.int8),
        players=np.array([s["player"] for s in samples], dtype=np.int8),
        pends=np.array([s["pending"] for s in samples], dtype=np.int16),
        tcs=np.array([s["tc"] for s in samples], dtype=np.int16),
        wts=np.array([s["wt"] for s in samples], dtype=np.int16),
        game_ids=np.array([s["game_id"] for s in samples], dtype=np.int32),
        sources=np.array([s["source"] for s in samples], dtype=np.int8),
        lmoves=np.stack([s["lmoves"] for s in samples]).astype(np.int16),
        lvisits=np.stack([s["lvisits"] for s in samples]).astype(np.int32),
        lwr=np.stack([s["lwr"] for s in samples]).astype(np.float32),
    )


def _merge(parts: Path, out: Path) -> int:
    files = sorted(parts.glob("g*.npz"))
    arrays = {k: [] for k in ("boards", "players", "pends", "tcs", "wts",
                              "game_ids", "sources", "lmoves", "lvisits", "lwr")}
    for f in files:
        z = np.load(f)
        if "boards" not in z:
            continue
        for k in arrays:
            arrays[k].append(z[k])
    if not arrays["boards"]:
        print("暂无已完成对局")
        return 0
    boards = np.concatenate(arrays["boards"])
    seen, keep = set(), []
    for i in range(len(boards)):
        k = boards[i].tobytes()
        if k not in seen:
            seen.add(k)
            keep.append(i)
    keep = np.asarray(keep)
    np.savez(out, **{k: np.concatenate(v)[keep] for k, v in arrays.items()})
    print(f"合并 {len(files)} 个分片 → {out}:{len(keep)} 个局面")
    return len(keep)


def main() -> None:
    d = _engine_defaults()
    p = argparse.ArgumentParser(description="KataGo 蒸馏数据构建")
    p.add_argument("--engine", default=d["engine"])
    p.add_argument("--weight", default=d["weight"])
    p.add_argument("--config", default=d["config"])
    p.add_argument("--visits", type=int, default=20000)
    p.add_argument("--play-visits", type=int, default=2000,
                   help="KataGo 对局落子访问数")
    p.add_argument("--games-selfplay", type=int, default=200)
    p.add_argument("--games-katago", type=int, default=12)
    p.add_argument("--sample-start", type=int, default=10)
    p.add_argument("--sample-step", type=int, default=3)
    p.add_argument("--sample-cap", type=int, default=12)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--minutes", type=float, default=15.0,
                   help="本次运行的时间上限(到点后不再开新局)")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--merge-only", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    parts = out_dir / "distill_parts"
    pool = out_dir / "distill_pool.npz"
    parts.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        _merge(parts, pool)
        return
    if not (args.engine and args.weight and args.config):
        p.error("未找到 KataGo 路径:请用 --engine/--weight/--config 指定")

    specs = []
    idx = 0
    for i in range(args.games_selfplay):
        specs.append({"idx": idx, "kind": "selfplay",
                      "seed": 70000 + i * 17,
                      "budget": 0.5 if i % 2 == 0 else 2.0,
                      "start": args.sample_start, "step": args.sample_step,
                      "cap": args.sample_cap, "play_visits": args.play_visits})
        idx += 1
    for i in range(args.games_katago):
        specs.append({"idx": idx, "kind": "katago",
                      "seed": 90000 + i * 29,
                      "budget": 1.0 if i % 2 == 0 else 2.0,
                      "heur_black": i % 2 == 0,
                      "start": args.sample_start, "step": args.sample_step,
                      "cap": args.sample_cap, "play_visits": args.play_visits})
        idx += 1

    done = len(list(parts.glob("g*.npz")))
    print(f"总局数 {len(specs)},已完成 {done},开始并行标注"
          f"(workers={args.workers}, visits={args.visits})", flush=True)
    deadline = time.time() + args.minutes * 60
    wargs = []
    for w in range(args.workers):
        wargs.append({"wid": w, "specs": specs[w::args.workers],
                      "deadline": deadline, "parts": str(parts),
                      "engine": args.engine, "weight": args.weight,
                      "config": args.config, "visits": args.visits})
    with mp.Pool(args.workers) as pool_mp:
        counts = pool_mp.map(_worker, wargs)
    print("本次完成对局:", sum(counts), flush=True)
    _merge(parts, pool)
    total = len(specs)
    done = len(list(parts.glob("g*.npz")))
    print(f"进度 {done}/{total}", flush=True)


if __name__ == "__main__":
    main()
