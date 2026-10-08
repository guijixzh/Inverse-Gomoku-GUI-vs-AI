"""开局库生成工具(KataGo GTP,默认 20k visits)

生成 `src/antifive/openbook_data.py`:空盘首手(黑)+ 每个 D4 等价类首手的
白方应手。查找端见 `antifive.openbook`(D4 对称规范化,包内纯数据,网页可用)。

用法:
    python -m antifive.tools.openbook_gen --visits 20000
    python -m antifive.tools.openbook_gen --engine <katago.exe> --weight <bin> \
        --config config/gtp_test.cfg

引擎/权重/配置默认从 config/engines.local.json(优先)或 engines.json 读取。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ..paths import app_base, config_dir, resolve
from ..reversegomoku import PLACE_SIZE, SYM_FLAT

LETTERS = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


def idx_to_gtp(i: int) -> str:
    r, c = divmod(int(i), 15)
    return f"{LETTERS[c]}{15 - r}"


def gtp_to_idx(coord: str) -> int:
    c = LETTERS.index(coord[0].upper())
    return (15 - int(coord[1:])) * 15 + c


def canonical_bytes(flat: np.ndarray) -> tuple:
    """字典序最小的 D4 变换:transformed[SYM_FLAT[t]] = flat(与查找端一致)。"""
    best, best_t = None, 0
    for t in range(8):
        tb = np.zeros(PLACE_SIZE, dtype=np.int8)
        tb[SYM_FLAT[t]] = flat
        b = tb.tobytes()
        if best is None or b < best:
            best, best_t = b, t
    return best, best_t


class Engine:
    def __init__(self, exe: str, weight: str, cfg: str, visits: int):
        self.proc = subprocess.Popen(
            [exe, "gtp", "-config", cfg, "-model", weight,
             "-override-config",
             f"maxVisits={visits},ponderingEnabled=false"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
            encoding="utf-8", errors="replace")

    def _read(self) -> str:
        out = self.proc.stdout.readline()
        if out == "":
            raise RuntimeError("katago 进程退出(检查 exe/权重/配置路径)")
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

    def analyze(self, color: str) -> str:
        self.proc.stdin.write(f"kata-genmove_analyze {color} 50 maxmoves 1\n")
        self.proc.stdin.flush()
        while True:
            out = self._read()
            if out.startswith("play "):
                return out[len("play "):]
            if out.startswith("? "):
                raise RuntimeError(out)

    def close(self) -> None:
        try:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
            time.sleep(0.1)
        except Exception:
            pass
        self.proc.terminate()


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


def main() -> None:
    d = _engine_defaults()
    p = argparse.ArgumentParser(description="生成 KataGo 开局库数据模块")
    p.add_argument("--engine", default=d["engine"], help="katago 可执行文件")
    p.add_argument("--weight", default=d["weight"], help="权重 .bin")
    p.add_argument("--config", default=d["config"], help="GTP 配置")
    p.add_argument("--visits", type=int, default=20000)
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                        / "openbook_data.py"))
    args = p.parse_args()
    if not (args.engine and args.weight and args.config):
        p.error("未找到 KataGo 路径:请用 --engine/--weight/--config 指定,"
                "或在 config/engines.json 配置")

    reps: dict = {}
    for a in range(PLACE_SIZE):
        flat = np.zeros(PLACE_SIZE, dtype=np.int8)
        flat[a] = 1
        key, t = canonical_bytes(flat)
        reps.setdefault(key, int(SYM_FLAT[t][a]))
    print(f"首手等价类: {len(reps)}", flush=True)

    eng = Engine(args.engine, args.weight, args.config, args.visits)
    eng.simple("boardsize 15 15")
    entries = {}

    eng.simple("clear_board")
    coord = eng.analyze("B")
    eng.simple("undo")
    key, _ = canonical_bytes(np.zeros(PLACE_SIZE, dtype=np.int8))
    entries[(key.hex(), 1)] = gtp_to_idx(coord)
    print(f"黑首手: {coord}", flush=True)

    t0 = time.perf_counter()
    for n, (key, pos) in enumerate(sorted(reps.items()), 1):
        eng.simple("clear_board")
        eng.simple(f"play B {idx_to_gtp(pos)}")
        coord = eng.analyze("W")
        eng.simple("undo")
        entries[(key.hex(), 2)] = gtp_to_idx(coord)
        print(f"[{n}/{len(reps)}] 首手 {idx_to_gtp(pos)} -> 白 {coord} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    eng.close()

    lines = [
        '"""开局库数据(由 tools/openbook_gen.py 生成,勿手改)。',
        "",
        "键:(D4 规范化棋盘字节 hex, 行棋方);值:规范化坐标下的着法 idx。",
        f"来源:KataGo {args.visits} visits 分析。",
        '"""',
        "",
        "OPENING_BOOK = {",
    ]
    for k, mv in sorted(entries.items()):
        lines.append(f'    ("{k[0]}", {k[1]}): {mv},')
    lines.append("}")
    lines.append("")
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    print(f"已写入 {args.out}: {len(entries)} 条目")


if __name__ == "__main__":
    main()
