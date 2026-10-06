"""逆五子棋棋谱(.afg)读写:格式定义 + 保存/加载函数

设计目标:人类可读、可版本管理、可供后续"行为克隆预热训练"直接解析。
着法按引擎的"步"记录:普通落子 = 1 个着法,移子 = 2 个着法(占领 + 安置)。

格式(UTF-8 纯文本):
    [AntiFive 1.0]                              # 格式版本
    [Config 15x15 white_restrict=2 loss_start=8 mask_suicide=1]
    [Result 黑胜|白胜|和棋|未完]
    [Moves 34]                                  # 着法数(信息项,解析时忽略)
    moves:
    O5 O4 O7 A4 ...                             # 着法序列,'#' 后为注释
    [Analysis 34]                               # 可选:逐手分析数据
    1 50.0 H8 J8:48.5 K8:47.2 L8:46.1
    2 47.3 K8 ...

着法坐标:列字母 A-O + 行数字 0-14,与 GUI 棋盘标注一致(O5 = 第 15 列第 6 行)。
分析行格式:`手数 当前方胜率 最佳着法 [候选着法:胜率 ...]`(候选为 GTP 坐标:百分比)。
解析容错:允许缺省 [Config]/[Result]/[Analysis] 头;非 '[' 开头的行均视为着法序列。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .reversegomoku import (
    BLACK, BOARD_SIZE, DEFAULT_CONFIG, GameConfig, WHITE,
)

FORMAT_TAG = "AntiFive 1.0"
RESULT_NAMES = ("黑胜", "白胜", "和棋", "未完")


def idx_to_coord(idx: int) -> str:
    """平铺索引 → 坐标文本(列字母 A-O + 行数字 0-14)。"""
    r, c = divmod(int(idx), BOARD_SIZE)
    return f"{chr(65 + c)}{r}"


def coord_to_idx(token: str) -> int:
    """坐标文本 → 平铺索引。非法输入抛 ValueError。"""
    t = token.strip().upper()
    if len(t) < 2 or not ("A" <= t[0] <= chr(64 + BOARD_SIZE)):
        raise ValueError(f"非法坐标: {token!r}")
    c = ord(t[0]) - 65
    r = int(t[1:])
    if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
        raise ValueError(f"坐标越界: {token!r}")
    return r * BOARD_SIZE + c


def game_result(game) -> str:
    """终局 → 结果文本:黑胜/白胜/和棋/未完。"""
    if not game.game_over:
        return "未完"
    if game.is_draw:
        return "和棋"
    return "白胜" if game.loser == BLACK else "黑胜"


@dataclass
class Record:
    """解析后的棋谱。moves 为平铺索引步序列(含移子两步)。

    analysis: {手数: {"wr": 当前方胜率, "best": 最佳着法, "moves": [(候选着法, 胜率), ...]}}
    """
    moves: List[int]
    config: GameConfig
    result: str = "未完"
    analysis: Optional[dict] = None


def _config_text(config: GameConfig) -> str:
    return (f"15x15 white_restrict={config.white_restrict_turns} "
            f"loss_start={config.loss_start_turns} "
            f"mask_suicide={int(config.mask_suicide)}")


def _config_parse(body: str) -> GameConfig:
    cfg = GameConfig()
    for tok in body.split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        if k == "white_restrict":
            cfg.white_restrict_turns = int(v)
        elif k == "loss_start":
            cfg.loss_start_turns = int(v)
        elif k == "mask_suicide":
            cfg.mask_suicide = bool(int(v))
    return cfg


def dumps(moves: List[int], config: GameConfig = DEFAULT_CONFIG,
          result: str = "未完", analysis: Optional[dict] = None) -> str:
    """步序列 → 棋谱文本。analysis 可选:{手数: {"wr", "best", "moves": [(候选, 胜率)]}}。"""
    coords = " ".join(idx_to_coord(m) for m in moves)
    text = ("[AntiFive 1.0]\n"
            f"[Config {_config_text(config)}]\n"
            f"[Result {result}]\n"
            f"[Moves {len(moves)}]\n"
            f"moves:\n{coords}")
    if analysis:
        lines = [f"[Analysis {len(analysis)}]"]
        for step in sorted(analysis):
            a = analysis[step]
            cand = " ".join(f"{c}:{w:.1f}" for c, w in a.get("moves", []))
            lines.append(f"{step} {a['wr']:.1f} {a['best']} {cand}".rstrip())
        text += "\n" + "\n".join(lines)
    return text + "\n"


def _parse_analysis_line(line: str) -> Optional[tuple]:
    """解析分析行 `step wr best cand:wr ...` → (step, entry) 或 None。"""
    parts = line.split()
    if len(parts) < 3:
        return None
    try:
        step = int(parts[0])
        wr = float(parts[1])
        best = parts[2].upper()
        moves = []
        for tok in parts[3:]:
            c, w = tok.rsplit(":", 1)
            moves.append((c.upper(), float(w)))
    except (ValueError, IndexError):
        return None
    return step, {"wr": wr, "best": best, "moves": moves}


def loads(text: str) -> Record:
    """棋谱文本 → Record。缺省头用默认值,非法着法报 ValueError。"""
    moves: List[int] = []
    config = GameConfig()
    result = "未完"
    analysis: Optional[dict] = None
    in_analysis = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            in_analysis = False
            body = line.strip("[]").strip()
            if body.startswith("Config"):
                config = _config_parse(body[len("Config"):])
            elif body.startswith("Result"):
                result = body[len("Result"):].strip()
            elif body.startswith("Analysis"):
                in_analysis = True
                analysis = {} if analysis is None else analysis
            continue
        if in_analysis:
            parsed = _parse_analysis_line(line)
            if parsed is not None:
                step, entry = parsed
                analysis[step] = entry
            continue
        if line.lower().startswith("moves"):
            line = line[len("moves"):].lstrip(": ").strip()
            if not line:
                continue
        for tok in line.split():
            moves.append(coord_to_idx(tok))
    return Record(moves, config, result, analysis)


def save_game(path: str, game, analysis: Optional[dict] = None,
              moves: Optional[list] = None) -> str:
    """把对局存为 .afg,返回结果文本。analysis 为可选逐手分析数据。
    moves: 可选,传入原始着法序列(复盘/试下后保存原谱用);None 则取 game.record()。

    移子待定且未终局(未完成安置)拒绝保存;若终局且最后一手是占领步
    (占领完成己方连五立即判负,安置未发生),该未完成步从记录中剔除,
    结果由 [Result] 头保留。
    """
    if moves is None:
        moves = game.record()
    if game.pending >= 0:
        if not game.game_over:
            raise ValueError("移子待定中:请先完成安置再保存")
        moves = moves[:-1]
    text = dumps(moves, game.config, game_result(game), analysis)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return game_result(game)


def load_game(path: str) -> Record:
    """读取 .afg → Record。"""
    with open(path, encoding="utf-8") as f:
        return loads(f.read())
