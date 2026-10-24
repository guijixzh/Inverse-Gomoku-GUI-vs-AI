"""启发式小 AI(强化版「大师」):极小化搜索 + 全局局面评估 + 选择性候选着法

v6(2026-10-13,防守强化;「高级」已冻结为 tools/heuristic_v6.py):
- 深化 VCF(对高级 +112 Elo 的主要来源):强制杀链 d8/6 万节点/0.6s →
  d12/24 万节点/≤1.8s(攻防同预算;无对局限时用确定性节点上限);
- P0 时间与防守资源:进攻 VCF + 主搜索 + 防守校验的总墙钟从 choose 入口
  计时并 ≤ 时限(修复此前进攻 VCF 不计入预算、平均数秒/步的超时);
  防守校验覆盖全部候选、节点上限与进攻一致、共享 deadline;防守预留自适应
  (仅己方已有危险四连/对方可杀四时预留大预算,常规中盘只留 0.1s);
  迭代加深时限驱动(上限 MAX_DYNAMIC_DEPTH 层);
- P1 预防性防守候选(ANTIFIVE_PREGIFT=1 启用,默认关闭):投送预威胁枚举
  (对方一步把己方子投进己方三连/跳形即成四/五的枢纽),对确成杀的投送
  收集射线封堵格补入根候选;严格等时对拍净负(根候选变多、搜索变浅);
- P2 潜伏威胁评估(ANTIFIVE_GIFT3=1 启用,默认关闭):E_SELF_GIFT3(己方
  三连"可被投送成四"专项负债);严格等时对拍净负;参数隔离机制保留
  (data/params_hell.json 优先,缺失回退包内 antifive/params.json);
- P3 复核返工:多回合"以攻代守"在杀对已成立时不成立(不封堵则进攻方下一
  回合立即完成杀),仅有"立刻反杀"是逃生(原 has_kill 检查已覆盖);不引入
  不成立的放宽,补回归测试锁定该语义(见 test_threat_search.py);
- 已否决(默认关闭,代码保留):根 α 传播 + PVS 与 LMR 速度收益大,但搜索树
  依赖搜索历史(TT/killer 会补进搜索集合),零窗边界失效(对拍 25%~38%),
  待候选集合确定性改造后再评估。

v5.3(2026-10-10,step30 复盘归因的两处低风险修复):
- "上一手吃子"保送槽:搜索递归携带 opp_last,应招节点强制保留"占领对手
  最近落子"的候选(此前被内节点宽度剪枝漏掉,浅层假高分选出 E0 随手);
- 安置评分新增"投送双威胁"项(O_GIFT_D2/D3:落子后两轴各成连二/活三,
  I4 类枢纽),修复单轴公式看不见双活三导致关键吃子排序靠后的问题。

v5(2026-10-08,基于 KataGo 自对弈复盘归因的修复):
- 修复 choose 根搜索两处错误:①占领步后同一方继续安置,子节点视角与根
  相同,旧实现却取了负并套用反向窄窗;②窄窗 fail-soft 返回的上界可能虚高,
  被直接采纳。现在 choose 与 analysis 共用同一根搜索实现(_root_iterative,
  逐层全窗口根评估),引擎落子与其自分析严格一致;
- 候选生成与搜索历史解耦(选取只看静态分,历史仅参与排序),并新增多样性
  候选(贴己方棋群/安静关键点型着法);KataGo 首选落在 ROOT_K 窗口内的
  比例显著提升(旧对局 16/52 → 12/52,新自对弈 6/33);
- 置换表在子节点被强制扩展时按 depth-1 保守存表,避免其他路径按标称深度
  复用到被抬高的分值。

v5.2(2026-10-08,低风险增强:开局库 + 防守 VCF):
- 开局库 openbook:首手(黑空盘/白应黑首手)用 KataGo 20k visits 离线结论,
  D4 对称规范化查表;包内纯数据模块,网页端可用,未命中自动回退搜索;
- 根节点防守校验:对最终排序前 3 的着法,走出且轮到对手时用 VCF 尝试证明
  对手存在强制杀;证明成功则该着降为必败分。预算从主搜索时间中预留
  (2.5%,夹在 [0.02s, 0.3s]),结论保守不误伤。

v5.1(2026-10-08,VCF 深杀修复):
- VCF 默认深度 6→8、节点上限 20000→60000;有对局限时时预算按剩余时间的
  5% 放宽(夹在 [0.05s, 0.6s],见 _vcf_slice)。复盘:用户局 step75 的白
  I13 强制胜须 d8/0.5s 才可证,旧 0.05s 漏掉;

v4:接入独立强制杀链搜索(threat_search,VCF 类;对弈与分析默认启用,
ANTIFIVE_VCF=0 可关闭),证明"无论对手如何防守都会在有限步内被完成连五"
的强制胜;找不到或超时自动回退常规搜索。

v3 优化(相对 tools/heuristic_v2.py 冻结快照):
- 调优参数运行时加载(包内 antifive/params.json 随 wheel 分发;桌面
  data/params.json 可覆盖;ANTIFIVE_NO_PARAMS=1 可关闭;
  置换表大小可用 ANTIFIVE_TT_BITS 调整,网页/低内存环境可用 19-20);
- 节点级双色线图/位置摘要缓存、kill_captures 无威胁早退、kill_placements
  复用合法掩码;节点/秒约 +25%,深度 6 节点数约 -25%;
- 根节点 α 传播、内部 PVS 零窗口 + 历史启发、TT/killer 强制入候选;
- 强制局面扩展(修复 ext_budget 恒 0 未生效的问题)沿强制链加深;
- 叶节点即时杀检测消除水平线效应(搜索耗尽也能看到下一手的杀)。

主要特性(相对旧版 heuristic_old.py 的改进):
- 评估函数:evaluate() 基于"线级模式提取"(_board_features)一次扫描统计
  双色材料,支持 跳三/跳四/断四/活三/冲四 等所有形状(旧版只数连续子),
  并新增双杀(两处可杀四)与"可杀四+三"双威胁结构评估;
- 可杀性:四形(直/跳)的完成格(开放端/缺口) + 借杀桥子可达性判定,
  _threats/_loses_to_kill 已覆盖断四(己方跳四被借杀可被识别与防御);
- 搜索:fail-soft alpha-beta + Zobrist 置换表 + PV 着法/killer 排序 +
  迭代加深根 PV 复用;深度自适应候选宽度(深层收窄);
- 必守/必杀剪枝:有杀只搜杀着;被对方必杀时只搜防守着(危险点倒推),
  防守链自动保持深度(ext_budget);
- 性能:静态 shift 索引表、fwd_run_lengths LRU 缓存(含填充版)、
  特征缓存、向量化候选排序——深度 8 单步从分钟级降到约 40 秒,
  批量生成配合 --budget 每步限时吞吐可控。

命令行:
    python -m antifive.heuristic --games 1000 --out records [--depth 1..8]
                       [--budget 秒/步] [--workers N]
        --depth 1: 贪心全局评估(快,批量生成用)
        --depth 2: 深度 2 极小化(默认,战术可靠)
        --depth 3+: 更深的迭代加深搜索(深度 8 建议配 --budget 3-5 秒)
"""

from __future__ import annotations

from operator import itemgetter

import argparse
import json
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from . import record as record_mod
from .paths import data_dir
from .reversegomoku import (
    BLACK, BOARD_SIZE, DIRECTIONS, EMPTY, GameConfig, LINE_IDX, MOVE_DIRS,
    MOVE_DISTS, PLACE_SIZE, RAYS, ReverseGomoku, WHITE, check_win_at,
    danger_map_for, fwd_run_lengths, runs_pad_for,
)
from . import distill
from . import distill2
from . import openbook
from . import threat_search
from .root_parallel import map_tasks as _map_root_tasks
from .tactics import (
    jump_through as _jump_through, kill_captures, kill_placements, kill_samples,
)

WIN = 1e6                       # 必杀/必败(与步数相关:WIN - ply)
MATE_LIMIT = WIN - 450          # 将死值阈值(步数上限 450):|v| ≥ 此值即为将死值,
                                # 依赖 ply 的将死值不进置换表(跨换位复用会失真)

# ---- 全局评估系数(叶节点) ----
E_KILLABLE4 = 0.9               # 对方可杀 4 连(我方资产)
E_DEAD4 = 0.2                   # 对方死 4 连(可后续造借杀子)
E_OPEN3 = 0.35                  # 对方开放 3 连(可构造为 4)
E_JUMP3 = 0.22                  # 对方跳三(一空档三连,同样可构造为 4)
E_CLOSED3 = 0.1
E_DANGER = 0.02                 # 对方连五危险格(其不能走的格)
E_SELF_KILLABLE4 = 0.95        # 我方可杀 4 连(最大负债)
E_SELF_DEAD4 = 0.3
E_SELF_OPEN3 = 0.5
E_SELF_JUMP3 = 0.25
E_SELF_CLOSED3 = 0.1
E_DOUBLE_KILL = 1.2             # 双杀:对方两处可杀四 = 必死(一次防守只能挡一处)
E_KILL_PLUS3 = 0.5              # 可杀四 + 活三/跳三并存 = 杀完挡完还能续杀(双威胁链)
E_PENDING_KILL = 0.95           # 持子且一步可杀
E_ADJ = 0.02                    # 同色相邻对数:己方集中=负债,对方集中=材料潜力
E_CENTER_OPP = 0.70             # 对方棋子中心度:被推中间=空间大易连五=我方资产
E_CENTER_SELF = 0.80            # 己方棋子中心度:自己居中同样易连五=负债(零和,落子偏中环)
E_CLUSTER = 0.03                # 己方棋子"居中×聚集"联合负债:被推中聚团=大危险
                                # (分散推中间不罚,聚团推中间重罚——直接反制推中战术)
E_SELF_GIFT3 = 0.15             # P2 潜伏威胁:己方三连(活三/跳三)被对方一步
                                # 投送成四的潜在负债(在 E_SELF_OPEN3/JUMP3 之上
                                # 追加的"可被投送"专项惩罚,hell 专用)

# ---- 候选排序分(搜索内部) ----
O_KILL = 1e5
O_DEFEND = 9e4
O_ROUTE = 8e4
O_BUILD4 = 150.0
O_BUILD3 = 40.0
O_BREAK4 = -250.0
O_BREAK3 = -80.0
O_OWN4 = -400.0
O_OWN3 = -50.0
O_FIGHT = 2.0
O_PUSH_OPP = 2.0               # 安置:把对方棋子推回对方棋群(对方越集中越易连五)
O_PUSH_CENTER = 5.0            # 安置:把对方棋子往中间/空间大处推(边角受边界限制不易连五)
O_PUSH_SELF = 1.0              # 安置:避免贴己方棋子
O_BLOCK_SELF4 = 30.0           # 安置:把对方子放到"我方落子即成 4/5"的格 → 封死自己的危险线
                               # (预防性防守:堵己方 3 连延伸,防止被对方扩成可杀四)
O_BLOCK_SELF3 = 10.0           # 安置:堵己方 2 连延伸(次要预防)
O_GIFT_D2 = 3.0                # 安置:落子后两轴各成连二(双二雏形)= 投送礼物
O_GIFT_D3 = 8.0                # 安置:落子后两轴各成活三(双活三,I4 类枢纽)= 大礼
                               # (单轴 lt 公式看不见"两轴各成三",靠 lt2 补齐)
O_LAST_CAP = 1e-6              # 保送槽分:吃对手上一手落子的强制入窗候选(仅排序)
O_PREVENT = 2e3                # P1 预防性封堵候选排序分:堵对方"投送己方三连成
                               # 可杀四"的射线路径(高于造材,低于必守)
PREGIFT_K = 3                  # P1 每次根候选生成最多补充的预防性封堵格数
PREGIFT_HUB_CAP = 48           # P1 投送枢纽枚举上限(成本控制)
PREGIFT_SIM_CAP = 96           # P1 预威胁模拟(投掷子,枢纽)次数上限
O_SPREAD = 1.0                 # 落子:远离己方棋子(打散自己)
O_MIDDLE = 0.0                 # 落子:不奖励中场落子(置 0 停用)。
                                # 棋理:己方棋子尽量在边角(安全,不给自己送连五材料),
                                # 推对方棋子往中间由 O_PUSH_CENTER/O_PUSH_OPP 负责。
O_NOISE = 0.02
O_HIST = 0.05                   # 历史启发权重(截断着法累计分,参与候选入选排序)
O_HIST_CAP = 10000.0            # 历史分上限(防止长期累积淹没静态分)
PVS_EPS = 1e-6                  # PVS 零窗口宽度(浮点评估值)
USE_PVS = True                  # 内部控制:PVS 零窗口(可关闭做一致性对拍)
USE_EXT = True                  # 内部控制:强制局面扩展(可关闭做一致性对拍)

ROOT_K = 8                      # 根节点候选数(每个类别:空位/占领各至多 k 个)
INNER_K = 6                     # 内部节点候选数(浅层)
DEFAULT_DEPTH = 2
MAX_DYNAMIC_DEPTH = 16          # 有时限时迭代加深的上限(深度参数不再是硬上限:
                                # 预算用满前逐层加深,防守需要的深手数由时限兜底)

# ---- LMR(晚着法降价):加深实际到达的层数 ----
# 实测 12s 下 d6≈8.5s、d7≈32s(根 18 候选),alpha-beta 实际只到第 6 层;
# 对排位靠后的非战术着法先按 d-1/d-2 零窗试搜,超上界再按全深度重搜。
# 保护项:战术着法(O_ROUTE 及以上)、前 3 个着法、TT 着法、强制节点、
# 加深扩展节点、浅层(d<3)不降价。
USE_LMR = False                 # 配对对拍显示净负(48 局 8:13),默认关闭;
LMR_MIN_DEPTH = 3               # 代码保留供 A/B(ANTIFIVE_NO_LMR 可关闭)
LMR_MIN_IDX = 3

# ---- 根 α 传播 + PVS(默认关闭,仅 A/B 研究用) ----
# 速度收益极大(根 10 候选 d7 由 ~10s 降到 ~3s),但当前搜索树依赖搜索历史
# (`_include_extras` 会把 TT/killer 着法补进搜索集合,扩展也随之变化):
# 零窗试搜与全窗搜索会走不同的树,零窗返回的"不优"上界可能失真(实测把
# 真实最优着法从 0.871 低估到 0.412,对拍胜率 25%~38%)。除非把候选集合
# 改为与搜索历史无关,否则不得启用。
USE_ROOT_PVS = False

# ---- 大师档能力开关(默认全开;仅供内部 A/B 归因与回退) ----
def _env_off(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_on(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "off", "")


NO_DYNAMIC = _env_off("ANTIFIVE_NO_DYNAMIC")    # 关 P0 时限动态加深
# P1/P2 默认关闭:严格 12s 等时对拍显示它们净负(见 docs/engines.md);
# ANTIFIVE_PREGIFT=1 / ANTIFIVE_GIFT3=1 可启用做 A/B。
NO_PREGIFT = not _env_on("ANTIFIVE_PREGIFT", False)
NO_GIFT3 = not _env_on("ANTIFIVE_GIFT3", False)
NO_HELL_PARAMS = _env_off("ANTIFIVE_NO_HELL_PARAMS")  # P2 参数回退 params.json
NO_LMR = _env_off("ANTIFIVE_NO_LMR")            # 关 LMR(晚着法降价)
NO_ROOT_PVS = _env_off("ANTIFIVE_NO_ROOT_PVS")  # 关根 α 传播 + PVS
# 根节点每类核心候选数(空位/占领;其余名额留给先验/多样性/预防性防守)。
# 8=保留完整根宽(默认,与原实现一致);4=收窄根宽换深度(A/B 用,
# 实测在完整根窗搜索下收益不稳定,故默认不启用)。
ROOT_K_CORE = _env_int("ANTIFIVE_ROOT_CORE", 8)

# ---- KataGo 蒸馏 v2(默认关;「传奇」冻结快照内改为常开) ----
# 开启后:evaluate 在 v1 价值校正之上叠加 distill2 MLP;先验走 18 维特征 +
# 5 格模式。distill_data2 缺失/维度不符时 distill2 自动退化,行为与关闭一致。
USE_DISTILL2 = _env_on("ANTIFIVE_DISTILL2", False)
_ACTIVE2 = bool(USE_DISTILL2 and distill2.ENABLED)


# ---- 大师档深化 VCF:杀链深度/节点/时限显著高于冻结档 ----
# 整局走向常由"占领+安置"强制杀链决定(对 KataGo 复盘:96/96 败局均为移子
# 杀),冻结档 VCF 深度 8/60000 节点/0.6s 在深杀链上漏杀,防守端也因此漏看。
VCF_MAX_DEPTH_HELL = _env_int("ANTIFIVE_HELL_VCF_DEPTH", 12)
VCF_NODE_CAP_HELL = _env_int("ANTIFIVE_HELL_VCF_NODES", 240000)
VCF_TIME_CAP_HELL = _env_float("ANTIFIVE_HELL_VCF_TIME", 1.8)
# 无对局限时(不限时/批量生成)的确定性节点上限:不用墙钟(结果可复现),
# 上限与旧 0.05s 墙钟的典型消耗相当,兼顾批量生成吞吐。
VCF_NODE_CAP_UNLIMITED = max(1000, threat_search.VCF_NODE_CAP // 16)


def _inner_k(depth: int) -> int:
    """深度自适应候选宽度:剩余深度越浅越收窄,用少量精度换大量深度。
    节点数从 6^n 降到 ~6*6*4*4*3^(n-4),深层战术由必守/必杀收窄兜底。"""
    if depth <= 2:
        return 3
    if depth <= 4:
        return 4
    return INNER_K

# ---- 调优参数运行时加载(包内 antifive/params.json,随 wheel 分发) ----
# 白名单:E_*/O_* 数值常量与 ROOT_K/INNER_K(与 tools/param_tune.py 的
# _PARAM_RANGES 对应)。加载只覆盖这些键,防止外部文件篡改搜索常量。
# 桌面覆盖:data/params_hell.json(大师档)优先,其次 data/params.json;
# 都缺失时回退包内文件——网页(Pyodide)没有仓库 data/ 目录,靠包内文件
# 才能吃到调优参数(此前静默回退内置常量的缺口已修)。
_PARAM_NAMES = frozenset(
    k for k, v in list(globals().items())
    if (k.startswith("E_") or k.startswith("O_"))
    and isinstance(v, (int, float)) and not isinstance(v, bool)
) | frozenset({"ROOT_K", "INNER_K"})
_PARAM_INT_KEYS = frozenset({"ROOT_K", "INNER_K"})
_PARAMS_LOCK = threading.Lock()
_PARAMS_LOADED = False


def package_params_path() -> Path:
    """随包分发的调优参数文件(antifive/params.json)。

    桌面仓库/安装包外的 data/params.json 缺失时回退到它;网页(Pyodide)
    只加载 wheel 内资源,必须靠这个文件才能吃到调优参数。"""
    return Path(__file__).resolve().with_name("params.json")


def _default_params_path() -> Path:
    """大师档默认参数文件解析(按优先级):

    1. `data/params_hell.json`(桌面大师档覆盖,可 ANTIFIVE_NO_HELL_PARAMS 关闭);
    2. `data/params.json`(桌面覆盖,param_tune 输出);
    3. 包内 `antifive/params.json`(随 wheel 分发;网页/Pyodide 与纯净安装)。"""
    if not NO_HELL_PARAMS:
        p = data_dir() / "params_hell.json"
        if p.exists():
            return p
    p = data_dir() / "params.json"
    if p.exists():
        return p
    return package_params_path()


def load_params(path: str | os.PathLike | None = None,
                force: bool = False) -> dict:
    """读取调优参数并应用(path=None 时按 _default_params_path 解析:
    data/params_hell.json → data/params.json → 包内 antifive/params.json)。

    返回实际应用的 {键: 值};文件缺失/非法时返回空 dict。线程安全,
    force=False 时进程内只加载一次(配合 ensure_params 懒加载)。"""
    global _PARAMS_LOADED
    with _PARAMS_LOCK:
        if _PARAMS_LOADED and not force:
            return {}
        _PARAMS_LOADED = True
        p = Path(path) if path is not None else _default_params_path()
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        applied = {}
        for k, v in data.items():
            if k not in _PARAM_NAMES or not isinstance(v, (int, float)) \
                    or isinstance(v, bool):
                continue
            value = int(v) if k in _PARAM_INT_KEYS else float(v)
            globals()[k] = value
            applied[k] = value
        return applied


def ensure_params() -> None:
    """首次调用时自动加载调优参数;环境变量 ANTIFIVE_NO_PARAMS=1 可关闭。"""
    if _PARAMS_LOADED:
        return
    if os.environ.get("ANTIFIVE_NO_PARAMS", "").strip().lower() in ("1", "true", "yes"):
        return
    load_params()

# ---- Zobrist 置换表(深度 4+ 的指数级提速来源) ----
_ZR = np.random.default_rng(20260811)
ZOB_BOARD = _ZR.integers(0, 2**63, size=(PLACE_SIZE, 3), dtype=np.uint64)
ZOB_PENDING = _ZR.integers(0, 2**63, size=(PLACE_SIZE + 1,), dtype=np.uint64)
ZOB_PLAYER = _ZR.integers(0, 2**63, size=(2,), dtype=np.uint64)
ZOB_TURN = _ZR.integers(0, 2**63, size=(64,), dtype=np.uint64)
ZOB_WHITE = _ZR.integers(0, 2**63, size=(64,), dtype=np.uint64)
_BOARD_IDX = np.arange(PLACE_SIZE, dtype=np.intp)


def _zobrist(board: np.ndarray, player: int, pending: int,
             turn_count: int, white_turns: int) -> int:
    """局面哈希(棋盘 + 玩家 + 手中棋子 + 回合/白棋回合计数)。"""
    flat = board.reshape(-1)
    key = np.bitwise_xor.reduce(ZOB_BOARD[_BOARD_IDX, flat])
    key ^= ZOB_PENDING[pending + 1]
    key ^= ZOB_PLAYER[player - 1]
    key ^= ZOB_TURN[min(turn_count, 63)]
    key ^= ZOB_WHITE[min(white_turns, 63)]
    return int(key)


class _TT:
    """置换表:单槽、深度优先替换、fail-soft 边界标记(1=精确, 2=下界, 3=上界)。"""
    FLAG_EXACT, FLAG_LOWER, FLAG_UPPER = 1, 2, 3

    def __init__(self, size: int | None = None):
        if size is None:
            # ANTIFIVE_TT_BITS 可调表大小(网页/Pyodide 内存敏感时设 19-20)
            try:
                bits = int(os.environ.get("ANTIFIVE_TT_BITS", "21"))
            except ValueError:
                bits = 21
            size = 1 << max(12, min(24, bits))
        self.mask = size - 1
        self.keys = np.zeros(size, dtype=np.uint64)
        self.values = np.zeros(size, dtype=np.float32)
        self.depths = np.zeros(size, dtype=np.int8)
        self.flags = np.zeros(size, dtype=np.int8)
        self.moves = np.zeros(size, dtype=np.int32)   # -1 = 无
        self.moves[:] = -1
        self.hits = 0
        self.misses = 0

    def probe(self, key: int):
        """返回 (value, depth, flag, move) 或 None(未命中)。"""
        i = key & self.mask
        if self.keys[i] != np.uint64(key):
            self.misses += 1
            return None
        self.hits += 1
        mv = int(self.moves[i])
        return (float(self.values[i]), int(self.depths[i]),
                int(self.flags[i]), mv if mv >= 0 else None)

    def store(self, key: int, depth: int, value: float, flag: int, move) -> None:
        i = key & self.mask
        if self.flags[i] and self.depths[i] > depth:
            return                      # 深度优先:已有更深条目不覆盖
        self.keys[i] = np.uint64(key)
        self.values[i] = np.float32(value)
        self.depths[i] = np.int8(depth)
        self.flags[i] = np.int8(flag)
        self.moves[i] = -1 if move is None else np.int32(move)


class _SearchCtx:
    """一次根搜索的共享状态:置换表 / 时限 / 截断标志 / killer 表 / 节点计数。"""
    __slots__ = ("tt", "deadline", "aborted", "killers", "ext_budget",
                 "nodes", "cb", "history")

    def __init__(self, tt: _TT, deadline: float | None, ext_budget: int = 1):
        self.tt = tt
        self.deadline = deadline
        self.aborted = False
        self.killers = [[-1, -1] for _ in range(64)]
        self.ext_budget = ext_budget
        self.nodes = 0                   # 已搜索节点数(GUI 实时计算量显示)
        self.cb = None                   # 可选进度回调 cb(node_count)
        self.history = np.zeros(PLACE_SIZE, dtype=np.float64)  # 历史启发

# 中心度图:0..1,棋盘中心=1,边角=0(空间越大越易连五)
_CENTER = BOARD_SIZE // 2
CENTER_MAP = np.array(
    [[1.0 - max(abs(r - _CENTER), abs(c - _CENTER)) / _CENTER
      for c in range(BOARD_SIZE)] for r in range(BOARD_SIZE)],
    dtype=np.float32)


def _build_shift_idx():
    """SHIFT_IDX[dr+1][dc+1]: out[r,c] = board2d[r-dr, c-dc] 的平铺索引(越界裁剪),
    SHIFT_OK[dr+1][dc+1]: 越界标记(0=越界,结果应置 0)。"""
    idx = np.empty((3, 3, BOARD_SIZE, BOARD_SIZE), dtype=np.intp)
    ok = np.zeros((3, 3, BOARD_SIZE, BOARD_SIZE), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r_idx = np.clip(np.arange(BOARD_SIZE) - dr, 0, BOARD_SIZE - 1)
            c_idx = np.clip(np.arange(BOARD_SIZE) - dc, 0, BOARD_SIZE - 1)
            idx[dr + 1, dc + 1] = r_idx[:, None] * BOARD_SIZE + c_idx[None, :]
            r_ok = (np.arange(BOARD_SIZE) - dr >= 0) & (np.arange(BOARD_SIZE) - dr < BOARD_SIZE)
            c_ok = (np.arange(BOARD_SIZE) - dc >= 0) & (np.arange(BOARD_SIZE) - dc < BOARD_SIZE)
            ok[dr + 1, dc + 1] = r_ok[:, None] & c_ok[None, :]
    return idx, ok


try:  # Cython 热路径(可选,缺失时全部走下面纯 Python 实现)
    from . import _fast as _FAST
except Exception:  # pragma: no cover - 未编译环境
    _FAST = None

SHIFT_IDX, SHIFT_OK = _build_shift_idx()
CENTER_FLAT = CENTER_MAP.reshape(-1)


def _shift(board2d: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """out[r,c] = board2d[r-dr, c-dc],越界补 0。静态索引查表,无 pad/clip 开销。"""
    if _FAST is not None:
        return _FAST.shift(board2d, dr, dc)
    out = board2d.reshape(-1)[SHIFT_IDX[dr + 1, dc + 1]].reshape(BOARD_SIZE, BOARD_SIZE)
    out *= SHIFT_OK[dr + 1, dc + 1]
    return out


def _other(color: int) -> int:
    return WHITE if color == BLACK else BLACK


# 8 邻域在棋盘内的实际邻居数(边界 3/5,内部 8),用于由 nb_b+nb_w 反推空邻数
_DEGREE = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
for _dr in (-1, 0, 1):
    for _dc in (-1, 0, 1):
        if _dr == 0 and _dc == 0:
            continue
        _DEGREE[max(0, -_dr):BOARD_SIZE - max(0, _dr),
                max(0, -_dc):BOARD_SIZE - max(0, _dc)] += 1

_LINE_MAP_CACHE: "OrderedDict[bytes, tuple]" = OrderedDict()
LINE_MAP_CACHE_MAX = 4096


def _line_maps_combo_full(board: np.ndarray) -> tuple:
    """双色线图一次算齐并按棋盘字节 LRU 缓存:
    (lt_black, lt_white, nb_black, nb_white, lt2_black, lt2_white)。
    lt 保持历史语义(供候选评分,不改行为)。"""
    key = board.tobytes()
    v = _LINE_MAP_CACHE.get(key)
    if v is not None:
        _LINE_MAP_CACHE.move_to_end(key)
        return v
    if _FAST is not None:
        res = _FAST.line_maps(np.ascontiguousarray(board, dtype=np.int8))
        if len(_LINE_MAP_CACHE) >= LINE_MAP_CACHE_MAX:
            _LINE_MAP_CACHE.popitem(last=False)
        _LINE_MAP_CACHE[key] = res
        return res
    lts = []
    lt2s = []
    for color in (BLACK, WHITE):
        runs = fwd_run_lengths(board == color)
        lt = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
        lt2 = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
        for d1, d2 in ((0, 1), (2, 3), (4, 5), (6, 7)):
            dr, dc = DIRECTIONS[d1]
            f = _shift(runs[d1], -dr, -dc)      # 越过 t 的前向连子(沿 d1 的邻居)
            b = _shift(runs[d2], dr, dc)        # 越过 t 的后向连子(沿 d2 的邻居)
            run = (1 + f + b).astype(np.uint8)
            # 维护四轴线的前两大:lt2 触发"双二/双活三"识别
            lt2 = np.maximum(lt2, np.minimum(lt, run))
            lt = np.maximum(lt, run)
        lts.append(lt)
        lt2s.append(lt2)
    nbs = []
    for color in (BLACK, WHITE):
        m = (board == color).astype(np.uint8)
        nb = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb += _shift(m, dr, dc)
        nbs.append(nb)
    res = (lts[0], lts[1], nbs[0], nbs[1], lt2s[0], lt2s[1])
    if len(_LINE_MAP_CACHE) >= LINE_MAP_CACHE_MAX:
        _LINE_MAP_CACHE.popitem(last=False)
    _LINE_MAP_CACHE[key] = res
    return res


def _line_maps_combo(board: np.ndarray) -> tuple:
    return _line_maps_combo_full(board)


def _line_maps(board: np.ndarray, color: int) -> tuple:
    """兼容接口:返回 (lt_color, nb_opp, nb_same)。"""
    opp = _other(color)
    lt_b, lt_w, nb_b, nb_w = _line_maps_combo(board)[:4]
    if color == BLACK:
        return lt_b, nb_w, nb_b
    return lt_w, nb_b, nb_w


def _loses_to_kill(board: np.ndarray, player: int, white_turns: int,
                   turn_count: int, config: GameConfig) -> bool:
    """当前局面下,对方是否有借我方连珠的必杀(我方可杀 4 连存在)。"""
    return (turn_count >= config.loss_start_turns
            and _run_material(board, player, white_turns, turn_count,
                              config)["killable4"] > 0)


def _defense_reserve_needed(board: np.ndarray, player: int, pending: int,
                            turn_count: int, white_turns: int,
                            config: GameConfig) -> bool:
    """是否需要为本步预留大额防守 VCF 预算。

    防守校验的价值集中在"己方已有危险四连(对方下一步可完成)/对方已有可杀
    四(攻防转换)"这类局面;仅有三连/跳三的常规中盘,己方着法造四的即时危险
    已由搜索硬判负覆盖,防守校验几乎无事可做,预留的大预算会被白扔——回收
    给主搜索加深(见 _root_iterative)。"""
    if pending >= 0:
        return True
    if turn_count < config.loss_start_turns:
        return False
    m_black, m_white, _, _, _, _ = _board_features(
        board, white_turns, turn_count, config)
    mine = m_black if player == BLACK else m_white
    theirs = m_white if player == BLACK else m_black
    return bool(mine["killable4"] or mine["dead4"] or theirs["killable4"])


def _threats(board: np.ndarray, player: int, white_turns: int,
             turn_count: int, config: GameConfig) -> tuple:
    """己方被杀威胁分析,返回 (ends, route):
    - ends :己方四形(直/跳,含断四/跳四)的完成格,对方可占领我方棋子后
            安置完成我方五连(危险点);
    - route:堵杀棋路线可用的中间格(完成格 T 与对方借用的我方棋子之间的空格)。
    判负窗口外或对方无法移子时无威胁。"""
    ends, route = set(), set()
    if turn_count < config.loss_start_turns:
        return ends, route
    opp = _other(player)
    if opp == WHITE and white_turns < config.white_restrict_turns:
        return ends, route
    flat = board.reshape(-1)
    _, _, fours_black, fours_white, _, _ = _board_features(board, white_turns, turn_count, config)
    fours_mine = fours_black if player == BLACK else fours_white
    for cells, T_list in fours_mine:
        for T in T_list:
            for d in range(MOVE_DIRS):
                is_bridge = False
                mid = set()               # 每根射线单独收集,仅桥射线并入 route
                for k in range(MOVE_DISTS):
                    cell = int(RAYS[T, d, k])
                    if cell < 0:
                        break
                    v = flat[cell]
                    if v == EMPTY:
                        mid.add(cell)
                        continue
                    if v == player and cell not in cells:
                        is_bridge = True
                    break
                if is_bridge:
                    ends.add(T)
                    route.update(mid)
    return ends, route


# ---- P1 投送预威胁(预防性防守候选) ----
def _gift_lethal(board: np.ndarray, player: int, c: int, t: int,
                 white_turns: int, turn_count: int, config: GameConfig) -> bool:
    """对方一个回合把己方棋子 c(被占领)投送到空位 t 后,是否立刻成杀威胁。

    t 满足"落己方子即成四/五"(由 _pregift_defense 的 lt_self ≥ 4 筛选)。
    模拟 c 离盘 + 己方子在 t 落定:
    - 直接在 t 完成己方五连 → 对方本回合即胜(必死);
    - 否则用 _threats 判定是否形成"对方下一步可完成的四"(有桥子可达完成格)。"""
    if board.reshape(-1)[c] != player or board.reshape(-1)[t] != EMPTY:
        return False
    b2 = board.copy()
    cr, cc = divmod(int(c), BOARD_SIZE)
    tr, tc = divmod(int(t), BOARD_SIZE)
    b2[cr, cc] = EMPTY
    b2[tr, tc] = player
    if check_win_at(b2, tr, tc, player):
        return True
    ends, _route = _threats(b2, player, white_turns, turn_count, config)
    return bool(ends)


def _pregift_defense(board: np.ndarray, player: int, white_turns: int,
                     turn_count: int, config: GameConfig) -> dict:
    """P1 预防性封堵格:{cell: 覆盖权重}。

    枚举"投送枢纽"t(空位且落己方子后经过 t 的最大连续数 ≥ 4,即对方一个
    回合可把己方棋子投送到此造出己方四连/五连),以及每个射向上可作为
    投掷子的己方棋子 c(c→t 路径全空);模拟判定该投送确为杀威胁后,把
    c→t 路径上的空格记为封堵点(占任一格即断该投送)。权重 = 覆盖的
    (投掷子, 枢纽) 对数,供根候选按"一子堵多条"排序。

    仅根候选生成调用(diversity=True),枚举上限 PREGIFT_HUB_CAP /
    PREGIFT_SIM_CAP 控制成本;结论不作为硬剪枝,交由搜索评估。"""
    if NO_PREGIFT or turn_count < config.loss_start_turns:
        return {}
    opp = _other(player)
    if opp == WHITE and white_turns < config.white_restrict_turns:
        return {}                       # 对手(白)禁移期无法投送
    flat = board.reshape(-1)
    # 枢纽 = 落己方子后经过该格的最大子数 ≥ 4(含跳形:跳三缺口/一空档延伸,
    # 单靠 lt_self 的连续游程看不见这些格,用 jump_through 精确判定)。
    hubs = []
    for t0 in np.nonzero(flat == EMPTY)[0]:
        ti = int(t0)
        r, c = divmod(ti, BOARD_SIZE)
        if _jump_through(board, r, c, player) >= 4:
            hubs.append(ti)
    if not hubs:
        return {}
    cover: dict = {}
    n_sim = 0
    for t in hubs[:PREGIFT_HUB_CAP]:
        ti = int(t)
        for d in range(MOVE_DIRS):
            path = []
            for k in range(MOVE_DISTS):
                cell = int(RAYS[ti, d, k])
                if cell < 0:
                    break
                v = flat[cell]
                if v == EMPTY:
                    path.append(cell)
                    continue
                if v == player and path:
                    if n_sim >= PREGIFT_SIM_CAP:
                        return cover
                    n_sim += 1
                    if _gift_lethal(board, player, cell, ti, white_turns,
                                    turn_count, config):
                        for x in path:
                            cover[x] = cover.get(x, 0.0) + 1.0
                break
    return cover


def _append_pregift(out: list, cover: dict, mask: np.ndarray,
                    limit: int = PREGIFT_K) -> None:
    """把覆盖权重最高的若干合法封堵格补进候选(去重、低分,O_PREVENT 级)。"""
    if not cover:
        return
    present = {m for _, m in out}
    items = [(w, c) for c, w in cover.items()
             if bool(mask[c]) and c not in present]
    if not items:
        return
    items.sort(key=lambda t: (-t[0], -t[1]))
    for w, c in items[:limit]:
        out.append((O_PREVENT + min(float(w), 99.0), int(c)))


# ---- 双色线级模式提取(Phase B:跳三/断四/双杀感知) ----
# 模块级预构建:4 条轴线主方向 (E, S, SE, SW) 每条线的格子索引(纯 Python 列表)
_LINES4 = [[[int(x) for x in row if x >= 0] for row in LINE_IDX[d]]
           for d in (3, 1, 7, 6)]
RAYS_LIST = RAYS.tolist()

_LINE_SEQ = tuple(line for axis in _LINES4 for line in axis)


def _make_line_getter(cells):
    if len(cells) == 1:
        i = cells[0]
        return lambda flat: (flat[i],)
    return itemgetter(*cells)


_LINE_GETTERS = tuple(_make_line_getter(cells) for cells in _LINE_SEQ)

_FEAT_CACHE: "OrderedDict[bytes, tuple]" = OrderedDict()
FEAT_CACHE_MAX = 4096
_EMPTY_FEAT = None


def _board_features(board: np.ndarray, white_turns: int,
                    turn_count: int, config: GameConfig) -> tuple:
    """双色线级模式提取(一次扫描同时统计黑白双方材料,跳三/断四感知)。

    返回 (m_black, m_white, fours_black, fours_white, dang_black, dang_white):
      m_color     : {killable4, dead4, open3, jump3, closed3}
      fours_color : [(四形格集合, 完成格列表[开放端/缺口]), ...]
      dang_color  : 该色落子即连五的完成格集合(其不能走的格)
    killable4:四形(直/跳)存在"完成格"且有借杀子可达(对方一步可杀)。
    结果按棋盘字节缓存(LRU),节点内多次调用(候选/威胁/评估)只扫一次。"""
    global _EMPTY_FEAT
    if turn_count < config.loss_start_turns:
        if _EMPTY_FEAT is None:
            empty_m = {"killable4": 0, "dead4": 0, "open3": 0,
                       "jump3": 0, "closed3": 0}
            _EMPTY_FEAT = (dict(empty_m), dict(empty_m), [], [],
                           set(), set())
        return _EMPTY_FEAT
    key = board.tobytes()
    v = _FEAT_CACHE.get(key)
    if v is not None:
        _FEAT_CACHE.move_to_end(key)
        return v
    if _FAST is not None:
        feats = _FAST.board_features(
            np.ascontiguousarray(board, dtype=np.int8), white_turns, turn_count,
            config.loss_start_turns, config.white_restrict_turns,
            _LINES4, RAYS_LIST)
        if len(_FEAT_CACHE) >= FEAT_CACHE_MAX:
            _FEAT_CACHE.popitem(last=False)
        _FEAT_CACHE[key] = feats
        return feats
    flat = board.reshape(-1).tolist()          # 纯 Python 列表,标量索引快 10 倍
    m_black = {"killable4": 0, "dead4": 0, "open3": 0, "jump3": 0, "closed3": 0}
    m_white = {"killable4": 0, "dead4": 0, "open3": 0, "jump3": 0, "closed3": 0}
    fours_black: list = []
    fours_white: list = []
    dang_black: set = set()
    dang_white: set = set()
    # 杀方能否移子:白受限则黑棋材料不可被杀
    cap_black = white_turns >= config.white_restrict_turns
    cap_white = True                            # 杀白棋的是黑,黑永不受限
    for li, line_cells in enumerate(_LINE_SEQ):
        vals = _LINE_GETTERS[li](flat)
        n = len(vals)
        i = 0
        while i < n:                        # 游程:(值, 起点, 终点)
            j = i + 1
            v = vals[i]
            while j < n and vals[j] == v:
                j += 1
            s, e = i, j
            i = j
            if v == 0:
                continue
            L = e - s
            is_black = v == BLACK
            m = m_black if is_black else m_white
            fours = fours_black if is_black else fours_white
            dang = dang_black if is_black else dang_white
            cap = cap_black if is_black else cap_white
            l_open = s > 0 and vals[s - 1] == 0
            r_open = e < n and vals[e] == 0
            if L == 4:
                _tally_four(flat, line_cells, m, fours, dang, v, s, e,
                            l_open, r_open, None, cap)
            elif L == 3:
                if l_open and r_open:
                    m["open3"] += 1
                else:
                    m["closed3"] += 1
            # 跳形:隔一个空位的同色连子(前段 + 空 + 后段)
            if e + 2 < n and vals[e] == 0 and vals[e + 1] == v:
                j2 = e + 1
                while j2 < n and vals[j2] == v:
                    j2 += 1
                total = L + (j2 - e - 1)
                if total >= 4:
                    # ≥4 子带 1 空档(含 3+2/2+3 拆分的"五带缺"如 XXX_XX):
                    # 填上空档即成连五,比四更致命,完成格=空档
                    _tally_four(flat, line_cells, m, fours, dang, v, s, j2,
                                l_open, j2 < n and vals[j2] == 0, e, cap)
                elif total == 3:
                    m["jump3"] += 1
    feats = (m_black, m_white, fours_black, fours_white,
             dang_black, dang_white)
    if len(_FEAT_CACHE) >= FEAT_CACHE_MAX:
        _FEAT_CACHE.popitem(last=False)
    _FEAT_CACHE[key] = feats
    return feats


def _tally_four(flat, cells, m, fours, dang, color, s, e, l_open, r_open,
                gap, killer_can_capture) -> None:
    """统计一个四形(直 s..e-1 或跳:两段 s..gap-1 ∪ gap+1..e-1)的材料与可杀性。
    完成格:直四 = 开放端;跳四 = 缺口(端部补子只得 5 子带缺,不成五)。"""
    end_Ts = []
    if gap is not None:
        end_Ts.append(cells[gap])
    else:
        if l_open:
            end_Ts.append(cells[s - 1])
        if r_open:
            end_Ts.append(cells[e])
    if gap is None:
        run_cells = set(cells[s:e])
    else:
        run_cells = set(cells[s:gap]) | set(cells[gap + 1:e])
    killable = False
    if killer_can_capture:
        for T in end_Ts:
            if _end_reachable_list(flat, T, color, run_cells):
                killable = True
                break
    if killable:
        m["killable4"] += 1
    else:
        m["dead4"] += 1
    fours.append((run_cells, end_Ts))
    dang.update(end_Ts)


def _end_reachable_list(flat: list, T: int, color: int, run: set) -> bool:
    """完成格 T 可被借杀:存在 color 的棋子(不在该四形上)射线直通 T。"""
    for d in range(MOVE_DIRS):
        for cell in RAYS_LIST[T][d]:
            if cell < 0:
                break
            v = flat[cell]
            if v == 0:
                continue
            if v == color and cell not in run:
                return True
            break
    return False


def _run_material(board: np.ndarray, color: int, white_turns: int,
                  turn_count: int, config: GameConfig) -> dict:
    """单色材料(兼容旧接口,内部用双色扫描)。"""
    m_black, m_white, _, _, _, _ = _board_features(board, white_turns, turn_count, config)
    return m_black if color == BLACK else m_white


_POS_CACHE: "OrderedDict[bytes, tuple]" = OrderedDict()
POS_CACHE_MAX = 4096


def _positional_summary(board: np.ndarray) -> tuple:
    """双色位置摘要一次算齐并按棋盘字节 LRU 缓存:
    (adj_black, adj_white, cluster_black, cluster_white,
     center_black, center_white)。
    adj  :同色 8 邻域相邻对数(聚集度);
    cluster:"居中×聚集"联合度(8 邻域同色数 × 中心度之和);
    center:棋子中心度之和。"""
    key = board.tobytes()
    v = _POS_CACHE.get(key)
    if v is not None:
        _POS_CACHE.move_to_end(key)
        return v
    mb = (board == BLACK).astype(np.int32)
    mw = (board == WHITE).astype(np.int32)
    # 双色叠成 (2,15,15),8 邻域用切片一次算完(越界补 0,与 _shift 等价)
    m = np.stack((mb, mw))
    nb = np.zeros_like(m)
    nb[:, 1:, :] += m[:, :-1, :]
    nb[:, :-1, :] += m[:, 1:, :]
    nb[:, :, 1:] += m[:, :, :-1]
    nb[:, :, :-1] += m[:, :, 1:]
    nb[:, 1:, 1:] += m[:, :-1, :-1]
    nb[:, :-1, :-1] += m[:, 1:, 1:]
    nb[:, 1:, :-1] += m[:, :-1, 1:]
    nb[:, :-1, 1:] += m[:, 1:, :-1]
    nb_b, nb_w = nb[0], nb[1]
    # 同色相邻对 = 8 邻域计数之和 / 2
    adj_b = int((mb * nb_b).sum()) // 2
    adj_w = int((mw * nb_w).sum()) // 2
    res = (adj_b, adj_w,
           float((mb * nb_b * CENTER_MAP).sum()),
           float((mw * nb_w * CENTER_MAP).sum()),
           float((mb * CENTER_MAP).sum()),
           float((mw * CENTER_MAP).sum()))
    if len(_POS_CACHE) >= POS_CACHE_MAX:
        _POS_CACHE.popitem(last=False)
    _POS_CACHE[key] = res
    return res


def evaluate(board: np.ndarray, player: int, pending: int, turn_count: int,
             white_turns: int, config: GameConfig) -> float:
    """全局局面评估,从 player 视角,值约 ∈ [-3, 3]。
    资产:对方 3/4 连材料(含跳三/断四)与可杀性、双杀结构、对方受困度、聚集度;
    负债:己方连珠被借杀风险、双杀负债、己方聚集度(集中=易连五)。"""
    opp = _other(player)
    m_black, m_white, _, _, dang_black, dang_white = _board_features(
        board, white_turns, turn_count, config)
    mine = m_black if player == BLACK else m_white
    theirs = m_white if player == BLACK else m_black
    danger_cnt = len(dang_white) if player == BLACK else len(dang_black)
    danger_mine = len(dang_black) if player == BLACK else len(dang_white)
    s = 0.0
    s += E_KILLABLE4 * theirs["killable4"] + E_DEAD4 * theirs["dead4"]
    s += E_OPEN3 * theirs["open3"] + E_JUMP3 * theirs["jump3"] \
        + E_CLOSED3 * theirs["closed3"]
    s -= E_SELF_KILLABLE4 * mine["killable4"] + E_SELF_DEAD4 * mine["dead4"]
    s -= E_SELF_OPEN3 * mine["open3"] + E_SELF_JUMP3 * mine["jump3"] \
        + E_SELF_CLOSED3 * mine["closed3"]
    # P2 潜伏威胁:己方三连可被对方一步投送成四(对方造杀素材),在基础
    # 负债系数之上追加"可被投送"专项惩罚
    if not NO_GIFT3:
        s -= E_SELF_GIFT3 * (mine["open3"] + mine["jump3"])
    # 双杀:对方两处可杀四 = 一次防守只能挡一处,必死;可杀四+三 = 双威胁链
    if theirs["killable4"] >= 2:
        s += E_DOUBLE_KILL
    elif theirs["killable4"] == 1 and theirs["open3"] + theirs["jump3"] >= 1:
        s += E_KILL_PLUS3
    if mine["killable4"] >= 2:
        s -= E_DOUBLE_KILL
    elif mine["killable4"] == 1 and mine["open3"] + mine["jump3"] >= 1:
        s -= E_KILL_PLUS3
    # 危险格:该色落子即连五的格(其不能走的格),由提取器完成格集合给出
    empty = PLACE_SIZE - int(np.count_nonzero(board))
    s += E_DANGER * min(danger_cnt, 20)
    s -= E_DANGER * min(danger_mine, 20)
    # 聚集度/中心度/"居中×聚集"联合度:双色一次算齐 + 按棋盘缓存
    adj_b, adj_w, cl_b, cl_w, ct_b, ct_w = _positional_summary(board)
    if player == BLACK:
        s += E_ADJ * (adj_w - adj_b)
        s += E_CENTER_OPP * ct_w - E_CENTER_SELF * ct_b
        s += E_CLUSTER * (cl_w - cl_b)
    else:
        s += E_ADJ * (adj_b - adj_w)
        s += E_CENTER_OPP * ct_b - E_CENTER_SELF * ct_w
        s += E_CLUSTER * (cl_b - cl_w)
    their_safe = empty - danger_cnt
    if their_safe <= 2:
        s += 0.8
    elif their_safe <= 5:
        s += 0.4
    elif their_safe <= 10:
        s += 0.2
    my_safe = empty - danger_mine
    if my_safe <= 2:
        s -= 0.8
    elif my_safe <= 5:
        s -= 0.4
    if pending >= 0 and turn_count >= config.loss_start_turns \
            and kill_placements(board, player, pending, turn_count,
                                white_turns, config).size:
        s += E_PENDING_KILL
    if _ACTIVE2 and distill2.HAS_VALUE:
        # KataGo 蒸馏 v2 价值校正(MLP):激活时**替换** v1——离线诊断显示
        # v2 与目标相关性显著更高(0.68 vs 0.33);叠加配置 12s×96 仅 47:49
        # (约五五开),替换后 75:21(78.1%),故采用替换语义。
        s += distill2.value_score(_value2_features(
            player, m_black, m_white, danger_cnt, danger_mine, empty,
            adj_b, adj_w, cl_b, cl_w, ct_b, ct_w,
            pending, turn_count, white_turns))
    elif distill.ENABLED:
        # KataGo 蒸馏 v1 价值校正(未训练/未启用时返回 0,不改变行为)
        s += distill.value_score(_value_features(
            player, m_black, m_white, danger_cnt, danger_mine, empty,
            adj_b, adj_w, cl_b, cl_w, ct_b, ct_w))
    return s


# ---- 候选静态排序(廉价查表,仅用于排序/剪枝) ----
def _place_targets(board: np.ndarray, c: int) -> list:
    """占领 c 后的可达安置格(基于空位射线,不模拟)。"""
    flat = board.reshape(-1)
    out = []
    for d in range(MOVE_DIRS):
        for k in range(MOVE_DISTS):
            cell = int(RAYS[c, d, k])
            if cell < 0 or flat[cell] != EMPTY:
                break
            out.append(cell)
    return out


_PRIOR_DIM = 14          # 蒸馏策略先验标量特征维数(与 distill_data 对齐)
_VALUE_DIM = 20          # 蒸馏价值校正特征维数


def _prior_features(board: np.ndarray, player: int, pending: int,
                    turn_count: int, white_turns: int, config: GameConfig,
                    moves) -> tuple:
    """蒸馏先验特征(与 tools/train_distill.py 同一实现,保证训练/推理一致)。

    返回 (scalars (M,_PRIOR_DIM) float64, tokens (M,8) uint8)。"""
    mv = np.asarray(moves, dtype=np.int64)
    flat = board.reshape(-1)
    opp = _other(player)
    lt_b, lt_w, nb_b, nb_w, _, _ = _line_maps_combo(board)
    lt_self_f = (lt_b if player == BLACK else lt_w).reshape(-1)
    lt_opp_f = (lt_w if player == BLACK else lt_b).reshape(-1)
    nb_self_f = (nb_b if player == BLACK else nb_w).reshape(-1)
    nb_opp_f = (nb_w if player == BLACK else nb_b).reshape(-1)
    ends, route = _threats(board, player, white_turns, turn_count, config)
    if pending >= 0:
        kill_cells = {int(x) for x in kill_placements(
            board, player, pending, turn_count, white_turns, config)}
    else:
        kill_cells = {int(c) for c, _ in kill_captures(
            board, player, pending, turn_count, white_turns, config)}
    open_dirs = np.zeros(len(mv), dtype=np.float64)
    for dr, dc in DIRECTIONS:
        rr = mv // BOARD_SIZE + dr
        cc = mv % BOARD_SIZE + dc
        ok = (rr >= 0) & (rr < BOARD_SIZE) & (cc >= 0) & (cc < BOARD_SIZE)
        idx = np.where(ok, rr * BOARD_SIZE + cc, 0)
        open_dirs += ((flat[idx] == EMPTY) & ok)
    jumps_self = np.empty(len(mv), dtype=np.float64)
    jumps_opp = np.empty(len(mv), dtype=np.float64)
    for i, m in enumerate(mv):
        r, c = divmod(int(m), BOARD_SIZE)
        jumps_self[i] = _jump_through(board, r, c, player)
        jumps_opp[i] = _jump_through(board, r, c, opp)
    ends_arr = np.fromiter(ends, dtype=np.int64, count=len(ends))
    route_arr = np.fromiter(route, dtype=np.int64, count=len(route))
    kill_arr = np.fromiter(kill_cells, dtype=np.int64, count=len(kill_cells))
    sc = np.empty((len(mv), _PRIOR_DIM), dtype=np.float64)
    sc[:, 0] = 1.0
    sc[:, 1] = lt_self_f[mv] / 5.0
    sc[:, 2] = lt_opp_f[mv] / 5.0
    sc[:, 3] = nb_self_f[mv] / 8.0
    sc[:, 4] = nb_opp_f[mv] / 8.0
    sc[:, 5] = CENTER_FLAT[mv]
    sc[:, 6] = np.minimum(jumps_self, 8.0) / 8.0
    sc[:, 7] = np.minimum(jumps_opp, 8.0) / 8.0
    sc[:, 8] = (flat[mv] == opp).astype(np.float64)
    sc[:, 9] = 1.0 if pending >= 0 else 0.0
    sc[:, 10] = np.isin(mv, ends_arr).astype(np.float64)
    sc[:, 11] = np.isin(mv, route_arr).astype(np.float64)
    sc[:, 12] = np.isin(mv, kill_arr).astype(np.float64)
    sc[:, 13] = open_dirs / 8.0
    return sc, distill.pattern_tokens(board, player, mv)


_PRIOR2_DIM = _PRIOR_DIM + 4     # v2 先验标量特征维数(与 distill_data2 对齐)


def _prior2_features(board: np.ndarray, player: int, pending: int,
                     turn_count: int, white_turns: int, config: GameConfig,
                     moves) -> tuple:
    """蒸馏 v2 先验特征:在 v1 的 14 维之上追加 4 维 push 专属特征(共 18 维),
    模式 token 换用 5 格窗口。返回 (scalars (M,_PRIOR2_DIM), tokens (M,8))。

    追加特征(列 14-17):
    - 14 空位候选(=1 普通落子);15 占领候选可达安置格数/16;
    - 16 该占领的强制杀安置数/8;17 安置候选落在杀格(=1,手中棋子一步杀)。"""
    mv = np.asarray(moves, dtype=np.int64)
    sc14, _ = _prior_features(board, player, pending, turn_count, white_turns,
                              config, mv)
    flat = board.reshape(-1)
    opp = _other(player)
    sc = np.empty((len(mv), _PRIOR2_DIM), dtype=np.float64)
    sc[:, :_PRIOR_DIM] = sc14
    sc[:, 14] = (flat[mv] == EMPTY).astype(np.float64)
    reach = np.zeros(len(mv), dtype=np.float64)
    kill_cnt = np.zeros(len(mv), dtype=np.float64)
    if len(mv) and pending < 0:
        kill_map: dict = {}
        for c, _ in kill_captures(board, player, pending, turn_count,
                                  white_turns, config, mv):
            kill_map[int(c)] = kill_map.get(int(c), 0) + 1
        for i, m in enumerate(mv):
            mi = int(m)
            if flat[mi] != opp:
                continue
            reach[i] = len(_place_targets(board, mi))
            kill_cnt[i] = kill_map.get(mi, 0)
    sc[:, 15] = np.minimum(reach, 16.0) / 16.0
    sc[:, 16] = np.minimum(kill_cnt, 8.0) / 8.0
    if pending >= 0:
        kill_set = {int(x) for x in kill_placements(
            board, player, pending, turn_count, white_turns, config)}
        if kill_set:
            sc[:, 17] = np.isin(
                mv, np.fromiter(kill_set, dtype=np.int64, count=len(kill_set))
            ).astype(np.float64)
        else:
            sc[:, 17] = 0.0
    else:
        sc[:, 17] = 0.0
    return sc, distill2.pattern_tokens5(board, player, mv)


def _prior_map(board: np.ndarray, player: int, pending: int, turn_count: int,
               white_turns: int, config: GameConfig, legal) -> np.ndarray:
    """全部合法着法的蒸馏先验打分(平铺 225 数组,非法位为 0)。
    v2 启用时走 MLP + 5 格模式;否则回退 v1 线性模型。"""
    pm = np.zeros(PLACE_SIZE, dtype=np.float64)
    if not len(legal):
        return pm
    if _ACTIVE2 and distill2.HAS_PRIOR:
        sc, tk = _prior2_features(board, player, pending, turn_count,
                                  white_turns, config, legal)
        pm[legal] = distill2.prior_scores(sc, tk)
    elif distill.HAS_PRIOR:
        sc, tk = _prior_features(board, player, pending, turn_count,
                                 white_turns, config, legal)
        pm[legal] = distill.prior_scores(sc, tk)
    return pm


def _append_prior_top(out: list, legal, prior_map: np.ndarray,
                      limit: int = 2) -> None:
    """把先验打分最高的至多 limit 个着法补进候选(去重、低分保底)。"""
    if prior_map is None or limit <= 0 or not len(legal):
        return
    present = {m for _, m in out}
    pool = np.asarray([int(t) for t in legal if int(t) not in present],
                      dtype=np.int64)
    if not pool.size:
        return
    n = min(limit, pool.size)
    top = pool[np.argsort(prior_map[pool])[-n:]]
    for m in top:
        out.append((0.0, int(m)))


def _value_features(player: int, m_black: dict, m_white: dict, danger_cnt: int,
                    danger_mine: int, empty: int, adj_b: float, adj_w: float,
                    cl_b: float, cl_w: float, ct_b: float,
                    ct_w: float) -> np.ndarray:
    """价值校正特征(我方/对方对称排列,训练端同实现)。"""
    mine, theirs = (m_black, m_white) if player == BLACK else (m_white, m_black)
    adj_m, adj_t = (adj_b, adj_w) if player == BLACK else (adj_w, adj_b)
    cl_m, cl_t = (cl_b, cl_w) if player == BLACK else (cl_w, cl_b)
    ct_m, ct_t = (ct_b, ct_w) if player == BLACK else (ct_w, ct_b)
    f = np.empty(_VALUE_DIM, dtype=np.float64)
    f[0] = 1.0
    for i, k in enumerate(("killable4", "dead4", "open3", "jump3", "closed3")):
        f[1 + i] = mine[k]
        f[6 + i] = theirs[k]
    f[11] = danger_mine
    f[12] = danger_cnt
    f[13] = empty
    f[14] = adj_m
    f[15] = adj_t
    f[16] = cl_m
    f[17] = cl_t
    f[18] = ct_m
    f[19] = ct_t
    return f


_VALUE2_DIM = _VALUE_DIM + 5     # v2 价值特征维数(与 distill_data2 对齐)


def _value2_features(player: int, m_black: dict, m_white: dict, danger_cnt: int,
                     danger_mine: int, empty: int, adj_b: float, adj_w: float,
                     cl_b: float, cl_w: float, ct_b: float, ct_w: float,
                     pending: int, turn_count: int,
                     white_turns: int) -> np.ndarray:
    """价值校正 v2 特征(25 维):v1 的 20 维 + 回合状态/手中棋子/双方安全空位
    (安全空位=空位数-该色连五危险格数,越大越不容易被送连五)。"""
    f = np.empty(_VALUE2_DIM, dtype=np.float64)
    f[:20] = _value_features(player, m_black, m_white, danger_cnt, danger_mine,
                             empty, adj_b, adj_w, cl_b, cl_w, ct_b, ct_w)
    f[20] = 1.0 if pending >= 0 else 0.0
    f[21] = min(turn_count, 63) / 63.0
    f[22] = min(white_turns, 63) / 63.0
    f[23] = (empty - danger_mine) / PLACE_SIZE
    f[24] = (empty - danger_cnt) / PLACE_SIZE
    return f


def _candidates(board: np.ndarray, player: int, pending: int, turn_count: int,
                white_turns: int, config: GameConfig, k: int,
                history=None, out_legal=None, diversity: bool = False,
                prior: bool = False, opp_last: int = -1,
                core_k: int | None = None) -> list:
    """选择性候选着法 [(排序分, move), ...] 降序。杀/解杀/堵路线 + 前 k 构造与位置着。
    history: 可选历史启发分数组(参与入选排序);
    out_legal: 可选 list,回填合法掩码数组(避免调用方重复计算掩码);
    diversity: 根节点专用,额外保留"贴己方棋群/安静关键点"型着法入候选;
    prior: 根节点专用,叠加 KataGo 蒸馏先验(缺失模型时自动忽略);
    opp_last: 可选,对手最近落下的己方子格(-1=未知);命中时"吃该子"保送入窗;
    core_k: 可选,核心类别(空位/占领/安置)各自入选上限;根节点用更小的
    核心宽度控制根宽(总候选 = 核心 + 先验/多样性/预防性防守等补充项)。"""
    kk = k if core_k is None else min(core_k, k)
    opp = _other(player)
    flat = board.reshape(-1)
    lt_b, lt_w, nb_b, nb_w, lt2_b, lt2_w = _line_maps_combo(board)
    nbe = _DEGREE - nb_b - nb_w
    lt_self = lt_b if player == BLACK else lt_w
    mask = ReverseGomoku.legal_mask_for(board, player, pending, white_turns,
                                        config, turn_count,
                                        pre_nb_empty=nbe)
    legal = np.nonzero(mask)[0]
    if out_legal is not None:
        out_legal.clear()
        out_legal.append(mask)
    if not legal.size:
        return []
    lt_self_f = lt_self.reshape(-1)
    lt_opp_f = (lt_w if player == BLACK else lt_b).reshape(-1)
    lt2_opp_f = (lt2_w if player == BLACK else lt2_b).reshape(-1)
    nb_self_f = (nb_b if player == BLACK else nb_w).reshape(-1)
    nb_opp_f = (nb_w if player == BLACK else nb_b).reshape(-1)
    hist_f = None
    if history is not None:
        hist_f = np.minimum(np.asarray(history, dtype=np.float64),
                            O_HIST_CAP) * O_HIST
    prior_map = None
    pw = 0.0
    if prior:
        if _ACTIVE2 and distill2.HAS_PRIOR:
            pw = float(distill2.PRIOR_W)
        elif distill.ENABLED and distill.HAS_PRIOR:
            pw = float(distill.PRIOR_W)
        if pw != 0.0:
            prior_map = _prior_map(board, player, pending, turn_count,
                                   white_turns, config, legal)
    out = []
    if pending >= 0:
        ends, route = _threats(board, player, white_turns, turn_count, config)
        kills = {int(x) for x in kill_placements(
            board, player, pending, turn_count, white_turns, config,
            legal=legal)}
        for t in kills:
            out.append((O_KILL, t))
        if kills:
            return sorted(out, reverse=True)   # 一步杀必成:只搜杀着
        for t in legal:
            ti = int(t)
            if ti in kills:
                continue
            if ti in ends:
                out.append((O_DEFEND, ti))
            elif ti in route:
                out.append((O_ROUTE, ti))
        rest = [int(t) for t in legal
                if int(t) not in kills and int(t) not in ends and int(t) not in route]
        rest_arr = np.asarray(rest, dtype=np.int64)
        if rest_arr.size:
            # 向量化安置排序分:造对方连珠 > 推回对方棋群/推中间 > 堵己方连珠 > 避免贴己方
            s_vec = np.where(lt_opp_f >= 5, O_KILL,
                             np.where(lt_opp_f == 4, O_BUILD4,
                                      np.where(lt_opp_f == 3, O_BUILD3, 0.0)))
            s_vec += (nb_opp_f * O_PUSH_OPP + CENTER_FLAT * O_PUSH_CENTER
                      - nb_self_f * O_PUSH_SELF)
            s_vec += (O_BLOCK_SELF4 * (lt_self_f >= 4)
                      + O_BLOCK_SELF3 * (lt_self_f == 3))
            # 投送双威胁:安置对方子后两轴各成连二/活三(单轴公式看不见)
            s_vec += O_GIFT_D2 * ((lt_opp_f >= 2) & (lt2_opp_f >= 2))
            s_vec += O_GIFT_D3 * ((lt_opp_f >= 3) & (lt2_opp_f >= 3))
            scored = s_vec[rest_arr] + (nb_self_f[rest_arr] + nb_opp_f[rest_arr]) * 1e-4
            if prior_map is not None:
                scored = scored + pw * prior_map[rest_arr]
            # 选取只用静态分+先验(与搜索顺序/历史无关,保证候选集合可复现);
            # 历史启发仅参与已入选着法的最终排序
            m = min(kk, rest_arr.size)
            idx = np.argpartition(scored, -m)[-m:]
            if hist_f is not None:
                order = scored[idx] + hist_f[rest_arr[idx]]
            else:
                order = scored[idx]
            idx = idx[np.argsort(order)[::-1]]
            out.extend((float(s_vec[rest_arr[j]]
                              + (pw * prior_map[rest_arr[j]]
                                 if prior_map is not None else 0.0)),
                        int(rest_arr[j])) for j in idx)
        if prior_map is not None:
            _append_prior_top(out, legal, prior_map)
        if diversity:
            _append_diversity(out, legal, nb_self_f)
            # P1:预防性封堵(对方下一步投送己方三连成杀)的射线格入根候选
            _append_pregift(out, _pregift_defense(board, player, white_turns,
                                                  turn_count, config), mask)
        return sorted(out, reverse=True)
    ends, route = _threats(board, player, white_turns, turn_count, config)
    kc = kill_captures(board, player, pending, turn_count, white_turns,
                       config, legal)
    kill_cells = {c for c, _ in kc}
    for c in kill_cells:
        out.append((O_KILL, c))
    if kill_cells:
        return sorted(out, reverse=True)   # 两步杀必成:只搜杀着
    if ends or route:
        # 解杀占领:占领后可达危险点/路线
        for c in legal:
            ci = int(c)
            if flat[ci] != opp or ci in kill_cells:
                continue
            for t in _place_targets(board, ci):
                if t in ends:
                    out.append((O_DEFEND, ci))
                    break
                if t in route:
                    out.append((O_ROUTE, ci))
                    break
        # 堵路线普通着法
        for t in legal:
            ti = int(t)
            if flat[ti] == EMPTY and ti in route:
                out.append((O_ROUTE - 1, ti))
    # 必守剪枝:对方可杀四在握(无解必败/只剩防守链)时,只保留 杀着+防守着,
    # 其余候选全部裁掉——防守选项本来就极少,穷举即可,不浪费深度在无关着法上。
    if not kill_cells and ends and _loses_to_kill(board, player, white_turns,
                                                  turn_count, config):
        return sorted(out, reverse=True)
    # 向量化基础分(空位) + 逐格特殊分(占领格,数量少)
    rest = [int(t) for t in legal if int(t) not in kill_cells]
    rest_arr = np.asarray(rest, dtype=np.int64)
    empty_f = flat == EMPTY
    base = np.where(lt_self_f >= 4, O_OWN4,
                    np.where(lt_self_f == 3, O_OWN3, 0.0))
    base += nb_opp_f * O_FIGHT - nb_self_f * O_SPREAD
    # 平局键:分降序、move 降序(与旧实现完全一致,保持树形稳定)
    dens = -(np.arange(PLACE_SIZE) * 1e-9).astype(np.float64)
    empty_cells = rest_arr[empty_f[rest_arr]]
    if empty_cells.size:
        # 选取用静态分(与历史/搜索顺序无关,保证候选集合可复现);
        # 历史启发只参与入选后的排序
        key = base[empty_cells] + dens[empty_cells]
        if prior_map is not None:
            key = key + pw * prior_map[empty_cells]
        m = min(kk, empty_cells.size)
        idx = np.argpartition(key, -m)[-m:]
        if hist_f is not None:
            order = base[empty_cells][idx] + hist_f[empty_cells[idx]] \
                + dens[empty_cells][idx]
        else:
            order = key[idx]
        if prior_map is not None:
            order = order + pw * prior_map[empty_cells[idx]]
        idx = idx[np.argsort(order)[::-1]]
        out.extend((float(base[empty_cells[j]]
                          + (pw * prior_map[empty_cells[j]]
                             if prior_map is not None else 0.0)),
                    int(empty_cells[j])) for j in idx)
    cap_cells = rest_arr[~empty_f[rest_arr]]
    if cap_cells.size:
        # 向量化安置评分图,占领格取"可达安置格中的最优"作拆连珠代价
        pmap = np.where(lt_opp_f >= 5, O_KILL,
                        np.where(lt_opp_f == 4, O_BUILD4,
                                 np.where(lt_opp_f == 3, O_BUILD3, 0.0)))
        pmap += nb_opp_f * O_PUSH_OPP + CENTER_FLAT * O_PUSH_CENTER \
            - nb_self_f * O_PUSH_SELF \
            + O_BLOCK_SELF4 * (lt_self_f >= 4) \
            + O_BLOCK_SELF3 * (lt_self_f == 3)
        # 投送双威胁:安置对方子后两轴各成连二/活三(单轴公式看不见)
        pmap += O_GIFT_D2 * ((lt_opp_f >= 2) & (lt2_opp_f >= 2))
        pmap += O_GIFT_D3 * ((lt_opp_f >= 3) & (lt2_opp_f >= 3))
        cap_scores = np.empty(cap_cells.size)
        for i, ti in enumerate(cap_cells):
            self_line = lt_self_f[ti]
            opp_line = lt_opp_f[ti]
            s = O_OWN4 if self_line >= 4 else (O_OWN3 if self_line == 3 else 0.0)
            s += O_BREAK4 if opp_line >= 4 else (O_BREAK3 if opp_line >= 3 else 0.0)
            ray = RAYS[ti].reshape(MOVE_DIRS * MOVE_DISTS)
            val = (ray >= 0) & empty_f[ray]
            val = np.minimum.accumulate(
                val.reshape(MOVE_DIRS, MOVE_DISTS), axis=1).reshape(-1)
            if val.any():
                s += float(pmap[ray[val]].max())
            cap_scores[i] = s
        m = min(kk, cap_cells.size)
        key = cap_scores + dens[cap_cells]
        if prior_map is not None:
            key = key + pw * prior_map[cap_cells]
        idx = np.argpartition(key, -m)[-m:]
        idx = idx[np.argsort(key[idx])[::-1]]
        out.extend((float(cap_scores[j]
                          + (pw * prior_map[cap_cells[j]]
                             if prior_map is not None else 0.0)),
                    int(cap_cells[j])) for j in idx)
    if prior_map is not None:
        _append_prior_top(out, legal, prior_map)
    if diversity:
        _append_diversity(out, legal, nb_self_f)
        # P1:预防性封堵(对方下一步投送己方三连成杀)的射线格入根候选
        _append_pregift(out, _pregift_defense(board, player, white_turns,
                                              turn_count, config), mask)
    # 保送槽:吃对手上一手落子(浅层宽度剪枝最容易漏掉的反击,强制入窗)
    if opp_last >= 0 and int(flat[opp_last]) == opp and bool(mask[opp_last]) \
            and all(mm != opp_last for _, mm in out):
        out.append((O_LAST_CAP, int(opp_last)))
    return sorted(out, reverse=True)


def _immediate_kill(board: np.ndarray, player: int, pending: int,
                    turn_count: int, white_turns: int,
                    config: GameConfig) -> bool:
    """当前行棋方是否握有一步/两步必杀(杀棋立即终局,无水平线问题)。
    复用 _board_features 的材料统计(按棋盘缓存),无额外扫描代价。"""
    if turn_count < config.loss_start_turns:
        return False
    m_black, m_white, _, _, dang_black, dang_white = _board_features(
        board, white_turns, turn_count, config)
    opp = _other(player)
    if pending >= 0:
        dang = dang_black if opp == BLACK else dang_white
        if not dang:
            return False
        return kill_placements(board, player, pending, turn_count,
                               white_turns, config).size > 0
    m_opp = m_black if opp == BLACK else m_white
    return m_opp["killable4"] > 0


def _bring_front(cands: list, m) -> None:
    """把候选着法 m 提到最前(若存在),保持其余相对顺序。"""
    for j, (_, mm) in enumerate(cands):
        if mm == m and j:
            cands[0], cands[j] = cands[j], cands[0]
            return


def _include_extras(cands: list, extras, mask) -> None:
    """把 TT/killer 等着法补进候选(按合法掩码过滤,插在特殊着之后)。

    同一着法已存在则跳过;extras 按传入优先级依次占据插入位置。"""
    if mask is None:
        return
    present = {m for _, m in cands}
    valid = []
    for m in extras:
        if m is None or m < 0 or m >= PLACE_SIZE or m in present:
            continue
        if bool(mask[int(m)]):
            valid.append(int(m))
            present.add(int(m))
    if not valid:
        return
    pos = 0
    while pos < len(cands) and cands[pos][0] >= O_ROUTE:
        pos += 1
    for m in reversed(valid):
        cands.insert(pos, (1e-6, m))


def _append_diversity(out: list, legal, nb_self_f: np.ndarray,
                      limit: int = 2) -> None:
    """补充至多 limit 个"己方邻接度最高"的着法(去重、低分)。

    静态分体系(推中/造对方连珠/避贴己方)会系统性埋没"贴着己方棋群/安静
    关键点"型着法,而实战复盘显示 kata 的首选常属此类;这里额外保留少量
    此类着法进入根候选,保证它们至少被搜索看到(不影响原有候选优先级)。"""
    if limit <= 0:
        return
    present = {m for _, m in out}
    pool = [int(t) for t in legal if int(t) not in present]
    if not pool:
        return
    arr = np.asarray(pool, dtype=np.int64)
    n = min(limit, arr.size)
    ii = np.argpartition(nb_self_f[arr], -n)[-n:]
    for j in ii:
        out.append((float(nb_self_f[arr[j]]) * 1e-3, int(arr[j])))


def _search(board: np.ndarray, player: int, pending: int, turn_count: int,
            white_turns: int, config: GameConfig, depth: int,
            alpha: float, beta: float, ctx: _SearchCtx,
            key: int | None = None, ext_budget: int = 0, ply: int = 0,
            opp_last: int = -1) -> float:
    """极小化搜索(fail-soft alpha-beta + 置换表 + PVS + killer 排序)。

    返回当前行棋方视角的值;将死值带步数距离:赢方取 WIN - ply(越快越大),
    输方取 -(WIN - ply)(越慢越大)——必败时自动选最长抵抗、必胜时自动选最快杀。
    超时置 ctx.aborted 并返回近似值(近似值不写置换表,根节点丢弃该深度)。"""
    ctx.nodes += 1
    if ctx.cb is not None and (ctx.nodes & 255) == 0:
        ctx.cb(ctx.nodes)
    if depth <= 0:
        # 叶节点先做战术终局检测:行棋方握有一步/两步杀则直接必胜,
        # 消除"搜索耗尽看不到下一手杀"的水平线效应
        if _immediate_kill(board, player, pending, turn_count, white_turns,
                           config):
            return WIN - ply
        return evaluate(board, player, pending, turn_count, white_turns, config)
    if key is None:
        key = _zobrist(board, player, pending, turn_count, white_turns)
    tt = ctx.tt
    entry = tt.probe(key)
    tt_move = None
    if entry is not None:
        v, td, flag, tt_move = entry
        if td >= depth:
            if flag == _TT.FLAG_EXACT:
                return v
            if flag == _TT.FLAG_LOWER and v >= beta:
                return v
            if flag == _TT.FLAG_UPPER and v <= alpha:
                return v
    opp = _other(player)
    legal_holder: list = []
    cands = _candidates(board, player, pending, turn_count, white_turns,
                        config, _inner_k(depth), history=ctx.history,
                        out_legal=legal_holder, opp_last=opp_last)
    if not cands:
        if np.count_nonzero(board) == PLACE_SIZE:
            return 0.0                    # 满盘和棋
        return -(WIN - ply)               # 无子可走(必守无解/全自杀),必败
    k1, k2 = ctx.killers[depth & 63]
    _include_extras(cands, (tt_move, k1, k2),
                    legal_holder[0] if legal_holder else None)
    forced = len(cands) <= 2 and cands[0][0] >= O_ROUTE
    best = -1e18
    best_move = None
    alpha0 = alpha
    searched = False
    ext_used = False                  # 本节点是否对子节点做过强制扩展(存表降标用)
    for idx, (cand_score, m) in enumerate(cands):
        if ctx.deadline is not None and (idx & 3) == 0 \
                and time.perf_counter() > ctx.deadline:
            ctx.aborted = True
            return best                   # 超时:近似值,不写 TT
        r, c = divmod(int(m), BOARD_SIZE)
        b2, p2, pend2, tc2, wt2, loser = ReverseGomoku.apply_step(
            board, m, player, pending, turn_count, white_turns, config)
        if loser == player:
            continue                      # 自杀着(掩码已剔除,防御)
        if loser == opp:
            v = WIN - ply                 # 有杀着:立即胜(步数越小越快)
        elif np.count_nonzero(b2) == PLACE_SIZE:
            v = 0.0                       # 满盘和棋
        elif pend2 >= 0:
            # 占领步后同一方继续安置,子节点视角与根一致,不取负
            v = _search(b2, p2, pend2, tc2, wt2, config, depth - 1,
                        alpha, beta, ctx, ext_budget=ext_budget, ply=ply + 1,
                        opp_last=opp_last)
        elif _jump_through(b2, pending // BOARD_SIZE if pending >= 0 else r,
                           pending % BOARD_SIZE if pending >= 0 else c,
                           player) >= 4 \
                and _loses_to_kill(b2, player, wt2, tc2, config):
            # 本回合己方新增子:普通落子在 (r,c),占领+安置在占领格(pending)。
            # 走完留己方可杀 4 连(含五带缺):对方下回合两步必杀(攻彼忘我!)
            v = -(WIN - (ply + 2))
        else:
            d2 = depth - 1
            # 保送槽传递:安置步后我方最新子=占领格;普通落子=刚下的 m
            child_opp_last = pending if pending >= 0 else int(m)
            extended = False
            child_ext = ext_budget
            if USE_EXT and ext_budget > 0 \
                    and (forced
                         or (pending >= 0
                             and _jump_through(b2, r, c, opp) >= 4
                             and _loses_to_kill(b2, opp, wt2, tc2, config))):
                # 强制局面(本节点只剩单一防守/造对方可杀四):对方应手极少,
                # 多搜一层几乎不增加分支,沿强制链看得更深(预算一次消耗:
                # 本节点首个触发扩展的子树获得加深,其余兄弟不扩展)
                d2 = depth
                child_ext = 0
                ext_budget = 0
                ext_used = True
                extended = True
            if not searched or not USE_PVS:
                v = -_search(b2, p2, pend2, tc2, wt2, config, d2,
                             -beta, -alpha, ctx, ext_budget=child_ext,
                             ply=ply + 1, opp_last=child_opp_last)
            else:
                # LMR:晚序的安静着法先按 d-1/d-2 零窗试搜,超上界再全深度重搜。
                # 战术着法(O_ROUTE 及以上)/前 3 着/TT 着法/强制或加深节点不降价。
                d_red = d2
                if (USE_LMR and not NO_LMR and not extended and not forced
                        and d2 >= LMR_MIN_DEPTH and idx >= LMR_MIN_IDX
                        and cand_score < O_ROUTE
                        and (tt_move is None or m != tt_move)):
                    d_red = max(1, d2 - (2 if idx >= 6 else 1))
                if d_red < d2:
                    v = -_search(b2, p2, pend2, tc2, wt2, config, d_red,
                                 -alpha - PVS_EPS, -alpha, ctx,
                                 ext_budget=child_ext, ply=ply + 1,
                                 opp_last=child_opp_last)
                    if v > alpha:
                        v = -_search(b2, p2, pend2, tc2, wt2, config, d2,
                                     -alpha - PVS_EPS, -alpha, ctx,
                                     ext_budget=child_ext, ply=ply + 1,
                                     opp_last=child_opp_last)
                else:
                    v = -_search(b2, p2, pend2, tc2, wt2, config, d2,
                                 -alpha - PVS_EPS, -alpha, ctx,
                                 ext_budget=child_ext, ply=ply + 1,
                                 opp_last=child_opp_last)
                if alpha < v < beta:
                    v = -_search(b2, p2, pend2, tc2, wt2, config, d2,
                                 -beta, -alpha, ctx, ext_budget=child_ext,
                                 ply=ply + 1, opp_last=child_opp_last)
            if ctx.aborted:
                return best               # 子树超时:整棵返回近似值
        searched = True
        if v > best:
            best = v
            best_move = m
            if best > alpha:
                alpha = best
            if alpha >= beta:
                if abs(best) < MATE_LIMIT:
                    ctx.history[int(m)] += depth * depth
                ks = ctx.killers[depth & 63]
                if ks[0] != m:
                    ks[1] = ks[0]
                    ks[0] = m
                break
    if not ctx.aborted:
        if abs(best) >= MATE_LIMIT:
            pass                          # 将死值依赖 ply,不进置换表
        else:
            if best <= alpha0:
                flag = _TT.FLAG_UPPER
            elif best >= beta:
                flag = _TT.FLAG_LOWER
            else:
                flag = _TT.FLAG_EXACT
            # 子节点被强制扩展时分值来自更深的实际搜索,按 depth-1 保守
            # 标注存表,防止其他路径按标称 depth 复用到被抬高的值
            tt.store(key, depth - 1 if ext_used else depth, best, flag, best_move)
    return best


def _vcf_slice(time_budget: float | None) -> float:
    """单步 VCF 预算:无对局限时时用基础预算(0.05s);有 budget 时按其 15%
    放宽并夹在 [VCF_TIME_CAP, VCF_TIME_CAP_HELL]。
    大师档杀链更深(d12/24万节点),需要更长证明预算;冻结档保持 5%/0.6s。"""
    base = threat_search.VCF_TIME_CAP
    if not time_budget:
        return base
    return min(VCF_TIME_CAP_HELL, max(base, 0.15 * time_budget))


def _defensive_vcf_slice(time_budget: float | None) -> float:
    """单步"防守 VCF"总预算(校验根候选走完后对手是否有强制杀)。

    P0 攻防对等:与进攻侧 `_vcf_slice` 同公式(限时取 5% 夹在
    [VCF_TIME_CAP, 0.6s],不限时用基础预算),从主搜索预算中预留
    (见 _root_iterative),不额外超时。"""
    return _vcf_slice(time_budget)


def _loss_value(turns: int) -> float:
    """VCF 已证对手强制杀时的必败分(保守下界):对手还需 turns 个进攻回合
    (每回合两步),损失发生在对手第 turns+1 个回合,折算 plies 下界
    2*turns + 3(turns=0 即"对手下回合两步杀",与搜索中 -(WIN-3) 同尺度)。
    只用于排序,故取保守下界、不夸大存活步数;关键是不再把已证必败压成
    "一步即败" -(WIN-1) 从而反转"必败选最长抵抗"的排序(见 _search 的
    WIN - ply 将死值语义)。"""
    return -(WIN - (2 * turns + 3))


def _proven_loss_value(board, opp, tc, wt, config, deadline, node_cap):
    """行棋方为 opp 的局面:可证其强制杀时返回对应必败分,未证/超时 None。"""
    got = threat_search.find_forced_kill_depth(
        board, opp, -1, tc, wt, config,
        max_depth=VCF_MAX_DEPTH_HELL,
        deadline=deadline, node_cap=node_cap)
    if got is None:
        return None
    return _loss_value(int(got[1]))


def _occupy_turn_loss(b2, p2, pend2, tc2, wt2, config, opp,
                      deadline, node_cap):
    """占领步后同方继续安置:校验整回合(占领+安置)是否仍必败。

    枚举全部合法安置;任一安置走完后对手无强制杀 = 该回合安全,
    返回 (False, None);全部安置均必败时返回 (True, 最长存活必败分)。
    占领步是逆五移子的前半步,旧实现直接跳过不校验,导致本局"占 H5"
    与"占 H6"两条防守分支只能靠有限深度评估区分、选错较长抵抗。"""
    pmask = ReverseGomoku.legal_mask_for(b2, p2, pend2, wt2, config, tc2)
    best = None
    for pi in np.nonzero(pmask)[0]:
        b3, _, _, tc3, wt3, loser3 = ReverseGomoku.apply_step(
            b2, int(pi), p2, pend2, tc2, wt2, config)
        if loser3 == opp:
            return False, None             # 安置直接反杀:本回合安全
        if loser3 != 0:
            continue                       # 自杀安置(掩码外,保守跳过)
        if opp == WHITE and wt3 < config.white_restrict_turns:
            return False, None             # 对手(白)禁移期不可能有杀
        if deadline is not None and time.perf_counter() > deadline:
            return False, None             # 未完成证明:保守视为安全
        if threat_search.has_kill(b3, opp, -1, tc3, wt3, config):
            val = _loss_value(0)
        else:
            val = _proven_loss_value(b3, opp, tc3, wt3, config,
                                     deadline, node_cap)
            if val is None:
                return False, None         # 未证/超时:安全
        if best is None or val > best:
            best = val
    if best is None:
        return False, None                 # 无合法安置(不应出现)
    return True, best


def _filter_defensive_kills(ranked, board, player, pending, turn_count,
                            white_turns, config, total_budget: float | None,
                            top_n: int | None = None):
    """把"走完轮到对手、对手可证明强制杀"的根着法降为必败分。

    降分带距离(见 _loss_value):不再用 -(WIN-1),保证 VCF 已证必败不会
    排在搜索已证更短的杀之后;占领步按"占领+全部安置"整回合校验,全部
    安置必败才降(任一安置未被证明必败则该候选视为安全、不降)。
    VCF 结论保守(找不到 ≠ 安全),只用于避免确定性送杀;已有我方确定杀着
    (|v| ≥ MATE_LIMIT)不再检查。

    P0:默认校验全部候选(按根排序,最可能被选中的着法优先),节点上限与
    进攻 VCF 一致;候选共享同一总 deadline(total_budget;None = 不限墙钟,
    由节点上限保证确定性上界),速败候选快速证明后把剩余预算留给后面的
    候选——不再按候选数固定均分。"""
    if not ranked or turn_count < config.loss_start_turns:
        return ranked
    if ranked[0][1] >= MATE_LIMIT:
        return ranked                       # 己方已证必胜
    opp = _other(player)
    if opp == WHITE and white_turns < config.white_restrict_turns:
        return ranked                       # 对手(白)禁移期不可能有杀
    if top_n is None:
        top_n = len(ranked)
    node_cap = (VCF_NODE_CAP_HELL if total_budget is not None
                else VCF_NODE_CAP_UNLIMITED)
    deadline = (time.perf_counter() + total_budget
                if total_budget is not None else None)
    changed = False
    for i in range(min(top_n, len(ranked))):
        m, v = ranked[i]
        if abs(v) >= MATE_LIMIT:
            continue
        if deadline is not None and time.perf_counter() > deadline:
            break                           # 共享预算耗尽:剩余候选保守视为安全
        b2, p2, pend2, tc2, wt2, loser = ReverseGomoku.apply_step(
            board, m, player, pending, turn_count, white_turns, config)
        if loser != 0:
            continue                        # 自杀/反杀
        if pend2 >= 0:
            # 占领步:校验"占领+安置"整回合(安置枚举,任一安全即不降)
            all_lose, val = _occupy_turn_loss(b2, p2, pend2, tc2, wt2,
                                              config, opp, deadline, node_cap)
            if all_lose:
                ranked[i] = (m, val)        # 所有安置均遭强制杀:确定性送杀
                changed = True
            continue
        if tc2 < config.loss_start_turns:
            continue
        val = _proven_loss_value(b2, opp, tc2, wt2, config, deadline, node_cap)
        if val is not None:
            ranked[i] = (m, val)            # 对手有强制杀:确定性送杀
            changed = True
    if changed:
        ranked = sorted(ranked, key=lambda t: -t[1])
    return ranked


_WORKER_TT = None


def _worker_tt():
    """每个 worker 进程内持久置换表:跨深度迭代复用(不与父进程共享)。"""
    global _WORKER_TT
    if _WORKER_TT is None:
        _WORKER_TT = _TT()
    return _WORKER_TT


def _rebuild_config(fields):
    """子进程里没有父进程的 GameConfig 实例,按字段重建。"""
    from .reversegomoku import GameConfig
    return GameConfig(**fields)


def _root_child_task(payload):
    """并行根搜索的子任务:评估根节点的一个着法(与串行分支等价)。

    payload = (board_bytes, player, pending, turn_count, white_turns,
               cfg_fields, move, depth_left, wall_deadline)
    返回 (move, value | None, nodes, aborted);value=None 表示自杀着(跳过)。
    wall_deadline 是 time.time() 的绝对时刻(同机各进程可比),0 表示不限时;
    用墙钟而非"剩余秒数",排队等待的任务也能立刻发现超时并快速返回。"""
    (board_bytes, player, pending, turn_count, white_turns, cfg_fields,
     move, depth_left, wall_deadline) = payload
    board = np.frombuffer(board_bytes, dtype=np.int8).reshape(
        BOARD_SIZE, BOARD_SIZE).copy()
    config = _rebuild_config(cfg_fields)
    opp = _other(player)
    r, c = divmod(int(move), BOARD_SIZE)
    b2, p2, pend2, tc2, wt2, loser = ReverseGomoku.apply_step(
        board, move, player, pending, turn_count, white_turns, config)
    if loser == player:
        return (int(move), None, 0, False)
    if loser == opp:
        return (int(move), WIN - 1, 0, False)
    if np.count_nonzero(b2) == PLACE_SIZE:
        return (int(move), 0.0, 0, False)
    remaining = 0.0
    if wall_deadline > 0.0:
        remaining = wall_deadline - time.time()
        if remaining <= 0.0:
            return (int(move), None, 0, True)      # 排队期间已超时
    ctx = _SearchCtx(_worker_tt(),
                     time.perf_counter() + remaining if remaining > 0 else None)
    if pend2 >= 0:
        v = _search(b2, p2, pend2, tc2, wt2, config, depth_left,
                    -1e18, 1e18, ctx, ext_budget=1, ply=2)
    elif (_jump_through(b2, pending // BOARD_SIZE if pending >= 0 else r,
                        pending % BOARD_SIZE if pending >= 0 else c,
                        player) >= 4
          and _loses_to_kill(b2, player, wt2, tc2, config)):
        v = -(WIN - 3)
    else:
        v = -_search(b2, p2, pend2, tc2, wt2, config, depth_left,
                     -1e18, 1e18, ctx, ext_budget=1, ply=2,
                     opp_last=int(move))
    return (int(move), v, ctx.nodes, ctx.aborted)


def _root_iterative(board, player, pending, turn_count, white_turns, config,
                    rng, depth, time_budget, k, node_cb=None, report_cb=None,
                    tt=None, noise: float = 0.0, time_start: float | None = None,
                    workers: int = 1):
    """根节点迭代加深公共实现:choose 与 analysis 共用,保证两者同值同选。

    先验杀着与 VCF 由调用方处理;这里只做"候选生成 + 逐层全窗口根评估"。
    noise>0 时给每个根着法价值加 [0, noise) 随机量(仅对局侧用于同分打散,
    分析传 0)。node_cb(节点数) 与 report_cb(候选列表) 语义不同,分别对应
    choose 与 analysis 的进度回调。
    有 time_budget 时迭代加深由时限驱动(depth 仅决定至少完成的层数,上限
    MAX_DYNAMIC_DEPTH);不限时时 depth 为硬上限。
    time_start: 本步起始时刻(含进攻 VCF 的耗时),使"进攻 VCF + 主搜索 +
    防守校验"的总墙钟 ≤ time_budget;None 时以进入本函数为起点。
    防守预算是自适应的:只有存在战术材料时才预留大额 VCF 预算,安静局面
    预留小预算、把时钟让给搜索加深(见 _defense_reserve_needed)。
    返回 (ranked, ok):ranked = 最后一次完整深度的 [(着法, 价值)] 降序;
    无任何合法着法时 ok=False(调用方决定返回 None 还是空列表)。"""
    opp = _other(player)
    book_move = openbook.lookup(board, player, pending)
    if book_move is not None:
        mask = ReverseGomoku.legal_mask_for(board, player, pending,
                                            white_turns, config, turn_count)
        if mask[book_move]:
            return [(int(book_move), 0.0)], True
    cands = _candidates(board, player, pending, turn_count, white_turns,
                        config, ROOT_K, diversity=True, prior=True,
                        core_k=ROOT_K_CORE)
    if not cands:
        # 必守剪枝可能把候选全部剪掉(被绝杀且无防守招):此刻仍有合法着法,
        # 把全部合法着法交给迭代加深搜索——将死值带 ply 步数(见 _search),
        # 搜索自动挑"手数最长的必败招",挣扎到最后一手再被绝杀,自然终局;
        # 只有真正无合法着法(全为自杀被剔除)才由调用方判定失败。
        legal_all = np.nonzero(ReverseGomoku.legal_mask_for(
            board, player, pending, white_turns, config, turn_count))[0]
        if not legal_all.size:
            return [], False
        cands = [(0.0, int(m)) for m in legal_all]
    # 预留"防守 VCF"预算:主搜索提前 dvcf 秒收手,校验对手是否有强制杀。
    # 自适应(见 _defense_reserve_needed):无战术材料时只留小预算,回收时钟
    # 给搜索加深;总墙钟从 time_start(本步起点)起算,≤ time_budget。
    if turn_count >= config.loss_start_turns:
        dvcf = _defensive_vcf_slice(time_budget)
        if time_budget and not _defense_reserve_needed(
                board, player, pending, turn_count, white_turns, config):
            dvcf = min(dvcf, 0.1)
    else:
        dvcf = 0.0
    t0 = time_start if time_start is not None else time.perf_counter()
    search_deadline = None
    if time_budget:
        search_deadline = t0 + max(0.02, time_budget - dvcf)
    ctx = _SearchCtx(tt if tt is not None else _TT(), search_deadline)
    ctx.cb = node_cb
    # P0 动态加深:有时限时 depth 只是起始完成要求,预算用满前逐层加深到
    # MAX_DYNAMIC_DEPTH;不限时时保持原语义(depth 为硬上限,完整搜索)。
    max_depth = depth
    if time_budget and not NO_DYNAMIC:
        max_depth = max(depth, MAX_DYNAMIC_DEPTH)
    prev_m = None
    ranked = []
    for d in range(1, max_depth + 1):
        depth_moves = []
        completed = True
        cands2 = cands[:]
        if prev_m is not None:
            _bring_front(cands2, prev_m)   # 上一深度最佳着法优先(排序加速)
        # 根 α 传播 + PVS:首个根着法全窗口,其后根着法先零窗试搜(在 alpha
        # 处),仅失败高(可能更优)才按全窗口重搜。10 个根候选全窗口时 d7≈10s,
        # PVS 后 d7≈3s(实测),有效深度多 1-2 层;最优着法始终精确,编号靠后
        # 的着法为不高于 alpha 的上界(排序/唯一最优不受影响)。
        if workers > 1 and len(cands2) > 1:
            # 根并行:每层把全部根着法分发到 worker 进程(各自独立 TT、无 alpha 传播)
            _now = time.perf_counter()
            if ctx.deadline is not None:
                _remaining = max(0.0, ctx.deadline - _now)
                if _remaining <= 0.0:
                    completed = False
                    break
                _wall_deadline = time.time() + _remaining
            else:
                _wall_deadline = 0.0
            _cfg_fields = dict(
                loss_start_turns=config.loss_start_turns,
                white_restrict_turns=config.white_restrict_turns,
                mask_suicide=config.mask_suicide)
            _state = board.astype(np.int8).tobytes()
            _tasks = [(_state, player, pending, turn_count, white_turns,
                       _cfg_fields, int(m), d - 1, _wall_deadline)
                      for _, m in cands2]
            _res = _map_root_tasks("antifive.heuristic:_root_child_task",
                                   _tasks, workers)
            for _mv, _v, _nodes, _aborted in _res:
                ctx.nodes += _nodes
                if _aborted:
                    completed = False
                if _v is None:
                    continue
                if noise > 0.0:
                    _v += rng.uniform(0.0, noise)
                depth_moves.append((_mv, _v))
            if node_cb is not None:
                node_cb(ctx.nodes)
            if (not completed or ctx.aborted
                    or (ctx.deadline is not None
                        and time.perf_counter() > ctx.deadline)):
                completed = False
                break
        else:
            root_alpha = -1e18
            first = True
            for _, m in cands2:
                r, c = divmod(int(m), BOARD_SIZE)
                b2, p2, pend2, tc2, wt2, loser = ReverseGomoku.apply_step(
                    board, m, player, pending, turn_count, white_turns, config)
                if loser == player:
                    continue
                if loser == opp:
                    v = WIN - 1                # 根着法第 1 步即杀
                elif np.count_nonzero(b2) == PLACE_SIZE:
                    v = 0.0
                elif pend2 >= 0:
                    # 占领步后同一方继续安置,子节点视角与根一致,不取负;
                    # 同侧节点可用父 alpha 作下界(仅需判断是否超过 alpha)
                    if USE_ROOT_PVS and not NO_ROOT_PVS and not first:
                        v = _search(b2, p2, pend2, tc2, wt2, config, d - 1,
                                    root_alpha, 1e18, ctx, ext_budget=1, ply=2)
                    else:
                        v = _search(b2, p2, pend2, tc2, wt2, config, d - 1,
                                    -1e18, 1e18, ctx, ext_budget=1, ply=2)
                elif _jump_through(b2, pending // BOARD_SIZE if pending >= 0 else r,
                                   pending % BOARD_SIZE if pending >= 0 else c,
                                   player) >= 4 \
                        and _loses_to_kill(b2, player, wt2, tc2, config):
                    v = -(WIN - 3)             # 走完留可杀四:对手下回合两步杀
                elif USE_ROOT_PVS and not NO_ROOT_PVS and not first:
                    v = -_search(b2, p2, pend2, tc2, wt2, config, d - 1,
                                 -root_alpha - PVS_EPS, -root_alpha, ctx,
                                 ext_budget=1, ply=2, opp_last=int(m))
                    if v > root_alpha:
                        v = -_search(b2, p2, pend2, tc2, wt2, config, d - 1,
                                     -1e18, 1e18, ctx, ext_budget=1, ply=2,
                                     opp_last=int(m))
                else:
                    v = -_search(b2, p2, pend2, tc2, wt2, config, d - 1,
                                 -1e18, 1e18, ctx, ext_budget=1, ply=2,
                                 opp_last=int(m))
                first = False
                if noise > 0.0:
                    v += rng.uniform(0.0, noise)
                if v > root_alpha:
                    root_alpha = v
                depth_moves.append((int(m), v))
                if node_cb is not None:
                    node_cb(ctx.nodes)
                if ctx.aborted or (ctx.deadline is not None
                                   and time.perf_counter() > ctx.deadline):
                    completed = False
                    break
        if not completed or not depth_moves:
            break
        depth_moves.sort(key=lambda t: -t[1])
        ranked = depth_moves
        prev_m = ranked[0][0]
        if report_cb is not None:
            report_cb(ranked[:k])
        if ranked[0][1] >= WIN - 1:
            break                          # 根着法即必杀,无需更深;
                                           # 必败不提前停:继续加深细化抵抗
        if ctx.deadline is not None and time.perf_counter() > ctx.deadline:
            break
    if not ranked:
        ranked = [(int(cands[0][1]), 0.0)]  # 兜底:静态分最高(仅极端超时)
    if dvcf > 0:
        # 有时限:用本步剩余墙钟(搜索提前收手时余量全部归防守校验);
        # 无对局限时:不受墙钟约束(结果确定性),由节点上限兜底
        remaining = None
        if time_budget:
            remaining = max(0.0, t0 + time_budget - time.perf_counter())
        ranked = _filter_defensive_kills(
            ranked, board, player, pending, turn_count, white_turns, config,
            remaining)
    return ranked, True


def choose_heuristic_move(board, player, pending, turn_count, white_turns,
                          config, rng, depth: int | None = None,
                          time_budget: float | None = None,
                          progress_cb=None, tt: "_TT | None" = None,
                          root_prune: bool = True,
                          use_vcf: bool | None = None,
                          workers: int = 1) -> int | None:
    """返回着法 idx;无子可走(所有着法均自杀)返回 None。
    progress_cb(节点数) 可选:搜索途中定期回调,供 GUI 实时显示计算量。
    tt: 可选置换表(跨步复用,键含完整状态,安全);
    use_vcf: 是否启用强制杀链搜索(None=用 threat_search.USE_VCF 全局默认)。
    root_prune: 已废弃(保留仅为兼容旧调用);根搜索与 analysis_moves
    共用同一全窗口实现,保证"引擎落子与其自分析一致"。

    先验:存在一步/两步杀直接走(精确必胜,免搜索);
    否则迭代加深:从深度 1 逐层加深,time_budget(秒)内完成多少算多少,
    超时采用上一完整深度选出的着法;上一深度最佳着法在下一深度优先搜索。
    有时限时 depth 不再是硬上限(动态加深到 MAX_DYNAMIC_DEPTH 层)。
    被绝杀无防守招时,以全部合法着法为候选搜索"手数最长的必败招",
    挣扎到最后一手再被绝杀,对局自然终局(不提前认输)。"""
    ensure_params()
    t_start = time.perf_counter()        # 本步计时起点(含进攻 VCF)
    if depth is None:
        depth = DEFAULT_DEPTH
    opp = _other(player)
    if pending >= 0:
        kp = kill_placements(board, player, pending, turn_count,
                             white_turns, config)
        if kp.size:
            return int(rng.choice(kp))
    else:
        kc = kill_captures(board, player, pending, turn_count, white_turns, config)
        if kc:
            return int(rng.choice([c for c, _ in kc]))
    if threat_search.USE_VCF if use_vcf is None else use_vcf:
        vcf_slice = _vcf_slice(time_budget)
        if vcf_slice > 0:
            # 无对局限时时不用墙钟 deadline(避免时间抖动导致 choose 与
            # analysis 结果不一致),改由节点上限保证确定性上界。
            vcf_deadline = (t_start + vcf_slice) if time_budget else None
            mv = threat_search.find_forced_kill(
                board, player, pending, turn_count, white_turns, config,
                max_depth=VCF_MAX_DEPTH_HELL,
                deadline=vcf_deadline,
                node_cap=VCF_NODE_CAP_HELL if time_budget
                else VCF_NODE_CAP_UNLIMITED,
                progress_cb=progress_cb)
            if mv is not None:
                return int(mv)
    # 与 analysis 共用同一根搜索实现(全窗口逐着法评估),保证"引擎下的棋
    # 与其自分析一致";噪声只用于同分打散。root_prune 参数已废弃(保留兼容)。
    ranked, ok = _root_iterative(board, player, pending, turn_count,
                                 white_turns, config, rng, depth, time_budget,
                                 ROOT_K, node_cb=progress_cb, tt=tt,
                                 noise=O_NOISE, time_start=t_start,
                                 workers=workers)
    if not ok or not ranked:
        return None
    return int(ranked[0][0])


def _terminal_winner(board: np.ndarray):
    """棋盘已有连五则返回胜者(逆五:连五者负,对方胜);无连五返回 None。"""
    for color in (BLACK, WHITE):
        if (fwd_run_lengths(board == color) >= 5).any():
            return _other(color)
    return None


def analysis_moves(board, player, pending, turn_count, white_turns, config, rng,
                   depth=None, time_budget=None, k=8, progress_cb=None,
                   tt: "_TT | None" = None, use_vcf: bool | None = None,
                   workers: int = 1):
    """局面分析:返回 (pos_v, moves)。pos_v=当前行棋方局面价值;
    moves=[(着法idx, 价值), ...] 按价值降序,供外部 GUI 投影/胜率显示。
    tt: 可选置换表(跨步复用);分析保持全窗口精确值,不做根 α 剪枝。

    与 choose_heuristic_move 同源的迭代加深:先验一步/两步杀直接给出(价值=WIN),
    否则对候选着法做 fail-soft alpha-beta;每完成一层深度即回调 progress_cb(moves)
    (GUI 流式刷新选点)。价值尺度与对弈一致:位置价值约 ±几,必杀/必败为 ±WIN;
    终局局面按实际胜负返回(胜者 +WIN / 败者 -WIN)。"""
    ensure_params()
    t_start = time.perf_counter()
    winner = _terminal_winner(board)
    if winner is not None:
        # 终局:不再搜索,按实际胜负给出价值;仍回发一个有效着法,
        # 使 GUI 能记录该步胜率点(胜者 100%/败者 0%)。
        side = WIN if winner == player else -WIN
        cell = int(np.argmin(board.reshape(-1) != EMPTY))  # 首个空格(全满则取 0 位)
        moves = [(cell, side)]
        if progress_cb:
            progress_cb(moves[:k])
        return (side, moves[:k])
    if depth is None:
        depth = DEFAULT_DEPTH
    opp = _other(player)
    moves = []
    if pending >= 0:
        kp = kill_placements(board, player, pending, turn_count,
                             white_turns, config)
        if kp.size:
            moves = [(int(m), WIN - 1) for m in kp]
    else:
        kc = kill_captures(board, player, pending, turn_count, white_turns, config)
        if kc:
            moves = [(int(c), WIN - 1) for c, _ in kc]
    if moves:
        moves.sort(key=lambda t: -t[1])
        if progress_cb:
            progress_cb(moves[:k])
        return (moves[0][1], moves[:k])
    if threat_search.USE_VCF if use_vcf is None else use_vcf:
        vcf_slice = _vcf_slice(time_budget)
        if vcf_slice > 0:
            vcf_deadline = (t_start + vcf_slice) if time_budget else None
            mv = threat_search.find_forced_kill(
                board, player, pending, turn_count, white_turns, config,
                max_depth=VCF_MAX_DEPTH_HELL,
                deadline=vcf_deadline,
                node_cap=VCF_NODE_CAP_HELL if time_budget
                else VCF_NODE_CAP_UNLIMITED)
            if mv is not None:
                moves = [(int(mv), WIN - 1)]
                if progress_cb:
                    progress_cb(moves[:k])
                return (moves[0][1], moves[:k])
    ranked, ok = _root_iterative(board, player, pending, turn_count,
                                 white_turns, config, rng, depth, time_budget,
                                 k, report_cb=progress_cb, tt=tt,
                                 time_start=t_start, workers=workers)
    if not ok or not ranked:
        return (0.0, [])
    return (ranked[0][1], ranked[:k])


def play_game(config: GameConfig, rng: np.random.Generator, max_steps: int = 450,
              depth: int = DEFAULT_DEPTH, time_budget: float | None = None):
    """启发式自对弈一局,返回 (moves, result)。
    time_budget:每步限时(秒),迭代加深在时限内完成多少深度算多少。"""
    g = ReverseGomoku(config)
    while not g.game_over and g.move_count < max_steps:
        a = choose_heuristic_move(g.board, g.current_player, g.pending,
                                  g.turn_count, g.white_turns, config, rng,
                                  depth=depth, time_budget=time_budget)
        if a is None:
            g.game_over = True         # 无子可走,行棋方必败
            g.loser = g.current_player
            break
        g.make_move(a)
    if not g.game_over:
        g.game_over = True             # 超步数兜底,按和棋计
        g.is_draw = True
    return g.record(), record_mod.game_result(g)


def count_kill_positions(moves, config) -> int:
    """复盘统计棋谱中出现的杀棋局面数(判负窗口开启后)。"""
    g = ReverseGomoku(config)
    n = 0
    for m in moves:
        if g.turn_count >= config.loss_start_turns:
            n += len(kill_samples(g.board, g.current_player, g.pending,
                                  g.turn_count, g.white_turns, config))
        g.make_move(m)
    return n


def _play_task(args_task):
    """worker 任务:一局棋 → (moves, result)。"""
    seed, config, depth, budget = args_task
    return play_game(config, np.random.default_rng(seed), depth=depth,
                     time_budget=budget)


def main():
    import multiprocessing as mp     # 惰性导入:Pyodide 等无 multiprocessing 环境可正常 import 本模块
    p = argparse.ArgumentParser(description="启发式小 AI 批量生成基础棋谱")
    p.add_argument("--games", type=int, default=1000, help="生成局数")
    p.add_argument("--out", default="records", help="输出根目录(按结果分类子目录)")
    p.add_argument("--prefix", default="game")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--white-restrict", type=int, default=2)
    p.add_argument("--loss-start", type=int, default=8)
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                   help=f"搜索深度(1=贪心全局评估,2=深度2(默认),3+ 更深的迭代加深)")
    p.add_argument("--budget", type=float, default=0.0,
                   help="每步限时(秒,0=不限)。批量生成建议 2-5 秒/步,"
                        "保证深度 8 下整局可控时间")
    p.add_argument("--workers", type=int, default=1, help="并行进程数(批量生成提速)")
    p.add_argument("--dry-run", action="store_true", help="只统计,不写文件")
    args = p.parse_args()

    if not (1 <= args.depth <= 8):
        p.error("--depth 仅支持 1-8")
    budget = args.budget if args.budget > 0 else None
    config = GameConfig(loss_start_turns=args.loss_start,
                        white_restrict_turns=args.white_restrict)
    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)
    stats = {"黑胜": 0, "白胜": 0, "和棋": 0, "其他": 0}
    n_kill_total = n_kill_games = 0
    file_idx = 0

    def handle(moves, result):
        nonlocal n_kill_total, n_kill_games, file_idx
        n_kill = count_kill_positions(moves, config)
        n_kill_total += n_kill
        n_kill_games += (n_kill > 0)
        cat = result if result in stats else "其他"
        stats[cat] += 1
        if args.dry_run:
            return
        file_idx += 1
        subs = [cat]
        if n_kill > 0:
            subs.append("有杀棋")
        for sub in subs:
            d = os.path.join(args.out, sub)
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{args.prefix}_{file_idx:05d}.afg")
            with open(path, "w", encoding="utf-8") as f:
                f.write(record_mod.dumps(moves, config, result))

    if args.workers > 1 and args.games > 1:
        tasks = [(args.seed + i * 1000, config, args.depth, budget)
                 for i in range(args.games)]
        with mp.Pool(args.workers) as pool:
            for moves, result in pool.imap_unordered(_play_task, tasks, chunksize=4):
                handle(moves, result)
    else:
        rng = np.random.default_rng(args.seed)
        for _ in range(args.games):
            handle(*play_game(config, rng, depth=args.depth, time_budget=budget))
    if not args.dry_run:
        with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"生成: 启发式 AI (heuristic.py, 极小化搜索 depth={args.depth} "
                    f"+ 全局评估 + 威胁空间着法 + 置换表)"
                    + (f" 每步限时 {args.budget:.1f}s" if budget else "")
                    + "\n"
                    f"规则: white_restrict={config.white_restrict_turns} "
                    f"loss_start={config.loss_start_turns}\n"
                    f"局数: {args.games}\n"
                    f"黑胜: {stats['黑胜']} | 白胜: {stats['白胜']} | "
                    f"和棋: {stats['和棋']} | 其他: {stats['其他']}\n"
                    f"含杀棋局: {n_kill_games} "
                    f"(杀棋局面 {n_kill_total}, 均 {n_kill_total / max(args.games, 1):.1f}/局)\n"
                    f"目录: 黑胜/ 白胜/ 和棋/ 按结果分类; "
                    f"有杀棋/ 为含杀棋局面的副本(训练价值最高)\n")
    print(f"共 {args.games} 局(depth={args.depth}) | 黑胜 {stats['黑胜']} | "
          f"白胜 {stats['白胜']} | 和棋 {stats['和棋']} | 含杀棋局 {n_kill_games} | "
          f"杀棋局面 {n_kill_total}(均 {n_kill_total / max(args.games, 1):.1f}/局)"
          + ("" if args.dry_run else f" → 已分类存入 {args.out}/"))


if __name__ == "__main__":
    main()
