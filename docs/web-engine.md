# 网页版 TypeScript 引擎

网页版已从 Pyodide + numpy + Python 搜索迁移为 TypeScript 原生引擎，运行在 Web Worker 中。
构建产物约 90KB，不再下载 Python 运行时、numpy 或 wheel。

## 目录

| 文件 | 职责 |
| --- | --- |
| `web/src/engine/rules.ts` | 15x15 棋盘、占领/安置、白棋禁推、自杀筛、判负、增量 Zobrist、合法着法 |
| `web/src/engine/eval.ts` | 线级材料 + 位置摘要；移植 Python 评估，并支持蒸馏价值校正 |
| `web/src/engine/tactics.ts` | 安置杀、占领+安置两步杀、杀对 route、跳形延伸 |
| `web/src/engine/vcf.ts` | 多层强制杀链（VCF），移植 `threat_search.py` 的攻防证明与预算 |
| `web/src/engine/profiles.ts` | 五档引擎：初级/中级/高级/大师/传奇，VCF/开局库/蒸馏能力 |
| `web/src/engine/distill.ts` | KataGo 蒸馏 v1 线性 / v2 MLP 价值校正 |
| `web/src/engine/search.ts` | 迭代加深 alpha-beta、置换表、根/内部候选裁剪、根分析 |
| `web/src/engine/openbook.ts` | KataGo 20k 首手/白应手表（0/1 子） |
| `web/src/engine/variety.ts` | AI 变化档位：根着法软化、开局加宽、开局库概率、每局种子 |
| `web/src/engine/api.ts` | 会话命令、`state()` 字段、`.afg` 导入导出、AI/提示/射线 |
| `web/src/worker.ts` | Worker 协议：init/call → ready/status/progress/result/fatal |

## 一致性

- 规则与合法着法：600 个随机状态 + 40 局随机对局与 Python `reversegomoku` 对拍，0 差异。
- 状态/命令：同手顺的 `state()`（board/player/pending/turn/legal/kinds/numbers/danger）与 Python `web_api` 对拍一致。
- 评估：25 个随机局面在无蒸馏/v1/v2 三种模式下与 Python `heuristic.evaluate` 逐值一致，最大误差 < 4e-7。
- 战术：80 个随机局面的 `kill_pairs` 与 Python `tactics` 完全一致；30 个局面 VCF 深度 1–3 与 Python `find_forced_kill` 完全一致。
- 档位：`beginner` 无 VCF/开局库/蒸馏；`mid` 无蒸馏；`advanced` v1；`hell` v1；`legend` v2。
  （`hell` 在 Python 端是不带蒸馏的独立引擎；网页端 2026-10-23 起改用 v1，见 `records/web_ab_distill_12s/report.md`——
  不开 v1 时大师档会比高级低约 60–70 Elo，开 v1 后两者持平，更符合「大师 ≥ 高级」的预期。）
- 深度：`ai()` 默认 `uniformDepth`（网页端与 Python `heuristic_versions.choose` 同语义——同一 `depth` 原样给五档）；`uniformDepth:false` 才回到按档位夹取的旧行为（初级 2 / 中级 4 / 高级 8 / 大师 +1 / 传奇 +2）。
- 防守：`hell`/`legend` 启用移植自 Python `heuristic.py` 的防守机制（防守预留、证明必败降权、预防性封堵），见下节。
- 搜索：深度 1–3 与无置换表暴力 negamax 分值一致；根与内部候选均按 Python `_candidates` 移植（空位/占领各 k、杀着/防守路线、diversity、history/killer 补充），内部搜索加入 PVS、TT 边界、killer、强制扩展与 `_jump_through` 送杀判定；20 局面深度 4 与 Python 同着法 17/20。
- 提示：`hint` 走根候选全窗口分析，返回 `pos_v` 与按分值排序的前 5 着，形状与 Python `analysis_moves` 一致。
- 变化：引擎层缺省 `variety=off`，同手顺逐位可复现（与旧版一致，`tests/variety.test.mjs` 覆盖）；
  只有 `ai` 请求显式带 `variety` 才启用采样。

## 大师/传奇档防守机制

`hell`/`legend` 两档在搜索侧启用一套移植自 Python `src/antifive/heuristic.py`
（大师/传奇）的防守机制，其余档位保持原样（与 Python `heuristic_v6` 等一致）：

| 机制 | 网页位置 | Python 出处 |
| --- | --- | --- |
| 防守预留时间片：先扣 `min(1.8s, 15% 时限)`，剩余时间才做迭代加深；无战术材料时只扣 100ms | `search.ts` `defensiveSliceMs` / `defenseReserveNeeded` / `searchBest` | `_vcf_slice`、`_defense_reserve_needed` |
| 证明必败降权：根着法逐一手推，走完即被对手证明必败的降到 `-(MATE-(2*turns+3))`（尽量晚死） | `search.ts` `filterProvenLosses` / `provenKillTurns` / `occupyTurnLoss` | `_filter_defensive_kills`、`_proven_loss_value`、`_occupy_turn_loss` |
| 预防性封堵（先手礼防御）：枚举"投送枢纽"与可作为投掷子的己方棋子，模拟成杀后把射线空格补进根候选 | `candidates.ts` `pregiftCover` / `appendPregift` | `_pregift_defense`、`_append_pregift` |

开关在 `profiles.ts` 的 `Profile.defense`（仅 `hell`/`legend` 为 `true`）；
`web/tests/defense.test.mjs` 覆盖档位开关、封堵覆盖与时限行为。

## AI 变化（去固定化）

旧版网页每次都是同一条确定性路径：同手顺 + 同深度/时限 + 同置换表 ⇒ 同应手，换局、
刷新、换设备都一样，玩家连下同一手顺会觉得「AI 在背棋谱」。现在引擎加入**变化档位**
（`web/src/engine/variety.ts`），三层机制叠加：

| 层 | 机制 | 位置 |
| --- | --- | --- |
| 根着法软化 | 在根节点全窗口分值上按 `exp((v-v_max)/T)` 加权抽样，只对「差距 ≤ window」且分值前 K 的着法开放 | `search.ts` `searchBest` / `sampleRoot` |
| 开局加宽 | 前 N 个半手用更大的 T/window/K（开局评估差小、真实价值差也小） | `variety.ts` `varietyAt` |
| 开局库概率化 | `bookChance<1` 时按概率跳过 KataGo 首手/白应手表，交给搜索决定 | `api.ts` `ai()` |

| 档位 | 引擎层缺省 | 网页缺省 | T | window | K | 开局 T/W/K | 开局库概率 |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| `off` | ✓ | | 0 | 0 | 1 | — | 1.0 |
| `low` | | | 0.06 | 0.15 | 2 | 0.12 / 0.3 / 3 | 0.55 |
| `mid` | | ✓ | 0.10 | 0.30 | 3 | 0.22 / 0.5 / 3 | 0.35 |
| `high` | | | 0.20 | 0.60 | 4 | 0.35 / 0.8 / 4 | 0.20 |
| `custom` | | | 逐项可调（模板 = `mid`） | | | | |

**自定义档**：设置面板选「自定义」后展开 8 个参数框（中盘 T/window/K、开局手数 N、开局
T/window/K、开局库概率），改任一数值即自动切到该档；参数由引擎侧统一校验（非有限值丢弃、
整数项取整、其余夹到下表区间），所以前端传越界值也不会破坏搜索：

| 参数 | 键 | 区间 | 步长 |
| --- | --- | --- | --- |
| 中盘 T | `temperature` | 0 – 1 | 0.02 |
| 中盘 window | `window` | 0 – 3 | 0.05 |
| 中盘 K | `poolCap` | 1 – 8 | 1 |
| 开局手数 | `openingPlies` | 0 – 20 | 1 |
| 开局 T | `openingTemperature` | 0 – 1.5 | 0.02 |
| 开局 window | `openingWindow` | 0 – 4 | 0.05 |
| 开局 K | `openingPoolCap` | 1 – 8 | 1 |
| 开局库概率 | `bookChance` | 0 – 1 | 0.05 |

只覆盖给到的字段（`varietyFor("custom", params)`），`T=0 / K=1` 等价于 `off` 的确定性；
`params` 只对 `custom` 生效，预设档位传参会被忽略（避免预设被悄悄改掉）。参数元数据是单一
事实来源（`variety.ts` 的 `PARAM_FIELDS`），面板与校验共用。

安全边界：

- **能赢的局一定赢**：一手杀（`findWinningMove`）与 VCF 找到的强制杀优先返回；
  `|v| ≥ MATE-450`（已证明的必胜/必败）整段不参与抽样。
- **分析不随机**：`hint` / 根分析不接变化档位，结论始终可复现（与 Python 一致：随机只在对局侧）。
- **稀疏局面必须用 K 收口**：空盘/一子局的根着法分值差普遍 < 0.1，只用 window 会把棋盘
  另一端的远端点也算成「同分」（实测曾抽出角点）；先按分值取前 K 个（同分保持候选启发式
  顺序，即中心优先），diversity 补充的端点不会进池。
- **每局一种子**：`crypto.getRandomValues` 取种子 + `mulberry32` 序列；`new` 可传 `seed`
  复现整局。同一局内悔棋后重搜会给出不同应手（随机序列继续前进），这是刻意的。

实测（`node scripts/variety-probe.mjs`；深度 4、每步 ≤6 万节点；固定手顺 黑 112 / 白 52 /
黑 172 / 白 34，12 次独立开局）：

| 档位 | 不同应手 | 分布 |
| --- | ---: | --- |
| `off` | 1/12 | 34×12 |
| `low` | 1/12 | 34×12 |
| `mid` | 3/12 | 34×9 18×2 52×1 |
| `high` | 4/12 | 52×5 18×3 37×2 34×2 |

换成网页默认口径（高级 + 深度 8、每步 ≤30 万节点 ≈「AI 读秒 15 秒」，8 次独立开局）：

| 档位 | 中腹 4 手后 | 白方首应手 |
| --- | --- | --- |
| `off` | 19×8 | 136×8 |
| `low` | 19×6 52×2 | 136×7 1×1 |
| `mid` | 52×3 19×3 37×2 | 136×5 111×3 |
| `high` | 19×3 52×3 18×2 | 1×4 136×3 111×1 |

即 `off` 在给定深度/时限下永远同一手（原有保证不变），`mid` 有 3/8–5/8 的局面会走出
与 `off` 不同的应手，`high` 约一半；中盘分布已经比较平（最高频着法只占 3/8）。

棋力代价（`node scripts/variety-probe.mjs --ab`：同一随机开局交换颜色各一盘，开局库不参与，
`variety` 对 `off`）：

| 每步节点上限 | 档位 | 局数 | 得分 | Elo |
| ---: | --- | ---: | ---: | ---: |
| 3,000 | `mid` | 320 | 46.3% | ≈ −27 |
| 20,000 | `mid` | 200 | 46.0% | ≈ −28 |
| 3,000 | `high` | 320 | 39.4% | ≈ −75 |
| 20,000 | `high` | 200 | 37.5% | ≈ −93 |
| 合计 | `mid` | 520 | 46.2% | ≈ −27 ± 38 |
| 合计 | `high` | 520 | 38.7% | ≈ −92 ± 37 |

A/B 每步只给 3,000/20,000 节点（实际深度约 2–4），评估噪声远大于网页默认「高级 + 读秒 15s」
（深度 8+），所以上表是变化档位的**棋力代价上界**：`mid` 在 520 局里与 `off` 不可区分，
`high` 大致让 80–90 Elo 换取明显变化。

### 随机只碰「价值相近」的着法

判据是**同一次搜索自己的根着法分值表**（`searchBest` 返回的 `rootScored`），不依赖跨搜索
的基线（节点上限会让不同次搜索完成不同层数）：

- 候选池 = 分值前 K **且** 与最优差 ≤ `window` 的根着法；`sampleRoot` 只从池子里抽，
  池外着法的概率恒为 0（代码是 `filter` 掉，不是「大概率不选」）。
- 池子只剩一个着法时就是它，没有任何随机余地。
- `|v| ≥ MATE-450`（已证明的必胜/必败）、一手杀（`findWinningMove`）、VCF 强制杀都不进抽样。

实测（`node scripts/variety-probe.mjs --deviation`；每次搜索前清空置换表、只换种子，
因此偏差只可能来自采样）：

| 深度 / 每步节点 | 样本 | 池外采样 | 池=1 时选中最优 | 中盘 `mid` 最大偏差 / 窗口 |
| --- | ---: | ---: | ---: | --- |
| 4 / 6 万 | 252 | 0 | 72/72 | 0.221 / 0.30 |
| 8 / 30 万（网页默认） | 126 | 0 | 本组无池=1 局面 | 0.151 / 0.30 |

其中「池=1」的 6 个局面/档位组合共 72 次抽样全部落在唯一候选上——分值明显领先时
（如 `中盘8手` 最优与次优差 0.504 > `mid` 窗口 0.30），变化档位不会换手。反过来，
`白首应手` 这类近同分局面各候选分值完全相同（差 0.000），换手不带来任何棋力损失。

已证明胜势的局面（最优与次优差 999 992）在深度 8 / 30 万节点下，`low`/`mid`/`high`
各 12 个种子全部下同一手（36/36），且根着法表为空——说明根本进不到抽样。

## 性能

Chrome（headless，固定局面）实测：

| 模式 | 节点 | 墙钟 | 节点/秒 |
| --- | ---: | ---: | ---: |
| 无预算、候选裁剪 | 0.1–1 万 | < 0.1s | 10–100 万 |
| 10s 预算（含 VCF/蒸馏/PVS） | ~62 万 | 10.0s | ~6.2 万 |

旧 Pyodide 版约 2.3k 节点/秒；单节点成本提升约 20–40 倍，首屏资源从约 17MB 降到约 90KB。

## 固定节点对照（旧网页 15s）

旧网页 20 局面实测 15s 平均约 2.84 万节点（21,696–33,082），平均墙钟 13.5s。
把 TS 每步硬限制到对应旧节点数后，平均墙钟约 0.78s，达到深度 6–7，与旧版同着法 12/20。
固定节点对局（旧 Pyodide 15s vs TS 28,422 节点）两局 **1–1**，两局均为白方获胜；
样本很小，只能说明未观察到明显棋力差，不能据此判定强弱。

## 命令

```bash
cd web
npm run build   # tsc --noEmit && vite build
npm test        # Node 规则/API/战术/档位/棋谱测试
npm run smoke   # 浏览器冒烟测试（需 Chrome/Edge）
npm run bench   # 浏览器搜索基准
node scripts/variety-probe.mjs        # AI 变化档位:同一手顺的应手分布
node scripts/variety-probe.mjs --ab   # AI 变化档位:棋力 A/B(可多进程分区,见脚本头注释)
node scripts/variety-probe.mjs --deviation  # 采样边界:池外采样必须为 0、池=1 必须选它
OLD_ENGINE_ROOT=<旧引擎检出目录> node scripts/selfplay-ab.mjs  # 两个检出之间的棋力 A/B(见脚本头注释)
```
