# 更新日志

本文件按发行版记录主要变化;更早的历史见 [git tag](https://github.com/guijixzh/Inverse-Gomoku-GUI-vs-AI/tags)。

## v2.5(2026-10-25)

**主题:单进程性能、两端一致性,以及第一次把「桌面端 vs 网页端」真正打了一遍。**

### 引擎性能

- **Cython 热路径**(`src/antifive/_fast.pyx` + `setup_fast.py`):把 `_shift`、
  `_line_maps_combo`、`_board_features`、`legal_mask_for`/`_line5_map_for`、
  `fwd_run_lengths` 编成本地扩展,单进程 **1.5–1.8×**;扩展缺失或导入失败时自动回退纯
  Python,400 组随机状态与固定深度选点逐位一致。
- **根节点并行**(`src/antifive/root_parallel.py`,高级/大师/传奇):`ANTIFIVE_ROOT_WORKERS=4`
  在 15s 下提升 **2.9–5.3×** 节点数;默认 1 = 串行,不改变单步行为。时限用绝对墙钟
  (`wall_deadline`)传给 worker,排队中的任务不会超时。
- 两者叠加后,15s 每步节点中位数 16.4k → 36.9k。

### 桌面端 vs 网页端(首次同墙钟对拍)

新增 `src/antifive/tools/match_web_py.py` + `web/scripts/engine-bridge.mjs`
(JSONL 桥:new / move / gen / hint / state),同档位、同 15s、黑白互换、10 路并发:

| 档位 | 局数 | 桌面 : 网页 | 桌面胜率 |
| --- | ---: | ---: | --- |
| 初级 | 40 | 18 : 22 | 45.0% |
| 中级 | 40 | 20 : 20 | 50.0% |
| 高级 | 40 | 30 : 10 | 75.0% |
| 大师 | 40 | 33 : 7 | 82.5% |
| 传奇 | 40 | 29 : 11 | 72.5% |
| 合计 | 200 | 130 : 70 | **65.0%** |

优化前同一赛程是 26.8%(400 局)。**并发是实验条件**:10 路时 65%,提到 30 路后双方每步
节点腰斩、结论翻转(56%,CI 跨 50%)。

### 网页端引擎

- `uniformDepth` 成为默认:`ai()` 把同一个 `depth` 原样给五档,与 Python
  `heuristic_versions.choose` 同语义;`uniformDepth:false` 保留旧的按档位夹取,仅供复现。
- 大师/传奇移植 Python 的防守三件套:防守时间片预留、证明必败降权、预防性封堵
  (`search.ts` / `candidates.ts`,`Profile.defense` 开关)。
- `hell` 档开启蒸馏 v1(此前只有 `advanced` 有):不开时大师比高级低约 60–70 Elo。
- 五档 15s 标定(300 局):高级 +182(8.2k)、大师 +121(8.8k)、传奇 +74(9.3k)、
  中级 +61(9.4k)、初级 0(10.0k,锚点);2s 同赛程五档只差 92 Elo。

### 文档

- README 中英版补「15s 等时复测」「网页端与桌面端不是同一套引擎」「单进程性能」三节;
- `docs/engines.md` 开头新增五档两端对照[速查表](docs/engines.md),并补跨引擎同墙钟小节;
- `docs/web-engine.md` 补五档标定、与桌面对照、命令清单。

### 兼容性

- 版本号 `2.0.0 → 2.5.0`;命令行/GUI/棋谱格式与 `.afg` 无变化;
- 无新增运行时依赖;Cython 只在需要提速时手动编译。

## v2.0(2026-10-17)

- 新增第五档引擎「传奇」= 大师 + KataGo 蒸馏 v2(MLP 价值项替换 v1;12s×96 等时
  对拍 75 : 21 胜大师);
- 「地狱」更名「大师」(内部键 `hell` 不变),四档 UI 升级为五档;
- 蒸馏 v2 全链路(`build_distill2_data` / `train_distill2`)与包内权重;调优参数随包分发。
