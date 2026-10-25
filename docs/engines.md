# AI 引擎与 KataGo 配置

本项目包含四类对局引擎,均通过对战接口 `choose(game) -> 着法 idx` 与 GUI 解耦。

| 引擎 | 依赖 | 强度 | 说明 |
|---|---|---|---|
| 随机先验 MCTS | numpy | 低 | 无需任何额外文件,均匀先验搜索,默认兜底 |
| 自研神经网络 MCTS | numpy + torch + 权重 | **不足(训练失败)** | 最多稍弱于启发式,仅供研究 |
| 启发式 AI | numpy | 中高 | 迭代加深 alpha-beta + 全局评估 + 置换表,深度 1–8 |
| KataGo(逆五) | 外部 katago.exe + `.bin` 权重 | 高 | 自训练逆五规则权重,推荐实用引擎 |

## 自研神经网络(失败尝试)

权重位于 `models/warm_model.pth`、`models/best_model.pth`。训练总体失败,
强度不足,详见 `docs/training-log.md`。加载方式:

```bash
python -m antifive.gui --model models/best_model.pth
```

未安装 torch 时神经网络引擎自动禁用(设置面板中不显示该项),其余功能不受影响。

## 启发式 AI

内置算法,无需额外文件;可选 `ANTIFIVE_ROOT_WORKERS` 开根节点并行、
`setup_fast.py` 编译 Cython 热路径(见下文两节);GUI 中可选深度 1–8(默认 8 + AI 读秒 12 秒,五档通用;
大师/传奇在时限内动态加深,不受 8 层硬上限),并可切换五档引擎版本与 VCF
强制杀链开关(初级无 VCF;默认高级 + VCF 开):

| 版本 | 实现 | 组成 |
|---|---|---|
| 初级 | `tools/heuristic_v2.py` 冻结经典版 | 老算法,无 VCF/开局库/蒸馏 |
| 中级 | `tools/heuristic_v5.py` 冻结快照 | 搜索一致性修复 + VCF v5.1 + 开局库 + 防守 VCF |
| 高级(默认) | `tools/heuristic_v6.py` 冻结快照 | 中级 + KataGo 蒸馏价值校正(旧 `antifive.heuristic` 行为) |
| 大师 | `antifive.heuristic` | 高级 + 防守强化:深化 VCF(d12/24 万节点/≤1.8s)、严格时限预算管理与自适应防守预留、防守校验全候选 + 时限动态加深(2026-10 前旧称「地狱」,内部键仍为 `hell`) |
| 传奇 | `tools/heuristic_v7.py` 冻结快照 | 大师 + KataGo 蒸馏 v2:激活时 MLP 价值项**替换** v1(权重 0.5;`ANTIFIVE_DISTILL2=0` 退回大师)、5 格模式先验可选(`ANTIFIVE_PRIOR2_W`,默认 0) |

「大师」「传奇」是**引擎版本**(困难/高级之上的强度档,可与任意搜索深度搭配),
网页与桌面都在「引擎版本」单选里选择;网页「AI 难度」仍为三档
(简单/中等/困难 = 深度 2/4/8),不放引擎档位,避免同一入口重复且互相覆盖
(选大师/传奇难度会静默忽略版本单选)。网页/桌面默认仍为困难/高级,
不默认大师/传奇。
调优参数内置在包内 `antifive/params.json`,随 wheel 分发(网页/Pyodide 与
纯净安装都生效,此前网页端因无仓库 `data/` 目录而静默回退内置常量的问题已修);
桌面可在 `data/params.json`(param_tune 输出)或 `data/params_hell.json`
(仅大师档优先读取;传奇/高级/中级共用 `data/params.json`)放覆盖文件。
A/B 开关:`ANTIFIVE_PREGIFT=1` / `ANTIFIVE_GIFT3=1`(P1/P2,默认关闭)、
`ANTIFIVE_NO_DYNAMIC=1`、`ANTIFIVE_NO_LMR=1`、`ANTIFIVE_NO_ROOT_PVS=1`
(实验特性)、VCF 参数 `ANTIFIVE_HELL_VCF_DEPTH/NODES/TIME`;
根节点并行 `ANTIFIVE_ROOT_WORKERS`(默认 1=串行,高级/大师/传奇有效,见下文
「根节点并行」;注意与 `match_web_py` 的 `--workers` 相乘,并发会改变结论)。

命令行行为 `--engine heuristic --heuristic-depth N --limit none|ai --ai-time S
--heuristic-version beginner|mid|advanced|hell|legend --no-vcf`
(兼容旧值 old/new、「地狱」→hell)。
参数内置 `antifive/params.json`(随包分发),桌面覆盖文件由
`src/antifive/tools/param_tune.py` 调优输出(默认 `data/params.json`)。

### 大师档等时对拍(12s/步 × 96 局/对,seed 0,严格时限)

> 历史口径:下表测量于 2026-10-13,当时的「地狱」即现「大师」(仅显示名变更,
> 实现与环境变量不变);「传奇 vs 大师」见下方蒸馏 v2 小节(12s×96:75 : 21)。

早期 1s 口径下低档被时钟饿死(高级对初级 98% 等);12s 口径下人人能完成
深度 8,档间差距大幅收敛。2026-10-13 实测(`match_ai` 直接对拍,96 局/对;
大师档已修复"进攻 VCF 不计入本步预算"的超时问题,与对手同为严格 12s):

| 配对(新 vs 旧) | 比分 | 近似 Elo 差(95% CI) |
|---|---|---|
| 大师 vs 高级 | 63 : 33(65.6%) | +112(+41 ~ +184) |
| 大师 vs 中级 | 67 : 29(69.8%) | +145(+72 ~ +219) |
| 大师 vs 初级 | 76 : 20(79.2%) | +232(+149 ~ +314) |
| 高级 vs 中级 | 48 : 48(50.0%) | 0(−68 ~ +68) |
| 高级 vs 初级 | 51 : 45(53.1%) | +22(−46 ~ +89) |
| 中级 vs 初级 | 68 : 28(70.8%) | +154(+78 ~ +225) |

以初级为 1000 的相对 Elo(12s 口径,直接对初级换算):中级 ≈1154、
高级 ≈1022、大师 ≈1232;旧三档在 12s 下差距收敛且存在风格不一致
(高级对中级 50%,中级对初级 70.8%),单对 CI 约 ±70 Elo。**大师在三个
直接配对中均显著领先(对高级 +112)**,是当时四档中的最强者。

> 复盘注:大师档此前的"57.3% vs 高级"含预算超时(进攻 VCF 的时间未计入
> 本步预算,平均多出数秒/步);修复后须在严格时限内取胜,当前成绩为
> 修复后重新实测。

### 防守强化机制(大师档相对高级)

- **深化 VCF**:强制杀链从 d8/6 万节点/0.6s 提升到 d12/24 万节点/≤1.8s,
  攻防两侧共用同一预算(`_vcf_slice`);无对局限时改用确定性节点上限
  (`VCF_NODE_CAP_UNLIMITED`),保证 choose 与 analysis 结果一致;这是
  对高级 +112 Elo 的主要来源;
- **预算对等与严格计时**(P0):根防守校验覆盖全部候选、节点上限与进攻
  一致、共享同一 deadline;本步总墙钟从 choose 入口计时,"进攻 VCF + 主
  搜索 + 防守校验"合计 ≤ 时限(修复此前平均数秒/步的超时);
- **自适应防守预留**:仅当己方已有危险四连(对方下一步可完成)或对方已有
  可杀四时预留大额防守 VCF 预算,常规中盘只留 0.1s,回收时钟给搜索;
- **时限动态加深**:`MAX_DYNAMIC_DEPTH=16`,预算用满前逐层加深;
- **P1 预防性防守候选**(`ANTIFIVE_PREGIFT=1` 启用):枚举对方一步"投送
  己方子成四/五"的枢纽(连续形 + 跳形缺口,jump_through 判定),对确成杀的
  投送收集射线封堵格,按覆盖数排序补入根候选。严格等时对拍显示其净负
  (根候选变多、搜索变浅),默认关闭;
- **P2 潜伏威胁评估**(`ANTIFIVE_GIFT3=1` 启用):`E_SELF_GIFT3` 惩罚己方
  三连"可被投送"负债;严格等时对拍净负,默认关闭;
- **P3 复核**:多回合"以攻代守"在杀对已成立时不成立(不封堵则进攻方下一
  回合立即完成杀),仅有"立刻反杀"是逃生(原 has_kill 检查已覆盖),故不做
  不成立的放宽,并补回归测试锁定该语义;
- **已否决的实验**:根 α 传播 + PVS 与 LMR 速度收益大(有效深度 +1~3 层),
  但当前搜索树依赖搜索历史(`_include_extras` 会把 TT/killer 着法补进搜索
  集合),零窗试搜与全窗搜索走不同的树,边界失效(实测把最优着法从 0.871
  低估到 0.412,对拍 25%~38%),默认关闭(`USE_ROOT_PVS=False`、`USE_LMR=False`),
  代码与开关保留待确定性改造后再启用。

开局库:`antifive/openbook_data.py` 由 `tools/openbook_gen.py` 用 KataGo 生成
(默认 20k visits;黑空盘首手 + 每个 D4 等价类首手的白方应手),运行时按 D4
对称规范化查表,`ANTIFIVE_NO_BOOK=1` 可关闭。重新生成(需本地 katago):

```bash
python -m antifive.tools.openbook_gen --visits 20000
```

### KataGo 蒸馏(策略先验 + 价值校正)

`antifive/distill_data.py` 由 `tools/build_distill_data.py`(并行采样局面并用
KataGo 20k visits 标注 top-20 候选分布,可断点续跑)与 `tools/train_distill.py`
(列表式 softmax 策略先验 + 价值线性回归)生成:

```bash
python -m antifive.tools.build_distill_data --workers 4 --visits 20000 \
    --games-selfplay 200 --games-katago 12 --minutes 22   # 分块运行,续跑
python -m antifive.tools.train_distill --pool checkpoints/distill_pool.npz
```

运行时:**价值校正默认开启**(`evaluate` 混入,权重 `ANTIFIVE_VALUE_W` 默认
0.5,`ANTIFIVE_NO_DISTILL=1` 关闭);策略先验默认关闭(`ANTIFIVE_PRIOR_W`
默认 0,设为正数开启)。等时对拍:价值校正 vs 修复前 144 局 129:15(89.6%,
两个时限),策略先验单独启用 96 局 47:49(49%,故默认关闭)。

### KataGo 蒸馏 v2(「传奇」;与 v1 平行,不污染冻结档)

`antifive/distill2.py` 的权重由 `tools/build_distill2_data.py`(规模化采样 +
回放既有棋谱 + 两层标注:批量 2k visits、硬样本 20k visits 带 `tier`)与
`tools/train_distill2.py`(D4×8 增广、按局切分;18 维先验 + 5 格模式表、
25 维价值,单隐层 MLP)生成到包内 `antifive/distill_data2.py`:

```bash
python -m antifive.tools.build_distill2_data --games-selfplay 300 \
    --games-katago 48 --replay-dir records/stage2_12s/games --workers 4 --minutes 40
python -m antifive.tools.build_distill2_data --merge-only      # 分块续跑后合并
python -m antifive.tools.train_distill2                          # 优先 distill2_pool.npz
```

运行时:大师档需 `ANTIFIVE_DISTILL2=1` 开启(默认关);**传奇快照
`tools/heuristic_v7.py` 默认常开**(`ANTIFIVE_DISTILL2=0` 退回大师)。激活
时 v2 价值项**替换** v1(而非叠加):离线诊断 v2 与 KataGo 价值相关性 0.68
(v1 仅 0.33);叠加配置 12s×96 仅 47 : 49(约五五开),替换后 75 : 21。权重
`ANTIFIVE_VALUE2_W`(默认 0.5;12s 下 0.5 优于 1.0:78.1% vs 70%)/
`ANTIFIVE_PRIOR2_W`(默认 0=关)可缩放;数据缺失/维度不符时自动退化。

正式数据:460 局(自对弈 300 + KataGo 对局 48 + 棋谱回放 112)标注出
7664 个局面(硬样本 1016)。离线指标:价值 val MSE 0.592 / 符号 85.1%
(v1 为 1.147 / 77.0%),先验 val top1 45.2% / top3 69.7%(v1 为 22.0% /
46.6%)。**12s×96 等时对拍:传奇 75 : 21 大师(78.1%,近似 +210 Elo)**。
策略先验的负结果:开启(`ANTIFIVE_PRIOR2_W=30`)后 8s×40 对拍仅 4 : 36(10%),
大幅劣化,默认保持关闭;离线 top1 质量未转化为实战收益。

对拍(等时强弱回归;冻结快照:`heuristic_v2`=初级、`heuristic_v5`=中级、
`heuristic_v6`=高级、`antifive.heuristic`=大师、`heuristic_v7`=传奇、
`heuristic_v4`=搜索一致性修复前的旧快照):

```bash
# 大师 vs 高级(防守强化回归)
python -m antifive.tools.match_ai --games 96 --depth 8 --budget 12 --workers 8 \
    --new-module antifive.heuristic --old-module antifive.tools.heuristic_v6
# 传奇 vs 大师(蒸馏 v2 回归;传奇默认开启 v2,大师可用 ANTIFIVE_DISTILL2=1 对齐)
python -m antifive.tools.match_ai --games 96 --depth 8 --budget 12 --workers 8 \
    --new-module antifive.tools.heuristic_v7 --old-module antifive.heuristic
# 高级 vs 中级
python -m antifive.tools.match_ai --games 96 --depth 8 --budget 12 --workers 8 \
    --new-module antifive.tools.heuristic_v6 \
    --old-module antifive.tools.heuristic_v5
# 中级 vs 初级
python -m antifive.tools.match_ai --games 96 --depth 8 --budget 12 --workers 8 \
    --new-module antifive.tools.heuristic_v5 \
    --old-module antifive.tools.heuristic_v2
```

## 桌面端 vs 网页端(15s 同墙钟,2026-10)

两端的同名档位**不是同一套引擎**(桌面端是五套独立实现,网页端是同一份约 500 行
alpha-beta 的参数化,见 [`web-engine.md`](web-engine.md)),所以「网页端某档 ≈ Python
某档」这类推断没有依据,只能对打。工具:`antifive.tools.match_web_py` +
`web/scripts/engine-bridge.mjs`(同档位对同档位、4 步随机开局同步喂双方、黑白互换、
10 路并发、每步 15 秒墙钟、每局自检两侧都真的下过棋):

| 档位 | 局数 | 桌面 : 网页 | 桌面胜率(95% CI) | Elo 差 |
| --- | ---: | ---: | --- | ---: |
| 初级 | 40 | 18 : 22 | 45.0%(30.7–60.2) | −35 |
| 中级 | 40 | 20 : 20 | 50.0%(35.2–64.8) | 0 |
| 高级 | 40 | 30 : 10 | **75.0%**(59.8–85.8) | +191 |
| 大师 | 40 | 33 : 7 | **82.5%**(68.0–91.3) | +269 |
| 传奇 | 40 | 29 : 11 | **72.5%**(57.2–83.9) | +168 |
| 合计 | 200 | 130 : 70 | **65.0%**(58.2–71.3) | +108 |

换算成算力:桌面端每步节点中位数 36,898(高档档位 60k 上下),网页端 404,236——
桌面端用约 1/11 的节点赢下 65%,差在每节点质量(五套独立引擎 + 防守/预算机制 + Cython)。

**并发数是实验条件,不是调参项**:同一套 harness 把并发从 10 提到 30,双方每步节点
被腰斩(桌面 25k→13k、网页 40 万→20 万),桌面端胜率从 65% 掉到 56%(CI 跨 50%),
结论会翻转。复现时固定并发。

```bash
# 桌面端 vs 网页端:同档同墙钟(建议 --workers 10,别再往上加)
python -m antifive.tools.match_web_py run --out records/web_vs_py \
    --pairs advanced:advanced --games-per-pair 40 --py-time 15 --web-budget 15 --workers 10
# 出表 + 图(只统计两侧都真的下过棋的对局,并按棋谱重判胜负)
python src/antifive/tools/analyze_web_py.py --out records/web_vs_py --budget 15
```

## 根节点并行(高级/大师/传奇,2026-10-24)

Python 启发式单核约 1.5k 节点/秒,而 alpha-beta 在根节点对每个候选着法都是
**全窗口独立评估**——天然可并行。`antifive.root_parallel` 提供一个常驻 spawn
进程池,`heuristic_v6`(高级)、`antifive.heuristic`(大师)与
`tools/heuristic_v7`(传奇)的根迭代加深都可以通过它把"根着法评估"分发到
多个进程,深度/时限语义与串行完全一致。

用法(环境变量,默认 1=串行、行为与旧版逐位相同):

```bash
# 4 进程跑高级引擎的根并行
ANTIFIVE_ROOT_WORKERS=4 python -m antifive.tools.match_web_py run ...
```

实测(15s/步、同一中盘局面、depth 8):

| 进程数 | 节点/步 | 节点/秒 | 相对串行 |
|---:|---:|---:|---:|
| 1 | 21,455 | 1,456 | 1.00x |
| 4 | 61,613 | 4,183 | 2.87x |
| 8 | 77,906 | 5,291 | 3.63x |

固定深度(无时限)时并行与串行的**选点完全一致**(三档均已回归验证);带时限
时并行搜索更深,选点可能不同(属于更强的结果,不是行为差异)。既定代价:每个
worker 各自一份置换表(不再与父进程共享),内存约 36MB/进程(可用
`ANTIFIVE_TT_BITS=19` 压到 9MB)。

大师/传奇(8s 预算、同一中盘局面)的实测:

| 引擎 | 1 进程 | 4 进程 | 8 进程 |
| --- | ---: | ---: | ---: |
| 大师(hell) | 11,355 节点 | 27,904(2.5x) | 43,752(3.9x) |
| 传奇(legend) | 24,968 节点 | 60,853(2.4x) | 71,896(2.9x) |

对局层面的效果(15s/步、同档位对网页端,串行基线每档 80 局):

| 档位 | 串行 | 根并行(4 进程/步) | Fisher p | 报告 |
| --- | ---: | ---: | ---: | --- |
| 高级 | 20.0%(16:64) | **55.0%**(11:9) | 0.0037 | `records/web_vs_py_par15s/` |
| 大师 | 16.2%(13:67) | **90.0%**(9:1) | <0.001 | `records/web_vs_py_par_all15s/` |
| 传奇 | 40.0%(32:48) | **90.0%**(9:1) | 0.005 | `records/web_vs_py_par_all15s/` |

并行版每步节点从 1.6–2.0 万涨到 8–9.6 万(约 5x),这正是此前「网页端强
175 Elo」的主要来源——单线程 Python 的算力劣势。样本仍小(高级 20 局、
大师/传奇各 10 局),只用于定方向。

## KataGo(逆五规则)

### 权重

- `models/katago/antifive15-init.bin`:初始权重(随机初始化,棋力无意义);
- `models/katago/antifive15-20260821-083223.bin`:自训练权重(最新)。

训练规则:逆五子棋(连五者输,`white_restrict=2`),基于
[hzyhhzy/KataGomo](https://github.com/hzyhhzy/KataGomo)(KataGo 分支,MIT)。

### 可执行文件(不随仓库分发)

`katago.exe`、CUDA/cuDNN 运行库因许可与体积原因不随仓库分发,KataGo 的训练
程序与依赖也不在开源范围内(本仓库只含运行配置与成品权重)。运行环境自检可
执行 `python scripts/setup_env.py`,它会按显卡直接给出该用哪个版本。

KataGo 官方预编译包**不含逆五规则**(`whiteRestrictTurns` /
`newgame_gomokuamazons`),必须从 [hzyhhzy/KataGomo](https://github.com/hzyhhzy/KataGomo)
源码构建。**同一份 `.bin` 权重与 `gtp_test.cfg` 通用**,只是按显卡换构建后端:

| 硬件 | 用哪个版本 | 运行期还需要 | 说明 |
| --- | --- | --- | --- |
| NVIDIA(驱动较新) | **CUDA**(最快) | CUDA Toolkit + cuDNN 的 DLL | 官方教程的默认路线 |
| AMD 独显 / APU | **OpenCL** | 显卡驱动(Adrenalin 自带 OpenCL) | KataGo 官方支持,不需要 CUDA |
| Intel 核显 | **OpenCL** | 核显驱动 | 能跑,速度约独显的数分之一 |
| 无独显 / 无可用显卡 | **Eigen(纯 CPU)** | 无 | 任何 x64 机器可跑,AVX2 多线程 |
| 其他/不确定 | 先 OpenCL,不行换 Eigen | — | 通用兜底 |

#### 构建(Windows,VS2022 + CMake,KataGomo 源码)

```bat
:: CUDA(NVIDIA)
cmake -S cpp -B build_cuda -G "Visual Studio 17 2022" -A x64 ^
  -DUSE_BACKEND=CUDA -DNO_GIT_REVISION=1 -DNO_LIBZIP=1
cmake --build build_cuda --config Release -j

:: OpenCL(AMD 独显 / APU / Intel 核显)
cmake -S cpp -B build_opencl -G "Visual Studio 17 2022" -A x64 ^
  -DUSE_BACKEND=OPENCL -DNO_GIT_REVISION=1 -DNO_LIBZIP=1
cmake --build build_opencl --config Release -j

:: Eigen(纯 CPU,AVX2)
cmake -S cpp -B build_eigen -G "Visual Studio 17 2022" -A x64 ^
  -DUSE_BACKEND=EIGEN -DUSE_AVX2=1 -DEIGEN3_INCLUDE_DIRS=<eigen3 解压目录> ^
  -DNO_GIT_REVISION=1 -DNO_LIBZIP=1
cmake --build build_eigen --config Release -j
```

- OpenCL 构建:`find_package(OpenCL)` 找不到时会退回 CUDA Toolkit 自带的
  OpenCL 头/库(`CUDA::OpenCL`),所以 N 卡机器也能编出给 A 卡用的 exe;
  运行期用的是显卡驱动提供的系统 `OpenCL.dll`,不依赖 NVIDIA。
- Eigen 构建:纯头文件库,不需要任何 GPU 相关依赖;`-DEIGEN3_INCLUDE_DIRS`
  指向解压后的 eigen3 根目录(`Eigen/`、`unsupported/` 所在层)。
- 产物 `katago.exe` 与 `zlib1.dll` 放到仓库根目录 `katago/`;CUDA 版还要把
  CUDA/cuDNN 的 DLL 一起放进来。

#### 核显 / 无独显 / A 卡的参数建议

- **AMD / Intel 核显(OpenCL)**:`numSearchThreads` 保持 4 左右,
  `nnMaxBatchSize` 从 16 起(核显可 8);`config/gtp_test.cfg` 里的
  `nnMaxBatchSize = 256` 是按 CUDA 调的,OpenCL/CPU 建议临时改小。
  APU 与独显同时存在时,启动日志会列出全部 OpenCL 设备,用
  `openclDeviceToUse = 设备号` 指定;首次启动会自调优(tuner),慢几十秒属正常。
- **纯 CPU(Eigen)**:`numEigenThreadsPerModel` 设为物理核数;`nnMaxBatchSize`
  8 左右即可。15×15 小棋盘 + 11.8MB 小模型,桌面 CPU 通常能到独显 CUDA 的
  十分之一量级,仍明显强于启发式。
- 三档都建议先用 `katago benchmark` 实测再定参数:

```bat
katago.exe benchmark -model models/katago/antifive15-20260821-083223.bin ^
  -config config/gtp_test.cfg -boardsize 15
```

- 若 `katago.exe` 起不来并报 `[WinError 126] 找不到指定的模块`,基本是版本与
  显卡不匹配(在 A 卡上跑了 CUDA 版):按上表改换 OpenCL/Eigen 版即可。

### 配置

`config/engines.json`(相对路径以仓库根/EXE 所在目录解析):

```json
[
  { "name": "启发式", "kind": "heuristic", "depth": 8, "budget": 2.0 },
  {
    "name": "逆五KataGo",
    "engine_path": "katago/katago.exe",
    "weight_path": "models/katago/antifive15-init.bin",
    "config_path": "config/gtp_test.cfg",
    "auto_latest": true
  }
]
```

- `auto_latest=true`:自动改用权重目录中最新的 `antifive15-*.bin`
  (排除 `antifive15-init.bin`,避免随机初始化权重被误选);
- `config/gtp_test.cfg`:GTP 最小配置(`whiteRestrictTurns=2`、`maxVisits`、
  搜索线程、`maxTime` 等;GUI 会按读秒/模拟次数追加 `-override-config`);
- **本机路径覆盖**:`config/engines.local.json`(格式同上,已被 gitignore)
  优先于 `engines.json` 读取,适合放置机器相关的 `katago.exe` 绝对路径;
  在 GUI「设定 → KataGo」中用「浏览」选好路径后点「确定」会自动写入该文件。
  相对路径按 EXE 同级目录、`_internal`、仓库根、当前目录依次探测解析;
  配置指向的文件缺失时,还会在 `katago/` 等常见位置自动探测可执行文件与权重。

### 自训练与对拍

- 训练/自对弈实验使用 KataGomo 的 `selfplay` 与 GTP 工具,不属于本仓库;
- 分级检验(随机→d2→d4→d8 阶梯):
  `python -m antifive.tools.katago_vs_heur --model models/katago/antifive15-init.bin
  --engine katago/katago.exe --engine-cfg config/gtp_test.cfg`;
- 等时正式对拍(配对开局、黑白互换、断点续跑、落谱自检):
  `python -m antifive.tools.match_katago --games 96 --katago-time 12
  --heur-budget 12 --workers 2`;
  输出 `records/katago_vs_advanced_<时间戳>/`(.afg 棋谱 + 逐局 JSON +
  Wilson 置信区间的 summary.json);中断后对同一 `--out` 重跑即续跑。
  注意:该工具对启发式使用 `mask_suicide=True`(GUI 同款);不能改为
  `False`——否则启发式的候选生成会走出"占领即成己方五连"的自杀着。

### 高级启发式 vs KataGo(12s 等时,96 局)

2026-10-11~12,RTX 5070 Laptop;KataGo v1.12.3(CUDA 编译)+ 权重
`antifive15-20260821-083223.bin`;高级启发式 = VCF + 开局库 + 蒸馏价值校正 +
调优参数,深度 8;双方每步 12 秒墙钟,48 个配对开局黑白互换,
数据在 `records/stage2_12s/`(不入库):

| 指标 | 结果 |
|---|---|
| 局分 | **启发式 0 : 96 KataGo**(KataGo 执黑 48:0、执白 48:0) |
| 比分 95% CI | 0%(0% ~ 3.9%);近似 Elo 差 ≤ −559 |
| 终局形态 | 96/96 由 KataGo 的"占领+安置"移子送连五给启发式落败;无认输/和棋 |
| 局长 | 中位 40 步(21~189) |
| 步时 | 启发式仅 21% 的步吃满 12s(迭代加深提前搜完);KataGo 全部 12s |
| 访问量 | KataGo 约 600~2400 visits/s(2 局并发,GPU 共享) |

**结论**:当前自训练权重的最强 KataGo 在 12s 等时下对"最强启发式"取得
96:0,强度差距大于约 560 Elo;该差距主要来自战术强制力(移子送礼连五),
而非启发式自损着(全程 0 次自我致命着)。

## 单进程提速:B(去重复计算)+ C(Cython 热路径)

-- 目标:在不改变任何搜索决策的前提下,让同一个进程每秒能搜更多节点。

**C:Cython 热路径**(`src/antifive/_fast.pyx`,构建 `python setup_fast.py build_ext --inplace`)

移植的 kernel(全部整数运算,与 numpy/Python 版逐位等价):

| kernel | 原实现 | 说明 |
| --- | --- | --- |
| `shift` | `heuristic_v6._shift` | 225 格平移,替换花式索引 + `out *= mask` |
| `line_maps` | `_line_maps_combo` | 8 方向游程 → `lt/lt2`(两轴最大/次大连子)+ 8 邻域计数,冷缓存 289 µs → 27 µs |
| `board_features` | `_board_features` | 双色线级模式扫描(四/跳四/活三/受困格),104 µs → 7.3 µs |
| `neighbor_counts` | `_positional_summary` 的 nb/adj 部分 | 8 邻域同色计数与同色相邻对数 |

三个引擎(`heuristic_v6` 高级、`heuristic.py` 大师、`heuristic_v7.py` 传奇)在各自
`_shift` / `_line_maps_combo` / `_board_features` / `_positional_summary` 的缓存 miss 分支里
优先调用扩展,扩展缺失时**自动回退**到原实现(行为逐位不变)。

**B:减少重复计算**

- 三个 4096 条 LRU(`_FEAT_CACHE`/`_LINE_MAP_CACHE`/`_POS_CACHE`)实测命中率只有 0.58,
  但把上限加到 16384 只换来 +12%,C 落地后这 12% 也消失了(特征重算已降到 7 µs)。
  结论:缓存不是瓶颈,保持 4096。
- 真正重复的是「同一局面在候选生成/威胁分析/评估里各算一次」,现在这三次调用都命中同一个
  kernel 与缓存,单次成本降到 2 µs 级。

**实测(同局面、同 seed、15 s 预算、串行 `workers=1`)**

| 引擎 | 改前节点/步 | 改后节点/步 | 倍数 |
| --- | ---: | ---: | ---: |
| 高级 heuristics_v6 | 48,817 | 89,697 | **1.84×** |
| 大师 heuristic | 50,357 | 76,488 | **1.52×** |
| 传奇 heuristic_v7 | 43,858 | 74,449 | **1.70×** |

固定深度回归(证明搜索路径不变):高级 depth 7 32.05 s → 16.17 s(节点数 99,196 与着法
完全一致);大师 depth 5 1.85×、传奇 depth 5 2.16×,选点完全相同。

**等价性验证**:300 张随机棋盘(`line_maps` 六张图逐元素相等)、200 张(`board_features`
返回的 dict/fours/dang 完全相等)、60 张 × 两方(`evaluate` 返回值逐位相等)、固定深度选点一致;
`python -m pytest tests` 121 passed / 1 skipped。

### 追加:合法掩码 / 连五图 kernel(2026-10-25)

`legal_mask_for` 与 `_line5_map_for` 原本要走 `fwd_run_lengths_padded`(8 方向 cumsum +
LRU)+ numpy 掩码拼装,是单节点里除特征外的第二大开销。现在由 `_fast.legal_mask` 与
`_fast.line5_mask` 直接算:

| 函数 | 原实现 | 现实现 |
| --- | --- | --- |
| `legal_mask_for` | runs cache + numpy 布尔拼装 | 逐格射线扫描(安置步)/ 8 邻域探测(占领)/ 连五剔除 |
| `_line5_map_for` | runs cache + 4 轴切片求和 | 直接双向走线计数 |

冷缓存微基准(380 个实战局面,平均):`legal_mask_for` 10.9 µs → 9.0 µs(1.22×),
`_line5_map_for` 5.9 µs → 4.1 µs(1.44×);但两者都**不再触碰 runs LRU**,搜索里省下的
是缓存管理与 numpy 小数组分配,15 s 实测节点数提升远大于单次微基准。

**15 s 预算、串行、同局面(叠加前面的 C 之后)**:高级 89,697 → **135,990**(1.52×)、
大师 76,488 → **109,481**(1.43×)、传奇 74,449 → **117,011**(1.57×)。

等价性:400 组随机状态(含安置步/白棋限制/自杀窗口)`legal_mask` 与 Python 版逐位相等;
固定深度下高级 depth 7、大师 depth 5、传奇 depth 5 的选点与 `evaluate` 返回值完全一致;
`pytest` 121 passed。
