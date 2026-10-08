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

内置算法,无需额外文件;GUI 中可选深度 1–8(默认 8 + AI 读秒 12 秒),并可切换
引擎版本(新版强化 / 老版经典)与 VCF 强制杀链开关(默认新版 + VCF 开),命令行行为
`--engine heuristic --heuristic-depth N --limit none|ai --ai-time S
--heuristic-version new|old --no-vcf`。参数可被 `data/params.json` 复现,
调优工具见 `src/antifive/tools/param_tune.py`。

开局库:`antifive/openbook_data.py` 由 `tools/openbook_gen.py` 用 KataGo 生成
(默认 20k visits;黑空盘首手 + 每个 D4 等价类首手的白方应手),运行时按 D4
对称规范化查表,`ANTIFIVE_NO_BOOK=1` 可关闭。重新生成(需本地 katago):

```bash
python -m antifive.tools.openbook_gen --visits 20000
```

对拍(等时强弱回归;`heuristic_v4.py` 为 2026-10-08 搜索一致性修复前的冻结快照):

```bash
python -m antifive.tools.match_ai --games 40 --depth 8 --budget 1 --workers 8 \
    --old-module antifive.tools.heuristic_v4
```

## KataGo(逆五规则)

### 权重

- `models/katago/antifive15-init.bin`:初始权重(随机初始化,棋力无意义);
- `models/katago/antifive15-20260821-083223.bin`:自训练权重(最新)。

训练规则:逆五子棋(连五者输,`white_restrict=2`),基于
[hzyhhzy/KataGomo](https://github.com/hzyhhzy/KataGomo)(KataGo 分支,MIT)。

### 可执行文件(不随仓库分发)

`katago.exe` 与 CUDA/cuDNN 运行库因 NVIDIA 许可与体积原因不随仓库分发,
KataGo 的训练程序与依赖也不在开源范围内(本仓库只含运行配置与成品权重)。
运行环境自检可执行 `python scripts/setup_env.py`。
使用步骤:

1. 从 [hzyhhzy/KataGomo](https://github.com/hzyhhzy/KataGomo) 获取源码,
   编译支持逆五规则(white-restrict / `newgame_gomokuamazons` 相关改动)的
   `katago.exe`;
2. 将 `katago.exe` 及依赖 DLL 放到仓库根目录 `katago/`;
3. 依赖:GPU 版需要 CUDA/cuDNN(NVIDIA 官方渠道获取,不随仓库分发);
4. 在 `config/engines.json` 中确认路径。

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
- 对拍脚本:`python -m antifive.tools.katago_vs_heur --model models/katago/antifive15-init.bin
  --engine katago/katago.exe --engine-cfg config/gtp_test.cfg`。
