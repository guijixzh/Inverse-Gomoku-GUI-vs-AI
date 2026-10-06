# 权重说明

本目录包含两类权重:**自研策略价值网络**(`.pth`,训练总体失败,仅供复现研究)
与 **KataGo 逆五规则权重**(`katago/*.bin`,实用引擎)。

## 自研策略价值网络(训练总体失败)

> **重要提示**:以下自研网络训练总体失败,性能不足,最多维持稍弱于启发式引擎
> (`src/antifive/heuristic.py`)的强度,不具备实用棋力。保留权重与训练代码仅为
> 复现与负结果记录,正式对局请使用启发式 AI 或 KataGo。

| 文件 | 架构 | 参数量 | step | 说明 |
|---|---|---|---|---|
| `warm_model.pth` | 128ch × 4res | 1,544,338 | 3 | 行为克隆(BC)预热起点,失败尝试的一部分 |
| `best_model.pth` | 128ch × 4res | 1,544,338 | 0 | 自对弈训练中的最优快照,强度仍不足 |

加载方式:

```bash
python -m antifive.gui --model models/best_model.pth      # GUI 中作为 MCTS 先验
python -m antifive.cli --model models/best_model.pth      # 命令行对战
python -m antifive.training.train --resume models/warm_model.pth
```

网络结构见 `src/antifive/network.py`(AlphaZero 风格 ResNet 策略/价值双头,
输入 5 通道 15×15)。

## KataGo 逆五权重(实用)

| 文件 | 说明 |
|---|---|
| `katago/antifive15-init.bin` | 初始权重(随机初始化,棋力无意义,仅作训练起点) |
| `katago/antifive15-20260821-083223.bin` | 自训练权重(最新),配合逆五规则 KataGo 使用 |

KataGo 权重训练规则为逆五子棋(连五者输,white_restrict=2),基于
[hzyhhzy/KataGomo](https://github.com/hzyhhzy/KataGomo)(KataGo 分支,MIT)。

`katago.exe` 及 CUDA/cuDNN 运行库因许可与体积原因**不随仓库分发**。使用方式:

1. 自行编译/获取支持逆五(white-restrict)规则的 KataGo/KataGomo 可执行文件;
2. 将 `katago.exe` 及依赖 DLL 放入仓库根目录 `katago/`;
3. 引擎路径在 `config/engines.json` 中配置(默认 `katago/katago.exe`),
   GTP 配置为 `config/gtp_test.cfg`;
4. `auto_latest=true` 会自动选用 `models/katago/` 中最新的
   `antifive15-*.bin`(排除 init)。

详见 `docs/engines.md`。
