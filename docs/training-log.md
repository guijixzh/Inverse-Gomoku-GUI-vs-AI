# 自研神经网络训练记录(负结果)

> **结论先行:自研策略价值网络的训练总体失败。** 受自对弈数据规模与训练预算限制,
> 最终强度不足,最多维持稍弱于启发式引擎(`src/antifive/heuristic.py`)的水平,
> 不具备实用棋力。本仓库保留代码与权重(`warm_model.pth`、`best_model.pth`)
> 仅为复现与负结果记录。**负结果也是结果。**

## 设计

- 规则引擎:`src/antifive/reversegomoku.py`(纯 numpy)与批量 GPU 镜像
  `src/antifive/training/reverse_gomoku_torch.py`;
- 网络:`src/antifive/network.py`,AlphaZero 风格 ResNet(5 输入通道,
  默认 128 通道 × 4 残差块,策略/价值双头,1,544,338 参数);
- 搜索:`src/antifive/mcts.py`,按"步"建模的 PUCT 搜索,批量叶子评估;
- 自对弈:`src/antifive/training/selfplay.py`,样本内嵌杀棋监督
  (安置杀/两步杀 one-hot + value=+1);
- 训练主循环:`src/antifive/training/train.py`(自对弈 → 回放缓存 → 训练 →
  评估 → 存档)与 `src/antifive/training/train_vs_heur.py`(对启发式课程训练)。

## 复现流程

```bash
# 1. 用启发式 AI 生成棋谱(按结果/有杀棋分类存放)
python -m antifive.heuristic --games 1000 --out records

# 2. 行为克隆 + 杀棋监督预热
python -m antifive.training.pretrain_bc --records records \
    --out checkpoints/warm_model.pth

# 3. AlphaZero 自对弈训练(或对启发式的课程训练)
python -m antifive.training.train --resume models/warm_model.pth
python -m antifive.training.train_vs_heur --resume models/warm_model.pth
```

训练产物写入 `checkpoints/`(已加入 `.gitignore`),包括
`best_model.pth` / `latest_model.pth` / `warm_model.pth`。

## 失败表现

- 自对弈/训练循环本身可稳定运行(代码路径已验证),但模型棋力始终无法超过
  启发式基线;仓库保留的 `best_model.pth`(step=0)与 `warm_model.pth`(step=3)
  即为该阶段的快照;
- 项目最终以启发式 AI + KataGo(逆五权重)作为实用引擎,自研网络仅作研究记录。

## 可能的原因(未做系统消融,仅记录)

- 逆五规则下"移子"为两步动作、白方禁移与自杀判负等机制使策略空间与价值目标
  更复杂,自对弈样本效率低于普通五子棋;
- 训练预算(单机 GPU、有限自对弈局数)与网络规模相对任务复杂度不足;
- 未实现对 KataGo 式的规则特征输入与更长时间的自对弈课程。

## 经验与保留价值

- 规则引擎、批量 GPU 环境、MCTS、棋谱格式均可独立复用(启发式生成、对拍、
  工具链均依赖它们);
- 杀棋监督样本(tactics.py)与 `.afg` 棋谱格式可供后续工作直接使用。
