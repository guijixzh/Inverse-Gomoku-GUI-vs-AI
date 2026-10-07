# 逆五子棋 Antifive —— AI 对战 GUI

[English](README_EN.md)

![逆五子棋 GUI 截图](docs/prevpic.png)

**English abstract.** Antifive is an open-source project for **Inverse Gomoku**
(Chinese: 逆五子棋; the player who first makes five in a row *loses*). It ships
a pygame GUI, a pure-numpy rules engine, a heuristic AI,
an AlphaZero-style policy/value network (training attempt **failed**, kept for
reproduction), and a KataGo integration with self-trained inverse-gomoku
weights. All code is MIT-licensed and heavily AI-generated.

> ## AI 使用声明
>
> **请注意**:本项目代码与文本主体由 AI 生成,经过人工初步核查但仍需要仔细甄别内容正误。
>
> > **AI Usage Statement:**
> >
> > **Please note** that the code and text content of this project are generated
> > by AI, preliminarily reviewed by humans, but still require careful
> > verification of accuracy.

---

## 目录

- [项目简介](#项目简介)
- [在线网页版](#在线网页版)
- [逆五子棋与发明人](#逆五子棋与发明人)
- [项目状态(重要,先读)](#项目状态重要先读)
- [安装](#安装)
- [依赖自检与一键配置](#依赖自检与一键配置推荐)
- [快速开始](#快速开始)
- [界面与快捷键](#界面与快捷键)
- [AI 引擎与权重](#ai-引擎与权重)
- [棋谱格式](#棋谱格式)
- [训练(负结果)](#训练负结果)
- [仓库结构](#仓库结构)
- [环境与已知边界](#环境与已知边界)
- [第三方与致谢](#第三方与致谢)
- [License](#license)
- [找到我](#找到我)

---

## 项目简介

**逆五子棋**在 15×15 棋盘上进行,黑先。每回合二选一:**落子**(空位放子)或
**移子**(占领对方一子,再沿直线把它安置到空位——占两个步但只算一个回合);
连成五子的一方判负。白棋前 2 回合禁止移子的平衡规则由
[hzyhhzy](https://github.com/hzyhhzy)(KataGo/KataGomo 的作者)提出。

> **名称说明**:本项目采用「逆五子棋」这一名称,以区别于**没有移子规则**的
> 反五子棋;两者胜负条件相同(连五者判负),但逆五子棋包含占领与安置的移子
> 机制。完整规则与动作编码见 [`docs/rules.md`](docs/rules.md)。

项目包含:

- **pygame 图形界面**:人机/人人/机机对战、复盘与试下分支、危险提示、移子箭头与
  辅助射线、读秒、合成音效、`.afg` 棋谱保存/读取;
- **规则引擎**:纯 numpy(训练用批量 GPU 镜像在 `training/`);
- **AI 引擎**:随机先验 MCTS、自研策略价值网络 MCTS、启发式 AI(alpha-beta)、
  KataGo(逆五自训练权重);
- **训练代码**:AlphaZero 风格自对弈训练与对启发式的课程训练(总体失败,
  负结果保留);
- **权重**:自研网络 `.pth`(失败,仅复现用)与 KataGo 逆五 `.bin`(实用)。

## 在线网页版

不安装任何环境,浏览器直接开玩:

**https://guijixzh.github.io/Inverse-Gomoku-GUI-vs-AI/**

- 实现方式:Pyodide(WASM)在 Web Worker 中直接运行本仓库的 numpy 规则引擎与启发式 AI,
  前端为 TypeScript + Canvas 静态站,托管在 GitHub Pages;
- 首次打开需下载运行环境(约 10-20MB),之后访问走 Service Worker 缓存;
  三档难度对应启发式深度 2 / 4 / 8(带时间预算,超时取上一层完整搜索结果);
- 功能对齐 pygame GUI:落子/占领/安置、危险提示、手数、移子箭头、辅助射线、音效、
  读秒、复盘手数轴与试下分支、`.afg` 棋谱导入导出、人机/人人/机机模式与全部快捷键;
- 浏览器内无法运行 KataGo 与 torch 权重,网页版仅提供启发式 AI;需要 KataGo 请用桌面版。

本地开发(需要 Node.js 20+ 与 Python 3.10+,自动构建纯 Python wheel 并准备 Pyodide 资源):

```bash
cd web
npm install
npm run dev      # 本地开发服务器
npm run build    # 静态产物在 web/dist
npm run smoke    # 构建后浏览器冒烟测试(需本机 Chrome/Edge)
```

部署:push 到 `main` 后由 `.github/workflows/pages.yml` 自动构建发布到 GitHub Pages;
使用自定义域名或用户主页(`<user>.github.io`)时,把工作流里的 `VITE_BASE` 改为 `/`。

## 逆五子棋与发明人

逆五子棋玩法由 **日出333** 发明:

- 发明人 B站主页:[space.bilibili.com/484235404](https://space.bilibili.com/484235404)
- 规则介绍视频:[BV1n84y1G7jT](https://www.bilibili.com/video/BV1n84y1G7jT)
- GUI 展示视频:[BV1i1eB6REcr](https://www.bilibili.com/video/BV1i1eB6REcr)

本项目是该玩法的开源 AI 对战与训练实现。

## 项目状态(重要,先读)

- **实用引擎是启发式 AI 与 KataGo。** 自研策略价值网络(`network.py` +
  `training/`)**训练总体失败**:受自对弈数据规模与训练预算限制,最终强度不足,
  最多维持稍弱于启发式引擎的水平,不具备实用棋力。仓库保留其代码与权重
  (`models/warm_model.pth`、`models/best_model.pth`)仅为复现与负结果记录,
  详见 [`docs/training-log.md`](docs/training-log.md)。负结果也是结果。
- KataGo 权重(`models/katago/*.bin`)为本项目基于
  [KataGomo](https://github.com/hzyhhzy/KataGomo) 自训练的逆五规则权重;
  `katago.exe` 与 CUDA/cuDNN 运行库因许可与体积原因**不随仓库分发**,
  使用方式见 [`docs/engines.md`](docs/engines.md)。
- 开发环境:Windows / Python 3.13 / torch 2.9.1+cu130 / pygame 2.6.1 /
  numpy 2.2.6(支持 Python ≥ 3.10)。

## 安装

```bash
git clone https://github.com/guijixzh/Inverse-Gomoku-GUI-vs-AI.git
cd Inverse-Gomoku-GUI-vs-AI

conda create -n antifive python=3.13 -y
conda activate antifive

# GPU(推荐):按 CUDA 版本安装 torch,例如 CUDA 13.0
pip install torch --index-url https://download.pytorch.org/whl/cu130
# 或 CPU 版:pip install torch

pip install -e ".[gui,nn]"
```

不安装 torch 也可运行 GUI 与启发式 AI(神经网络引擎会自动禁用):

```bash
pip install -e ".[gui]"
```

## 依赖自检与一键配置(推荐)

仓库只包含源码、配置与成品权重;运行所需的第三方可执行文件(如 KataGo)**不随仓库分发**。
根目录提供自检/安装脚本(纯标准库,可在裸 Python 下运行):

```bash
python scripts/setup_env.py            # 检查依赖与资源是否齐全(不修改环境)
python scripts/setup_env.py --install  # 自动安装缺失依赖(按硬件选择 torch)
```

`--install` 会自动检测 NVIDIA GPU:有 GPU 装 CUDA 版 torch,没有 GPU
自动装 CPU 版(功能完整,速度较慢);还支持 `--cpu` 强制 CPU 版、
`--torch-index URL` 自定义下载源。脚本同时校验权重/配置文件是否齐全,
并提示 KataGo 可执行文件与 DLL 是否就位。

### 可选:启用 KataGo 引擎(最简流程)

1. 自行构建/获取支持逆五规则的 `katago.exe`:见 [docs/engines.md](docs/engines.md)
   (源码来自 KataGomo;GPU 版另需 NVIDIA CUDA/cuDNN 运行库,均**不随仓库分发**);
2. 把 `katago.exe` 与依赖 DLL 放到 `katago/` 目录(或任意位置);
3. 启动 GUI →「设定 → KataGo」→ 用「浏览」选择可执行文件/权重/GTP 配置 →
   「确定」(路径会保存到本机 `config/engines.local.json`,已被 gitignore);
4. 再次运行 `python scripts/setup_env.py` 复检。

> 说明:本仓库**不包含 KataGo 训练程序与训练依赖**;`models/katago/*.bin`
> 是外部训练好的成品权重,可直接用于对局。

## 快速开始

### 图形界面对局

```bash
python -m antifive                       # 等价于 antifive(GUI 入口)
python -m antifive --engine heuristic    # 直接使用启发式 AI(深度 4)
python -m antifive --model models/best_model.pth    # 自研网络作 MCTS 先验(研究)
python -m antifive --engine katago       # KataGo(需自行放置 katago.exe,见 docs/engines.md)
python -m antifive --limit ai --ai-time 15   # 仅 AI 限时读秒,每步 15 秒
```

Windows 下也可直接双击仓库根目录的 [`run_gui.bat`](run_gui.bat) 一键启动
(自动把 `src/` 加入 `PYTHONPATH`,支持传入与上面相同的参数)。若系统默认
Python 缺少依赖,可先运行 `python scripts/setup_env.py --install` 一键补齐;
也可在根目录新建 `run_gui.local.bat`(已被 gitignore),写入一行
`set PY="你的python.exe路径"` 指定解释器(例如 conda 环境)。

启动后点「设定」可随时更换引擎与参数;1–4 切换人执黑/人执白/人人/机机模式。

### 命令行对战

```bash
python -m antifive.cli --model models/best_model.pth --sims 400
```

### 测试

```bash
pip install -e ".[dev,nn]"
pytest -q
```

测试包含自杀掩码回归、numpy↔torch 引擎一致性、自对弈冒烟;规则一致性测试需要
C++ `rules_probe`(由 KataGomo 构建,设置 `ANTIFIVE_RULES_PROBE` 后启用,否则跳过)。

## 界面与快捷键

| 快捷键 | 功能 |
|---|---|
| R | 新局 |
| U | 悔棋 |
| D | 危险提示开关 |
| N | 棋子手数显示开关 |
| A | 移子箭头指示开关 |
| H | 辅助射线开关 |
| M | 音效开关 |
| S | 保存棋谱 |
| L | 读取棋谱 |
| E | 设定面板 |
| 1–4 | 人执黑 / 人执白 / 人人对战 / 机机观战 |
| ← → | 复盘时逐手回退 / 前进 |
| Esc | 退出 |

鼠标左键:空位落子;点对方棋子发起占领;安置阶段点绿色高亮格放下手中棋子。
终局或读取棋谱后自动进入复盘,可点手数轴跳转、试下分支、一键恢复原谱。

## AI 引擎与权重

| 引擎 | 依赖 | 强度 | 说明 |
|---|---|---|---|
| 随机先验 MCTS | numpy | 低 | 无额外文件的默认兜底 |
| 自研网络 MCTS | torch + `.pth` | 不足(训练失败) | 研究/复现用途 |
| 启发式 AI | numpy | 中高 | 深度 1–8,默认 4 |
| KataGo(逆五) | 外部 exe + `.bin` | 高 | 推荐实用引擎 |

权重明细、加载方式与 KataGo 配置见 [`models/README.md`](models/README.md)
与 [`docs/engines.md`](docs/engines.md)。

## 棋谱格式

`.afg` 为 UTF-8 纯文本棋谱,含规则配置、结果与着法序列(支持逐手分析数据);
示例见 `data/samples/`,格式定义见 [`docs/afg-format.md`](docs/afg-format.md)。

## 训练(负结果)

`training/` 提供完整的 AlphaZero 风格训练链路:

```bash
python -m antifive.heuristic --games 1000 --out records   # 生成棋谱
python -m antifive.training.pretrain_bc --records records \
    --out checkpoints/warm_model.pth                       # 行为克隆预热
python -m antifive.training.train --resume models/warm_model.pth   # 自对弈训练
```

**该训练链路最终失败**(强度不足,最多稍弱于启发式),流程与可能原因见
[`docs/training-log.md`](docs/training-log.md)。

## 仓库结构

```
Inverse-Gomoku-GUI-vs-AI/
├── src/antifive/
│   ├── reversegomoku.py        # 规则引擎(纯 numpy)
│   ├── training/reverse_gomoku_torch.py  # 批量 GPU 规则镜像
│   ├── mcts.py                 # AlphaZero 风格 MCTS(torch 可选)
│   ├── network.py              # 策略价值网络(训练失败)
│   ├── heuristic.py            # 启发式 AI(alpha-beta + 全局评估)
│   ├── tactics.py              # 杀棋检测
│   ├── record.py               # .afg 棋谱读写
│   ├── gui.py                  # pygame 图形界面
│   ├── cli.py                  # 命令行对战
│   ├── web_api.py              # 网页版 JSON 接口(Pyodide 复用引擎与启发式 AI)
│   ├── paths.py                # 资源路径定位
│   ├── training/               # 自对弈与训练
│   └── tools/                  # 对拍、诊断、参数调优
├── web/                        # 网页版(Vite + TypeScript + Canvas + Pyodide)
├── scripts/setup_env.py        # 环境自检与依赖自动安装(GPU/CPU 自适应)
├── config/                     # engines.json 与 gtp_test.cfg
├── data/                       # 启发式参数、调参池、示例棋谱
├── models/                     # 权重(说明见 models/README.md)
├── docs/                       # 规则、棋谱格式、引擎、训练记录
├── tests/                      # pytest 回归测试
├── packaging/                  # PyInstaller 打包(无 torch 版)
├── run_gui.bat                 # Windows 一键启动 GUI
├── pyproject.toml
└── THIRD_PARTY_NOTICES.md
```

## 环境与已知边界

- **平台**:在 Windows 11 上开发与测试;GUI 使用 Windows 中文字体回退,
  其他平台未验证;
- **Python**:≥ 3.10(开发环境 3.13);
- **torch 可选**:未安装时 GUI/启发式/KataGo 均可用,神经网络引擎自动禁用;
- **KataGo**:可执行文件与 CUDA/cuDNN 不随仓库分发,需自行构建/获取
  (见 `docs/engines.md`);
- **自研网络**:训练失败,仅供研究;
- **规则一致性测试**:依赖外部 `rules_probe.exe`,默认跳过。

## 第三方与致谢

- [lightvector/KataGo](https://github.com/lightvector/KataGo)(MIT):
  KataGo 引擎基础;
- [hzyhhzy/KataGomo](https://github.com/hzyhhzy/KataGomo)(继承 KataGo 许可):
  逆五规则支持与权重训练基础;
- [lxsgx23/Ivy-Gomoku](https://github.com/lxsgx23/Ivy-Gomoku)(GPL-3.0):
  引擎配置格式参考,**其代码未包含在本仓库中**;
- NVIDIA CUDA/cuDNN:不随仓库分发。

完整许可见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## License

[MIT](LICENSE)

## 找到我

- GitHub:[https://github.com/guijixzh](https://github.com/guijixzh)
- B站:[https://space.bilibili.com/74464185](https://space.bilibili.com/74464185)
- 邮箱:1722532014@qq.com(推荐)/ guijixzh@gmail.com
- QQ:1722532014(请注明来意)
