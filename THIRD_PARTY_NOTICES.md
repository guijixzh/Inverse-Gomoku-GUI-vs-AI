# 第三方组件与致谢

本仓库以 MIT 许可证发布(`LICENSE`)。以下第三方组件/项目被引用、借鉴或作为
可选依赖使用;本仓库**未重新分发**其中的可执行文件、二进制依赖或 GPL 代码。

## KataGo

- 项目:[lightvector/KataGo](https://github.com/lightvector/KataGo)
- 许可证:MIT(仓库内部分 `cpp/external/` 第三方库与 `cpp/core/sha2.cpp`
  保留各自许可证,详见其 LICENSE)
- 本仓库关系:KataGo 是本项目 KataGo 引擎与逆五权重训练的基础。仓库**不包含**
  `katago.exe` 及 CUDA/cuDNN 运行库;`config/gtp_test.cfg` 为面向逆五规则的
  最小 GTP 配置(参考 KataGo/KataGomo 配置格式改写)。

## KataGomo

- 项目:[hzyhhzy/KataGomo](https://github.com/hzyhhzy/KataGomo)
- 许可证:继承 KataGo 的 MIT 许可(仓库标注为 Other/NOASSERTION,主体源自
  KataGo)
- 本仓库关系:逆五子棋(连五者输,white-restrict)规则支持与自训练 KataGo
  权重基于该分支;仓库**不包含**其源码或构建产物。`models/katago/*.bin` 为本
  项目使用 KataGomo 自行训练得到的权重。

## Ivy-Gomoku

- 项目:[lxsgx23/Ivy-Gomoku](https://github.com/lxsgx23/Ivy-Gomoku)
- 许可证:GPL-3.0
- 本仓库关系:本项目的 `config/engines.json` 引擎配置格式受其参考;
  **其 GPL-3.0 代码未包含在本仓库中**,故本项目可整体以 MIT 发布。
  若你需要保留基于 Ivy-Gomoku 的 Tk GUI 特化版,请按其 GPL-3.0 条款单独分发。

## 运行依赖(通过 pip 安装,不随仓库分发)

| 组件 | 许可证 | 用途 |
|---|---|---|
| [NumPy](https://numpy.org/) | BSD-3-Clause | 规则引擎、搜索、数据处理 |
| [pygame](https://www.pygame.org/) | LGPL-2.1 | 图形界面 |
| [PyTorch](https://pytorch.org/) | BSD-3-Clause | 自研神经网络与训练(可选) |
| [pytest](https://pytest.org/) | MIT | 测试(开发依赖,可选) |

## NVIDIA CUDA / cuDNN

KataGo GPU 版运行所需的 CUDA/cuDNN 运行库受 NVIDIA 许可约束,**不随本仓库
分发**;请通过 NVIDIA 官方渠道获取并遵守其许可条款。
