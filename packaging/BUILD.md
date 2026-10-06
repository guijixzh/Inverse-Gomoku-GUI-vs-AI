# 打包为 Windows 可执行文件(无 torch 版)

本项目源码树即唯一来源,无 torch 分发版通过 PyInstaller **排除 torch** 构建,
无需维护第二份代码。

## 步骤

```powershell
# 1. 准备环境(使用只含 numpy + pygame 的解释器最干净)
conda create -n antifive-pack python=3.13 -y
conda activate antifive-pack
pip install numpy pygame pyinstaller

# 2. 安装本项目(可编辑安装,供 PyInstaller 解析包)
pip install -e .

# 3. 构建(仓库根目录执行)
pyinstaller packaging/Antifive.spec --noconfirm
```

产物为 `dist/Antifive/`(含 `Antifive.exe`),`torch` 已在 spec 中排除,
神经网络引擎会自动禁用,其余功能(随机先验 MCTS、启发式、KataGo)完整可用。

## 分发

将以下内容放到 `Antifive.exe` 同级目录后压缩分发:

```
Antifive/
├── Antifive.exe
├── config/                  # 构建时已自动包含(engines.json、gtp_test.cfg)
├── models/                  # 自研 .pth(可选)与 katago/*.bin
└── katago/                  # 自行放置 katago.exe 与依赖 DLL
```

- `models/` 与 `katago/` 不随 EXE 构建产物自动打包,请按需复制;
- KataGo 可执行文件与 CUDA/cuDNN 不随仓库分发,获取方式见 `docs/engines.md`;
- 资源定位逻辑见 `src/antifive/paths.py`(EXE 同级目录优先,兼容
  PyInstaller `_internal` 布局,可用 `ANTIFIVE_HOME` 覆盖)。
