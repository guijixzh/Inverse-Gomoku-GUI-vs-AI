"""逆五子棋运行环境自检与依赖安装助手(纯标准库,裸 Python 即可运行)。

用法:
    python scripts/setup_env.py                    # 检查依赖与资源是否齐全
    python scripts/setup_env.py --install          # 自动安装缺失的 Python 依赖
    python scripts/setup_env.py --install --cpu    # 强制 CPU 版 torch(无 GPU 时自动)
    python scripts/setup_env.py --torch-index URL  # 自定义 torch 下载源

检查内容:
    1. Python 版本(>= 3.10);
    2. numpy / pygame(必需)、torch(可选,神经网络引擎)、tkinter(可选,文件对话框);
    3. NVIDIA GPU 检测(可选):有 GPU 时 --install 自动装 CUDA 版 torch,否则 CPU 版;
    4. 仓库资源文件(规则引擎、权重、配置)是否齐全;
    5. KataGo 引擎(可选):按 config/engines.local.json > engines.json 解析
       可执行文件 / 权重 / GTP 配置,并统计 exe 同级 DLL 数量。

注意:本仓库不包含 KataGo 训练程序与训练依赖;自带 .bin 权重为外部训练好的
成品,可直接用于对局。KataGo 可执行文件需自行从 KataGomo 构建/获取,
详见 docs/engines.md。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIN_PY = (3, 10)
TORCH_CUDA_TPL = "https://download.pytorch.org/whl/{}"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

REQUIRED_FILES = (
    "pyproject.toml",
    "src/antifive/__init__.py",
    "config/engines.json",
    "config/gtp_test.cfg",
    "data/params.json",
)
OPTIONAL_FILES = (
    "data/eval_pool.npz",
    "models/warm_model.pth",
    "models/best_model.pth",
)


def _line(status: str, msg: str) -> None:
    print(f"  [{status}] {msg}")


def check_python() -> bool:
    ok = sys.version_info >= MIN_PY
    ver = ".".join(map(str, sys.version_info[:3]))
    _line("OK" if ok else "错误", f"Python {ver}(需要 >= {MIN_PY[0]}.{MIN_PY[1]})")
    return ok


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def detect_gpu():
    """检测 NVIDIA GPU:返回 (名称, 驱动 CUDA 版本) 或 None。"""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                             timeout=15).stdout
        names = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    name = names.splitlines()[0].strip() if names else "NVIDIA GPU"
    ver = ""
    m = re.search(r"CUDA Version:\s*([\d.]+)", out)
    if m:
        ver = m.group(1)
    return (name, ver)


def cuda_index_tag(cuda_ver: str) -> str:
    """按驱动 CUDA 版本选择 PyTorch CUDA 轮子标签。"""
    try:
        v = tuple(int(x) for x in cuda_ver.split(".")[:2])
    except ValueError:
        return "cu126"
    if v >= (13, 0):
        return "cu130"
    if v >= (12, 8):
        return "cu128"
    if v >= (12, 4):
        return "cu126"
    if v >= (12, 1):
        return "cu121"
    return "cu118"


def check_packages() -> dict:
    status = {}
    for name, required in (("numpy", True), ("pygame", True),
                           ("torch", False), ("tkinter", False)):
        ok = has_module(name)
        status[name] = ok
        tag = "OK" if ok else ("缺失" if required else "可选未装")
        _line(tag, f"{name}{'' if required else '(可选)'}")
    if status.get("torch"):
        try:
            import torch  # noqa: PLC0415
            if torch.cuda.is_available():
                _line("OK", f"torch {torch.__version__} · CUDA 可用 · "
                            f"{torch.cuda.get_device_name(0)}")
            else:
                _line("OK", f"torch {torch.__version__} · 仅 CPU")
        except Exception as e:                        # pragma: no cover
            _line("警告", f"torch 导入失败: {e}")
    return status


def install_packages(status: dict, force_cpu: bool, torch_index: str) -> None:
    missing = [n for n in ("numpy", "pygame") if not status.get(n)]
    if missing:
        print(f"\n安装: {' '.join(missing)}")
        _pip(["install", *missing])
    if not status.get("torch"):
        gpu = None if force_cpu else detect_gpu()
        if torch_index:
            index = torch_index
        elif gpu:
            index = TORCH_CUDA_TPL.format(cuda_index_tag(gpu[1]))
            print(f"\n检测到 GPU:{gpu[0]}(驱动 CUDA {gpu[1] or '未知'}),"
                  f"安装 CUDA 版 torch({index})")
        else:
            index = TORCH_CPU_INDEX
            print("\n未检测到 NVIDIA GPU,安装 CPU 版 torch")
        if _pip(["install", "torch", "--index-url", index]) != 0 \
                and index != TORCH_CPU_INDEX:
            print("CUDA 版安装失败,回退 CPU 版 …")
            _pip(["install", "torch", "--index-url", TORCH_CPU_INDEX])


def _pip(args: list) -> int:
    cmd = [sys.executable, "-m", "pip", *args]
    print("  $ " + " ".join(cmd))
    try:
        return subprocess.run(cmd).returncode
    except OSError as e:
        print(f"  pip 调用失败: {e}")
        return 1


def _load_kata_entry() -> dict:
    for name in ("engines.local.json", "engines.json"):
        p = ROOT / "config" / name
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for e in data if isinstance(data, list) else []:
            if isinstance(e, dict) and e.get("kind") != "heuristic":
                return e
    return {}


def _resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    for root in (ROOT, Path.cwd()):
        if (root / p).exists():
            return root / p
    return ROOT / p


def check_resources() -> bool:
    all_ok = True
    for rel in REQUIRED_FILES:
        ok = (ROOT / rel).exists()
        all_ok &= ok
        _line("OK" if ok else "缺失", rel)
    for rel in OPTIONAL_FILES:
        _line("OK" if (ROOT / rel).exists() else "可选未找到", rel)
    bins = sorted((ROOT / "models" / "katago").glob("*.bin")) \
        if (ROOT / "models" / "katago").is_dir() else []
    if bins:
        newest = max(bins, key=lambda f: f.stat().st_mtime)
        _line("OK", f"KataGo 权重 {len(bins)} 个(最新: {newest.name})")
    else:
        _line("可选未找到", "models/katago/*.bin")
    return all_ok


def check_katago() -> None:
    entry = _load_kata_entry()
    if not entry:
        _line("可选未配置", "config/engines.json 无 KataGo 条目")
        return
    engine = _resolve(entry.get("engine_path", ""))
    weight = _resolve(entry.get("weight_path", ""))
    config = _resolve(entry.get("config_path", ""))
    for label, p in (("可执行", engine), ("权重", weight), ("GTP 配置", config)):
        if not entry.get({"可执行": "engine_path", "权重": "weight_path",
                          "GTP 配置": "config_path"}[label]):
            _line("可选未配置", f"KataGo {label}")
        else:
            _line("OK" if p.exists() else "可选未找到", f"KataGo {label}: {p}")
    if engine.exists():
        dlls = list(engine.parent.glob("*.dll"))
        _line("OK" if dlls else "提示",
              f"exe 同级 DLL {len(dlls)} 个"
              + ("" if dlls else"(GPU 版还需 CUDA/cuDNN DLL,不随仓库分发)"))
    else:
        print("        获取方式:见 docs/engines.md(KataGo/KataGomo,自行构建,"
              "本仓库不含其训练程序与依赖)")


def main() -> int:
    ap = argparse.ArgumentParser(description="逆五子棋环境自检/依赖安装")
    ap.add_argument("--install", action="store_true", help="自动安装缺失的 Python 依赖")
    ap.add_argument("--cpu", action="store_true", help="强制安装 CPU 版 torch")
    ap.add_argument("--torch-index", default="", help="自定义 torch 下载源 index-url")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print(f"仓库目录: {ROOT}\n")
    print("[1] Python 版本")
    py_ok = check_python()
    print("\n[2] Python 依赖")
    status = check_packages()
    print("\n[3] GPU(可选)")
    gpu = detect_gpu()
    if gpu:
        _line("OK", f"{gpu[0]} · 驱动 CUDA {gpu[1] or '未知'}")
    else:
        _line("可选无 GPU", "将使用 CPU(兼容运行,速度较慢)")

    if args.install:
        print("\n[4] 安装缺失依赖")
        install_packages(status, args.cpu, args.torch_index)
        print("\n安装后复检:")
        status = check_packages()
    else:
        print("\n[4] 安装(未执行;加 --install 自动安装缺失依赖)")

    print("\n[5] 仓库资源")
    res_ok = check_resources()
    print("\n[6] KataGo 引擎(可选)")
    check_katago()

    pkg_ok = bool(status.get("numpy")) and bool(status.get("pygame"))
    print("\n" + "=" * 56)
    if py_ok and pkg_ok and res_ok:
        print("结论:必需项齐全,可启动 GUI:  双击 run_gui.bat 或 python -m antifive")
        if not status.get("torch"):
            print("提示:未装 torch,神经网络引擎将自动禁用(启发式/KataGo 不受影响)")
        return 0
    print("结论:存在缺失项,请按上面提示处理(可加 --install 自动安装依赖)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
