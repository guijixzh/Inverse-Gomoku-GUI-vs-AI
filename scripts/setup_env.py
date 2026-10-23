"""逆五子棋运行环境自检与依赖安装助手(纯标准库,裸 Python 即可运行)。

用法:
    python scripts/setup_env.py                    # 检查依赖、显卡与资源
    python scripts/setup_env.py --install          # 自动安装缺失依赖(按显卡选 torch)
    python scripts/setup_env.py --install --cpu    # 强制 CPU 版 torch
    python scripts/setup_env.py --torch-index URL  # 自定义 torch 下载源
    python scripts/setup_env.py --katago opencl    # 覆写 KataGo 版本建议

检查内容:
    1. Python 版本(>= 3.10);
    2. numpy / pygame(必需)、torch(可选,神经网络引擎)、tkinter(可选,文件对话框);
    3. 显卡检测与引擎方案(可选):按 NVIDIA(有 CUDA 驱动)/ AMD / Intel 核显 /
       无可用显卡 / 其他厂商,分别给出 torch 与 KataGo 版本建议;
    4. 仓库资源文件(规则引擎、权重、配置)是否齐全;
    5. KataGo 引擎(可选):按 config/engines.local.json > engines.json 解析
       可执行文件 / 权重 / GTP 配置,并尽量识别 exe 的后端(CUDA/OpenCL/Eigen)
       是否与当前显卡匹配。

注意:本仓库不包含 KataGo 训练程序与训练依赖;自带 .bin 权重为外部训练好的
成品,可直接用于对局。KataGo 可执行文件需按显卡自行从 KataGomo 构建/获取,
详见 docs/engines.md。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIN_PY = (3, 10)
TORCH_CUDA_TPL = "https://download.pytorch.org/whl/{}"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
KATAGO_LABEL = {"cuda": "CUDA 版", "opencl": "OpenCL 版", "eigen": "Eigen 纯 CPU 版"}

# 虚拟/远程显示适配器不算显卡
VIRTUAL_GPU_KEYS = ("microsoft basic display", "microsoft remote", "virtual",
                    "vmware", "hyper-v", "citrix", "parsec", "todesk", "oray",
                    "asklink", "displaylink", "spacedesk", "sunlogin",
                    "anydesk", "teamviewer", "rustdesk")

REQUIRED_FILES = (
    "pyproject.toml",
    "src/antifive/__init__.py",
    "config/engines.json",
    "config/gtp_test.cfg",
    "src/antifive/params.json",
)
OPTIONAL_FILES = (
    "data/eval_pool.npz",
    "models/warm_model.pth",
    "models/best_model.pth",
)


@dataclass
class Gpu:
    name: str
    vendor: str = "other"          # nvidia / amd / intel / other
    cuda: str = ""                 # 驱动支持的 CUDA 版本(NVIDIA)


@dataclass
class Plan:
    torch_index: str
    torch_label: str
    katago: str                    # cuda / opencl / eigen
    notes: list = field(default_factory=list)


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


def _run_text(cmd: list, timeout: int = 25) -> str:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                             encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return ""
    return res.stdout or ""


def _windows_gpu_names() -> list:
    """Windows 下列出显示适配器名称(PowerShell CIM 优先,wmic 兜底)。"""
    cmds = (
        ["powershell", "-NoProfile", "-NonInteractive", "-Command",
         "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }"],
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command",
         "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }"],
        ["wmic", "path", "win32_VideoController", "get", "name"],
    )
    for cmd in cmds:
        out = _run_text(cmd)
        if not out.strip():
            continue
        names = [ln.strip() for ln in out.splitlines() if ln.strip()]
        names = [n for n in names if n.lower() != "name"]
        names = [n for n in names
                 if not any(k in n.lower() for k in VIRTUAL_GPU_KEYS)]
        if names:
            return names
    return []


def _nvidia_smi() -> tuple:
    """返回 (显卡名, 驱动 CUDA 版本);未装 nvidia-smi 时返回空。"""
    if shutil.which("nvidia-smi") is None:
        return "", ""
    out = _run_text(["nvidia-smi"])
    names = _run_text(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    name = names.splitlines()[0].strip() if names.strip() else ""
    m = re.search(r"CUDA Version:\s*([\d.]+)", out)
    return name, (m.group(1) if m else "")


def classify_vendor(name: str) -> str:
    low = name.lower()
    if any(k in low for k in ("nvidia", "geforce", "rtx", "gtx", "quadro", "tesla")):
        return "nvidia"
    if any(k in low for k in ("amd", "radeon", "ati ", "firepro")):
        return "amd"
    if "intel" in low:
        return "intel"
    return "other"


def detect_gpus() -> list:
    """检测显示适配器并归类厂商。"""
    cuda_name, cuda_ver = _nvidia_smi()
    names = _windows_gpu_names() if sys.platform == "win32" else []
    seen = {n.lower() for n in names}
    if cuda_name and cuda_name.lower() not in seen:
        names.append(cuda_name)
    gpus = []
    for name in names:
        vendor = classify_vendor(name)
        gpus.append(Gpu(name, vendor, cuda_ver if vendor == "nvidia" else ""))
    return gpus


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


def has_opencl_runtime() -> bool:
    """粗查系统是否有 OpenCL 运行库(Windows 由显卡驱动安装 OpenCL.dll)。"""
    if sys.platform == "win32":
        root = Path(os.environ.get("SystemRoot") or r"C:\Windows")
        if (root / "System32" / "OpenCL.dll").exists():
            return True
    for p in ("/usr/lib/x86_64-linux-gnu/libOpenCL.so.1",
              "/usr/lib64/libOpenCL.so.1", "/usr/lib/libOpenCL.so.1"):
        if Path(p).exists():
            return True
    return shutil.which("clinfo") is not None


def build_plan(gpus: list, override: str = "") -> Plan:
    """按硬件给出 torch 与 KataGo 版本方案;未知情况走通用兜底。"""
    nv = next((g for g in gpus if g.vendor == "nvidia"), None)
    amd = next((g for g in gpus if g.vendor == "amd"), None)
    intel = next((g for g in gpus if g.vendor == "intel"), None)
    other = next((g for g in gpus if g.vendor == "other"), None)
    notes = []

    if nv:
        if nv.cuda:
            index = TORCH_CUDA_TPL.format(cuda_index_tag(nv.cuda))
            label = f"CUDA {nv.cuda} 版"
        else:
            index, label = TORCH_CPU_INDEX, "CPU 版(未读到驱动 CUDA 版本)"
        katago = "cuda"
        notes.append("NVIDIA 显卡:KataGo 用 CUDA 版,另需 CUDA/cuDNN 运行库 DLL")
        if amd or intel:
            notes.append("同时有集成/其他显卡,仍按 NVIDIA 独立显卡配置")
    elif amd:
        index, label, katago = TORCH_CPU_INDEX, "CPU 版", "opencl"
        notes.append("AMD 显卡:KataGo 用 OpenCL 版(官方支持,驱动自带运行库,无需 CUDA)")
        notes.append("若是 APU/核显与独显并存,启动日志里用 openclDeviceToUse 选设备")
    elif intel:
        index, label, katago = TORCH_CPU_INDEX, "CPU 版", "opencl"
        notes.append("Intel 核显:KataGo 用 OpenCL 版(能跑,速度约为独显的数分之一)")
        notes.append("也可以再放一份 Eigen 纯 CPU 版兜底")
    elif other:
        index, label, katago = TORCH_CPU_INDEX, "CPU 版", "opencl"
        notes.append("未识别的显卡厂商:先试 OpenCL 版;无法启动再换 Eigen 版(通用兜底)")
    else:
        index, label, katago = TORCH_CPU_INDEX, "CPU 版", "eigen"
        notes.append("未检测到可用显卡:用 Eigen 纯 CPU 版(任何 x64 机器可跑)")

    if katago == "opencl" and not has_opencl_runtime():
        notes.append("未找到系统 OpenCL 运行库(OpenCL.dll):OpenCL 版可能起不来,"
                     "建议先用 Eigen 纯 CPU 版兜底")
    if override in KATAGO_LABEL:
        katago = override
        notes.append(f"已按命令行覆写:KataGo 使用 {KATAGO_LABEL[override]}")

    return Plan(index, label, katago, notes)


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


def install_packages(status: dict, plan: Plan, force_cpu: bool,
                     torch_index: str) -> None:
    missing = [n for n in ("numpy", "pygame") if not status.get(n)]
    if missing:
        print(f"\n安装: {' '.join(missing)}")
        _pip(["install", *missing])
    if not status.get("torch"):
        if force_cpu:
            index = TORCH_CPU_INDEX
            print("\n按 --cpu 安装 CPU 版 torch")
        elif torch_index:
            index = torch_index
        else:
            index = plan.torch_index
            print(f"\n按显卡方案安装 torch({plan.torch_label})")
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


def _scan_file(path: Path, needles: tuple) -> set:
    """在文件(分块)中查找字节串;用于识别 exe 链接了哪个后端。"""
    found = set()
    overlap = max(len(n) for n in needles) - 1
    tail = b""
    try:
        with path.open("rb") as f:
            while True:
                chunk = f.read(1 << 22)
                if not chunk:
                    break
                buf = tail + chunk
                found |= {n for n in needles if n in buf}
                if len(found) == len(needles):
                    break
                tail = buf[-overlap:] if overlap > 0 else b""
    except OSError:
        return set()
    return found


def detect_katago_backend(exe: Path) -> str:
    """粗识别 katago.exe 的后端:opencl / cuda;识别不出返回空串。"""
    hits = _scan_file(exe, (b"OpenCL.dll", b"cudnn", b"cublas", b"cudart"))
    if b"OpenCL.dll" in hits:
        return "opencl"
    if hits:
        return "cuda"
    return ""


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


def check_katago(plan: Plan) -> None:
    entry = _load_kata_entry()
    if not entry:
        _line("可选未配置", "config/engines.json 无 KataGo 条目")
        return
    engine = _resolve(entry.get("engine_path", ""))
    weight = _resolve(entry.get("weight_path", ""))
    config = _resolve(entry.get("config_path", ""))
    for label, key, p in (("可执行", "engine_path", engine),
                          ("权重", "weight_path", weight),
                          ("GTP 配置", "config_path", config)):
        if not entry.get(key):
            _line("可选未配置", f"KataGo {label}")
        else:
            _line("OK" if p.exists() else "可选未找到", f"KataGo {label}: {p}")
    if engine.exists():
        dlls = list(engine.parent.glob("*.dll"))
        _line("OK" if dlls else "提示",
              f"exe 同级 DLL {len(dlls)} 个"
              + ("" if dlls else "(CUDA 版需 CUDA/cuDNN DLL;OpenCL 版只需 zlib1.dll)"))
        backend = detect_katago_backend(engine)
        if backend:
            _line("OK" if backend == plan.katago else "警告",
                  f"exe 后端:{KATAGO_LABEL[backend]}")
            if backend != plan.katago:
                print(f"        本机方案是 {KATAGO_LABEL[plan.katago]};"
                      f"启动报错(如 [WinError 126])时请换对应版本,"
                      f"构建命令见 docs/engines.md")
        else:
            _line("OK", "exe 未检出 CUDA/OpenCL 依赖,可能是 Eigen 纯 CPU 版")
    else:
        print(f"        获取方式:按显卡构建 {KATAGO_LABEL[plan.katago]},"
              f"见 docs/engines.md")


def main() -> int:
    ap = argparse.ArgumentParser(description="逆五子棋环境自检/依赖安装")
    ap.add_argument("--install", action="store_true", help="自动安装缺失的 Python 依赖")
    ap.add_argument("--cpu", action="store_true", help="强制安装 CPU 版 torch")
    ap.add_argument("--torch-index", default="", help="自定义 torch 下载源 index-url")
    ap.add_argument("--katago", default="auto",
                    choices=("auto", "cuda", "opencl", "eigen"),
                    help="覆写 KataGo 版本建议(默认按显卡自动选择)")
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

    print("\n[3] 显卡与引擎方案")
    gpus = detect_gpus()
    if gpus:
        for g in gpus:
            extra = f" · 驱动 CUDA {g.cuda}" if g.cuda else ""
            _line("OK", f"{g.name}{extra}")
    else:
        _line("提示", "未检测到可用显卡(虚拟机/无显示设备)")
    plan = build_plan(gpus, "" if args.katago == "auto" else args.katago)
    for note in plan.notes:
        _line("方案", note)
    _line("推荐", f"torch:{plan.torch_label};KataGo:{KATAGO_LABEL[plan.katago]}")
    if status.get("torch") and plan.katago == "cuda" and plan.torch_index != TORCH_CPU_INDEX:
        try:
            import torch  # noqa: PLC0415
            if "+cu" not in torch.__version__:
                _line("提示", "当前 torch 是 CPU 版;需要 CUDA 版请先 "
                              "pip uninstall torch 再运行 --install")
        except Exception:
            pass

    if args.install:
        print("\n[4] 安装缺失依赖")
        install_packages(status, plan, args.cpu, args.torch_index)
        print("\n安装后复检:")
        status = check_packages()
    else:
        print("\n[4] 安装(未执行;加 --install 自动安装缺失依赖)")

    print("\n[5] 仓库资源")
    res_ok = check_resources()
    print("\n[6] KataGo 引擎(可选)")
    check_katago(plan)

    pkg_ok = bool(status.get("numpy")) and bool(status.get("pygame"))
    print("\n" + "=" * 56)
    if py_ok and pkg_ok and res_ok:
        print("结论:必需项齐全,可启动 GUI:  双击 run_gui.bat 或 python -m antifive")
        if not status.get("torch"):
            print("提示:未装 torch,神经网络引擎将自动禁用(启发式/KataGo 不受影响)")
        print(f"KataGo 建议:{KATAGO_LABEL[plan.katago]}"
              f"(构建/获取见 docs/engines.md)")
        return 0
    print("结论:存在缺失项,请按上面提示处理(可加 --install 自动安装依赖)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
