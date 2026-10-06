"""项目资源路径定位:兼容源码运行、可编辑安装与 PyInstaller 冻结。

config/、models/、data/ 等资源目录位于仓库根目录;冻结(EXE)时位于
EXE 所在目录或 PyInstaller 的 _internal 目录。可用环境变量 ANTIFIVE_HOME
覆盖资源根目录。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _roots() -> list:
    """候选资源根目录(按优先级)。"""
    roots = []
    env = os.environ.get("ANTIFIVE_HOME")
    if env:
        roots.append(Path(env).expanduser().resolve())
    if getattr(sys, "frozen", False):
        exe = Path(sys.executable).resolve().parent
        roots.extend([exe, exe / "_internal"])
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            roots.append(parent)
            break
    cwd = Path.cwd()
    if cwd not in roots:
        roots.append(cwd)
    return roots


def roots() -> list:
    """候选资源根目录(按优先级):EXE 同级 > _internal > 仓库根 > 当前目录。"""
    return _roots()


def app_base() -> Path:
    """资源根目录:优先取实际含有 config/ 或 models/ 的候选目录。"""
    roots = _roots()
    for root in roots:
        if (root / "config").is_dir() or (root / "models").is_dir():
            return root
    return roots[0]


def _sub(name: str) -> Path:
    for root in _roots():
        d = root / name
        if d.is_dir():
            return d
    return app_base() / name


def config_dir() -> Path:
    """config/ 目录(engines.json、gtp_test.cfg 等)。"""
    return _sub("config")


def models_dir() -> Path:
    """models/ 目录(自研网络 .pth 与 KataGo 权重)。"""
    return _sub("models")


def data_dir() -> Path:
    """data/ 目录(启发式参数、调参池、示例棋谱)。"""
    return _sub("data")


def resolve(path: str | os.PathLike) -> str:
    """把配置中的相对路径解析为绝对路径。

    依次在各候选根目录下探测:EXE 同级目录优先(打包分发时 models/、
    katago/ 放在 EXE 旁),其次 _internal、仓库根、当前目录;
    均不存在时回退到 app_base()。"""
    p = Path(path)
    if p.is_absolute():
        return str(p)
    for root in _roots():
        cand = root / p
        if cand.exists():
            return str(cand)
    return str(app_base() / p)
