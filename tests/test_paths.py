"""资源路径定位测试(不依赖 torch / pygame)。

重点覆盖:相对路径的绝对化、候选根目录探测优先级、ANTIFIVE_HOME 覆盖。
"""

from __future__ import annotations

import os
from pathlib import Path

from antifive import paths


def test_app_base_is_repo_root():
    base = paths.app_base()
    assert (base / "pyproject.toml").is_file()


def test_resolve_existing_relative_path():
    p = paths.resolve("models/README.md")
    assert os.path.isabs(p) and os.path.exists(p)


def test_resolve_missing_relative_falls_back_to_base():
    p = Path(paths.resolve("no_such_dir/no_such_file.xyz"))
    assert p.is_absolute()
    assert p == paths.app_base() / "no_such_dir" / "no_such_file.xyz"


def test_resolve_prefers_existing_candidate(tmp_path, monkeypatch):
    """相对路径在多个根目录下都存在时,按 roots() 优先级取第一个。"""
    home = tmp_path / "home"
    target = home / "models" / "katago" / "x.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"bin")
    monkeypatch.setenv("ANTIFIVE_HOME", str(home))
    resolved = Path(paths.resolve("models/katago/x.bin"))
    assert resolved.resolve() == target.resolve()


def test_roots_env_first(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("ANTIFIVE_HOME", str(home))
    assert paths.roots()[0] == home.resolve()
