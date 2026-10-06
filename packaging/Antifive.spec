# -*- mode: python ; coding: utf-8 -*-
# 无 torch 分发版构建配置。
# 用法(仓库根目录): pyinstaller packaging/Antifive.spec --noconfirm
# 产物: dist/Antifive/ ,将 models/、katago/ 复制到可执行文件旁即可使用。

import os

ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))  # noqa: F821
SRC = os.path.join(ROOT, "src")
CONFIG = os.path.join(ROOT, "config")

a = Analysis(
    [os.path.join(SPECPATH, "launcher.py")],  # noqa: F821
    pathex=[SRC],
    binaries=[],
    datas=[
        (os.path.join(CONFIG, "engines.json"), "config"),
        (os.path.join(CONFIG, "gtp_test.cfg"), "config"),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "torchvision", "torchaudio"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Antifive",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Antifive",
)
