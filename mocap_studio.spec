# -*- mode: python ; coding: utf-8 -*-
"""Сборка MoCap Studio в один .exe (Windows/Linux)."""
import os
import sys

block_cipher = None
# ресурс версии — только для Windows (на других ОС PyInstaller его не принимает)
version_file = "version_info.txt" if sys.platform == "win32" else None

a = Analysis(
    ["run_gui.py"],
    pathex=[],
    binaries=[],
    datas=[("models", "models")],
    hiddenimports=[
        "mediapipe.tasks.python.vision.pose_landmarker",
        "mediapipe.tasks.python.vision.hand_landmarker",
        "scipy.signal",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "notebook", "IPython"],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="MoCapStudio",
    version=version_file,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
