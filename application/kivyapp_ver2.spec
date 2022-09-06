# -*- mode: python ; coding: utf-8 -*-
from kivy_deps import sdl2, glew

block_cipher = None
excluded_modules = ['torch.distributions']

a = Analysis(
    ['kivyapp_ver2.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=['win32file', 'win32timezone'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_modules,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

a.datas+=[('Code\kivyapp_ver2.kv','C:\\Users\\DLPC\Desktop\\kivy\\application\kivyapp_ver2.kv','DATA')]
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='kivyapp_ver2',
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
coll = COLLECT(
    exe,Tree('C:\\Users\\DLPC\\Desktop\\kivy\\application'),
    a.binaries,
    a.zipfiles,
    a.datas,
    *[Tree(p) for p in (sdl2.dep_bins+glew.dep_bins)],
    strip=False,
    upx=True,
    upx_exclude=[],
    name='kivyapp_ver2',
)
