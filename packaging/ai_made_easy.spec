# PyInstaller spec — desktop bundle of the AI Made Easy designer.
# Build:  pyinstaller packaging/ai_made_easy.spec --noconfirm
# Training/export runs use a real Python environment (see python_executable()).
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent
icon = ROOT / "ai_made_easy" / "assets" / ("icon.icns" if sys.platform == "darwin" else "icon.png")

datas = collect_data_files("ai_made_easy", includes=["samples/*.json", "assets/*",
                                                     "worker/*.py"])
datas += collect_data_files("OdenGraphQt")
hidden = collect_submodules("ai_made_easy") + collect_submodules("OdenGraphQt")

a = Analysis(
    [str(ROOT / "ai_made_easy" / "__main__.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=hidden,
    excludes=["torch", "tensorflow", "keras", "jax", "transformers", "sklearn", "xgboost",
              "lightgbm", "catboost", "pandas", "tkinter", "matplotlib"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="AI Made Easy", console=False,
          icon=str(icon))
coll = COLLECT(exe, a.binaries, a.datas, name="AI Made Easy")
if sys.platform == "darwin":
    app = BUNDLE(coll, name="AI Made Easy.app", icon=str(icon),
                 bundle_identifier="dev.aimadeeasy.designer",
                 info_plist={"NSHighResolutionCapable": True,
                             "CFBundleShortVersionString": "1.0.0"})
