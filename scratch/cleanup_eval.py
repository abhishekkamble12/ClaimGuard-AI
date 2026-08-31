import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
eval_dir = ROOT / "eval"
if eval_dir.exists():
    shutil.rmtree(eval_dir, ignore_errors=True)
    print("Cleaned up legacy eval/ directory.")
