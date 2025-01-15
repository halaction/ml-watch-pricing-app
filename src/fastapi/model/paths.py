from pathlib import Path


MODULE_DIR = Path(__file__).absolute().parent
BASE_DIR = MODULE_DIR.parent

CACHE_DIR = BASE_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
