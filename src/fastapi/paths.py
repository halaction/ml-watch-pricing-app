from pathlib import Path


SOURCE_DIR = Path(__file__).absolute().parent
BASE_DIR = SOURCE_DIR.parent

CACHE_DIR = SOURCE_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = CACHE_DIR / "data"
MODEL_DIR = CACHE_DIR / "model"
