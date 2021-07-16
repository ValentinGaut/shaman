from pathlib import Path
import dill

CURRENT_DIR = Path(__file__).parent.resolve()
MODEL_PATH = (CURRENT_DIR / "models" / "model_linux.dill")

with open(MODEL_PATH, "rb") as f:
    model = dill.load(f)
