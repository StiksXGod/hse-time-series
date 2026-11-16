"""Find or download the original UCI Online Retail dataset."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retail_ts.data import file_sha256, locate_dataset

if __name__ == "__main__":
    path = locate_dataset(ROOT)
    print(f"Dataset: {path.relative_to(ROOT)}")
    print(f"SHA256: {file_sha256(path)}")
