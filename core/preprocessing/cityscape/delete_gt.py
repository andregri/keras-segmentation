import os
import sys
from pathlib import Path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Usage: main.py mask_path")

    gt_dir_path = Path(sys.argv[1])

    mask_path_gen = (path for path in gt_dir_path.rglob("*"))

    for p in mask_path_gen:
        if "labelIds" not in p.stem and not p.is_dir():
            os.remove(p)
