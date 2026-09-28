"""Train the PyTorch DSen2-CR implementation on SEN12MS-CR triplets."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dsen2cr_training_cli import main


if __name__ == "__main__":
    main()
