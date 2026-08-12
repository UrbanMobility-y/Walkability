import os
import sys
from pathlib import Path

# Keep BLAS/OpenMP libraries from leaving persistent worker pools in small CI/test runs.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
