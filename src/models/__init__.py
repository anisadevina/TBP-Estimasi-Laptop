# src/models/__init__.py
# Ekspor fungsi train dari ketiga model agar mudah di-import orchestrator.

from .linear_regression import train as train_linear
from .decision_tree     import train as train_dt
from .random_forest     import train as train_rf

__all__ = ["train_linear", "train_dt", "train_rf"]
