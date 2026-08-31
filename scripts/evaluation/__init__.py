"""Evaluation-dataset workspace management.

The package intentionally depends only on the Python standard library so it
can be called from the annotation editor after every save.
"""

from .eval_dataset_status import refresh_after_annotation

__all__ = ["refresh_after_annotation"]
