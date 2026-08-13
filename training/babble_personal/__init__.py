"""Local training tooling for the personalized Baballonia face adapter.

Python is a development-time dependency only. The Baballonia runtime loads the exported ONNX file
through the ONNX Runtime it already ships and never invokes Python.
"""

from . import schema  # noqa: F401  (re-exported for convenience)

__all__ = ["schema"]
